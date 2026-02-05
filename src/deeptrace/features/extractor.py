"""Video frame feature extractors.

Provides abstract base class and concrete implementations for efficient
batch extraction of video frame features.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class FeatureExtractor(ABC):
    """Feature extractor abstract base class.

    Defines the common interface for all feature extractors. All concrete
    implementations should inherit from this class.
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """Initialize feature extractor.

        Args:
            device: Device to run on ('cuda' or 'cpu')
            batch_size: Batch size for processing
            num_workers: Number of data loader worker processes
        """
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        logger.info(f"Initializing feature extractor, device: {device}, batch size: {batch_size}")

    @abstractmethod
    def extract_features(self, images: Union[list[Image.Image], torch.Tensor]) -> np.ndarray:
        """Extract image features.

        Args:
            images: List of PIL images or Tensor

        Returns:
            Feature array of shape (N, D), where N is the number of images
            and D is the feature dimension
        """
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get feature dimension.

        Returns:
            Dimension of the feature vector
        """
        pass

    def extract_from_folder(
        self,
        folder_path: Union[str, Path],
        pattern: str = "*.jpg",
        save_path: Optional[Union[str, Path]] = None,
        numeric_sort: bool = True,
    ) -> np.ndarray:
        """Extract features from all images in a folder.

        Images are extracted in sorted order by filename to ensure consistency.

        Args:
            folder_path: Path to image folder
            pattern: File matching pattern (e.g., '*.png', '*.jpg')
            save_path: Optional path to save .npy file
            numeric_sort: Whether to perform natural numeric sorting of filenames
                (e.g., 2 comes before 10)

        Returns:
            Feature array of shape (N, D)
        """
        folder_path = Path(folder_path)

        # Get all image files and sort them
        image_files = list(folder_path.glob(pattern))
        if numeric_sort:
            def _nat_key(p: Path):
                name = p.stem
                # Extract consecutive digit strings from name; if none exist,
                # return original name for stable sorting
                import re
                m = re.search(r"(\d+)", name)
                if m:
                    try:
                        return (int(m.group(1)), name)
                    except Exception:
                        return (float('inf'), name)
                return (float('inf'), name)
            image_files.sort(key=_nat_key)
        else:
            image_files.sort()

        if not image_files:
            logger.warning(f"No files matching {pattern} found in {folder_path}")
            return np.array([])

        logger.info(f"Found {len(image_files)} frame files in {folder_path}")

        # Create dataset and dataloader
        dataset = ImageFileDataset(image_files, self.get_preprocessor())
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle, maintain order
            num_workers=self.num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )

        # Batch extract features
        all_features = []
        for batch_images in dataloader:
            batch_features = self.extract_features(batch_images)
            all_features.append(batch_features)

        # Concatenate all features
        features = np.vstack(all_features)
        logger.info(f"Extraction complete, feature shape: {features.shape}")

        # Save features
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, features)
            logger.info(f"Features saved to {save_path}")

        return features

    @abstractmethod
    def get_preprocessor(self) -> Any:
        """Get preprocessing function/transform.

        Returns:
            Preprocessing function or torchvision.transforms
        """
        pass


class ImageFileDataset(Dataset):
    """Image file dataset for batch loading images.

    A PyTorch Dataset implementation that loads images from file paths
    and applies preprocessing transformations.
    """

    def __init__(self, image_files: list[Path], preprocessor: Any):
        """Initialize dataset.

        Args:
            image_files: List of image file paths
            preprocessor: Preprocessing function
        """
        self.image_files = image_files
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        return self.preprocessor(image)
