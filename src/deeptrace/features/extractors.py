"""Concrete feature extractor implementations.

This module contains specific implementations of the FeatureExtractor
base class for various pre-trained models including ISC21, SSCD, and DINOv3.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from loguru import logger
from PIL import Image

from .extractor import FeatureExtractor


class DINOv3OutputMode(str, Enum):
    """DINOv3 feature extraction output mode."""
    CLS = "cls"
    PATCH = "patch"
    BOTH = "both"


class ISC21FeatureExtractor(FeatureExtractor):
    """ISC21 Feature Extractor.

    Extracts features using the Facebook AI Image Similarity Challenge 2021
    winning model. This model excels at image retrieval and copy detection tasks.

    Reference: https://github.com/lyakaap/ISC21-Descriptor-Track-1st
    """

    def __init__(
        self,
        weight_name: str = "isc_ft_v107",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        num_workers: int = 4,
        use_amp: bool = False,
    ):
        """Initialize ISC21 feature extractor.

        Args:
            weight_name: Model weight name, options:
                - 'isc_ft_v107' (recommended, 256-dim)
                - 'isc_ft_v110' (512-dim)
            device: Device to run on ('cuda' or 'cpu')
            batch_size: Batch size for processing
            num_workers: Number of data loader worker processes
            use_amp: Whether to use automatic mixed precision (CUDA only)
        """
        super().__init__(device, batch_size, num_workers)

        # Import ISC21 library
        try:
            from isc_feature_extractor import create_model
        except ImportError:
            raise ImportError(
                "isc_feature_extractor not found. Please install:\n"
                "pip install git+https://github.com/lyakaap/ISC21-Descriptor-Track-1st"
            )

        logger.info(f"Loading ISC21 model: {weight_name}")
        self.model, self.preprocessor = create_model(
            weight_name=weight_name,
            device=device,
        )
        self.model.eval()
        self.use_amp = use_amp and (device == "cuda")

        # Determine feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_output = self.model(dummy_input)
            self._feature_dim = dummy_output.shape[1]

        logger.info(f"ISC21 model loaded successfully, feature dimension: {self._feature_dim}")

    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self._feature_dim

    def get_preprocessor(self) -> Any:
        """Get preprocessing function."""
        return self.preprocessor

    @torch.no_grad()
    def extract_features(self, images: Union[list[Image.Image], torch.Tensor]) -> np.ndarray:
        """Extract image features.

        Args:
            images: List of PIL images or preprocessed Tensor

        Returns:
            L2-normalized feature array of shape (N, D)
        """
        # Preprocess if PIL image list
        if isinstance(images, list):
            images = torch.stack([self.preprocessor(img) for img in images])

        # Move to device
        images = images.to(self.device)

        # Extract features with AMP support
        if self.use_amp:
            with torch.cuda.amp.autocast():
                features = self.model(images)
        else:
            features = self.model(images)

        # Convert to numpy and normalize
        features = features.cpu().numpy()
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

        return features


class SSCDFeatureExtractor(FeatureExtractor):
    """SSCD (Self-Supervised Copy Detection) Feature Extractor.

    Uses Facebook Research's SSCD model to extract image copy detection features.
    This model is based on self-supervised contrastive learning and performs
    excellently on image copy detection tasks.

    Reference: https://github.com/facebookresearch/sscd-copy-detection
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        num_workers: int = 4,
        use_amp: bool = False,
        resize_size: int = 288,
        resize_mode: str = "small_edge",
    ):
        """Initialize SSCD feature extractor.

        Args:
            model_path: Path to TorchScript model file. If None, uses default path.
                Recommended models:
                - sscd_disc_mixup.torchscript.pt (ResNet50, 512-dim, recommended)
                - sscd_disc_large.torchscript.pt (ResNeXt101, 1024-dim, higher accuracy)
            device: Device to run on ('cuda' or 'cpu')
            batch_size: Batch size for processing
            num_workers: Number of data loader worker processes
            use_amp: Whether to use automatic mixed precision (CUDA only)
            resize_size: Image preprocessing size
            resize_mode: Resize mode:
                - 'small_edge': Preserve aspect ratio, resize short edge to resize_size (recommended)
                - 'square': Force to square (resize_size, resize_size)
        """
        super().__init__(device, batch_size, num_workers)

        # Determine model path
        if model_path is None:
            # Default: use models/sscd_disc_mixup.torchscript.pt in project root
            from deeptrace.utils import get_project_root
            model_path = get_project_root() / "models" / "sscd_disc_mixup.torchscript.pt"
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"SSCD model file not found: {model_path}\n"
                f"Please download TorchScript model from:\n"
                f"  - sscd_disc_mixup: https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt\n"
                f"  - sscd_disc_large: https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt"
            )

        logger.info(f"Loading SSCD TorchScript model: {model_path}")
        self.model = torch.jit.load(str(model_path), map_location=device)
        self.model.eval()
        self.use_amp = use_amp and (device == "cuda")

        # Create preprocessor
        from torchvision import transforms

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if resize_mode == "small_edge":
            # Preserve aspect ratio, resize short edge to specified size
            self.preprocessor = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                normalize,
            ])
        elif resize_mode == "square":
            # Force to square (not preserving aspect ratio, recommended by SSCD for better batching efficiency)
            self.preprocessor = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            raise ValueError(f"Unknown resize_mode: {resize_mode}")

        # Determine feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, resize_size, resize_size).to(device)
            dummy_output = self.model(dummy_input)
            self._feature_dim = dummy_output.shape[1]

        logger.info(f"SSCD model loaded successfully")
        logger.info(f"  - Feature dimension: {self._feature_dim}")
        logger.info(f"  - Resize size: {resize_size} ({resize_mode})")
        logger.info(f"  - Device: {device}")

    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self._feature_dim

    def get_preprocessor(self) -> Any:
        """Get preprocessing function."""
        return self.preprocessor

    @torch.no_grad()
    def extract_features(self, images: Union[list[Image.Image], torch.Tensor]) -> np.ndarray:
        """Extract image features.

        Args:
            images: List of PIL images or preprocessed Tensor

        Returns:
            L2-normalized feature array of shape (N, D)
            Note: SSCD output features are already L2-normalized
        """
        # Preprocess if PIL image list
        if isinstance(images, list):
            images = torch.stack([self.preprocessor(img) for img in images])

        # Move to device
        images = images.to(self.device)

        # Extract features with AMP support
        if self.use_amp:
            with torch.cuda.amp.autocast():
                features = self.model(images)
        else:
            features = self.model(images)

        # Convert to numpy
        # Note: SSCD TorchScript model already outputs L2-normalized features
        features = features.cpu().numpy()

        # Ensure normalization (just in case)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

        return features


class DINOv3FeatureExtractor(FeatureExtractor):
    """DINOv3 Feature Extractor.

    Uses Facebook Research's DINOv3 model via Hugging Face transformers.
    DINOv3 is a self-supervised Vision Transformer that performs excellently
    across various vision tasks.

    Reference: https://github.com/facebookresearch/dinov3
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        num_workers: int = 4,
        use_amp: bool = False,
        output_mode: Union[str, DINOv3OutputMode] = DINOv3OutputMode.CLS,
    ):
        """Initialize DINOv3 feature extractor.

        Args:
            model_name_or_path: Hugging Face model name or local path, options:
                - 'facebook/dinov3-vits16-pretrain-lvd1689m' (ViT-S/16, 384-dim, fastest)
                - 'facebook/dinov3-vitb16-pretrain-lvd1689m' (ViT-B/16, 768-dim, balanced)
                - 'facebook/dinov3-vitl16-pretrain-lvd1689m' (ViT-L/16, 1024-dim, high accuracy)
                - 'facebook/dinov3-vith16plus-pretrain-lvd1689m' (ViT-H/16+, 1280-dim)
                - 'facebook/dinov3-vit7b16-pretrain-lvd1689m' (ViT-g/16, 1536-dim, highest accuracy)
                - Or a local path like 'pretrained/dinov3/dinov3-vitb16'
            device: Device to run on ('cuda' or 'cpu')
            batch_size: Batch size for processing
            num_workers: Number of data loader worker processes
            use_amp: Whether to use automatic mixed precision (CUDA only)
            output_mode: Output mode - 'cls', 'patch', or 'both'
        """
        super().__init__(device, batch_size, num_workers)

        # Validate and store output mode
        if isinstance(output_mode, str):
            output_mode = DINOv3OutputMode(output_mode)
        self.output_mode = output_mode

        from transformers import AutoImageProcessor, AutoModel

        logger.info(f"Loading DINOv3 model: {model_name_or_path}")

        # Load image processor and model from Hugging Face
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)

        # Determine dtype for model
        dtype = torch.float16 if use_amp and device == "cuda" else torch.float32

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            dtype=dtype,
        )
        self.model.to(device)
        self.model.eval()

        self.use_amp = use_amp and (device == "cuda")
        self.model_name_or_path = model_name_or_path

        # Get feature dimension from model config
        self._feature_dim = self.model.config.hidden_size

        # Get number of register tokens
        self.num_register_tokens = self.model.config.num_register_tokens

        logger.info(f"DINOv3 model loaded successfully")
        logger.info(f"  - Model: {model_name_or_path}")
        logger.info(f"  - Feature dimension: {self._feature_dim}")
        logger.info(f"  - Output mode: {self.output_mode.value}")
        logger.info(f"  - Register tokens: {self.num_register_tokens}")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - Dtype: {dtype}")

    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self._feature_dim

    def get_preprocessor(self) -> Any:
        """Get preprocessing function.

        Returns a callable that takes PIL Image and returns preprocessed tensor.
        """
        def preprocess(image: Image.Image) -> torch.Tensor:
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs.pixel_values.squeeze(0)

        return preprocess

    @torch.no_grad()
    def extract_features(self, images: Union[list[Image.Image], torch.Tensor]) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Extract image features.

        Args:
            images: List of PIL images or preprocessed Tensor

        Returns:
            Depending on output_mode:
            - CLS: L2-normalized feature array of shape (N, D)
            - PATCH: L2-normalized feature array of shape (N, num_layers, num_patches, D)
            - BOTH: Tuple of (cls_features, patch_features)
        """
        # Preprocess if PIL image list
        if isinstance(images, list):
            # Use processor for batch processing
            inputs = self.processor(images=images, return_tensors="pt")
            images = inputs.pixel_values

        # Move to device
        images = images.to(self.device)
        if self.use_amp:
            images = images.to(torch.float16)

        # Determine if we need hidden states
        need_hidden_states = self.output_mode in (DINOv3OutputMode.PATCH, DINOv3OutputMode.BOTH)

        # Extract features with AMP support
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                outputs = self.model(pixel_values=images, output_hidden_states=need_hidden_states)
        else:
            outputs = self.model(pixel_values=images, output_hidden_states=need_hidden_states)

        if self.output_mode == DINOv3OutputMode.CLS:
            # CLS token only (current behavior)
            features = outputs.pooler_output
            features = features.cpu().float().numpy()
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            return features

        elif self.output_mode == DINOv3OutputMode.PATCH:
            # Patch tokens from all transformer layers
            hidden_states = outputs.hidden_states  # tuple of 13 tensors

            # Extract patch tokens from layers 1-12 (skip embedding layer at index 0)
            all_layer_patches = []
            for layer_hidden in hidden_states[1:]:  # 12 transformer layers
                # Skip CLS token (index 0) and register tokens (indices 1 to num_register_tokens)
                patches = layer_hidden[:, 1 + self.num_register_tokens:, :]  # (batch, 196, 768)
                all_layer_patches.append(patches)

            # Stack into (batch, num_layers, num_patches, hidden_size)
            patch_features = torch.stack(all_layer_patches, dim=1)
            patch_features = patch_features.cpu().float().numpy()

            # L2 normalize each patch vector (normalize along hidden_size dimension)
            norms = np.linalg.norm(patch_features, axis=-1, keepdims=True)
            patch_features = patch_features / norms

            return patch_features

        else:  # BOTH
            # Extract both CLS and patch tokens
            cls_features = outputs.pooler_output
            cls_features = cls_features.cpu().float().numpy()
            cls_features = cls_features / np.linalg.norm(cls_features, axis=1, keepdims=True)

            # Extract patch tokens (same logic as PATCH mode)
            hidden_states = outputs.hidden_states
            all_layer_patches = []
            for layer_hidden in hidden_states[1:]:
                patches = layer_hidden[:, 1 + self.num_register_tokens:, :]
                all_layer_patches.append(patches)

            patch_features = torch.stack(all_layer_patches, dim=1)
            patch_features = patch_features.cpu().float().numpy()
            norms = np.linalg.norm(patch_features, axis=-1, keepdims=True)
            patch_features = patch_features / norms

            return cls_features, patch_features

    def extract_from_folder(
        self,
        folder_path: Union[str, Path],
        pattern: str = "*.jpg",
        save_path: Optional[Union[str, Path]] = None,
        numeric_sort: bool = True,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Extract features from all images in a folder.

        For BOTH mode, saves to separate _cls.npy and _patch.npy files.

        Args:
            folder_path: Path to image folder
            pattern: File matching pattern (e.g., '*.png', '*.jpg')
            save_path: Optional path to save .npy file(s)
            numeric_sort: Whether to perform natural numeric sorting

        Returns:
            For CLS/PATCH: Feature array
            For BOTH: Tuple of (cls_features, patch_features)
        """
        if self.output_mode != DINOv3OutputMode.BOTH:
            # Use base class implementation for CLS and PATCH modes
            return super().extract_from_folder(folder_path, pattern, save_path, numeric_sort)

        # Custom handling for BOTH mode
        folder_path = Path(folder_path)

        # Get all image files and sort them (same logic as base class)
        image_files = list(folder_path.glob(pattern))
        if numeric_sort:
            import re
            def _nat_key(p: Path):
                name = p.stem
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
            return np.array([]), np.array([])

        logger.info(f"Found {len(image_files)} frame files in {folder_path}")

        # Create dataset and dataloader
        from .extractor import ImageFileDataset
        from torch.utils.data import DataLoader

        dataset = ImageFileDataset(image_files, self.get_preprocessor())
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )

        # Batch extract features
        all_cls_features = []
        all_patch_features = []

        for batch_images in dataloader:
            cls_batch, patch_batch = self.extract_features(batch_images)
            all_cls_features.append(cls_batch)
            all_patch_features.append(patch_batch)

        # Concatenate all features
        cls_features = np.vstack(all_cls_features)
        patch_features = np.vstack(all_patch_features)

        logger.info(f"Extraction complete, CLS shape: {cls_features.shape}, Patch shape: {patch_features.shape}")

        # Save features
        if save_path is not None:
            save_path = Path(save_path)
            # Remove .npy extension if present to add suffixes
            base_path = save_path.with_suffix('')

            cls_path = base_path.parent / f"{base_path.name}_cls.npy"
            patch_path = base_path.parent / f"{base_path.name}_patch.npy"

            save_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(cls_path, cls_features)
            np.save(patch_path, patch_features)

            logger.info(f"CLS features saved to {cls_path}")
            logger.info(f"Patch features saved to {patch_path}")

        return cls_features, patch_features
