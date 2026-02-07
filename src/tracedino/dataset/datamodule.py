"""Data module for managing train/valid/test datasets."""

from pathlib import Path
from typing import Optional, Union

from torch.utils.data import DataLoader, Dataset

from .augmentations import TraceDINOTransform
from .dataset import TraceDINODataset
from .preprocessed_dataset import PreprocessedTraceDINODataset


class TraceDINODataModule:
    """
    Data module for TraceDINO training and evaluation.

    Manages train, validation, and test datasets with appropriate
    configurations for each split.

    Args:
        train_csv: Path to training CSV
        valid_csv: Path to validation CSV
        test_csv: Path to test CSV
        query_video_dir: Directory containing query videos
        source_frame_dir: Directory containing source frames
        image_size: Input image size
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        num_workers: Number of DataLoader workers
        n_anchor_frames: Number of anchor frames per video
        n_hard_negatives: Number of hard negatives
        safety_radius_sec: Safety radius for hard negative sampling
        use_preprocessed: Whether to use preprocessed dataset
        preprocessed_dir: Directory containing preprocessed data
    """

    def __init__(
        self,
        train_csv: Path,
        valid_csv: Path,
        test_csv: Path,
        query_video_dir: Path,
        source_frame_dir: Path,
        image_size: int = 224,
        batch_size: int = 256,
        eval_batch_size: int = 64,
        num_workers: int = 8,
        n_anchor_frames: int = 2,
        n_hard_negatives: int = 3,
        safety_radius_sec: float = 15.0,
        use_preprocessed: bool = False,
        preprocessed_dir: Optional[Path] = None,
    ):
        self.train_csv = Path(train_csv)
        self.valid_csv = Path(valid_csv)
        self.test_csv = Path(test_csv)
        self.query_video_dir = Path(query_video_dir)
        self.source_frame_dir = Path(source_frame_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.n_anchor_frames = n_anchor_frames
        self.n_hard_negatives = n_hard_negatives
        self.safety_radius_sec = safety_radius_sec
        self.use_preprocessed = use_preprocessed
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None

        # Initialize transforms
        self.train_transform = TraceDINOTransform(image_size=image_size, is_training=True)
        self.eval_transform = TraceDINOTransform(image_size=image_size, is_training=False)

        # Datasets (lazy initialization)
        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

    def _create_dataset(
        self,
        split: str,
        is_training: bool,
    ) -> Union[TraceDINODataset, PreprocessedTraceDINODataset]:
        """Create dataset for the given split."""
        transform = self.train_transform if is_training else self.eval_transform
        csv_path = getattr(self, f"{split}_csv")

        if self.use_preprocessed and self.preprocessed_dir:
            return PreprocessedTraceDINODataset(
                preprocessed_dir=self.preprocessed_dir / split,
                source_frame_dir=self.source_frame_dir,
                transform=transform,
                n_anchor_frames=self.n_anchor_frames,
                n_hard_negatives=self.n_hard_negatives,
                is_training=is_training,
            )
        else:
            return TraceDINODataset(
                metadata_csv=csv_path,
                query_video_dir=self.query_video_dir,
                source_frame_dir=self.source_frame_dir,
                transform=transform,
                n_anchor_frames=self.n_anchor_frames,
                n_hard_negatives=self.n_hard_negatives,
                safety_radius_sec=self.safety_radius_sec,
                is_training=is_training,
            )

    def train_dataset(self) -> Dataset:
        """Get training dataset."""
        if self._train_dataset is None:
            self._train_dataset = self._create_dataset("train", is_training=True)
        return self._train_dataset

    def valid_dataset(self) -> Dataset:
        """Get validation dataset."""
        if self._valid_dataset is None:
            self._valid_dataset = self._create_dataset("valid", is_training=False)
        return self._valid_dataset

    def test_dataset(self) -> Dataset:
        """Get test dataset."""
        if self._test_dataset is None:
            self._test_dataset = self._create_dataset("test", is_training=False)
        return self._test_dataset

    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        return DataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def valid_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        return DataLoader(
            self.valid_dataset(),
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        return DataLoader(
            self.test_dataset(),
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
