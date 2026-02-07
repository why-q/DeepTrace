"""TraceDINO contrastive learning dataset."""

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import (
    FaceBlurAugmentation,
    HumanCropAugmentation,
    JPEGCompression,
    TraceDINOTransform,
)
from .frame_extractor import VideoFrameExtractor
from .metadata import VideoMetadata, load_metadata


class TraceDINODataset(Dataset):
    """
    Contrastive learning dataset for TraceDINO training.

    For each query video:
    - Anchors: 2 uniformly sampled frames from the deepfake video
    - Positives (3 types per anchor):
        1. Original source frame (time-aligned)
        2. Cropped variant (human detection + random crop)
        3. Face-blurred variant (face detection + Gaussian blur)
    - Hard negatives: 3 frames from same source video, outside 15-second radius
    - Batch negatives: Other samples in the batch

    Args:
        metadata_csv: Path to CSV file (train/valid/test.csv)
        query_video_dir: Directory containing query videos
        source_frame_dir: Directory containing source frames
        transform: Image transformation pipeline
        n_anchor_frames: Number of anchor frames per video
        n_hard_negatives: Number of hard negative samples
        safety_radius_sec: Safety radius for hard negative sampling
        is_training: Whether in training mode (applies augmentations)
    """

    def __init__(
        self,
        metadata_csv: Path,
        query_video_dir: Path,
        source_frame_dir: Path,
        transform: TraceDINOTransform,
        n_anchor_frames: int = 2,
        n_hard_negatives: int = 3,
        safety_radius_sec: float = 15.0,
        is_training: bool = True,
    ):
        self.metadata = load_metadata(metadata_csv)
        self.query_video_dir = Path(query_video_dir)
        self.source_frame_dir = Path(source_frame_dir)
        self.transform = transform
        self.n_anchor_frames = n_anchor_frames
        self.n_hard_negatives = n_hard_negatives
        self.safety_radius_sec = safety_radius_sec
        self.is_training = is_training

        # Initialize extractors and augmentations
        self.frame_extractor = VideoFrameExtractor()
        self.face_blur = FaceBlurAugmentation() if is_training else None
        self.human_crop = HumanCropAugmentation() if is_training else None
        self.jpeg_compress = JPEGCompression() if is_training else None

    def __len__(self) -> int:
        return len(self.metadata) * self.n_anchor_frames

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample consisting of anchor + positives + hard negatives.

        Returns:
            Dict with:
                - 'images': Tensor [7, C, H, W] (1 anchor + 3 positives + 3 negatives)
                - 'labels': Tensor [7] (positive samples have same label)
        """
        # Map flat index to (video_idx, anchor_idx)
        video_idx = idx // self.n_anchor_frames
        anchor_idx = idx % self.n_anchor_frames
        meta = self.metadata[video_idx]

        # Extract anchor frame from query video
        anchor_frame = self._get_anchor_frame(meta, anchor_idx)

        # Extract positives (3 types)
        positives = self._get_positives(meta, anchor_idx)

        # Extract hard negatives (3 frames)
        hard_negatives = self._get_hard_negatives(meta, anchor_idx)

        # Stack all images
        all_images = [anchor_frame] + positives + hard_negatives

        # Apply transforms
        all_tensors = [self.transform(img) for img in all_images]
        images = torch.stack(all_tensors)  # [7, C, H, W]

        # Create labels: anchor and positives share label 0, negatives have unique labels
        labels = torch.tensor([0, 0, 0, 0, 1, 2, 3], dtype=torch.long)

        return {"images": images, "labels": labels}

    def _get_anchor_frame(self, meta: VideoMetadata, anchor_idx: int) -> np.ndarray:
        """Extract anchor frame from query video using boundary sampling.

        Samples frames from the first and last second of the video to ensure
        significant temporal distance between anchor frames.
        """
        video_path = self.query_video_dir / f"{meta.id}.mp4"
        frames = self.frame_extractor.extract_boundary_frames(
            video_path, self.n_anchor_frames, boundary_seconds=1.0
        )
        if anchor_idx < len(frames):
            return frames[anchor_idx]
        # Fallback to first frame or placeholder
        if frames:
            return frames[0]
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def _get_positives(self, meta: VideoMetadata, anchor_idx: int) -> List[np.ndarray]:
        """Generate 3 positive samples: original, cropped, face-blurred."""
        # Get time-aligned source frame
        source_frame_no = self._map_anchor_to_source(meta, anchor_idx)
        source_frame = self.frame_extractor.get_source_frame(
            self.source_frame_dir, meta.origin_id, source_frame_no
        )

        if source_frame is None:
            # Fallback: use random frame from source
            source_frame = self._get_random_source_frame(meta)

        positives = []

        # 1. Original source frame (with optional JPEG compression)
        if self.is_training and self.jpeg_compress:
            pos1 = self.jpeg_compress(source_frame)
        else:
            pos1 = source_frame.copy()
        positives.append(pos1)

        # 2. Cropped variant
        if self.is_training and self.human_crop:
            pos2 = self.human_crop(source_frame)
            pos2 = self.jpeg_compress(pos2) if self.jpeg_compress else pos2
        else:
            pos2 = source_frame.copy()
        positives.append(pos2)

        # 3. Face-blurred variant
        if self.is_training and self.face_blur:
            pos3 = self.face_blur(source_frame)
            pos3 = self.jpeg_compress(pos3) if self.jpeg_compress else pos3
        else:
            pos3 = source_frame.copy()
        positives.append(pos3)

        return positives

    def _get_hard_negatives(self, meta: VideoMetadata, anchor_idx: int) -> List[np.ndarray]:
        """Sample hard negatives from same source video, outside safety radius."""
        anchor_source_frame = self._map_anchor_to_source(meta, anchor_idx)
        safety_radius_frames = int(self.safety_radius_sec * meta.fps)

        # Get available source frames
        source_dir = self.source_frame_dir / meta.origin_id
        available_frames = sorted([int(f.stem) for f in source_dir.glob("*.jpg")])

        # Handle empty source directory
        if len(available_frames) == 0:
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * self.n_hard_negatives

        # Filter out frames within safety radius
        valid_frames = [
            f
            for f in available_frames
            if abs(f - anchor_source_frame) > safety_radius_frames
        ]

        # If no valid frames outside safety radius, use all available frames
        if len(valid_frames) == 0:
            valid_frames = available_frames

        # Sample n_hard_negatives frames
        if len(valid_frames) < self.n_hard_negatives:
            # Not enough frames, sample with replacement
            selected_frames = random.choices(valid_frames, k=self.n_hard_negatives)
        else:
            selected_frames = random.sample(valid_frames, self.n_hard_negatives)

        # Load frames
        hard_negatives = []
        for frame_no in selected_frames:
            frame = self.frame_extractor.get_source_frame(
                self.source_frame_dir, meta.origin_id, frame_no
            )
            if frame is not None:
                hard_negatives.append(frame)

        # Ensure we have exactly n_hard_negatives
        while len(hard_negatives) < self.n_hard_negatives:
            hard_negatives.append(hard_negatives[0].copy() if hard_negatives else self._get_random_source_frame(meta))

        return hard_negatives[:self.n_hard_negatives]

    def _map_anchor_to_source(self, meta: VideoMetadata, anchor_idx: int) -> int:
        """Map anchor frame index to source frame number."""
        # Linearly map anchor index to source frame range
        ratio = anchor_idx / max(1, self.n_anchor_frames - 1)
        source_frame_no = int(meta.gt_start_f + ratio * (meta.gt_end_f - meta.gt_start_f))
        return source_frame_no

    def _get_random_source_frame(self, meta: VideoMetadata) -> np.ndarray:
        """Get a random frame from source video as fallback."""
        source_dir = self.source_frame_dir / meta.origin_id
        frames = list(source_dir.glob("*.jpg"))
        if frames:
            return self.frame_extractor.get_source_frame(
                self.source_frame_dir, meta.origin_id, int(random.choice(frames).stem)
            )
        # Fallback to black image
        return np.zeros((224, 224, 3), dtype=np.uint8)
