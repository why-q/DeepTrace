"""Preprocessed dataset for TraceDINO training.

This dataset reads from preprocessed data directory containing:
- anchor frames (anchor_0.jpg, anchor_1.jpg)
- hard negative frames (negative_00.jpg, ..., negative_19.jpg)
- metadata with detection bboxes (meta.json)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import JPEGCompression, TraceDINOTransform


class PreprocessedTraceDINODataset(Dataset):
    """
    Dataset for TraceDINO training using preprocessed data.

    For each sample:
    - Anchor: loaded from preprocessed anchor_X.jpg
    - Positives (3 types):
        1. Original source frame (from vorpus_f)
        2. Cropped variant (using saved human_bbox)
        3. Face-blurred variant (using saved face_bbox)
    - Hard negatives: randomly selected from preprocessed negative_XX.jpg

    Args:
        preprocessed_dir: Directory containing preprocessed data for this split
        source_frame_dir: Directory containing source frames (vorpus_f)
        transform: Image transformation pipeline
        n_anchor_frames: Number of anchor frames per video
        n_hard_negatives: Number of hard negatives to sample per anchor
        is_training: Whether in training mode (applies augmentations)
        blur_sigma: Gaussian blur sigma for face blurring
        crop_ratio_range: Crop ratio range for human cropping
    """

    def __init__(
        self,
        preprocessed_dir: Path,
        source_frame_dir: Path,
        transform: TraceDINOTransform,
        n_anchor_frames: int = 2,
        n_hard_negatives: int = 3,
        is_training: bool = True,
        blur_sigma: float = 40.0,
        crop_ratio_range: Tuple[float, float] = (0.25, 0.75),
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.source_frame_dir = Path(source_frame_dir)
        self.transform = transform
        self.n_anchor_frames = n_anchor_frames
        self.n_hard_negatives = n_hard_negatives
        self.is_training = is_training
        self.blur_sigma = blur_sigma
        self.crop_ratio_range = crop_ratio_range

        # JPEG compression for training augmentation
        self.jpeg_compress = JPEGCompression() if is_training else None

        # Load all video metadata and filter out invalid samples
        self.video_dirs = []
        skipped = 0
        for d in sorted(self.preprocessed_dir.iterdir()):
            if not d.is_dir() or not (d / "meta.json").exists():
                continue
            if self._is_valid_sample(d):
                self.video_dirs.append(d)
            else:
                skipped += 1

        if skipped > 0:
            print(f"Skipped {skipped} samples with out-of-range frame numbers")

        # Cache metadata
        self._metadata_cache: Dict[str, dict] = {}

    def _is_valid_sample(self, video_dir: Path) -> bool:
        """Check if sample has valid source frame numbers."""
        try:
            with open(video_dir / "meta.json") as f:
                meta = json.load(f)

            origin_id = meta["origin_id"]
            vorpus_dir = self.source_frame_dir / origin_id

            if not vorpus_dir.exists():
                return False

            # Get max available frame number
            frame_files = list(vorpus_dir.glob("*.jpg"))
            if not frame_files:
                return False
            max_frame = max(int(f.stem) for f in frame_files)

            # Check all anchor source frames are in range
            for anchor in meta["anchors"]:
                if anchor["source_frame_no"] > max_frame:
                    return False

            return True
        except Exception:
            return False

    def __len__(self) -> int:
        return len(self.video_dirs) * self.n_anchor_frames

    def _load_metadata(self, video_dir: Path) -> dict:
        """Load and cache metadata for a video."""
        video_id = video_dir.name
        if video_id not in self._metadata_cache:
            with open(video_dir / "meta.json") as f:
                self._metadata_cache[video_id] = json.load(f)
        return self._metadata_cache[video_id]

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

        video_dir = self.video_dirs[video_idx]
        meta = self._load_metadata(video_dir)

        # Load anchor frame
        anchor_frame = self._load_anchor(video_dir, anchor_idx)

        # Generate positives
        positives = self._get_positives(meta, anchor_idx)

        # Load hard negatives
        hard_negatives = self._get_hard_negatives(video_dir)

        # Stack all images
        all_images = [anchor_frame] + positives + hard_negatives

        # Apply transforms
        all_tensors = [self.transform(img) for img in all_images]
        images = torch.stack(all_tensors)  # [7, C, H, W]

        # Create labels: anchor and positives share label 0, negatives have unique labels
        labels = torch.tensor([0, 0, 0, 0, 1, 2, 3], dtype=torch.long)

        return {"images": images, "labels": labels}

    def _load_anchor(self, video_dir: Path, anchor_idx: int) -> np.ndarray:
        """Load anchor frame from preprocessed directory."""
        anchor_path = video_dir / f"anchor_{anchor_idx}.jpg"
        if anchor_path.exists():
            frame = cv2.imread(str(anchor_path))
            if frame is not None:
                return frame
        # Fallback to black image
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def _get_positives(self, meta: dict, anchor_idx: int) -> List[np.ndarray]:
        """Generate 3 positive samples: original, cropped, face-blurred."""
        anchor_data = meta["anchors"][anchor_idx]
        source_frame_no = anchor_data["source_frame_no"]
        origin_id = meta["origin_id"]

        # Load source frame
        source_frame_path = self.source_frame_dir / origin_id / f"{source_frame_no:04d}.jpg"
        source_frame = cv2.imread(str(source_frame_path))

        if source_frame is None:
            # Fallback to black image
            source_frame = np.zeros((224, 224, 3), dtype=np.uint8)

        # Get detection data
        detections = meta.get("detections", {}).get(str(source_frame_no), {})
        human_bbox = detections.get("human_bbox")
        face_bbox = detections.get("face_bbox")

        positives = []

        # 1. Original source frame (with optional JPEG compression)
        if self.is_training and self.jpeg_compress:
            pos1 = self.jpeg_compress(source_frame)
        else:
            pos1 = source_frame.copy()
        positives.append(pos1)

        # 2. Cropped variant using human bbox
        pos2 = self._apply_human_crop(source_frame, human_bbox)
        if self.is_training and self.jpeg_compress:
            pos2 = self.jpeg_compress(pos2)
        positives.append(pos2)

        # 3. Face-blurred variant using face bbox
        pos3 = self._apply_face_blur(source_frame, face_bbox)
        if self.is_training and self.jpeg_compress:
            pos3 = self.jpeg_compress(pos3)
        positives.append(pos3)

        return positives

    def _apply_human_crop(
        self,
        image: np.ndarray,
        bbox: Optional[List[int]],
    ) -> np.ndarray:
        """Apply human-centered crop using saved bbox."""
        if bbox is None:
            # Fallback to center crop
            return self._center_crop(image)

        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # Random crop ratio
        crop_ratio = random.uniform(*self.crop_ratio_range) if self.is_training else 0.5
        new_h, new_w = int(h * (1 - crop_ratio)), int(w * (1 - crop_ratio))

        # Ensure crop contains the bbox
        max_top = min(y1, h - new_h)
        min_top = max(0, y2 - new_h)
        max_left = min(x1, w - new_w)
        min_left = max(0, x2 - new_w)

        if max_top >= min_top and max_left >= min_left:
            top = random.randint(min_top, max_top) if self.is_training else (min_top + max_top) // 2
            left = random.randint(min_left, max_left) if self.is_training else (min_left + max_left) // 2
        else:
            # Fallback to center crop
            top = (h - new_h) // 2
            left = (w - new_w) // 2

        return image[top:top + new_h, left:left + new_w]

    def _center_crop(self, image: np.ndarray, crop_ratio: float = 0.5) -> np.ndarray:
        """Center crop with given ratio."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * (1 - crop_ratio)), int(w * (1 - crop_ratio))
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return image[top:top + new_h, left:left + new_w]

    def _apply_face_blur(
        self,
        image: np.ndarray,
        bbox: Optional[List[int]],
    ) -> np.ndarray:
        """Apply face blur using saved bbox."""
        if bbox is None:
            # No face detected, return original
            return image.copy()

        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # Add padding (10% of face size)
        face_w, face_h = x2 - x1, y2 - y1
        pad_x, pad_y = int(0.1 * face_w), int(0.1 * face_h)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        # Create blurred copy
        blurred = image.copy()
        roi = blurred[y1:y2, x1:x2]

        if roi.size > 0:
            roi_blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=self.blur_sigma)
            blurred[y1:y2, x1:x2] = roi_blurred

        return blurred

    def _get_hard_negatives(self, video_dir: Path) -> List[np.ndarray]:
        """Load random hard negatives from preprocessed directory."""
        # Find all negative files
        negative_files = sorted(video_dir.glob("negative_*.jpg"))

        if len(negative_files) == 0:
            # Fallback to black images
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * self.n_hard_negatives

        # Randomly select n_hard_negatives
        if len(negative_files) <= self.n_hard_negatives:
            selected_files = negative_files.copy()
            while len(selected_files) < self.n_hard_negatives:
                selected_files.append(random.choice(negative_files))
        else:
            selected_files = random.sample(negative_files, self.n_hard_negatives)

        # Load frames
        hard_negatives = []
        for neg_path in selected_files:
            frame = cv2.imread(str(neg_path))
            if frame is not None:
                hard_negatives.append(frame)
            else:
                hard_negatives.append(np.zeros((224, 224, 3), dtype=np.uint8))

        return hard_negatives
