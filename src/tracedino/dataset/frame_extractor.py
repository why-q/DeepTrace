"""Video frame extraction utilities with caching support."""

import hashlib
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


class VideoFrameExtractor:
    """
    Video frame extractor with disk caching support.

    Provides efficient frame extraction from MP4 videos with optional
    disk caching to avoid repeated extraction.

    Args:
        cache_dir: Directory for caching extracted frames (optional)
        use_cache: Whether to use caching
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_cache: bool = False,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache

        if self.use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, video_path: Path, frame_idx: int) -> Path:
        """
        Generate cache file path for a specific frame.

        Args:
            video_path: Path to video file
            frame_idx: Frame index

        Returns:
            Path to cache file
        """
        # Create unique cache key from video path and frame index
        key = f"{video_path.stem}_{frame_idx}"
        hash_key = hashlib.md5(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{hash_key}.jpg"

    def extract_frame(
        self,
        video_path: Path,
        frame_idx: int,
    ) -> Optional[np.ndarray]:
        """
        Extract a single frame from video.

        Args:
            video_path: Path to video file
            frame_idx: Frame index to extract (0-indexed)

        Returns:
            Frame as numpy array [H, W, C] in BGR format, or None if failed
        """
        # Check cache first
        if self.use_cache and self.cache_dir:
            cache_path = self._get_cache_path(video_path, frame_idx)
            if cache_path.exists():
                return cv2.imread(str(cache_path))

        # Extract frame from video
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None

        # Save to cache
        if self.use_cache and self.cache_dir:
            cache_path = self._get_cache_path(video_path, frame_idx)
            cv2.imwrite(str(cache_path), frame)

        return frame

    def extract_uniform_frames(
        self,
        video_path: Path,
        n_frames: int = 2,
    ) -> List[np.ndarray]:
        """
        Extract uniformly sampled frames from video.

        Args:
            video_path: Path to video file
            n_frames: Number of frames to extract

        Returns:
            List of frames as numpy arrays [H, W, C] in BGR format
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames == 0:
            return []

        # Calculate uniform frame indices
        if n_frames == 1:
            frame_indices = [total_frames // 2]
        else:
            frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

        # Extract frames
        frames = []
        for frame_idx in frame_indices:
            frame = self.extract_frame(video_path, frame_idx)
            if frame is not None:
                frames.append(frame)

        return frames

    def get_source_frame(
        self,
        source_frame_dir: Path,
        origin_id: str,
        frame_no: int,
    ) -> Optional[np.ndarray]:
        """
        Load a frame from the source frame directory.

        The source frames are pre-extracted and stored in:
        {source_frame_dir}/{origin_id}/{frame_no:04d}.jpg

        Args:
            source_frame_dir: Root directory containing source frames
            origin_id: Source video ID (4-digit zero-padded)
            frame_no: Frame number

        Returns:
            Frame as numpy array [H, W, C] in BGR format, or None if not found
        """
        frame_path = source_frame_dir / origin_id / f"{frame_no:04d}.jpg"

        if not frame_path.exists():
            return None

        return cv2.imread(str(frame_path))

    def get_video_info(self, video_path: Path) -> dict:
        """
        Get basic video information.

        Args:
            video_path: Path to video file

        Returns:
            Dict with 'fps', 'frame_count', 'width', 'height', 'duration'
        """
        cap = cv2.VideoCapture(str(video_path))

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }

        if info["fps"] > 0:
            info["duration"] = info["frame_count"] / info["fps"]
        else:
            info["duration"] = 0.0

        cap.release()

        return info

    def extract_boundary_frames(
        self,
        video_path: Path,
        n_frames: int = 2,
        boundary_seconds: float = 1.0,
    ) -> List[np.ndarray]:
        """
        Extract frames from the first and last boundary regions of a video.

        Randomly samples frames from the first `boundary_seconds` and last
        `boundary_seconds` of the video. This ensures sampled frames have
        significant temporal distance for more meaningful training.

        Args:
            video_path: Path to video file
            n_frames: Number of frames to extract (must be even, half from each boundary)
            boundary_seconds: Duration of boundary region in seconds (default: 1.0)

        Returns:
            List of frames as numpy arrays [H, W, C] in BGR format
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames == 0 or fps == 0:
            return []

        # Calculate boundary region in frames
        boundary_frames_count = int(boundary_seconds * fps)
        # Ensure boundary doesn't exceed half of video length
        boundary_frames_count = max(1, min(boundary_frames_count, total_frames // 2))

        frames_per_boundary = n_frames // 2

        frame_indices = []

        # Sample from first boundary (first second)
        for _ in range(frames_per_boundary):
            first_idx = random.randint(0, boundary_frames_count - 1)
            frame_indices.append(first_idx)

        # Sample from last boundary (last second)
        last_boundary_start = max(0, total_frames - boundary_frames_count)
        for _ in range(frames_per_boundary):
            last_idx = random.randint(last_boundary_start, total_frames - 1)
            frame_indices.append(last_idx)

        # Extract frames
        frames = []
        for frame_idx in frame_indices:
            frame = self.extract_frame(video_path, frame_idx)
            if frame is not None:
                frames.append(frame)

        return frames
