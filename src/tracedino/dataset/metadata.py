"""Metadata parsing for DeepTrace dataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class VideoMetadata:
    """
    Metadata for a single deepfake video in the DeepTrace dataset.

    Attributes:
        id: Query video UUID (32-char hex string)
        origin_id: Source video ID (4-digit zero-padded string)
        gt_start_f: Ground truth start frame in source video
        gt_end_f: Ground truth end frame in source video
        fps: Frames per second
        frames: Total number of frames in query video
        category: Forgery category (e.g., "face_rvc_textmismatch")
        celebrity: Name of the person in the video
        v_no: Video number identifier
        scene_no: Scene number in source video
        scene_sub_no: Sub-scene number (optional)
        face_no: Face number in frame (optional)
        method: Forgery method (e.g., "simswap", "roop", "infoswap", optional)
    """

    id: str
    origin_id: str
    gt_start_f: int
    gt_end_f: int
    fps: int
    frames: int
    category: str
    celebrity: str
    v_no: int
    scene_no: int
    scene_sub_no: str  # Can be None or empty
    face_no: str  # Can be None or empty
    method: str  # Can be None or empty

    @property
    def duration_seconds(self) -> float:
        """Calculate query video duration in seconds."""
        return self.frames / self.fps

    @property
    def source_duration_frames(self) -> int:
        """Calculate source segment duration in frames."""
        return self.gt_end_f - self.gt_start_f

    @property
    def source_duration_seconds(self) -> float:
        """Calculate source segment duration in seconds."""
        return self.source_duration_frames / self.fps


def load_metadata(csv_path: Path) -> List[VideoMetadata]:
    """
    Load and parse metadata from CSV file.

    Args:
        csv_path: Path to the CSV file (train.csv, valid.csv, or test.csv)

    Returns:
        List of VideoMetadata objects
    """
    df = pd.read_csv(csv_path)

    metadata_list = []
    for _, row in df.iterrows():
        metadata = VideoMetadata(
            id=row["id"],
            origin_id=str(row["origin_id"]).zfill(4),  # Zero-pad to 4 digits
            gt_start_f=int(row["gt_start_f"]),
            gt_end_f=int(row["gt_end_f"]),
            fps=int(row["fps"]),
            frames=int(row["frames"]),
            category=row["category"],
            celebrity=row["celebrity"],
            v_no=int(row["v_no"]),
            scene_no=int(row["scene_no"]),
            scene_sub_no=str(row["scene_sub_no"]) if pd.notna(row["scene_sub_no"]) else "",
            face_no=str(row["face_no"]) if pd.notna(row["face_no"]) else "",
            method=row["method"] if pd.notna(row["method"]) else "",
        )
        metadata_list.append(metadata)

    return metadata_list


def load_similar_videos(csv_path: Path) -> dict:
    """
    Load similar video mapping from similar.csv.

    Args:
        csv_path: Path to similar.csv

    Returns:
        Dict mapping origin_id → list of similar video IDs
    """
    df = pd.read_csv(csv_path)

    similar_map = {}
    for _, row in df.iterrows():
        origin_id = str(row["id"]).zfill(4)
        similar_ids = [s.strip().zfill(4) for s in row["sim"].split("-")]
        similar_map[origin_id] = similar_ids

    return similar_map
