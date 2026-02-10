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
        origin_id: Source video ID (32-char hex string, UUID)
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
        gt_start_img: Ground truth start image index (1-based)
        gt_end_img: Ground truth end image index (1-based)
        frame_count: Total frame count after extraction
        output_fps: Output FPS after extraction
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
    gt_start_img: int
    gt_end_img: int
    frame_count: int
    output_fps: float

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
            origin_id=str(row["origin_id"]),  # 32-char hex string
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
            gt_start_img=int(row["gt_start_img"]),
            gt_end_img=int(row["gt_end_img"]),
            frame_count=int(row["frame_count"]),
            output_fps=float(row["output_fps"]),
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
        origin_id = str(row["id"])
        similar_ids = [s.strip() for s in row["sim"].split("-")]
        similar_map[origin_id] = similar_ids

    return similar_map
