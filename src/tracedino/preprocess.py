"""Preprocessing script for TraceDINO dataset.

This script preprocesses the dataset by:
1. Extracting anchor frames from query videos
2. Running face and human detection on source frames
3. Pre-sampling hard negative frames

Usage:
    python -m src.tracedino.preprocess --split train --limit 100
    python -m src.tracedino.preprocess --split train
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.tracedino.config import TraceDINOConfig
from src.tracedino.dataset.metadata import VideoMetadata, load_metadata


class FaceDetector:
    """Face detector using InsightFace."""

    def __init__(self):
        from insightface.app import FaceAnalysis

        self.detector = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.detector.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, image: np.ndarray) -> Optional[List[int]]:
        """Detect face and return bbox [x1, y1, x2, y2] or None."""
        faces = self.detector.get(image)
        if not faces:
            return None

        # Return largest face
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        bbox = largest_face.bbox.astype(int).tolist()
        return bbox


class HumanDetector:
    """Human detector using YOLOv8."""

    def __init__(self, model_path: str = "pretrained/detection/yolov8n.pt"):
        from ultralytics import YOLO

        self.detector = YOLO(model_path)
        self.detector.to("cuda")

    def detect(self, image: np.ndarray) -> Optional[List[int]]:
        """Detect human and return bbox [x1, y1, x2, y2] or None."""
        results = self.detector(image, classes=[0], verbose=False)

        if len(results[0].boxes) == 0:
            return None

        # Return largest person
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_idx = areas.argmax()
        bbox = boxes[largest_idx].astype(int).tolist()
        return bbox


class VideoFrameExtractor:
    """Extract frames from video files."""

    def extract_boundary_frames(
        self,
        video_path: Path,
        n_frames: int = 2,
        boundary_seconds: float = 1.0,
        seed: int = 42,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Extract frames from first and last boundary regions.

        Returns list of (frame_idx, frame) tuples.
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0 or fps == 0:
            cap.release()
            return []

        # Calculate boundary region in frames
        boundary_frames_count = int(boundary_seconds * fps)
        boundary_frames_count = max(1, min(boundary_frames_count, total_frames // 2))

        frames_per_boundary = n_frames // 2

        # Use fixed seed for deterministic sampling
        rng = random.Random(seed)

        frame_indices = []

        # Sample from first boundary
        for _ in range(frames_per_boundary):
            idx = rng.randint(0, boundary_frames_count - 1)
            frame_indices.append(idx)

        # Sample from last boundary
        last_boundary_start = max(0, total_frames - boundary_frames_count)
        for _ in range(frames_per_boundary):
            idx = rng.randint(last_boundary_start, total_frames - 1)
            frame_indices.append(idx)

        # Extract frames
        results = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                results.append((frame_idx, frame))

        cap.release()
        return results


def map_anchor_to_source(meta: VideoMetadata, anchor_idx: int, n_anchors: int = 2) -> int:
    """Map anchor frame index to source frame number in vorpus_f (1 FPS).

    vorpus_f frames are extracted at 1 FPS, so we need to convert from
    original frame number (at original fps) to 1 FPS frame number.
    """
    ratio = anchor_idx / max(1, n_anchors - 1)
    original_frame_no = int(meta.gt_start_f + ratio * (meta.gt_end_f - meta.gt_start_f))
    # Convert to 1 FPS frame number
    vorpus_frame_no = original_frame_no // meta.fps
    return vorpus_frame_no


def get_hard_negative_frames(
    source_frame_dir: Path,
    origin_id: str,
    anchor_source_frames: List[int],
    fps: int,
    safety_radius_sec: float = 15.0,
    n_negatives: int = 20,
    seed: int = 42,
) -> List[int]:
    """
    Sample hard negative frame numbers from source video.

    Args:
        source_frame_dir: Root directory containing source frames
        origin_id: Source video ID
        anchor_source_frames: List of source frame numbers in vorpus_f (1 FPS)
        fps: Original video FPS (not used since vorpus_f is 1 FPS)
        safety_radius_sec: Safety radius in seconds
        n_negatives: Number of negatives to sample
        seed: Random seed

    Returns:
        List of frame numbers for hard negatives (in vorpus_f 1 FPS format)
    """
    source_dir = source_frame_dir / origin_id
    available_frames = sorted([int(f.stem) for f in source_dir.glob("*.jpg")])

    if len(available_frames) == 0:
        return []

    # vorpus_f is at 1 FPS, so safety_radius in frames = safety_radius in seconds
    safety_radius_frames = int(safety_radius_sec)

    # Filter out frames within safety radius of any anchor
    valid_frames = []
    for f in available_frames:
        is_valid = True
        for anchor_frame in anchor_source_frames:
            if abs(f - anchor_frame) <= safety_radius_frames:
                is_valid = False
                break
        if is_valid:
            valid_frames.append(f)

    # If no valid frames, use all available
    if len(valid_frames) == 0:
        valid_frames = available_frames

    # Sample with fixed seed
    rng = random.Random(seed)
    if len(valid_frames) <= n_negatives:
        selected = valid_frames.copy()
        # Sample with replacement if needed
        while len(selected) < n_negatives:
            selected.append(rng.choice(valid_frames))
    else:
        selected = rng.sample(valid_frames, n_negatives)

    return selected


def process_video(
    meta: VideoMetadata,
    query_video_dir: Path,
    source_frame_dir: Path,
    output_dir: Path,
    frame_extractor: VideoFrameExtractor,
    face_detector: FaceDetector,
    human_detector: HumanDetector,
    n_anchors: int = 2,
    n_negatives: int = 20,
    safety_radius_sec: float = 15.0,
    seed: int = 42,
) -> bool:
    """
    Process a single video and save preprocessed data.

    Returns True if successful, False otherwise.
    """
    video_id = meta.id
    video_output_dir = output_dir / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    meta_path = video_output_dir / "meta.json"
    if meta_path.exists():
        return True

    # Extract anchor frames from query video
    video_path = query_video_dir / f"{video_id}.mp4"
    if not video_path.exists():
        return False

    anchor_frames = frame_extractor.extract_boundary_frames(
        video_path, n_frames=n_anchors, boundary_seconds=1.0, seed=seed
    )

    if len(anchor_frames) < n_anchors:
        return False

    # Process each anchor
    anchors_data = []
    detections_data = {}

    for anchor_idx, (frame_idx, anchor_frame) in enumerate(anchor_frames):
        # Save anchor frame
        anchor_path = video_output_dir / f"anchor_{anchor_idx}.jpg"
        cv2.imwrite(str(anchor_path), anchor_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Map to source frame
        source_frame_no = map_anchor_to_source(meta, anchor_idx, n_anchors)
        anchors_data.append({
            "frame_idx": frame_idx,
            "source_frame_no": source_frame_no,
        })

        # Load source frame and run detection
        source_frame_path = source_frame_dir / meta.origin_id / f"{source_frame_no:04d}.jpg"
        if source_frame_path.exists():
            source_frame = cv2.imread(str(source_frame_path))
            if source_frame is not None:
                # Run detections
                face_bbox = face_detector.detect(source_frame)
                human_bbox = human_detector.detect(source_frame)

                detections_data[str(source_frame_no)] = {
                    "human_bbox": human_bbox,
                    "face_bbox": face_bbox,
                }

    # Get hard negative frames
    anchor_source_frames = [a["source_frame_no"] for a in anchors_data]
    negative_frame_nos = get_hard_negative_frames(
        source_frame_dir=source_frame_dir,
        origin_id=meta.origin_id,
        anchor_source_frames=anchor_source_frames,
        fps=meta.fps,
        safety_radius_sec=safety_radius_sec,
        n_negatives=n_negatives,
        seed=seed,
    )

    # Copy negative frames
    for neg_idx, frame_no in enumerate(negative_frame_nos):
        src_path = source_frame_dir / meta.origin_id / f"{frame_no:04d}.jpg"
        dst_path = video_output_dir / f"negative_{neg_idx:02d}.jpg"
        if src_path.exists():
            shutil.copy2(src_path, dst_path)

    # Save metadata
    meta_dict = {
        "video_id": video_id,
        "origin_id": meta.origin_id,
        "anchors": anchors_data,
        "detections": detections_data,
    }

    with open(meta_path, "w") as f:
        json.dump(meta_dict, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess TraceDINO dataset")
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "valid", "test"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos to process (for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    args = parser.parse_args()

    # Load config
    config = TraceDINOConfig()

    # Determine paths
    csv_path = getattr(config, f"{args.split}_csv")
    output_dir = config.preprocessed_dir / args.split

    print(f"Processing {args.split} split")
    print(f"CSV: {csv_path}")
    print(f"Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print("Loading metadata...")
    metadata = load_metadata(csv_path)
    if args.limit:
        metadata = metadata[:args.limit]
    print(f"Total videos: {len(metadata)}")

    # Initialize detectors (in main process)
    print("Initializing detectors...")
    frame_extractor = VideoFrameExtractor()
    face_detector = FaceDetector()
    human_detector = HumanDetector()
    print("Detectors initialized")

    # Process videos
    success_count = 0
    fail_count = 0

    for meta in tqdm(metadata, desc="Processing videos"):
        try:
            success = process_video(
                meta=meta,
                query_video_dir=config.query_video_dir,
                source_frame_dir=config.source_frame_dir,
                output_dir=output_dir,
                frame_extractor=frame_extractor,
                face_detector=face_detector,
                human_detector=human_detector,
                n_anchors=config.n_anchor_frames,
                n_negatives=config.n_presampled_negatives,
                safety_radius_sec=config.safety_radius_seconds,
                seed=args.seed,
            )
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"Error processing {meta.id}: {e}")
            fail_count += 1

    print(f"\nProcessing complete!")
    print(f"Success: {success_count}, Failed: {fail_count}")


if __name__ == "__main__":
    main()
