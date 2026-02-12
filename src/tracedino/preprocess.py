"""Preprocessing script for TraceDINO dataset.

This script preprocesses the dataset by:
1. Extracting anchor frames from query videos
2. Running face and human detection on source frames
3. Pre-sampling hard negative frames

Usage:
    python -m src.tracedino.preprocess --split train --limit 100
    python -m src.tracedino.preprocess --split train
    python -m src.tracedino.preprocess --split all  # Process all splits

Optimized mode (default):
    python -m src.tracedino.preprocess --split train --batch-size 32 --io-workers 8

Legacy mode:
    python -m src.tracedino.preprocess --split train --legacy
"""

import argparse
import gc
import json
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.tracedino.config import TraceDINOConfig
from src.tracedino.dataset.metadata import VideoMetadata, load_metadata


@dataclass
class VideoTask:
    """单个视频的处理任务"""
    meta: VideoMetadata
    video_output_dir: Path


@dataclass
class PreparedVideoData:
    """视频帧提取后的数据"""
    task: VideoTask
    anchor_frames: List[Tuple[int, np.ndarray]]  # (frame_idx, frame)
    source_frames: List[Tuple[int, np.ndarray]]  # (source_frame_no, frame)
    negative_frame_nos: List[int]
    anchors_data: List[dict]  # 锚点元数据


@dataclass
class ProcessedVideoData:
    """检测完成后的数据"""
    task: VideoTask
    anchor_frames: List[Tuple[int, np.ndarray]]
    detections: Dict[str, Dict[str, Optional[List[int]]]]
    negative_frame_nos: List[int]
    anchors_data: List[dict]


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

    def detect_batch(self, images: List[np.ndarray]) -> List[Optional[List[int]]]:
        """批量检测多张图像"""
        results = []
        for image in images:
            bbox = self.detect(image)
            results.append(bbox)
        return results


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

    def detect_batch(self, images: List[np.ndarray], batch_size: int = 16) -> List[Optional[List[int]]]:
        """批量检测多张图像（利用 YOLOv8 原生批量推理）"""
        if not images:
            return []

        results = self.detector(images, classes=[0], verbose=False, batch=batch_size)

        output = []
        for result in results:
            if len(result.boxes) == 0:
                output.append(None)
            else:
                boxes = result.boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                largest_idx = areas.argmax()
                output.append(boxes[largest_idx].astype(int).tolist())
        return output


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


def map_anchor_to_source(
    meta: VideoMetadata,
    anchor_idx: int,
    n_anchors: int = 2,
    max_available_frame: Optional[int] = None,
) -> int:
    """
    Map anchor frame index to source frame image index.

    New dataset has pre-computed gt_start_img and gt_end_img, use them directly.

    Args:
        meta: Video metadata containing gt_start_img and gt_end_img
        anchor_idx: Index of the anchor frame (0 to n_anchors-1)
        n_anchors: Total number of anchor frames
        max_available_frame: Maximum available frame number in source video.
                            If provided, clamps the result to avoid out-of-range errors.

    Returns:
        Source frame image index (clamped to max_available_frame if provided)
    """
    ratio = anchor_idx / max(1, n_anchors - 1)
    # Directly use pre-computed image indices
    source_img_idx = int(meta.gt_start_img + ratio * (meta.gt_end_img - meta.gt_start_img))

    # Clamp to available range if provided (fixes off-by-one errors)
    if max_available_frame is not None:
        source_img_idx = min(source_img_idx, max_available_frame)

    return source_img_idx


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
        origin_id: Source video ID (32-char hex string)
        anchor_source_frames: List of source frame image indices
        fps: Original video FPS (not used since frames are at 1 FPS)
        safety_radius_sec: Safety radius in seconds
        n_negatives: Number of negatives to sample
        seed: Random seed

    Returns:
        List of frame numbers for hard negatives (image indices)
    """
    source_dir = source_frame_dir / origin_id
    available_frames = sorted([int(f.stem) for f in source_dir.glob("*.jpg")])

    if len(available_frames) == 0:
        return []

    # Frames are at 1 FPS, so safety_radius in frames = safety_radius in seconds
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

    # Get max available frame in source video to avoid out-of-range errors
    source_dir = source_frame_dir / meta.origin_id
    if not source_dir.exists():
        return False
    available_frames = [int(f.stem) for f in source_dir.glob("*.jpg")]
    if not available_frames:
        return False
    max_available_frame = max(available_frames)

    # Process each anchor
    anchors_data = []
    detections_data = {}

    for anchor_idx, (frame_idx, anchor_frame) in enumerate(anchor_frames):
        # Save anchor frame
        anchor_path = video_output_dir / f"anchor_{anchor_idx}.jpg"
        cv2.imwrite(str(anchor_path), anchor_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Map to source frame (with clamping to avoid out-of-range)
        source_frame_no = map_anchor_to_source(meta, anchor_idx, n_anchors, max_available_frame)
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


def process_split(
    split: str,
    config: TraceDINOConfig,
    frame_extractor: VideoFrameExtractor,
    face_detector: FaceDetector,
    human_detector: HumanDetector,
    limit: Optional[int] = None,
    seed: int = 42,
) -> Tuple[int, int]:
    """
    Process a single dataset split.

    Returns:
        Tuple of (success_count, fail_count)
    """
    # Determine paths
    csv_path = getattr(config, f"{split}_csv")
    output_dir = config.preprocessed_dir / split

    print(f"\n{'='*60}")
    print(f"Processing {split} split")
    print(f"CSV: {csv_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print("Loading metadata...")
    metadata = load_metadata(csv_path)
    if limit:
        metadata = metadata[:limit]
    print(f"Total videos: {len(metadata)}")

    # Process videos
    success_count = 0
    fail_count = 0

    for meta in tqdm(metadata, desc=f"Processing {split}"):
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
                seed=seed,
            )
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"Error processing {meta.id}: {e}")
            fail_count += 1

    print(f"\n{split.upper()} split complete!")
    print(f"Success: {success_count}, Failed: {fail_count}")

    return success_count, fail_count


def prepare_single_video(
    task: VideoTask,
    frame_extractor: VideoFrameExtractor,
    query_video_dir: Path,
    source_frame_dir: Path,
    n_anchors: int,
    n_negatives: int,
    safety_radius_sec: float,
    seed: int,
) -> Optional[PreparedVideoData]:
    """在 I/O 线程中提取单个视频的帧数据"""
    meta = task.meta
    video_path = query_video_dir / f"{meta.id}.mp4"

    if not video_path.exists():
        return None

    # 提取锚点帧
    anchor_frames = frame_extractor.extract_boundary_frames(
        video_path, n_frames=n_anchors, boundary_seconds=1.0, seed=seed
    )
    if len(anchor_frames) < n_anchors:
        return None

    # Get max available frame in source video to avoid out-of-range errors
    source_dir = source_frame_dir / meta.origin_id
    if not source_dir.exists():
        return None
    available_frames = [int(f.stem) for f in source_dir.glob("*.jpg")]
    if not available_frames:
        return None
    max_available_frame = max(available_frames)

    # 读取源帧并构建锚点元数据
    source_frames = []
    anchors_data = []
    for anchor_idx, (frame_idx, _) in enumerate(anchor_frames):
        source_frame_no = map_anchor_to_source(meta, anchor_idx, n_anchors, max_available_frame)
        anchors_data.append({
            "frame_idx": frame_idx,
            "source_frame_no": source_frame_no,
        })

        source_frame_path = source_frame_dir / meta.origin_id / f"{source_frame_no:04d}.jpg"
        if source_frame_path.exists():
            source_frame = cv2.imread(str(source_frame_path))
            if source_frame is not None:
                source_frames.append((source_frame_no, source_frame))

    # 获取负样本帧号
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

    return PreparedVideoData(
        task=task,
        anchor_frames=anchor_frames,
        source_frames=source_frames,
        negative_frame_nos=negative_frame_nos,
        anchors_data=anchors_data,
    )


def save_single_video(
    data: ProcessedVideoData,
    source_frame_dir: Path,
) -> bool:
    """在 I/O 线程中保存单个视频的处理结果"""
    task = data.task
    video_output_dir = task.video_output_dir
    meta = task.meta

    try:
        # 保存锚点帧
        for anchor_idx, (frame_idx, anchor_frame) in enumerate(data.anchor_frames):
            anchor_path = video_output_dir / f"anchor_{anchor_idx}.jpg"
            cv2.imwrite(str(anchor_path), anchor_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 复制负样本
        for neg_idx, frame_no in enumerate(data.negative_frame_nos):
            src_path = source_frame_dir / meta.origin_id / f"{frame_no:04d}.jpg"
            dst_path = video_output_dir / f"negative_{neg_idx:02d}.jpg"
            if src_path.exists():
                shutil.copy2(src_path, dst_path)

        # 保存元数据
        meta_dict = {
            "video_id": meta.id,
            "origin_id": meta.origin_id,
            "anchors": data.anchors_data,
            "detections": data.detections,
        }
        meta_path = video_output_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta_dict, f, indent=2)

        return True
    except Exception as e:
        print(f"Error saving {meta.id}: {e}")
        return False
    finally:
        # 显式释放内存
        del data.anchor_frames
        gc.collect()


def process_split_optimized(
    split: str,
    config: TraceDINOConfig,
    frame_extractor: VideoFrameExtractor,
    face_detector: FaceDetector,
    human_detector: HumanDetector,
    limit: Optional[int] = None,
    seed: int = 42,
    batch_size: int = 32,
    io_workers: int = 8,
) -> Tuple[int, int]:
    """
    优化后的分片处理流程（三阶段流水线）

    Returns:
        Tuple of (success_count, fail_count)
    """
    csv_path = getattr(config, f"{split}_csv")
    output_dir = config.preprocessed_dir / split
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载元数据
    print(f"\n{'='*60}")
    print(f"Processing {split} split (optimized)")
    print(f"{'='*60}")

    metadata = load_metadata(csv_path)
    if limit:
        metadata = metadata[:limit]

    # 过滤已处理的视频
    tasks = []
    for meta in metadata:
        video_output_dir = output_dir / meta.id
        video_output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = video_output_dir / "meta.json"
        if not meta_path.exists():
            tasks.append(VideoTask(meta=meta, video_output_dir=video_output_dir))

    print(f"Total: {len(metadata)}, Pending: {len(tasks)}, Skipped: {len(metadata) - len(tasks)}")

    if not tasks:
        return len(metadata) - len(tasks), 0

    success_count = 0
    fail_count = 0

    # 分批处理
    with ThreadPoolExecutor(max_workers=io_workers) as executor:
        for batch_start in tqdm(range(0, len(tasks), batch_size), desc=f"Processing {split}"):
            batch_tasks = tasks[batch_start:batch_start + batch_size]

            # === 阶段 1: 并行帧提取 ===
            prepare_futures = {
                executor.submit(
                    prepare_single_video,
                    task, frame_extractor, config.query_video_dir, config.source_frame_dir,
                    config.n_anchor_frames, config.n_presampled_negatives,
                    config.safety_radius_seconds, seed
                ): task for task in batch_tasks
            }

            prepared_data = []
            for future in as_completed(prepare_futures):
                try:
                    result = future.result()
                    if result is not None:
                        prepared_data.append(result)
                    else:
                        fail_count += 1
                except Exception as e:
                    print(f"Prepare error: {e}")
                    fail_count += 1

            if not prepared_data:
                continue

            # === 阶段 2: 批量 GPU 检测 ===
            # 收集所有需要检测的图像
            all_images = []
            image_mapping = []  # (data_idx, source_frame_no)
            for data_idx, data in enumerate(prepared_data):
                for source_frame_no, source_frame in data.source_frames:
                    all_images.append(source_frame)
                    image_mapping.append((data_idx, source_frame_no))

            # 批量人脸检测
            face_results = face_detector.detect_batch(all_images) if all_images else []
            # 批量人体检测
            human_results = human_detector.detect_batch(all_images, batch_size=batch_size) if all_images else []

            # 组装检测结果
            processed_data = []
            detections_by_video: Dict[int, Dict[str, Dict]] = {i: {} for i in range(len(prepared_data))}

            for idx, (data_idx, source_frame_no) in enumerate(image_mapping):
                detections_by_video[data_idx][str(source_frame_no)] = {
                    "face_bbox": face_results[idx] if idx < len(face_results) else None,
                    "human_bbox": human_results[idx] if idx < len(human_results) else None,
                }

            for data_idx, data in enumerate(prepared_data):
                processed_data.append(ProcessedVideoData(
                    task=data.task,
                    anchor_frames=data.anchor_frames,
                    detections=detections_by_video[data_idx],
                    negative_frame_nos=data.negative_frame_nos,
                    anchors_data=data.anchors_data,
                ))

            # 显式释放源帧内存
            for data in prepared_data:
                del data.source_frames
            del all_images, face_results, human_results
            gc.collect()

            # === 阶段 3: 并行保存 ===
            save_futures = {
                executor.submit(save_single_video, data, config.source_frame_dir): data
                for data in processed_data
            }

            for future in as_completed(save_futures):
                try:
                    if future.result():
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    print(f"Save error: {e}")
                    fail_count += 1

            # 批次结束后强制 GC
            del prepared_data, processed_data
            gc.collect()

    print(f"\n{split.upper()} complete! Success: {success_count}, Failed: {fail_count}")
    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(description="Preprocess TraceDINO dataset")
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "valid", "test", "all"],
        help="Dataset split to process (use 'all' to process train, valid, and test)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos to process per split (for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for GPU inference (optimized mode only)",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=8,
        help="Number of I/O worker threads (optimized mode only)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy sequential processing instead of optimized pipeline",
    )
    args = parser.parse_args()

    # Load config
    config = TraceDINOConfig()

    # Initialize detectors (in main process)
    print("Initializing detectors...")
    frame_extractor = VideoFrameExtractor()
    face_detector = FaceDetector()
    human_detector = HumanDetector()
    print("Detectors initialized")

    # Determine which splits to process
    if args.split == "all":
        splits = ["train", "valid", "test"]
    else:
        splits = [args.split]

    # Process each split
    total_success = 0
    total_fail = 0

    for split in splits:
        if args.legacy:
            success, fail = process_split(
                split=split,
                config=config,
                frame_extractor=frame_extractor,
                face_detector=face_detector,
                human_detector=human_detector,
                limit=args.limit,
                seed=args.seed,
            )
        else:
            success, fail = process_split_optimized(
                split=split,
                config=config,
                frame_extractor=frame_extractor,
                face_detector=face_detector,
                human_detector=human_detector,
                limit=args.limit,
                seed=args.seed,
                batch_size=args.batch_size,
                io_workers=args.io_workers,
            )
        total_success += success
        total_fail += fail

    # Print overall summary if processing multiple splits
    if len(splits) > 1:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"Total Success: {total_success}, Total Failed: {total_fail}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
