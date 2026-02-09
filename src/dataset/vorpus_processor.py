"""Main processing logic for Vorpus dataset frame extraction and metadata update."""

import argparse
import fcntl
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from loguru import logger
from tqdm import tqdm

from .utils import (
    extract_frames,
    frame_to_image_idx,
    generate_video_hash,
    get_video_info,
)


@dataclass
class ProgressTracker:
    """追踪处理进度，支持断点续传"""

    progress_dir: Path

    def __post_init__(self):
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.origins_file = self.progress_dir / "progress_origins.jsonl"
        self.records_file = self.progress_dir / "progress_records.jsonl"
        self.lock_file = self.progress_dir / ".progress.lock"

    def get_completed_origins(self) -> set[str]:
        """获取已完成的 origin_id 集合"""
        if not self.origins_file.exists():
            return set()
        completed = set()
        with open(self.origins_file, "r") as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line)["origin_id"])
        return completed

    def get_saved_records(self) -> list[dict]:
        """获取已保存的记录"""
        if not self.records_file.exists():
            return []
        records = []
        with open(self.records_file, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def save_origin_progress(self, origin_id: str, records: list[dict]):
        """保存单个 origin 的处理结果（线程安全）"""
        with open(self.lock_file, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                # 追加记录
                with open(self.records_file, "a") as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                # 标记 origin 完成
                with open(self.origins_file, "a") as f:
                    f.write(json.dumps({"origin_id": origin_id}) + "\n")
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def clear(self):
        """清除进度文件"""
        for f in [self.origins_file, self.records_file, self.lock_file]:
            if f.exists():
                f.unlink()


def group_records_by_origin(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Group records by origin_id.

    Args:
        df: DataFrame with origin_id column

    Returns:
        Dictionary mapping origin_id (string) to DataFrame of records
    """
    grouped = {}
    for origin_id in df["origin_id"].unique().sort():
        grouped[origin_id] = df.filter(pl.col("origin_id") == origin_id)
    return grouped


def cluster_gt_ranges(
    records: pl.DataFrame,
    max_gap_frames: int,
) -> list[pl.DataFrame]:
    """
    Cluster records by GT frame ranges using greedy grouping.

    Records are sorted by gt_start_f, then greedily merged into clusters
    such that each cluster's total span (from min gt_start to max gt_end)
    does not exceed max_gap_frames.

    Args:
        records: DataFrame with gt_start_f and gt_end_f columns
        max_gap_frames: Maximum total span in frames for each cluster

    Returns:
        List of DataFrames, each containing records in one cluster
    """
    if len(records) == 0:
        return []

    # Sort by gt_start_f
    sorted_df = records.sort("gt_start_f")
    gt_starts = sorted_df["gt_start_f"].to_list()
    gt_ends = sorted_df["gt_end_f"].to_list()

    # Find cluster boundaries
    cluster_indices = []  # List of (start_idx, end_idx) for each cluster
    cluster_start = 0
    current_end = gt_ends[0]

    for i in range(1, len(gt_starts)):
        # Calculate potential span if we merge this record into current cluster
        potential_end = max(current_end, gt_ends[i])
        cluster_start_frame = gt_starts[cluster_start]
        potential_span = potential_end - cluster_start_frame + 1

        if potential_span <= max_gap_frames:
            # Merged span is within max duration, merge into current cluster
            current_end = potential_end
        else:
            # Merged span exceeds max duration, start new cluster
            cluster_indices.append((cluster_start, i))
            cluster_start = i
            current_end = gt_ends[i]

    # Add last cluster
    cluster_indices.append((cluster_start, len(gt_starts)))

    # Extract DataFrames for each cluster
    clusters = []
    for start_idx, end_idx in cluster_indices:
        cluster_df = sorted_df[start_idx:end_idx]
        clusters.append(cluster_df)

    return clusters


def calculate_clip_range(
    gt_min: int,
    gt_max: int,
    max_frames: int,
    total_frames: int,
    random_offset: bool = True,
) -> tuple[int, int]:
    """
    Calculate the frame range [start, end) for video clipping.

    Ensures that all GT frames are within the clipped range, and randomly
    selects a start point if the video is long enough.

    Args:
        gt_min: Minimum GT frame (inclusive)
        gt_max: Maximum GT frame (inclusive)
        max_frames: Maximum duration in frames
        total_frames: Total frames in the original video
        random_offset: If True, randomly select start point within valid range

    Returns:
        Tuple of (start_frame, end_frame)
    """
    gt_span = gt_max - gt_min + 1

    if gt_span > max_frames:
        # GT range exceeds max duration, clip exactly around GT
        logger.warning(
            f"GT range ({gt_span} frames) exceeds max duration ({max_frames} frames). "
            f"Clipping exactly around GT range."
        )
        return gt_min, gt_max + 1

    # Calculate valid start range
    # start must satisfy: start <= gt_min and start + max_frames >= gt_max + 1
    earliest_start = max(0, gt_max + 1 - max_frames)
    latest_start = min(gt_min, max(0, total_frames - max_frames))

    if earliest_start > latest_start:
        # Not enough room, clip from earliest possible
        start = max(0, gt_max + 1 - max_frames)
    elif random_offset:
        # Randomly select start within valid range
        start = random.randint(earliest_start, latest_start)
    else:
        # Use earliest valid start
        start = earliest_start

    end = min(start + max_frames, total_frames)

    return start, end


def process_non_gt_video(
    video_path: Path,
    output_dir: Path,
    max_duration_sec: int,
    output_fps: float,
    pack_tar: bool = False,
) -> dict | None:
    """
    Process a video without GT: randomly clip a segment and extract frames.

    Args:
        video_path: Path to the video file
        output_dir: Output frames directory
        max_duration_sec: Maximum duration in seconds
        output_fps: Frame extraction rate (frames per second)
        pack_tar: If True, pack frames into a tar file

    Returns:
        Dictionary with video metadata, or None if failed
    """
    try:
        video_info = get_video_info(video_path)
        total_frames = video_info["total_frames"]
        video_fps = video_info["fps"]
        duration_sec = video_info["duration_sec"]
    except Exception as e:
        logger.error(f"Failed to get video info for {video_path}: {e}")
        return None

    # Calculate max_duration_frames using actual video fps
    max_duration_frames = int(max_duration_sec * video_fps)

    # Randomly select start frame
    if total_frames <= max_duration_frames:
        # Video is shorter than max duration, use entire video
        clip_start = 0
        clip_end = total_frames
    else:
        # Randomly select start point
        max_start = total_frames - max_duration_frames
        clip_start = random.randint(0, max_start)
        clip_end = clip_start + max_duration_frames

    # Generate new origin ID
    new_origin_hash = generate_video_hash()
    frames_output_dir = output_dir / new_origin_hash

    # Extract frames
    frame_count = extract_frames(
        video_path,
        frames_output_dir,
        clip_start,
        clip_end,
        video_fps,
        output_fps,
        pack_tar=pack_tar,
    )

    if frame_count == 0:
        logger.error(f"Failed to extract frames for {video_path.name}")
        return None

    return {
        "origin_id": new_origin_hash,
        "original_video": video_path.stem,
        "frame_count": frame_count,
        "output_fps": output_fps,
        "clip_start_sec": clip_start / video_fps,
        "clip_end_sec": clip_end / video_fps,
    }


def process_non_gt_video_multi(
    video_path: Path,
    output_dir: Path,
    max_duration_sec: int,
    output_fps: float,
    num_clips: int = 1,
    pack_tar: bool = False,
) -> list[dict]:
    """
    从单个非 GT 视频随机提取多个不重叠的片段。

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        max_duration_sec: 每个片段的最大时长（秒）
        output_fps: 抽帧频率
        num_clips: 要提取的片段数量
        pack_tar: 是否打包为 tar

    Returns:
        提取的片段元数据列表
    """
    try:
        video_info = get_video_info(video_path)
        total_frames = video_info["total_frames"]
        video_fps = video_info["fps"]
    except Exception as e:
        logger.error(f"Failed to get video info for {video_path}: {e}")
        return []

    max_duration_frames = int(max_duration_sec * video_fps)

    # 计算可以提取多少个不重叠的片段
    max_possible_clips = total_frames // max_duration_frames
    actual_clips = min(num_clips, max(1, max_possible_clips))

    # 生成随机的、不重叠的起始位置
    # 可用的起始位置：0, max_duration_frames, 2*max_duration_frames, ...
    available_starts = list(
        range(0, total_frames - max_duration_frames + 1, max_duration_frames)
    )

    if len(available_starts) < actual_clips:
        # 如果可用位置不足，使用所有可用位置
        selected_starts = available_starts
    else:
        # 随机选择 actual_clips 个不重叠的起始位置
        selected_starts = random.sample(available_starts, actual_clips)

    # 按起始位置排序，避免乱序处理
    selected_starts.sort()

    results = []
    for i, clip_start in enumerate(selected_starts):
        clip_end = min(clip_start + max_duration_frames, total_frames)

        # 生成新的 origin_id
        new_origin_hash = generate_video_hash()
        frames_output_dir = output_dir / new_origin_hash

        # 抽帧
        frame_count = extract_frames(
            video_path,
            frames_output_dir,
            clip_start,
            clip_end,
            video_fps,
            output_fps,
            pack_tar=pack_tar,
        )

        if frame_count > 0:
            results.append({
                "origin_id": new_origin_hash,
                "original_video": video_path.stem,
                "frame_count": frame_count,
                "output_fps": output_fps,
                "clip_index": i,
                "clip_start_sec": clip_start / video_fps,
                "clip_end_sec": clip_end / video_fps,
            })

    return results


def process_single_origin(
    origin_id: str,
    records: pl.DataFrame,
    input_dir: Path,
    output_dir: Path,
    max_duration_sec: int,
    output_fps: float,
    pack_tar: bool = False,
) -> list[dict]:
    """
    Process a single origin video: cluster GT ranges and extract frames.

    Args:
        origin_id: Original video ID (string, e.g., "1379")
        records: DataFrame of records for this origin
        input_dir: Input video directory
        output_dir: Output frames directory
        max_duration_sec: Maximum duration in seconds
        output_fps: Frame extraction rate (frames per second)
        pack_tar: If True, pack frames into a tar file

    Returns:
        List of updated record dictionaries
    """
    # Get original video path (origin_id is like "1379", video is "1379.mp4")
    origin_video_path = input_dir / f"{origin_id}.mp4"

    if not origin_video_path.exists():
        logger.error(f"Origin video not found: {origin_video_path}")
        return []

    # Get video info
    try:
        video_info = get_video_info(origin_video_path)
        total_frames = video_info["total_frames"]
        video_fps = video_info["fps"]
    except Exception as e:
        logger.error(f"Failed to get video info for {origin_video_path}: {e}")
        return []

    # Calculate max_duration_frames using actual video fps
    max_duration_frames = int(max_duration_sec * video_fps)

    # Cluster GT ranges - returns list of DataFrames
    clusters = cluster_gt_ranges(records, max_duration_frames)

    updated_records = []

    for cluster_idx, cluster_df in enumerate(clusters):
        # Calculate GT range from cluster DataFrame
        gt_min = cluster_df["gt_start_f"].min()
        gt_max = cluster_df["gt_end_f"].max()

        # Calculate clip range
        clip_start, clip_end = calculate_clip_range(
            gt_min, gt_max, max_duration_frames, total_frames
        )

        # Generate new origin ID (now used as frames directory name)
        new_origin_hash = generate_video_hash()
        frames_output_dir = output_dir / new_origin_hash

        # Extract frames instead of clipping video
        frame_count = extract_frames(
            origin_video_path,
            frames_output_dir,
            clip_start,
            clip_end,
            video_fps,
            output_fps,
            pack_tar=pack_tar,
        )

        if frame_count == 0:
            logger.error(
                f"Failed to extract frames for origin {origin_id}, cluster {cluster_idx}"
            )
            continue

        # Update records - preserve all original columns
        for row_idx in range(len(cluster_df)):
            row_dict = cluster_df.row(row_idx, named=True)

            # Calculate new GT frame numbers (relative to clip start)
            new_gt_start_f = int(row_dict["gt_start_f"] - clip_start)
            new_gt_end_f = int(row_dict["gt_end_f"] - clip_start)

            # Convert to image indices
            gt_start_img = int(frame_to_image_idx(new_gt_start_f, video_fps, output_fps))
            gt_end_img = int(frame_to_image_idx(new_gt_end_f, video_fps, output_fps))

            # Update record
            row_dict["origin_id"] = new_origin_hash
            row_dict["gt_start_f"] = new_gt_start_f
            row_dict["gt_end_f"] = new_gt_end_f
            row_dict["gt_start_img"] = gt_start_img
            row_dict["gt_end_img"] = gt_end_img
            row_dict["frame_count"] = int(frame_count)
            row_dict["output_fps"] = float(output_fps)

            updated_records.append(row_dict)

    return updated_records


def process_vorpus(
    input_dir: Path,
    output_dir: Path,
    csv_dir: Path,
    output_csv_dir: Path,
    max_duration_sec: int = 300,
    output_fps: float = 1.0,
    workers: int = 4,
    limit: int | None = None,
    pack_tar: bool = False,
    resume: bool = True,
) -> None:
    """
    Main entry point for processing Vorpus dataset.

    Args:
        input_dir: Input video directory
        output_dir: Output frames directory
        csv_dir: Input CSV directory
        output_csv_dir: Output CSV directory
        max_duration_sec: Maximum segment duration in seconds
        output_fps: Frame extraction rate (frames per second)
        workers: Number of worker threads
        limit: If set, only process this many origin videos (for testing)
        pack_tar: If True, pack frames into tar files
        resume: If True, resume from previous progress (default: True)
    """
    logger.info("Starting Vorpus dataset processing")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Max duration: {max_duration_sec}s")
    logger.info(f"Output FPS: {output_fps}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Pack tar: {pack_tar}")
    logger.info(f"Resume: {resume}")
    if limit:
        logger.info(f"Limit: {limit} origins (test mode)")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    # Initialize progress tracker
    progress_dir = output_csv_dir / ".progress"
    tracker = ProgressTracker(progress_dir)

    if not resume:
        tracker.clear()
        logger.info("Disabled resume, starting fresh")

    # Read all.csv
    all_csv_path = csv_dir / "all.csv"
    logger.info(f"Reading {all_csv_path}")
    df = pl.read_csv(all_csv_path, schema_overrides={"origin_id": pl.Utf8})

    logger.info(f"Total records: {len(df)}")
    logger.info(f"Unique origins: {df['origin_id'].n_unique()}")

    # Group by origin
    grouped = group_records_by_origin(df)
    logger.info(f"Total origin videos: {len(grouped)}")

    # Apply limit for testing
    if limit:
        origin_ids = list(grouped.keys())[:limit]
        grouped = {k: grouped[k] for k in origin_ids}
        logger.info(f"Processing {len(grouped)} origin videos (limited)")

    # Get completed origins and saved records
    completed_origins = tracker.get_completed_origins()
    all_updated_records = tracker.get_saved_records()

    if completed_origins:
        logger.info(
            f"Resuming: {len(completed_origins)} origins already completed, "
            f"{len(all_updated_records)} records saved"
        )

    # Filter out completed origins
    pending_grouped = {k: v for k, v in grouped.items() if k not in completed_origins}
    logger.info(f"Pending origins to process: {len(pending_grouped)}")

    if not pending_grouped:
        logger.info("All origins already processed, generating CSV...")
    else:
        # Process remaining origins in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_single_origin,
                    origin_id,
                    records,
                    input_dir,
                    output_dir,
                    max_duration_sec,
                    output_fps,
                    pack_tar,
                ): origin_id
                for origin_id, records in pending_grouped.items()
            }

            with tqdm(total=len(futures), desc="Processing origins") as pbar:
                for future in as_completed(futures):
                    origin_id = futures[future]
                    try:
                        updated_records = future.result()
                        # Save progress immediately
                        tracker.save_origin_progress(origin_id, updated_records)
                        all_updated_records.extend(updated_records)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing origin {origin_id}: {e}")
                        pbar.update(1)

    # Create new DataFrame
    logger.info(f"Total updated records: {len(all_updated_records)}")

    if not all_updated_records:
        logger.error("No records were processed successfully!")
        return

    new_df = pl.DataFrame(all_updated_records, infer_schema_length=None)

    # Sort by id
    new_df = new_df.sort("id")

    # Save all.csv
    output_all_csv = output_csv_dir / "all.csv"
    new_df.write_csv(output_all_csv)
    logger.info(f"Saved {output_all_csv}")

    # Split into train/valid/test based on original splits
    for split in ["train", "valid", "test"]:
        split_csv_path = csv_dir / f"{split}.csv"
        if not split_csv_path.exists():
            logger.warning(f"Split file not found: {split_csv_path}")
            continue

        split_df = pl.read_csv(split_csv_path)
        split_ids = set(split_df["id"].to_list())

        # Filter new_df by split_ids
        split_new_df = new_df.filter(pl.col("id").is_in(split_ids))

        output_split_csv = output_csv_dir / f"{split}.csv"
        split_new_df.write_csv(output_split_csv)
        logger.info(f"Saved {output_split_csv} ({len(split_new_df)} records)")

    logger.info("Processing complete!")


def process_non_gt_videos(
    input_dir: Path,
    output_dir: Path,
    csv_dir: Path,
    max_duration_sec: int = 300,
    output_fps: float = 1.0,
    workers: int = 4,
    limit: int | None = None,
    pack_tar: bool = False,
) -> list[dict]:
    """
    Process videos that are not in the GT CSV (random clip and extract frames).

    Args:
        input_dir: Input video directory
        output_dir: Output frames directory
        csv_dir: Input CSV directory (to find which videos have GT)
        max_duration_sec: Maximum segment duration in seconds
        output_fps: Frame extraction rate (frames per second)
        workers: Number of worker threads
        limit: If set, only process this many videos (for testing)
        pack_tar: If True, pack frames into tar files

    Returns:
        List of processed video metadata dictionaries
    """
    logger.info("Starting non-GT video processing")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Max duration: {max_duration_sec}s")
    logger.info(f"Output FPS: {output_fps}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Pack tar: {pack_tar}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read GT CSV to find which videos have GT
    all_csv_path = csv_dir / "all.csv"
    df = pl.read_csv(all_csv_path, schema_overrides={"origin_id": pl.Utf8})
    gt_origins = set(df["origin_id"].unique().to_list())
    logger.info(f"Videos with GT: {len(gt_origins)}")

    # Find all videos in input directory
    all_videos = list(input_dir.glob("*.mp4"))
    logger.info(f"Total videos in directory: {len(all_videos)}")

    # Filter to non-GT videos
    non_gt_videos = [v for v in all_videos if v.stem not in gt_origins]
    logger.info(f"Videos without GT: {len(non_gt_videos)}")

    # Apply limit for testing
    if limit:
        non_gt_videos = non_gt_videos[:limit]
        logger.info(f"Processing {len(non_gt_videos)} videos (limited)")

    # Process videos in parallel
    all_results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_non_gt_video,
                video_path,
                output_dir,
                max_duration_sec,
                output_fps,
                pack_tar,
            ): video_path
            for video_path in non_gt_videos
        }

        with tqdm(total=len(futures), desc="Processing non-GT videos") as pbar:
            for future in as_completed(futures):
                video_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Error processing {video_path.name}: {e}")
                    pbar.update(1)

    logger.info(f"Successfully processed: {len(all_results)} videos")

    if not all_results:
        logger.warning("No non-GT videos were processed!")
        return []

    logger.info("Non-GT video processing complete!")
    return all_results


def generate_vorpus_csv(
    gt_csv_path: Path,
    non_gt_results: list[dict],
    output_csv_path: Path,
) -> None:
    """
    Generate unified vorpus.csv with both GT and non-GT videos.

    Args:
        gt_csv_path: Path to the GT all.csv file
        non_gt_results: List of non-GT video metadata dictionaries
        output_csv_path: Output path for vorpus.csv
    """
    logger.info("Generating unified vorpus.csv")

    # Read GT data
    gt_df = pl.read_csv(gt_csv_path, schema_overrides={"origin_id": pl.Utf8})

    # Get unique GT origins with their metadata
    gt_origins = gt_df.group_by("origin_id").agg([
        pl.col("frame_count").first(),
        pl.col("output_fps").first(),
    ])

    # Add label column
    gt_origins = gt_origins.with_columns(pl.lit("gt").alias("label"))

    # Create non-GT DataFrame
    if non_gt_results:
        non_gt_df = pl.DataFrame(non_gt_results)
        non_gt_df = non_gt_df.select([
            pl.col("origin_id"),
            pl.col("frame_count"),
            pl.col("output_fps"),
        ]).with_columns(pl.lit("non_gt").alias("label"))

        # Combine GT and non-GT
        vorpus_df = pl.concat([gt_origins, non_gt_df])
    else:
        vorpus_df = gt_origins

    # Sort by origin_id
    vorpus_df = vorpus_df.sort("origin_id")

    # Save
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    vorpus_df.write_csv(output_csv_path)
    logger.info(f"Saved {output_csv_path} ({len(vorpus_df)} records: "
                f"{len(gt_origins)} GT, {len(non_gt_results)} non-GT)")


def process_non_gt_only(
    input_dir: Path,
    output_dir: Path,
    csv_dir: Path,
    output_csv_dir: Path,
    target_total: int = 5000,
    max_duration_sec: int = 300,
    output_fps: float = 1.0,
    workers: int = 4,
    pack_tar: bool = False,
    resume: bool = True,
    min_duration_sec: int = 600,
) -> None:
    """
    只处理非 GT 视频，生成增量片段以达到目标总数。

    Args:
        input_dir: 输入视频目录
        output_dir: 输出帧目录
        csv_dir: 输入 CSV 目录
        output_csv_dir: 输出 CSV 目录
        target_total: 目标片段总数
        max_duration_sec: 每个片段的最大时长
        output_fps: 抽帧频率
        workers: 工作线程数
        pack_tar: 是否打包为 tar
        resume: 是否启用断点续传
        min_duration_sec: 最小视频时长（秒），过滤短视频
    """
    logger.info("=" * 60)
    logger.info("Processing non-GT videos only (incremental)")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取现有 vorpus.csv，计算当前片段数
    vorpus_csv = output_csv_dir / "vorpus.csv"
    if not vorpus_csv.exists():
        logger.error(f"vorpus.csv not found: {vorpus_csv}")
        logger.error("Please run 'all' or 'gt' command first to generate vorpus.csv")
        return

    existing_df = pl.read_csv(vorpus_csv)
    current_count = len(existing_df)

    # 初始化进度追踪器
    progress_dir = output_csv_dir / ".progress_non_gt"
    tracker = ProgressTracker(progress_dir)

    if not resume:
        tracker.clear()
        logger.info("Disabled resume, starting fresh")

    # 获取已保存的记录
    saved_results = tracker.get_saved_records()
    total_with_saved = current_count + len(saved_results)

    needed = target_total - total_with_saved

    logger.info(f"Current in vorpus.csv: {current_count}")
    logger.info(f"Saved in progress: {len(saved_results)}")
    logger.info(f"Target: {target_total}")
    logger.info(f"Need to generate: {needed}")

    if needed <= 0:
        logger.info("Already reached target, updating vorpus.csv with saved results...")
        if saved_results:
            new_non_gt_df = pl.DataFrame(saved_results, infer_schema_length=None).select([
                pl.col("origin_id"),
                pl.col("frame_count"),
                pl.col("output_fps"),
            ]).with_columns(pl.lit("non_gt").alias("label"))

            updated_df = pl.concat([existing_df, new_non_gt_df])
            updated_df = updated_df.sort("origin_id")
            updated_df.write_csv(vorpus_csv)
            logger.info(f"Updated {vorpus_csv}: {len(updated_df)} total clips")
        return

    # 2. 获取非 GT 视频列表
    gt_csv = csv_dir / "all.csv"
    gt_df = pl.read_csv(gt_csv, schema_overrides={"origin_id": pl.Utf8})
    gt_origins = set(gt_df["origin_id"].unique().to_list())

    all_videos = sorted(input_dir.glob("*.mp4"))
    non_gt_videos = [v for v in all_videos if v.stem not in gt_origins]
    logger.info(f"Total non-GT videos: {len(non_gt_videos)}")

    # 3. 过滤短视频，只保留长视频
    logger.info(f"Filtering videos shorter than {min_duration_sec}s ({min_duration_sec/60:.1f} min)...")
    long_videos = []
    for video_path in non_gt_videos:
        try:
            video_info = get_video_info(video_path)
            duration_sec = video_info["duration_sec"]
            if duration_sec >= min_duration_sec:
                long_videos.append((video_path, duration_sec))
        except Exception as e:
            logger.warning(f"Failed to get info for {video_path.name}: {e}")

    logger.info(f"Long videos (>= {min_duration_sec}s): {len(long_videos)}")

    if not long_videos:
        logger.error("No long videos available!")
        return

    # 4. 按时长降序排序（最长的在前）
    long_videos.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Longest video: {long_videos[0][1]:.1f}s ({long_videos[0][1]/60:.1f} min)")
    logger.info(f"Shortest video: {long_videos[-1][1]:.1f}s ({long_videos[-1][1]/60:.1f} min)")

    # 5. 获取已完成的视频并过滤
    completed_videos = tracker.get_completed_origins()
    if completed_videos:
        logger.info(f"Already completed videos: {len(completed_videos)}")

    pending_videos = [(v, d) for v, d in long_videos if v.stem not in completed_videos]
    logger.info(f"Pending long videos to process: {len(pending_videos)}")

    if not pending_videos:
        logger.warning("No more videos to process!")
        return

    # 6. 计算采样分配
    base_clips = needed // len(pending_videos)  # 商：每个视频的基础采样次数
    extra_clips = needed % len(pending_videos)  # 余数：需要额外采样的视频数量

    logger.info(f"Sampling strategy:")
    logger.info(f"  - All {len(pending_videos)} videos: {base_clips} clips each")
    if extra_clips > 0:
        logger.info(f"  - Longest {extra_clips} videos: +1 extra clip")

    # 7. 为每个视频分配采样次数
    video_clip_counts = {}
    for i, (video_path, duration) in enumerate(pending_videos):
        if i < extra_clips:
            # 最长的 extra_clips 个视频多采样 1 次
            video_clip_counts[video_path] = base_clips + 1
        else:
            video_clip_counts[video_path] = base_clips

    # 8. 并行处理
    all_results = list(saved_results)  # 从已保存的结果开始

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_non_gt_video_multi,
                video_path,
                output_dir,
                max_duration_sec,
                output_fps,
                video_clip_counts[video_path],  # 使用分配的采样次数
                pack_tar,
            ): video_path
            for video_path, _ in pending_videos
        }

        with tqdm(total=len(futures), desc="Processing non-GT videos") as pbar:
            for future in as_completed(futures):
                video_path = futures[future]
                try:
                    results = future.result()
                    if results:
                        tracker.save_origin_progress(video_path.stem, results)
                        all_results.extend(results)
                    pbar.update(1)

                    # 检查是否已达到目标
                    if len(all_results) >= needed:
                        logger.info("Reached target, stopping early")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                except Exception as e:
                    logger.error(f"Error processing {video_path.name}: {e}")
                    pbar.update(1)

    # 6. 更新 vorpus.csv
    logger.info(f"Total new clips generated: {len(all_results)}")

    if all_results:
        new_non_gt_df = pl.DataFrame(all_results, infer_schema_length=None).select([
            pl.col("origin_id"),
            pl.col("frame_count"),
            pl.col("output_fps"),
        ]).with_columns(pl.lit("non_gt").alias("label"))

        updated_df = pl.concat([existing_df, new_non_gt_df])
        updated_df = updated_df.sort("origin_id")
        updated_df.write_csv(vorpus_csv)

        logger.info(f"Updated {vorpus_csv}: {len(updated_df)} total clips")

    logger.info("=" * 60)
    logger.info("Non-GT processing complete!")
    logger.info("=" * 60)


def process_all(
    input_dir: Path,
    output_dir: Path,
    csv_dir: Path,
    output_csv_dir: Path,
    max_duration_sec: int = 300,
    output_fps: float = 1.0,
    workers: int = 4,
    gt_limit: int | None = None,
    non_gt_limit: int | None = None,
    pack_tar: bool = False,
    resume: bool = True,
) -> None:
    """
    Process all videos (GT and non-GT) and generate unified vorpus.csv.

    Args:
        input_dir: Input video directory
        output_dir: Output frames directory
        csv_dir: Input CSV directory
        output_csv_dir: Output CSV directory
        max_duration_sec: Maximum segment duration in seconds
        output_fps: Frame extraction rate (frames per second)
        workers: Number of worker threads
        gt_limit: Limit number of GT origins to process (for testing)
        non_gt_limit: Limit number of non-GT videos to process (for testing)
        pack_tar: If True, pack frames into tar files
        resume: If True, resume from previous progress (default: True)
    """
    logger.info("=" * 60)
    logger.info("Processing all Vorpus videos (GT + non-GT)")
    logger.info("=" * 60)

    # Step 1: Process GT videos
    process_vorpus(
        input_dir=input_dir,
        output_dir=output_dir,
        csv_dir=csv_dir,
        output_csv_dir=output_csv_dir,
        max_duration_sec=max_duration_sec,
        output_fps=output_fps,
        workers=workers,
        limit=gt_limit,
        pack_tar=pack_tar,
        resume=resume,
    )

    # Step 2: Process non-GT videos
    non_gt_results = process_non_gt_videos(
        input_dir=input_dir,
        output_dir=output_dir,
        csv_dir=csv_dir,
        max_duration_sec=max_duration_sec,
        output_fps=output_fps,
        workers=workers,
        limit=non_gt_limit,
        pack_tar=pack_tar,
    )

    # Step 3: Generate unified vorpus.csv
    gt_csv_path = output_csv_dir / "all.csv"
    vorpus_csv_path = output_csv_dir / "vorpus.csv"
    generate_vorpus_csv(gt_csv_path, non_gt_results, vorpus_csv_path)

    logger.info("=" * 60)
    logger.info("All processing complete!")
    logger.info("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process Vorpus dataset: extract frames and update metadata"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments for all subcommands
    def add_common_args(subparser):
        subparser.add_argument(
            "--input-dir",
            type=Path,
            required=True,
            help="Input video directory",
        )
        subparser.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="Output frames directory",
        )
        subparser.add_argument(
            "--csv-dir",
            type=Path,
            required=True,
            help="Input CSV directory",
        )
        subparser.add_argument(
            "--max-duration",
            type=int,
            default=300,
            help="Maximum segment duration in seconds (default: 300)",
        )
        subparser.add_argument(
            "--output-fps",
            type=float,
            default=1.0,
            help="Frame extraction rate in FPS (default: 1.0)",
        )
        subparser.add_argument(
            "--workers",
            type=int,
            default=8,
            help="Number of worker threads (default: 8)",
        )
        subparser.add_argument(
            "--pack-tar",
            action="store_true",
            help="Pack frames into tar files to save disk space",
        )

    # Subcommand: all (process all videos)
    all_parser = subparsers.add_parser("all", help="Process all videos (GT + non-GT)")
    add_common_args(all_parser)
    all_parser.add_argument(
        "--output-csv-dir",
        type=Path,
        required=True,
        help="Output CSV directory",
    )
    all_parser.add_argument(
        "--gt-limit",
        type=int,
        default=None,
        help="Limit number of GT origins to process (for testing)",
    )
    all_parser.add_argument(
        "--non-gt-limit",
        type=int,
        default=None,
        help="Limit number of non-GT videos to process (for testing)",
    )
    all_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume, reprocess all origins",
    )

    # Subcommand: gt (process videos with GT only)
    gt_parser = subparsers.add_parser("gt", help="Process videos with Ground Truth only")
    add_common_args(gt_parser)
    gt_parser.add_argument(
        "--output-csv-dir",
        type=Path,
        required=True,
        help="Output CSV directory",
    )
    gt_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of origins to process (for testing)",
    )
    gt_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume, reprocess all origins",
    )

    # Subcommand: non-gt (process non-GT videos only, incremental)
    non_gt_parser = subparsers.add_parser(
        "non-gt", help="Process non-GT videos only (incremental)"
    )
    add_common_args(non_gt_parser)
    non_gt_parser.add_argument(
        "--output-csv-dir",
        type=Path,
        required=True,
        help="Output CSV directory (must contain existing vorpus.csv)",
    )
    non_gt_parser.add_argument(
        "--target-total",
        type=int,
        default=5000,
        help="Target total number of clips (default: 5000)",
    )
    non_gt_parser.add_argument(
        "--min-duration",
        type=int,
        default=600,
        help="Minimum video duration in seconds (default: 600, i.e., 10 minutes)",
    )
    non_gt_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume, reprocess all non-GT videos",
    )

    args = parser.parse_args()

    if args.command == "all":
        process_all(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            csv_dir=args.csv_dir,
            output_csv_dir=args.output_csv_dir,
            max_duration_sec=args.max_duration,
            output_fps=args.output_fps,
            workers=args.workers,
            gt_limit=args.gt_limit,
            non_gt_limit=args.non_gt_limit,
            pack_tar=args.pack_tar,
            resume=not args.no_resume,
        )
    elif args.command == "gt":
        process_vorpus(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            csv_dir=args.csv_dir,
            output_csv_dir=args.output_csv_dir,
            max_duration_sec=args.max_duration,
            output_fps=args.output_fps,
            workers=args.workers,
            limit=args.limit,
            pack_tar=args.pack_tar,
            resume=not args.no_resume,
        )
    elif args.command == "non-gt":
        process_non_gt_only(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            csv_dir=args.csv_dir,
            output_csv_dir=args.output_csv_dir,
            target_total=args.target_total,
            max_duration_sec=args.max_duration,
            output_fps=args.output_fps,
            workers=args.workers,
            pack_tar=args.pack_tar,
            resume=not args.no_resume,
            min_duration_sec=args.min_duration,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
