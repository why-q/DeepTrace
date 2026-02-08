"""Main processing logic for Vorpus dataset video clipping and metadata update."""

import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from loguru import logger
from tqdm import tqdm

from .utils import (
    clip_video,
    frames_to_seconds,
    generate_video_hash,
    get_video_info,
)


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

    Records are sorted by gt_start_f, then grouped such that if the gap
    between consecutive GT ranges is <= max_gap_frames, they are merged
    into the same cluster.

    Args:
        records: DataFrame with gt_start_f and gt_end_f columns
        max_gap_frames: Maximum gap in frames to merge clusters

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
        gap = gt_starts[i] - current_end

        if gap <= max_gap_frames:
            # Merge into current cluster
            current_end = max(current_end, gt_ends[i])
        else:
            # Save current cluster and start new one
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


def process_single_origin(
    origin_id: str,
    records: pl.DataFrame,
    input_dir: Path,
    output_dir: Path,
    max_duration_frames: int,
) -> list[dict]:
    """
    Process a single origin video: cluster GT ranges and clip videos.

    Args:
        origin_id: Original video ID (string, e.g., "1379")
        records: DataFrame of records for this origin
        input_dir: Input video directory
        output_dir: Output video directory
        max_duration_frames: Maximum duration in frames

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

        # Generate new origin ID
        new_origin_hash = generate_video_hash()
        new_origin_path = output_dir / f"{new_origin_hash}.mp4"

        # Clip video
        start_sec = frames_to_seconds(clip_start, video_fps)
        duration_sec = frames_to_seconds(clip_end - clip_start, video_fps)

        success = clip_video(
            origin_video_path,
            new_origin_path,
            start_sec,
            duration_sec,
            copy_streams=False,
        )

        if not success:
            logger.error(
                f"Failed to clip video for origin {origin_id}, cluster {cluster_idx}"
            )
            continue

        # Update records - preserve all original columns
        for row_idx in range(len(cluster_df)):
            row_dict = cluster_df.row(row_idx, named=True)

            # Update origin_id and gt frame numbers
            row_dict["origin_id"] = new_origin_hash
            row_dict["gt_start_f"] = row_dict["gt_start_f"] - clip_start
            row_dict["gt_end_f"] = row_dict["gt_end_f"] - clip_start

            updated_records.append(row_dict)

    return updated_records


def process_vorpus(
    input_dir: Path,
    output_dir: Path,
    csv_dir: Path,
    output_csv_dir: Path,
    max_duration_sec: int = 300,
    workers: int = 4,
    limit: int | None = None,
) -> None:
    """
    Main entry point for processing Vorpus dataset.

    Args:
        input_dir: Input video directory
        output_dir: Output video directory
        csv_dir: Input CSV directory
        output_csv_dir: Output CSV directory
        max_duration_sec: Maximum video duration in seconds
        workers: Number of worker threads
        limit: If set, only process this many origin videos (for testing)
    """
    logger.info("Starting Vorpus dataset processing")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Max duration: {max_duration_sec}s")
    logger.info(f"Workers: {workers}")
    if limit:
        logger.info(f"Limit: {limit} origins (test mode)")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    # Read all.csv
    all_csv_path = csv_dir / "all.csv"
    logger.info(f"Reading {all_csv_path}")
    df = pl.read_csv(all_csv_path, schema_overrides={"origin_id": pl.Utf8})

    logger.info(f"Total records: {len(df)}")
    logger.info(f"Unique origins: {df['origin_id'].n_unique()}")

    # Get FPS from CSV (all videos should have same fps)
    fps = df["fps"].head(1).item()
    logger.info(f"Video FPS (from CSV): {fps}")

    max_duration_frames = int(max_duration_sec * fps)

    # Group by origin
    grouped = group_records_by_origin(df)
    logger.info(f"Total origin videos: {len(grouped)}")

    # Apply limit for testing
    if limit:
        origin_ids = list(grouped.keys())[:limit]
        grouped = {k: grouped[k] for k in origin_ids}
        logger.info(f"Processing {len(grouped)} origin videos (limited)")

    # Process origins in parallel
    all_updated_records = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_single_origin,
                origin_id,
                records,
                input_dir,
                output_dir,
                max_duration_frames,
            ): origin_id
            for origin_id, records in grouped.items()
        }

        with tqdm(total=len(futures), desc="Processing origins") as pbar:
            for future in as_completed(futures):
                origin_id = futures[future]
                try:
                    updated_records = future.result()
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

    new_df = pl.DataFrame(all_updated_records)

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


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process Vorpus dataset: clip videos and update metadata"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input video directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output video directory",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        required=True,
        help="Input CSV directory",
    )
    parser.add_argument(
        "--output-csv-dir",
        type=Path,
        required=True,
        help="Output CSV directory",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=300,
        help="Maximum video duration in seconds (default: 300)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of origins to process (for testing)",
    )

    args = parser.parse_args()

    process_vorpus(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        csv_dir=args.csv_dir,
        output_csv_dir=args.output_csv_dir,
        max_duration_sec=args.max_duration,
        workers=args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
