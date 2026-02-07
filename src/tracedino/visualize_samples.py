"""Visualization script for TraceDINO training samples.

This script helps verify the correctness of:
1. Hard negative sampling (should be >15 seconds away from anchor)
2. Positive sample generation (original, cropped, face-blurred)
3. Anchor frame sampling (from first and last second of query video)
"""

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.tracedino.config import TraceDINOConfig
from src.tracedino.dataset.augmentations import TraceDINOTransform
from src.tracedino.dataset.dataset import TraceDINODataset
from src.tracedino.dataset.metadata import load_metadata


def visualize_sample(
    dataset: TraceDINODataset,
    idx: int,
    save_path: Path = None,
    show: bool = True,
):
    """
    Visualize a single training sample.

    Args:
        dataset: TraceDINODataset instance
        idx: Sample index
        save_path: Path to save the visualization (optional)
        show: Whether to display the plot
    """
    # Get raw sample data (before transform)
    video_idx = idx // dataset.n_anchor_frames
    anchor_idx = idx % dataset.n_anchor_frames
    meta = dataset.metadata[video_idx]

    # Get anchor frame
    anchor_frame = dataset._get_anchor_frame(meta, anchor_idx)

    # Get positives
    positives = dataset._get_positives(meta, anchor_idx)

    # Get hard negatives
    hard_negatives = dataset._get_hard_negatives(meta, anchor_idx)

    # Calculate frame numbers for annotation
    anchor_source_frame = dataset._map_anchor_to_source(meta, anchor_idx)
    safety_radius_frames = int(dataset.safety_radius_sec * meta.fps)

    # Get hard negative frame numbers
    source_dir = dataset.source_frame_dir / meta.origin_id
    available_frames = sorted([int(f.stem) for f in source_dir.glob("*.jpg")])
    valid_frames = [
        f for f in available_frames
        if abs(f - anchor_source_frame) > safety_radius_frames
    ]

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"Sample {idx} | Video: {meta.id[:8]}... | Source: {meta.origin_id}\n"
        f"Anchor maps to source frame {anchor_source_frame} | Safety radius: {safety_radius_frames} frames ({dataset.safety_radius_sec}s)",
        fontsize=12,
    )

    # Convert BGR to RGB for display
    def bgr_to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Row 1: Anchor + 3 Positives
    axes[0, 0].imshow(bgr_to_rgb(anchor_frame))
    axes[0, 0].set_title(f"Anchor\n(Query video, anchor_idx={anchor_idx})")
    axes[0, 0].axis("off")

    positive_titles = ["Positive 1\n(Original source)", "Positive 2\n(Cropped)", "Positive 3\n(Face-blurred)"]
    for i, (pos, title) in enumerate(zip(positives, positive_titles)):
        axes[0, i + 1].imshow(bgr_to_rgb(pos))
        axes[0, i + 1].set_title(title)
        axes[0, i + 1].axis("off")

    # Row 2: 3 Hard Negatives + Info
    for i, neg in enumerate(hard_negatives):
        axes[1, i].imshow(bgr_to_rgb(neg))
        axes[1, i].set_title(f"Hard Negative {i + 1}")
        axes[1, i].axis("off")

    # Info panel
    axes[1, 3].axis("off")
    info_text = (
        f"Video Metadata:\n"
        f"  Query ID: {meta.id[:16]}...\n"
        f"  Source ID: {meta.origin_id}\n"
        f"  Category: {meta.category}\n"
        f"  Celebrity: {meta.celebrity}\n"
        f"  FPS: {meta.fps}\n"
        f"  GT Range: [{meta.gt_start_f}, {meta.gt_end_f}]\n\n"
        f"Sampling Info:\n"
        f"  Anchor source frame: {anchor_source_frame}\n"
        f"  Safety radius: {safety_radius_frames} frames\n"
        f"  Valid neg frames: {len(valid_frames)}/{len(available_frames)}\n"
    )
    axes[1, 3].text(0.1, 0.9, info_text, transform=axes[1, 3].transAxes,
                    fontsize=10, verticalalignment="top", fontfamily="monospace")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_hard_negative_distribution(
    dataset: TraceDINODataset,
    n_samples: int = 100,
    save_path: Path = None,
):
    """
    Visualize the distribution of hard negative frame distances.

    Args:
        dataset: TraceDINODataset instance
        n_samples: Number of samples to analyze
        save_path: Path to save the visualization (optional)
    """
    distances = []
    violations = 0

    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    for idx in indices:
        video_idx = idx // dataset.n_anchor_frames
        anchor_idx = idx % dataset.n_anchor_frames
        meta = dataset.metadata[video_idx]

        anchor_source_frame = dataset._map_anchor_to_source(meta, anchor_idx)
        safety_radius_frames = int(dataset.safety_radius_sec * meta.fps)

        # Get hard negative frame numbers
        source_dir = dataset.source_frame_dir / meta.origin_id
        available_frames = sorted([int(f.stem) for f in source_dir.glob("*.jpg")])
        valid_frames = [
            f for f in available_frames
            if abs(f - anchor_source_frame) > safety_radius_frames
        ]

        if len(valid_frames) >= dataset.n_hard_negatives:
            selected = random.sample(valid_frames, dataset.n_hard_negatives)
        elif len(valid_frames) > 0:
            selected = random.choices(valid_frames, k=dataset.n_hard_negatives)
        else:
            selected = random.choices(available_frames, k=dataset.n_hard_negatives)
            violations += 1

        for frame_no in selected:
            dist_frames = abs(frame_no - anchor_source_frame)
            dist_seconds = dist_frames / meta.fps
            distances.append(dist_seconds)

    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(distances, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(x=dataset.safety_radius_sec, color="r", linestyle="--",
                    label=f"Safety radius ({dataset.safety_radius_sec}s)")
    axes[0].set_xlabel("Distance from anchor (seconds)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Hard Negative Distance Distribution")
    axes[0].legend()

    # Statistics
    axes[1].axis("off")
    stats_text = (
        f"Hard Negative Sampling Statistics\n"
        f"{'=' * 40}\n\n"
        f"Samples analyzed: {n_samples}\n"
        f"Total hard negatives: {len(distances)}\n\n"
        f"Distance Statistics (seconds):\n"
        f"  Min: {min(distances):.2f}\n"
        f"  Max: {max(distances):.2f}\n"
        f"  Mean: {np.mean(distances):.2f}\n"
        f"  Median: {np.median(distances):.2f}\n\n"
        f"Safety radius: {dataset.safety_radius_sec}s\n"
        f"Violations (no valid frames): {violations}/{n_samples}\n"
        f"Below threshold: {sum(1 for d in distances if d < dataset.safety_radius_sec)}/{len(distances)}"
    )
    axes[1].text(0.1, 0.9, stats_text, transform=axes[1].transAxes,
                 fontsize=12, verticalalignment="top", fontfamily="monospace")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved distribution plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize TraceDINO training samples")
    parser.add_argument("--mode", type=str, default="sample",
                        choices=["sample", "distribution"],
                        help="Visualization mode: 'sample' for single sample, 'distribution' for statistics")
    parser.add_argument("--idx", type=int, default=None,
                        help="Sample index to visualize (random if not specified)")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples for distribution analysis")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save visualizations")
    args = parser.parse_args()

    # Load config
    config = TraceDINOConfig()

    # Create dataset
    transform = TraceDINOTransform(image_size=config.image_size, is_training=True)
    dataset = TraceDINODataset(
        metadata_csv=config.train_csv,
        query_video_dir=config.query_video_dir,
        source_frame_dir=config.source_frame_dir,
        transform=transform,
        n_anchor_frames=config.n_anchor_frames,
        n_hard_negatives=config.n_hard_negatives,
        safety_radius_sec=config.safety_radius_seconds,
        is_training=True,
    )

    print(f"Dataset loaded: {len(dataset)} samples")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "sample":
        idx = args.idx if args.idx is not None else random.randint(0, len(dataset) - 1)
        save_path = save_dir / f"sample_{idx}.png" if save_dir else None
        visualize_sample(dataset, idx, save_path=save_path)

    elif args.mode == "distribution":
        save_path = save_dir / "hard_negative_distribution.png" if save_dir else None
        visualize_hard_negative_distribution(dataset, n_samples=args.n_samples, save_path=save_path)


if __name__ == "__main__":
    main()
