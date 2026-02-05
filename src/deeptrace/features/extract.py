"""Batch feature extraction script for video frames.

Extracts features from a root directory containing multiple video frame
subfolders, saving each video as a single .npy file.
"""

import argparse
from pathlib import Path
from typing import Optional
import random

from loguru import logger

from .extractor import FeatureExtractor
from .extractors import (
    DINOv3FeatureExtractor,
    DINOv3OutputMode,
    ISC21FeatureExtractor,
    SSCDFeatureExtractor,
)


def extract_features_from_videos(
    frames_dir: Path,
    output_dir: Path,
    feature_extractor: FeatureExtractor,
    frame_pattern: str = "*.png",
    overwrite: bool = False,
    numeric_sort: bool = True,
) -> None:
    """Batch extract features from video frame directories.

    Expected directory structure:
    frames_dir/
        video1/
            00001.png
            00002.png
            ...
        video2/
            00001.png
            00002.png
            ...

    Output structure:
    output_dir/
        video1.npy
        video2.npy
        ...

    Args:
        frames_dir: Root directory of video frames
        output_dir: Feature output directory
        feature_extractor: Feature extractor instance
        frame_pattern: Frame file matching pattern
        overwrite: Whether to overwrite existing feature files
        numeric_sort: Whether to perform natural numeric sorting
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all video subfolders
    video_folders = sorted([d for d in frames_dir.iterdir() if d.is_dir()])

    # Shuffle video_folders list randomly
    random.shuffle(video_folders)

    if not video_folders:
        logger.error(f"No subfolders found in {frames_dir}")
        return

    logger.info(f"Found {len(video_folders)} video folders")

    # Process each video
    success_count = 0
    skip_count = 0
    error_count = 0

    for idx, video_folder in enumerate(video_folders, 1):
        video_name = video_folder.name

        # Check for DINOv3 "both" mode
        is_both_mode = (
            isinstance(feature_extractor, DINOv3FeatureExtractor) and
            feature_extractor.output_mode == DINOv3OutputMode.BOTH
        )

        if is_both_mode:
            output_file_cls = output_dir / f"{video_name}_cls.npy"
            output_file_patch = output_dir / f"{video_name}_patch.npy"
            # Check if both exist
            if output_file_cls.exists() and output_file_patch.exists() and not overwrite:
                logger.info(f"[{idx}/{len(video_folders)}] Skipping {video_name} (already exists)")
                skip_count += 1
                continue
            output_file = output_dir / f"{video_name}.npy"  # Base path for extract_from_folder
        else:
            output_file = output_dir / f"{video_name}.npy"
            if output_file.exists() and not overwrite:
                logger.info(f"[{idx}/{len(video_folders)}] Skipping {video_name} (already exists)")
                skip_count += 1
                continue

        try:
            logger.info(f"[{idx}/{len(video_folders)}] Processing {video_name}")

            # Extract and save features
            feature_extractor.extract_from_folder(
                folder_path=video_folder,
                pattern=frame_pattern,
                save_path=output_file,
                numeric_sort=numeric_sort,
            )

            success_count += 1

        except Exception as e:
            logger.error(f"Error processing {video_name}: {e}")
            error_count += 1

    # Output statistics
    logger.info("=" * 60)
    logger.info(f"Processing complete!")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Skipped: {skip_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Total: {len(video_folders)}")
    logger.info("=" * 60)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Batch extract features from video frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Root directory of video frames (containing multiple video subfolders)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Feature output directory",
    )

    parser.add_argument(
        "--extractor",
        type=str,
        default="isc21",
        choices=["isc21", "sscd", "dinov3"],
        help="Feature extractor type",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="isc_ft_v107",
        help="Model name or path: "
             "For DINOv3: facebook/dinov3-vits16-pretrain-lvd1689m, "
             "facebook/dinov3-vitb16-pretrain-lvd1689m, "
             "facebook/dinov3-vitl16-pretrain-lvd1689m, or local path like pretrained/dinov3/dinov3-vitb16; "
             "For ISC21: isc_ft_v107/isc_ft_v110",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="SSCD TorchScript model file path (for SSCD only)",
    )

    parser.add_argument(
        "--resize-size",
        type=int,
        default=None,
        help="Image preprocessing size (SSCD default 288, DINOv3 default 224)",
    )

    parser.add_argument(
        "--resize-mode",
        type=str,
        default="small_edge",
        choices=["small_edge", "square"],
        help="Resize mode (for SSCD only): small_edge (preserve aspect ratio) or square (force square)",
    )

    parser.add_argument(
        "--frame-pattern",
        type=str,
        default="*.png",
        help="Frame file matching pattern",
    )

    parser.add_argument(
        "--numeric-sort",
        action="store_true",
        help="Perform natural numeric sorting by filename",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Specify GPU number (e.g., 0, 1, 2), if not specified uses all visible GPUs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader worker processes",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision inference (CUDA only)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature files",
    )

    parser.add_argument(
        "--output-mode",
        type=str,
        default="cls",
        choices=["cls", "patch", "both"],
        help="DINOv3 output mode: 'cls' (CLS token only, default), "
             "'patch' (all-layer patch tokens), 'both' (saves _cls.npy and _patch.npy separately). "
             "Only applicable to DINOv3 extractor.",
    )

    # Single folder mode
    parser.add_argument(
        "--single-folder",
        type=str,
        default=None,
        help="Process only the specified single subfolder name, output as .npy with same name",
    )

    args = parser.parse_args()

    # Handle GPU parameter
    if args.gpu is not None:
        if args.device == "cpu":
            logger.warning("--gpu parameter specified but --device is cpu, ignoring --gpu")
            device = "cpu"
        else:
            device = f"cuda:{args.gpu}"
            logger.info(f"Using specified GPU: {args.gpu}")
    else:
        device = args.device

    # Create feature extractor
    if args.extractor == "isc21":
        extractor = ISC21FeatureExtractor(
            weight_name=args.model_name,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=args.amp,
        )
    elif args.extractor == "sscd":
        resize_size = args.resize_size if args.resize_size is not None else 288
        extractor = SSCDFeatureExtractor(
            model_path=args.model_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=args.amp,
            resize_size=resize_size,
            resize_mode=args.resize_mode,
        )
    elif args.extractor == "dinov3":
        resize_size = args.resize_size if args.resize_size is not None else 224
        # Support both online Hugging Face models and local paths
        model_path = args.model_name
        if not model_path.startswith("facebook/"):
            # Local path - resolve it relative to project root
            from pathlib import Path
            # If it's a relative path, make it absolute from current directory
            model_path = str(Path(args.model_name).resolve())

        extractor = DINOv3FeatureExtractor(
            model_name_or_path=model_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=args.amp,
            output_mode=args.output_mode,
        )
    else:
        raise ValueError(f"Unknown feature extractor: {args.extractor}")

    # Single subfolder mode
    if args.single_folder is not None:
        folder = Path(args.frames_dir) / args.single_folder
        output = Path(args.output_dir) / f"{args.single_folder}.npy"
        extractor.extract_from_folder(
            folder_path=folder,
            pattern=args.frame_pattern,
            save_path=output,
            numeric_sort=args.numeric_sort or True,
        )
    else:
        # Batch extract features
        extract_features_from_videos(
            frames_dir=Path(args.frames_dir),
            output_dir=Path(args.output_dir),
            feature_extractor=extractor,
            frame_pattern=args.frame_pattern,
            overwrite=args.overwrite,
            numeric_sort=args.numeric_sort or True,
        )


if __name__ == "__main__":
    main()
