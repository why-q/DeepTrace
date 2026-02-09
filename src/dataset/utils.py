"""Utility functions for video processing."""

import shutil
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger


def get_video_info(video_path: Path) -> dict:
    """
    Get video metadata including fps and total frame count.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing:
            - fps: Frames per second
            - total_frames: Total number of frames
            - duration_sec: Duration in seconds

    Raises:
        ValueError: If video cannot be opened or metadata cannot be read
    """
    if not video_path.exists():
        raise ValueError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0

        if fps <= 0 or total_frames <= 0:
            raise ValueError(f"Invalid video metadata: fps={fps}, frames={total_frames}")

        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration_sec,
        }
    finally:
        cap.release()


def clip_video(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float,
    copy_streams: bool = False,
) -> bool:
    """
    Clip a video segment using FFmpeg.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        start_sec: Start time in seconds
        duration_sec: Duration in seconds
        copy_streams: If True, copy streams without re-encoding (faster but less precise)

    Returns:
        True if successful, False otherwise
    """
    if not input_path.exists():
        logger.error(f"Input video not found: {input_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build FFmpeg command
    # -ss before -i for fast seek
    cmd = [
        "ffmpeg",
        "-ss", str(start_sec),
        "-i", str(input_path),
        "-t", str(duration_sec),
    ]

    if copy_streams:
        # Copy streams without re-encoding (fast but may not be frame-accurate)
        cmd.extend(["-c:v", "copy", "-c:a", "copy"])
    else:
        # Re-encode with H.264 and AAC (slower but frame-accurate)
        cmd.extend(["-c:v", "libx264", "-c:a", "aac"])

    cmd.extend(["-y", str(output_path)])

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        logger.debug(f"Clipped video: {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during video clipping: {e}")
        return False


def generate_video_hash() -> str:
    """
    Generate a 32-character UUID hash without hyphens.

    Returns:
        32-character hexadecimal string
    """
    return uuid.uuid4().hex


def frames_to_seconds(frames: int, fps: float) -> float:
    """
    Convert frame count to seconds.

    Args:
        frames: Number of frames
        fps: Frames per second

    Returns:
        Time in seconds
    """
    return frames / fps if fps > 0 else 0


def seconds_to_frames(seconds: float, fps: float) -> int:
    """
    Convert seconds to frame count.

    Args:
        seconds: Time in seconds
        fps: Frames per second

    Returns:
        Number of frames (rounded)
    """
    return int(round(seconds * fps))


def frame_to_image_idx(frame_num: int, video_fps: float, output_fps: float) -> int:
    """
    Convert original video frame number to extracted image index.

    Args:
        frame_num: Original video frame number
        video_fps: Original video frame rate (e.g., 30)
        output_fps: Extraction frame rate (e.g., 1)

    Returns:
        Image index (1-based)
    """
    second = frame_num / video_fps
    image_idx = int(second * output_fps) + 1  # 1-based index
    return image_idx


def pack_frames_to_tar(frames_dir: Path, tar_path: Path) -> bool:
    """
    Pack a directory of frames into a tar archive.

    Args:
        frames_dir: Directory containing frame images
        tar_path: Output tar file path

    Returns:
        True if successful, False otherwise
    """
    try:
        tar_path.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tar_path, "w") as tar:
            # Add all jpg files, sorted by name
            jpg_files = sorted(
                [f for f in frames_dir.glob("*.jpg") if not f.name.startswith("._")]
            )
            for jpg_file in jpg_files:
                tar.add(jpg_file, arcname=jpg_file.name)

        logger.debug(f"Packed {len(jpg_files)} frames to {tar_path.name}")
        return True

    except Exception as e:
        logger.error(f"Error packing frames to tar: {e}")
        return False


def extract_frames(
    input_path: Path,
    output_dir: Path,
    start_frame: int,
    end_frame: int,
    video_fps: float,
    output_fps: float = 1.0,
    pack_tar: bool = False,
) -> int:
    """
    Extract frames from a video at specified intervals using FFmpeg.

    Extracts middle frames of each second interval, not the first frame.
    For 1 FPS: extracts frame at 0.5s, 1.5s, 2.5s, etc.
    For 2 FPS: extracts frames at 0.33s, 0.67s, 1.33s, 1.67s, etc.

    Args:
        input_path: Path to input video
        output_dir: Directory to save extracted frames (or tar file path if pack_tar=True)
        start_frame: Start frame number (inclusive)
        end_frame: End frame number (inclusive)
        video_fps: Video frame rate
        output_fps: Extraction rate (frames per second)
        pack_tar: If True, pack frames into a tar file and delete the directory.
                  output_dir will be treated as the tar file path (without .tar extension).

    Returns:
        Number of frames extracted
    """
    if not input_path.exists():
        logger.error(f"Input video not found: {input_path}")
        return 0

    # If packing to tar, use a temp directory for extraction
    if pack_tar:
        temp_dir = Path(tempfile.mkdtemp())
        frames_dir = temp_dir
        tar_path = output_dir.parent / f"{output_dir.name}.tar"
    else:
        frames_dir = output_dir
        tar_path = None

    frames_dir.mkdir(parents=True, exist_ok=True)

    # Calculate time range
    start_sec = start_frame / video_fps
    end_sec = end_frame / video_fps
    duration_sec = end_sec - start_sec

    # Calculate offset to extract middle frames
    # For 1 FPS: offset = 0.5s (middle of each second)
    # For 2 FPS: offset = 0.167s (1/6 of a second, first middle point)
    offset_sec = 0.5 / output_fps

    # Adjust start time to get middle frames
    adjusted_start = start_sec + offset_sec

    # Build FFmpeg command
    # -hwaccel videotoolbox for Mac GPU acceleration (falls back to CPU if unavailable)
    # -ss before -i for fast seek
    # fps filter to extract at specified rate
    # -q:v 2 for high quality JPEG (scale 2-31, lower is better)
    cmd = [
        "ffmpeg",
        "-hwaccel", "videotoolbox",
        "-ss", str(adjusted_start),
        "-i", str(input_path),
        "-t", str(duration_sec - offset_sec),
        "-vf", f"fps={output_fps}",
        "-q:v", "2",
        "-start_number", "1",
        str(frames_dir / "%04d.jpg"),
        "-y",
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # Count extracted frames (exclude macOS ._ files)
        extracted_count = len([f for f in frames_dir.glob("*.jpg") if not f.name.startswith("._")])
        logger.debug(f"Extracted {extracted_count} frames to {frames_dir.name}")

        # Pack to tar if requested
        if pack_tar and extracted_count > 0:
            if pack_frames_to_tar(frames_dir, tar_path):
                # Clean up temp directory
                shutil.rmtree(temp_dir)
            else:
                logger.error(f"Failed to pack frames to tar, keeping temp dir: {temp_dir}")
                return 0

        return extracted_count

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error extracting frames: {e.stderr[:500]}")
        if pack_tar:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return 0
    except Exception as e:
        logger.error(f"Unexpected error during frame extraction: {e}")
        if pack_tar:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return 0

