"""Utility functions for video processing."""

import subprocess
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
