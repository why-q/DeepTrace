"""
Face Detection Agent

Extracts a random frame from each video in a directory and detects faces,
saving bounding box coordinates to a JSON file and visualization images.

This module provides an abstraction layer for face detection algorithms,
allowing easy switching between different detection methods.
"""

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


class BaseFaceDetector(ABC):
    """
    Abstract base class for face detectors.
    
    Any face detection algorithm should inherit from this class and implement
    the detect_faces method. This allows seamless integration of different
    detection algorithms without changing the rest of the pipeline.
    """

    @abstractmethod
    def detect_faces(
        self, frame: "cv2.typing.MatLike"
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame.

        Args:
            frame: BGR image frame from OpenCV

        Returns:
            List of tuples (x, y, width, height) for each detected face
        """
        pass

    @property
    def name(self) -> str:
        """Return the name of the detector for logging purposes."""
        return self.__class__.__name__


class HaarCascadeFaceDetector(BaseFaceDetector):
    """Detects faces using OpenCV's Haar Cascade classifier."""

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ):
        """
        Initialize the Haar Cascade face detector.

        Args:
            scale_factor: Parameter specifying how much the image size is reduced
                         at each image scale (default: 1.1)
            min_neighbors: How many neighbors each candidate rectangle should have
                          to retain it (default: 5)
            min_size: Minimum possible object size (default: (30, 30))
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        # Load pre-trained Haar Cascade model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_faces(
        self, frame: "cv2.typing.MatLike"
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame using Haar Cascade.

        Args:
            frame: BGR image frame from OpenCV

        Returns:
            List of tuples (x, y, width, height) for each detected face
        """
        # Convert to grayscale for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        # Convert numpy arrays to list of tuples
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


# Backward compatibility alias
FaceDetector = HaarCascadeFaceDetector


def get_random_frame(video_path: Path) -> Tuple["cv2.typing.MatLike", int]:
    """
    Extract a random frame from a video.

    Args:
        video_path: Path to the video file

    Returns:
        Tuple of (frame, frame_number)

    Raises:
        ValueError: If video cannot be opened or has no frames
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    # Select random frame number
    frame_number = random.randint(0, total_frames - 1)

    # Seek to the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Failed to read frame {frame_number} from {video_path}")

    return frame, frame_number


def draw_face_boxes(
    frame: "cv2.typing.MatLike",
    faces: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> "cv2.typing.MatLike":
    """
    Draw bounding boxes on detected faces.

    Args:
        frame: Input image frame
        faces: List of (x, y, width, height) tuples for each face
        color: BGR color for the bounding box (default: green)
        thickness: Line thickness in pixels (default: 2)

    Returns:
        Frame with drawn bounding boxes
    """
    annotated_frame = frame.copy()

    for idx, (x, y, w, h) in enumerate(faces, 1):
        # Draw rectangle
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)

        # Add label with face number
        label = f"Face {idx}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y - 10, label_size[1] + 10)

        # Draw label background
        cv2.rectangle(
            annotated_frame,
            (x, label_y - label_size[1] - 5),
            (x + label_size[0] + 5, label_y + 5),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            annotated_frame,
            label,
            (x + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return annotated_frame


def process_video_folder(
    video_dir: Path,
    output_json: Path,
    detector: Optional[BaseFaceDetector] = None,
    video_extensions: Optional[List[str]] = None,
    save_visualizations: bool = True,
    visualization_dir: Optional[Path] = None,
) -> Dict:
    """
    Process all videos in a folder, detect faces in random frames.

    Args:
        video_dir: Directory containing video files
        output_json: Path to save the JSON results
        detector: Face detector instance (default: HaarCascadeFaceDetector)
        video_extensions: List of video file extensions to process
                         (default: ['.mp4', '.avi', '.mov', '.mkv'])
        save_visualizations: Whether to save annotated images (default: True)
        visualization_dir: Directory to save visualization images
                          (default: same directory as output_json)

    Returns:
        Dictionary containing face detection results
    """
    if video_extensions is None:
        video_extensions = [".mp4", ".avi", ".mov", ".mkv"]

    if detector is None:
        detector = HaarCascadeFaceDetector()

    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise ValueError(f"Video directory does not exist: {video_dir}")

    # Setup output paths
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if visualization_dir is None:
        visualization_dir = output_json.parent / "visualizations"

    if save_visualizations:
        visualization_dir = Path(visualization_dir)
        visualization_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"**/*{ext}"))
        video_files.extend(video_dir.glob(f"**/*{ext.upper()}"))

    if not video_files:
        print(f"No video files found in {video_dir}")
        return {}

    print(f"{'=' * 60}")
    print(f"Face Detection Pipeline")
    print(f"{'=' * 60}")
    print(f"Detector: {detector.name}")
    print(f"Video directory: {video_dir}")
    print(f"Found {len(video_files)} video files")
    print(f"Save visualizations: {save_visualizations}")
    if save_visualizations:
        print(f"Visualization directory: {visualization_dir}")
    print(f"{'=' * 60}\n")

    # Process each video
    results = {}
    successful = 0
    failed = 0

    for video_path in video_files:
        try:
            # Get relative path for cleaner output
            rel_path = video_path.relative_to(video_dir)
            video_stem = video_path.stem

            # Extract random frame
            frame, frame_number = get_random_frame(video_path)

            # Detect faces
            faces = detector.detect_faces(frame)

            # Save visualization if requested
            visualization_path = None
            if save_visualizations:
                annotated_frame = draw_face_boxes(frame, faces)
                vis_filename = f"{video_stem}_frame{frame_number}.jpg"
                visualization_path = visualization_dir / vis_filename
                cv2.imwrite(str(visualization_path), annotated_frame)

            # Store results
            result_data = {
                "frame_number": frame_number,
                "frame_shape": {
                    "height": frame.shape[0],
                    "width": frame.shape[1],
                    "channels": frame.shape[2],
                },
                "faces": [
                    {"x": x, "y": y, "width": w, "height": h} for x, y, w, h in faces
                ],
                "face_count": len(faces),
            }

            if save_visualizations:
                result_data["visualization"] = str(
                    visualization_path.relative_to(output_json.parent)
                )

            results[str(rel_path)] = result_data

            successful += 1
            vis_info = (
                f" [saved to {vis_filename}]" if save_visualizations else ""
            )
            print(
                f"✓ {rel_path}: {len(faces)} face(s) detected "
                f"(frame {frame_number}){vis_info}"
            )

        except Exception as e:
            failed += 1
            print(f"✗ {rel_path}: Error - {str(e)}")
            results[str(rel_path)] = {"error": str(e)}

    # Save results to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Processing complete!")
    print(f"Successful: {successful} | Failed: {failed}")
    print(f"Results saved to: {output_json}")
    if save_visualizations:
        print(f"Visualizations saved to: {visualization_dir}")
    print(f"{'=' * 60}")

    return results


def main():
    """CLI entry point for face detection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect faces in random frames from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m deeptrace.preprocessing.face_detection --video-dir videos --output results.json

  # Without saving visualizations
  python -m deeptrace.preprocessing.face_detection --video-dir videos --output results.json --no-vis

  # Custom visualization directory
  python -m deeptrace.preprocessing.face_detection --video-dir videos --output results.json --vis-dir images

  # With custom detector parameters (via Python API)
  from deeptrace.preprocessing import HaarCascadeFaceDetector, process_video_folder
  detector = HaarCascadeFaceDetector(scale_factor=1.05, min_neighbors=3)
  process_video_folder("videos", "results.json", detector=detector)
        """,
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="face_detection_results.json",
        help="Output JSON file path (default: face_detection_results.json)",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".mp4", ".avi", ".mov", ".mkv"],
        help="Video file extensions to process (default: .mp4 .avi .mov .mkv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable saving visualization images",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        help="Directory to save visualization images (default: <output_dir>/visualizations)",
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Process videos
    process_video_folder(
        video_dir=Path(args.video_dir),
        output_json=Path(args.output),
        video_extensions=args.extensions,
        save_visualizations=not args.no_vis,
        visualization_dir=Path(args.vis_dir) if args.vis_dir else None,
    )


if __name__ == "__main__":
    main()


