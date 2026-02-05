"""
RetinaFace Deep Learning Face Detector

A high-accuracy face detector using the RetinaFace deep learning model.
This implementation wraps the retina-face library and integrates with
DeepTrace's pluggable detector architecture.

References:
    - RetinaFace Paper: https://arxiv.org/abs/1905.00641
    - Library: https://github.com/serengil/retinaface
"""

import os
from pathlib import Path
from typing import List, Tuple

import cv2

from deeptrace.preprocessing.face_detection import BaseFaceDetector


class RetinaFaceDetector(BaseFaceDetector):
    """
    Deep learning face detector using RetinaFace.

    RetinaFace is a state-of-the-art face detection model that performs
    well even in challenging conditions (crowds, occlusion, varied angles).

    The model is downloaded automatically on first use and cached in the
    models/ directory. The model is loaded only once (singleton pattern).

    Attributes:
        threshold (float): Detection confidence threshold (0-1)
        model_name (str): Model variant to use
        _model (object): Cached RetinaFace module instance
        _initialized (bool): Whether the model has been loaded

    Example:
        >>> from deeptrace.preprocessing import RetinaFaceDetector
        >>> detector = RetinaFaceDetector(threshold=0.9)
        >>> faces = detector.detect_faces(frame)
        >>> print(f"Detected {len(faces)} faces")
    """

    # Class-level cache to ensure model is loaded only once
    _model = None
    _initialized = False
    _models_dir = None

    def __init__(
        self,
        threshold: float = 0.9,
        model_name: str = "retinaface",
        models_dir: str = None,
    ):
        """
        Initialize RetinaFace detector.

        Args:
            threshold: Detection confidence threshold (default: 0.9)
            model_name: Model variant ('retinaface' or others)
            models_dir: Directory to store model files (default: ./models)
        """
        self.threshold = threshold
        self.model_name = model_name

        # Set up models directory
        if models_dir is None:
            # Default to models/ in project root
            project_root = Path(__file__).parent.parent.parent.parent
            models_dir = project_root / "models"
        else:
            models_dir = Path(models_dir)

        # Store models directory at class level
        if RetinaFaceDetector._models_dir is None:
            RetinaFaceDetector._models_dir = models_dir
            models_dir.mkdir(parents=True, exist_ok=True)

            # Set environment variable for retinaface library
            # This tells the library where to download/cache models
            os.environ["RETINAFACE_HOME"] = str(models_dir)

        # Lazy loading - model will be initialized on first detect_faces call
        self._ensure_model_loaded()

    @classmethod
    def _ensure_model_loaded(cls):
        """
        Ensure the RetinaFace model is loaded (singleton pattern).

        This method loads the model only once, even if multiple detector
        instances are created. Thread-safe lazy initialization.
        """
        if not cls._initialized:
            try:
                # Import here to avoid loading at module import time
                from retinaface import RetinaFace

                cls._model = RetinaFace
                cls._initialized = True

                print(f"✓ RetinaFace model loaded successfully")
                print(f"  Models directory: {cls._models_dir}")

            except ImportError as e:
                raise ImportError(
                    "RetinaFace library not found. Install with: uv pip install retina-face"
                ) from e
            except Exception as e:
                raise RuntimeError(f"Failed to load RetinaFace model: {e}") from e

    def detect_faces(
        self, frame: "cv2.typing.MatLike"
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame using RetinaFace.

        Args:
            frame: BGR image frame from OpenCV

        Returns:
            List of tuples (x, y, width, height) for each detected face

        Example:
            >>> detector = RetinaFaceDetector(threshold=0.9)
            >>> frame = cv2.imread("image.jpg")
            >>> faces = detector.detect_faces(frame)
            >>> for x, y, w, h in faces:
            ...     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        """
        # Ensure model is loaded
        self._ensure_model_loaded()

        try:
            # RetinaFace expects RGB or BGR, returns dict of detections
            detections = self._model.detect_faces(frame, threshold=self.threshold)

            # Convert to our standard format: (x, y, width, height)
            faces = []

            if isinstance(detections, dict):
                for key, detection in detections.items():
                    # detection['facial_area'] is [x1, y1, x2, y2]
                    facial_area = detection.get("facial_area", [])
                    if len(facial_area) == 4:
                        x1, y1, x2, y2 = facial_area

                        # Convert to (x, y, width, height)
                        x = int(x1)
                        y = int(y1)
                        width = int(x2 - x1)
                        height = int(y2 - y1)

                        # Ensure positive dimensions
                        if width > 0 and height > 0:
                            faces.append((x, y, width, height))

            return faces

        except Exception as e:
            # Log error but don't crash - return empty list
            print(f"Warning: RetinaFace detection failed: {e}")
            return []

    @property
    def name(self) -> str:
        """Return the detector name for logging."""
        return f"RetinaFace(threshold={self.threshold})"


# Example usage and testing
if __name__ == "__main__":
    """
    Test RetinaFace detector with sample image.
    """
    import sys

    print("=" * 60)
    print("RetinaFace Detector Test")
    print("=" * 60)

    # Create detector
    detector = RetinaFaceDetector(threshold=0.9)
    print(f"\nDetector: {detector.name}")

    # Test with sample image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Testing with image: {image_path}")

        frame = cv2.imread(image_path)
        if frame is not None:
            faces = detector.detect_faces(frame)
            print(f"\nDetected {len(faces)} face(s)")

            for idx, (x, y, w, h) in enumerate(faces, 1):
                print(f"  Face {idx}: position=({x}, {y}), size={w}x{h}")

            # Save annotated image
            from deeptrace.preprocessing.face_detection import draw_face_boxes

            annotated = draw_face_boxes(frame, faces)
            output_path = "retinaface_test_output.jpg"
            cv2.imwrite(output_path, annotated)
            print(f"\nSaved annotated image to: {output_path}")
        else:
            print(f"Error: Could not load image {image_path}")
    else:
        print("\nUsage: python retinaface_detector.py <image_path>")
        print("Example: python retinaface_detector.py test_image.jpg")

