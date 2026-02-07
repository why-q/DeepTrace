"""Data augmentation for TraceDINO training."""

import random
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms as T


class FaceBlurAugmentation:
    """
    Face blur augmentation using face detection + Gaussian blur.

    Uses insightface for face detection and applies strong Gaussian blur
    to the detected face regions.

    Args:
        blur_sigma: Gaussian blur sigma (default: 40.0)
        threshold: Face detection confidence threshold (default: 0.5)
    """

    def __init__(self, blur_sigma: float = 40.0, threshold: float = 0.5):
        self.blur_sigma = blur_sigma
        self.threshold = threshold
        self._detector = None

    @property
    def detector(self):
        """Lazy load face detector."""
        if self._detector is None:
            try:
                from insightface.app import FaceAnalysis

                self._detector = FaceAnalysis(providers=["CPUExecutionProvider"])
                self._detector.prepare(ctx_id=-1, det_size=(640, 640))
            except ImportError:
                print("Warning: insightface not installed. Face blur will be skipped.")
                self._detector = "unavailable"
        return self._detector

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply face blur to image.

        Args:
            image: Input image [H, W, C] in BGR format

        Returns:
            Image with blurred faces [H, W, C] in BGR format
        """
        if self.detector == "unavailable":
            return image

        # Detect faces
        faces = self.detector.get(image)

        if not faces:
            return image

        # Create a copy for blurring
        blurred = image.copy()

        # Blur each detected face
        for face in faces:
            if face.det_score < self.threshold:
                continue

            # Get face bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Add padding (10% of face size)
            w, h = x2 - x1, y2 - y1
            pad_x, pad_y = int(0.1 * w), int(0.1 * h)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(image.shape[1], x2 + pad_x)
            y2 = min(image.shape[0], y2 + pad_y)

            # Extract face ROI
            roi = blurred[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # Apply strong Gaussian blur
            roi_blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=self.blur_sigma)

            # Replace face region
            blurred[y1:y2, x1:x2] = roi_blurred

        return blurred


class HumanCropAugmentation:
    """
    Human detection + random crop augmentation.

    Uses YOLOv8 for human detection and performs random cropping
    while ensuring the detected person remains in the frame.

    Args:
        model_name: YOLOv8 model name (default: "yolov8n.pt")
        crop_ratio_range: Min and max crop ratio (default: (0.25, 0.75))
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        crop_ratio_range: Tuple[float, float] = (0.25, 0.75),
    ):
        self.model_name = model_name
        self.crop_ratio_range = crop_ratio_range
        self._detector = None

    @property
    def detector(self):
        """Lazy load human detector."""
        if self._detector is None:
            try:
                from ultralytics import YOLO

                self._detector = YOLO(self.model_name)
                # Force CPU to avoid CUDA issues in DataLoader workers
                self._detector.to("cpu")
            except ImportError:
                print("Warning: ultralytics not installed. Human crop will be skipped.")
                self._detector = "unavailable"
        return self._detector

    def __call__(
        self,
        image: np.ndarray,
        crop_ratio: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply human-centered random crop.

        Args:
            image: Input image [H, W, C] in BGR format
            crop_ratio: Crop ratio (if None, randomly sample from range)

        Returns:
            Cropped image [H', W', C] in BGR format
        """
        if self.detector == "unavailable":
            # Fallback to center crop
            return self._center_crop(image, crop_ratio or 0.5)

        # Detect humans (class 0 = person in COCO)
        results = self.detector(image, classes=[0], verbose=False)

        if len(results[0].boxes) == 0:
            # No human detected, use center crop
            return self._center_crop(image, crop_ratio or 0.5)

        # Get largest person bbox
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_idx = areas.argmax()
        x1, y1, x2, y2 = boxes[largest_idx].astype(int)

        # Random crop containing the person
        return self._random_crop_around_bbox(
            image, (x1, y1, x2, y2), crop_ratio or random.uniform(*self.crop_ratio_range)
        )

    def _center_crop(self, image: np.ndarray, crop_ratio: float) -> np.ndarray:
        """Center crop with given ratio."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * (1 - crop_ratio)), int(w * (1 - crop_ratio))
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return image[top : top + new_h, left : left + new_w]

    def _random_crop_around_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        crop_ratio: float,
    ) -> np.ndarray:
        """Random crop ensuring bbox is within the crop."""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # Calculate crop size
        new_h, new_w = int(h * (1 - crop_ratio)), int(w * (1 - crop_ratio))

        # Ensure crop contains the bbox
        max_top = min(y1, h - new_h)
        min_top = max(0, y2 - new_h)
        max_left = min(x1, w - new_w)
        min_left = max(0, x2 - new_w)

        # Random crop position
        if max_top >= min_top and max_left >= min_left:
            top = random.randint(min_top, max_top)
            left = random.randint(min_left, max_left)
        else:
            # Fallback to center crop
            top = (h - new_h) // 2
            left = (w - new_w) // 2

        return image[top : top + new_h, left : left + new_w]


class JPEGCompression:
    """
    Random JPEG compression augmentation.

    Simulates social media compression artifacts by encoding and
    decoding with random quality factors.

    Args:
        quality_range: Min and max JPEG quality (default: (30, 95))
    """

    def __init__(self, quality_range: Tuple[int, int] = (30, 95)):
        self.quality_range = quality_range

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random JPEG compression.

        Args:
            image: Input image [H, W, C] in BGR format

        Returns:
            Compressed image [H, W, C] in BGR format
        """
        quality = random.randint(*self.quality_range)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


class TraceDINOTransform:
    """
    Complete transformation pipeline for TraceDINO training.

    Args:
        image_size: Target image size (default: 224)
        is_training: Whether in training mode (applies augmentations)
        mean: ImageNet normalization mean
        std: ImageNet normalization std
    """

    def __init__(
        self,
        image_size: int = 224,
        is_training: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.image_size = image_size
        self.is_training = is_training

        # Base transforms (always applied)
        self.base_transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(
        self,
        image: np.ndarray,
        apply_jpeg: bool = False,
    ) -> torch.Tensor:
        """
        Apply transformations to image.

        Args:
            image: Input image [H, W, C] in BGR format
            apply_jpeg: Whether to apply JPEG compression

        Returns:
            Transformed tensor [C, H, W]
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply JPEG compression if specified
        if apply_jpeg and self.is_training:
            jpeg_compressor = JPEGCompression()
            image_bgr = jpeg_compressor(image)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Apply base transforms
        return self.base_transform(image_rgb)
