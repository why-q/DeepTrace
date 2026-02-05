"""
Image Cropper with Face Detection Support

Provides intelligent cropping for images based on face detection,
with correct crop degree semantics (0.25 = remove 25% of area, keep 75%).
"""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class ImageCropper:
    """
    Intelligent image cropper based on face detection results.
    
    Crops images using a "crop degree" parameter where:
    - crop_degree = 0.0 means no cropping (keep 100% of area)
    - crop_degree = 0.25 means remove 25% of area (keep 75%)
    - crop_degree = 0.5 means remove 50% of area (keep 50%)
    - crop_degree = 0.75 means remove 75% of area (keep 25%)
    
    The cropping maintains the aspect ratio and tries to keep detected
    faces centered or at a specified vertical offset.
    
    Attributes:
        crop_degree (float): Fraction of area to remove (0-1)
        face_vertical_offset (float): Vertical position of face in cropped frame
                                     (0=top, 0.5=center, 1.0=bottom)
    
    Example:
        >>> cropper = ImageCropper(crop_degree=0.25)  # Keep 75% of area
        >>> cropped = cropper.crop_image(
        ...     img,
        ...     face_bbox={"x": 100, "y": 100, "width": 200, "height": 200}
        ... )
    """
    
    def __init__(
        self,
        crop_degree: float = 0.0,
        face_vertical_offset: float = 0.3,
    ):
        """
        Initialize ImageCropper.
        
        Args:
            crop_degree: Fraction of area to remove (0-1). 
                        0.25 = remove 25% area, keep 75%
            face_vertical_offset: Vertical position of face in cropped frame
                                (0=top, 0.3=upper-center, 0.5=center)
        """
        if not 0 <= crop_degree < 1:
            raise ValueError("crop_degree must be in [0, 1)")
        if not 0 <= face_vertical_offset <= 1:
            raise ValueError("face_vertical_offset must be in [0, 1]")
        
        self.crop_degree = crop_degree
        self.face_vertical_offset = face_vertical_offset
        
        # Calculate the scale factor from crop degree
        # If we want to keep (1 - crop_degree) of the area:
        # area_ratio = (1 - crop_degree)
        # Since area = width * height and we scale uniformly:
        # scale = sqrt(area_ratio)
        retained_area = 1.0 - crop_degree
        self.crop_scale = math.sqrt(retained_area)
    
    def calculate_crop_region(
        self,
        face_bbox: Optional[Dict[str, int]],
        image_width: int,
        image_height: int,
    ) -> Tuple[int, int, int, int]:
        """
        Calculate crop region to keep face centered or use center crop if no face.
        
        Args:
            face_bbox: Face bounding box with keys: x, y, width, height
                      If None, uses center crop
            image_width: Original image width
            image_height: Original image height
        
        Returns:
            Tuple of (crop_x, crop_y, crop_width, crop_height)
        """
        # Calculate target crop dimensions based on crop_scale
        target_width = int(image_width * self.crop_scale)
        target_height = int(image_height * self.crop_scale)
        
        if face_bbox is None:
            # No face detected - use center crop
            crop_x = (image_width - target_width) // 2
            crop_y = (image_height - target_height) // 2
        else:
            # Calculate face center
            face_center_x = face_bbox["x"] + face_bbox["width"] // 2
            face_center_y = face_bbox["y"] + face_bbox["height"] // 2
            
            # Calculate crop region with face at specified vertical position
            crop_x = face_center_x - target_width // 2
            crop_y = face_center_y - int(target_height * self.face_vertical_offset)
        
        # Ensure crop region is within image boundaries
        crop_x = max(0, min(crop_x, image_width - target_width))
        crop_y = max(0, min(crop_y, image_height - target_height))
        
        # Ensure dimensions don't exceed image
        crop_width = min(target_width, image_width - crop_x)
        crop_height = min(target_height, image_height - crop_y)
        
        return (crop_x, crop_y, crop_width, crop_height)
    
    def crop_image(
        self,
        image: np.ndarray,
        face_bbox: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Crop image based on face detection or center crop.
        
        Args:
            image: Input image as numpy array (BGR format from cv2)
            face_bbox: Optional face bounding box dict with x, y, width, height
        
        Returns:
            Cropped image as numpy array
        """
        height, width = image.shape[:2]
        crop_x, crop_y, crop_w, crop_h = self.calculate_crop_region(
            face_bbox, width, height
        )
        
        # Perform the crop
        cropped = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        return cropped
    
    def crop_image_file(
        self,
        input_path: Path,
        output_path: Path,
        face_bbox: Optional[Dict[str, int]] = None,
    ) -> bool:
        """
        Crop an image file and save to output path.
        
        Args:
            input_path: Path to input image
            output_path: Path to save cropped image
            face_bbox: Optional face bounding box dict
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read image
            image = cv2.imread(str(input_path))
            if image is None:
                print(f"Error: Could not read image from {input_path}")
                return False
            
            # Crop image
            cropped = self.crop_image(image, face_bbox)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save cropped image
            cv2.imwrite(str(output_path), cropped)
            return True
            
        except Exception as e:
            print(f"Error cropping image: {e}")
            return False


def detect_face_bbox(image: np.ndarray) -> Optional[Dict[str, int]]:
    """
    Detect the largest face in an image using Haar Cascade.
    
    Args:
        image: Input image as numpy array (BGR format)
    
    Returns:
        Dictionary with x, y, width, height of the largest face,
        or None if no face detected
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load Haar Cascade classifier
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None
    
    # Return the largest face
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    
    return {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}

