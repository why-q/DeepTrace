"""DeepTrace Features Module.

Feature extraction module providing extraction capabilities from various
pre-trained models including DINOv3, ISC21, and SSCD.

Usage:
    from deeptrace.features import DINOv3FeatureExtractor

    extractor = DINOv3FeatureExtractor(
        model_name="dinov3_vitl16",
        device="cuda",
    )

    features = extractor.extract_from_folder(
        folder_path="path/to/frames",
        save_path="output.npy"
    )
"""

from .extractor import FeatureExtractor, ImageFileDataset
from .extractors import (
    DINOv3FeatureExtractor,
    DINOv3OutputMode,
    ISC21FeatureExtractor,
    SSCDFeatureExtractor,
)
from .extract import extract_features_from_videos

__version__ = "0.1.0"

__all__ = [
    "FeatureExtractor",
    "ImageFileDataset",
    "DINOv3FeatureExtractor",
    "DINOv3OutputMode",
    "ISC21FeatureExtractor",
    "SSCDFeatureExtractor",
    "extract_features_from_videos",
]
