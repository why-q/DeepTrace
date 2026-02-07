"""
TraceDINO: Feature Enhancement for Deepfake Video Tracing

A contrastive learning framework that fine-tunes DINOv3 features for
segment-level deepfake video tracing.
"""

from .config import TraceDINOConfig
from .dataset.datamodule import TraceDINODataModule
from .dataset.dataset import TraceDINODataset
from .losses.combined import TraceDINOLoss
from .models.tracedino import TraceDINO

__version__ = "0.1.0"

__all__ = [
    "TraceDINO",
    "TraceDINOConfig",
    "TraceDINOLoss",
    "TraceDINODataset",
    "TraceDINODataModule",
]