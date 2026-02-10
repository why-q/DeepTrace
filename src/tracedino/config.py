"""Configuration for TraceDINO training and evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class TraceDINOConfig:
    """Configuration for TraceDINO feature enhancement model."""

    # ========== Paths ==========
    backbone_path: Path = Path("pretrained/dinov3/dinov3-vitb16/")
    train_csv: Path = Path("asset/dataset/new/train.csv")
    valid_csv: Path = Path("asset/dataset/new/valid.csv")
    test_csv: Path = Path("asset/dataset/new/test.csv")
    query_video_dir: Path = Path("/datadrive2/pychen/deeptrace/query_v/")
    source_frame_dir: Path = Path("/datadrive2/pychen/deeptrace/new_vorpus/")
    checkpoint_dir: Path = Path("/datadrive2/pychen/deeptrace/checkpoints/tracedino/")
    log_dir: Path = Path("logs/tracedino/")
    preprocessed_dir: Path = Path("/datadrive2/pychen/deeptrace/new_preprocessed/")

    # ========== Model Architecture ==========
    output_dim: int = 512  # Final embedding dimension
    gem_p: float = 4.0  # GeM pooling exponent
    extract_layers: Tuple[int, ...] = (3, 6, 9, 12)  # Layers to extract from DINOv3
    fused_dim: int = 768  # Dimension after linear fusion
    freeze_backbone: bool = True  # Freeze DINOv3 backbone

    # ========== Training Hyperparameters ==========
    batch_size: int = 256  # Global batch size
    num_epochs: int = 50
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.05
    warmup_epochs: int = 5  # Learning rate warmup

    # ========== Loss Function ==========
    temperature: float = 0.05  # Temperature for SupCon loss
    koleo_weight: float = 30.0  # Weight for KoLeo entropy regularization

    # ========== Data Configuration ==========
    image_size: int = 224  # Input image size
    n_anchor_frames: int = 2  # Number of frames to sample from each query video
    n_positives: int = 3  # Number of positive samples per anchor
    n_hard_negatives: int = 3  # Number of hard negative samples per anchor
    n_presampled_negatives: int = 20  # Number of pre-sampled negatives in preprocessing
    safety_radius_seconds: float = 15.0  # Safety radius for hard negative sampling
    jpeg_quality_range: Tuple[int, int] = (30, 95)  # JPEG compression quality range
    crop_ratio_range: Tuple[float, float] = (0.25, 0.75)  # Crop ratio range for augmentation
    blur_sigma: float = 40.0  # Gaussian blur sigma for face blurring
    use_preprocessed: bool = True  # Use preprocessed dataset

    # ========== Distributed Training ==========
    num_workers: int = 8  # DataLoader workers
    distributed: bool = False  # Enable distributed training
    local_rank: int = 0  # Local rank for DDP
    world_size: int = 1  # World size for DDP

    # ========== Logging and Checkpointing ==========
    log_interval: int = 100  # Log every N batches
    save_interval: int = 1  # Save checkpoint every N epochs
    eval_interval: int = 1  # Evaluate on validation set every N epochs
    use_tensorboard: bool = True  # Enable tensorboard logging

    # ========== Mixed Precision Training ==========
    use_amp: bool = True  # Enable automatic mixed precision

    # ========== Evaluation ==========
    eval_batch_size: int = 64  # Batch size for evaluation

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Convert paths to absolute paths
        self.backbone_path = Path(self.backbone_path).resolve()
        self.train_csv = Path(self.train_csv).resolve()
        self.valid_csv = Path(self.valid_csv).resolve()
        self.test_csv = Path(self.test_csv).resolve()
