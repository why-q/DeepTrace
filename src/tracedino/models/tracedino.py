"""Complete TraceDINO model combining backbone and adapter."""

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from .backbone import DINOv3Backbone
from .adapter import TraceDINOAdapter


class TraceDINO(nn.Module):
    """
    Complete TraceDINO model for deepfake video tracing feature learning.

    Combines a frozen DINOv3 backbone for multi-layer feature extraction
    with trainable adapter layers for feature aggregation and projection.

    Args:
        backbone_path: Path to DINOv3 model directory
        extract_layers: Tuple of layer indices to extract (1-indexed)
        freeze_backbone: Whether to freeze the DINOv3 backbone
        fused_dim: Dimension after linear fusion
        output_dim: Final embedding dimension
        gem_p: GeM pooling exponent
    """

    def __init__(
        self,
        backbone_path: Path,
        extract_layers: Tuple[int, ...] = (3, 6, 9, 12),
        freeze_backbone: bool = True,
        fused_dim: int = 768,
        output_dim: int = 512,
        gem_p: float = 4.0,
    ):
        super().__init__()

        # Frozen DINOv3 backbone
        self.backbone = DINOv3Backbone(
            model_path=backbone_path,
            extract_layers=extract_layers,
            freeze=freeze_backbone,
        )

        # Trainable adapter
        self.adapter = TraceDINOAdapter(
            input_dim=self.backbone.hidden_size,  # 768
            num_layers=len(extract_layers),  # 4
            fused_dim=fused_dim,  # 768
            output_dim=output_dim,  # 512
            gem_p=gem_p,  # 4.0
        )

        self.freeze_backbone = freeze_backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TraceDINO.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            L2-normalized embeddings [B, output_dim]
        """
        # Extract multi-layer features from frozen backbone
        if self.freeze_backbone:
            with torch.no_grad():
                multi_layer_features = self.backbone(x)
        else:
            multi_layer_features = self.backbone(x)

        # Pass through trainable adapter
        embeddings = self.adapter(multi_layer_features)

        return embeddings

    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        Return only trainable adapter parameters.

        Returns:
            List of trainable parameters (only from adapter if backbone is frozen)
        """
        if self.freeze_backbone:
            return list(self.adapter.parameters())
        else:
            return list(self.parameters())

    def get_num_trainable_params(self) -> int:
        """
        Count the number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.get_trainable_params())

    def get_num_total_params(self) -> int:
        """
        Count the total number of parameters.

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())

    def freeze_bn(self):
        """Freeze all batch normalization layers (if any)."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
        """
        Extract features in batches for efficient inference.

        Args:
            x: Input images [N, C, H, W]
            batch_size: Batch size for processing

        Returns:
            Features [N, output_dim]
        """
        self.eval()
        features = []

        for i in range(0, x.size(0), batch_size):
            batch = x[i : i + batch_size]
            feat = self.forward(batch)
            features.append(feat.cpu())

        return torch.cat(features, dim=0)
