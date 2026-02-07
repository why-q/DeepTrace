"""Feature adapter modules for TraceDINO."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).

    Computes: f_GeM = (1/N * sum(z_i^p))^(1/p)

    Args:
        p: Pooling exponent (default: 4.0)
        eps: Small constant for numerical stability
    """

    def __init__(self, p: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GeM pooling.

        Args:
            x: Patch tokens [B, N, D]

        Returns:
            Pooled feature [B, D]
        """
        # Clamp to avoid numerical issues
        x = x.clamp(min=self.eps)
        # GeM pooling: (mean(x^p))^(1/p)
        return x.pow(self.p).mean(dim=1).pow(1.0 / self.p)


class MLPProjectionHead(nn.Module):
    """
    2-layer MLP projection head with GELU activation and L2 normalization.

    Architecture: input_dim → hidden_dim → output_dim
    Output is L2 normalized to lie on the unit hypersphere.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 768,
        output_dim: int = 512,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features to embedding space.

        Args:
            x: Input features [B, D]

        Returns:
            L2-normalized embeddings [B, output_dim]
        """
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        # L2 normalize to unit hypersphere
        x = F.normalize(x, p=2, dim=-1)
        return x


class TraceDINOAdapter(nn.Module):
    """
    Complete adapter module for TraceDINO feature enhancement.

    Architecture pipeline:
    1. Multi-layer concatenation: [B, N, K*D] from K layers
    2. Linear fusion: [B, N, K*D] → [B, N, D']
    3. GeM pooling: [B, N, D'] → [B, D']
    4. MLP projection: [B, D'] → [B, output_dim] (L2 normalized)

    Args:
        input_dim: Feature dimension per layer (D)
        num_layers: Number of layers to aggregate (K)
        fused_dim: Dimension after linear fusion (D')
        output_dim: Final embedding dimension
        gem_p: GeM pooling exponent
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_layers: int = 4,
        fused_dim: int = 768,
        output_dim: int = 512,
        gem_p: float = 4.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.fused_dim = fused_dim
        self.output_dim = output_dim

        # Linear fusion: K*D → D' with ReLU activation
        # ReLU ensures non-negative input for GeM pooling
        self.linear_fusion = nn.Sequential(
            nn.Linear(input_dim * num_layers, fused_dim),
            nn.ReLU(inplace=True),
        )

        # GeM pooling
        self.gem_pool = GeM(p=gem_p)

        # MLP projection head
        self.projection = MLPProjectionHead(
            input_dim=fused_dim,
            hidden_dim=fused_dim,
            output_dim=output_dim,
        )

    def forward(self, multi_layer_features: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Aggregate and project multi-layer features to embeddings.

        Args:
            multi_layer_features: Dict mapping layer_idx → [B, N, D]
                                  Expected to have num_layers entries

        Returns:
            L2-normalized embeddings [B, output_dim]
        """
        # Concatenate features from all layers along channel dimension
        # Sort by layer index to ensure consistent ordering
        features_list = [
            multi_layer_features[idx] for idx in sorted(multi_layer_features.keys())
        ]
        z_cat = torch.cat(features_list, dim=-1)  # [B, N, K*D]

        # Linear fusion: [B, N, K*D] → [B, N, D']
        z_fused = self.linear_fusion(z_cat)  # [B, N, 768]

        # GeM pooling: [B, N, D'] → [B, D']
        z_pooled = self.gem_pool(z_fused)  # [B, 768]

        # MLP projection + L2 norm: [B, D'] → [B, output_dim]
        embeddings = self.projection(z_pooled)  # [B, 512]

        return embeddings
