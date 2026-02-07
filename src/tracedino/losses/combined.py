"""Combined loss function for TraceDINO training."""

from typing import Dict

import torch
import torch.nn as nn

from .supcon import SupConLoss
from .koleo import KoLeoLoss


class TraceDINOLoss(nn.Module):
    """
    Combined loss for TraceDINO training.

    L_total = L_SupCon + λ * L_KoLeo

    where:
        - L_SupCon: Supervised Contrastive Loss
        - L_KoLeo: KoLeo Entropy Regularization Loss
        - λ: Weight for entropy regularization (default: 30)

    Args:
        temperature: Temperature for SupCon loss (default: 0.05)
        koleo_weight: Weight for KoLeo loss (default: 30.0)
    """

    def __init__(
        self,
        temperature: float = 0.05,
        koleo_weight: float = 30.0,
    ):
        super().__init__()
        self.supcon = SupConLoss(temperature=temperature)
        self.koleo = KoLeoLoss()
        self.koleo_weight = koleo_weight

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            features: L2-normalized embeddings [B, D]
            labels: Integer labels for positive grouping [B]

        Returns:
            Dict with 'total', 'supcon', 'koleo' losses
        """
        # Compute individual losses
        loss_supcon = self.supcon(features, labels)
        loss_koleo = self.koleo(features)

        # Combined loss
        loss_total = loss_supcon + self.koleo_weight * loss_koleo

        return {
            "total": loss_total,
            "supcon": loss_supcon,
            "koleo": loss_koleo,
        }
