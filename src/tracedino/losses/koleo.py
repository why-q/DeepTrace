"""KoLeo entropy regularization loss for uniform feature distribution."""

import torch
import torch.nn as nn


class KoLeoLoss(nn.Module):
    """
    KoLeo Entropy Regularization Loss.

    Encourages uniform distribution of embeddings on the hypersphere
    by maximizing the entropy of k-nearest neighbor distances.

    Loss formula:
        L_entropy = -1/N * sum_i log(min_{j≠i} ||z_i - z_j||)

    This maximizes the minimum distance to the nearest neighbor for each sample,
    encouraging features to spread uniformly on the unit hypersphere.

    Reference: Used in SSCD (https://arxiv.org/abs/2202.10261) and DINOv3
    """

    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute KoLeo entropy regularization loss.

        Args:
            features: L2-normalized embeddings [B, D]

        Returns:
            Scalar loss value
        """
        batch_size = features.size(0)

        if batch_size == 1:
            # Cannot compute pairwise distances with single sample
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Compute pairwise cosine similarity matrix
        # For L2-normalized features: ||z_i - z_j||^2 = 2 - 2*z_i·z_j
        similarity = torch.matmul(features, features.T)  # [B, B]

        # Set diagonal to -inf to exclude self-distances
        similarity.fill_diagonal_(-float("inf"))

        # Find nearest neighbor (maximum similarity = minimum distance)
        nn_similarity, _ = similarity.max(dim=1)  # [B]

        # Convert similarity to squared distance
        # d^2 = 2 - 2*sim (for unit vectors)
        # Clamp similarity to avoid numerical issues when vectors are nearly identical
        nn_similarity = nn_similarity.clamp(max=1.0 - 1e-6)
        nn_dist_sq = 2.0 - 2.0 * nn_similarity

        # Clamp for numerical stability
        nn_dist_sq = nn_dist_sq.clamp(min=1e-6)

        # Compute log distance
        # log(d) = 0.5 * log(d^2)
        log_dist = 0.5 * torch.log(nn_dist_sq)

        # Entropy loss: -mean(log(d))
        # We want to maximize entropy (spread embeddings), so minimize -entropy
        loss = -log_dist.mean()

        return loss
