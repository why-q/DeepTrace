"""Supervised Contrastive Loss for TraceDINO training."""

from typing import Callable, Optional

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
    Paper: https://arxiv.org/abs/2004.11362

    Loss formula:
        L = -1/|P(i)| * sum_{p in P(i)} log(
            exp(z_i · z_p / τ) / sum_{a in A(i)} exp(z_i · z_a / τ)
        )

    where:
        - P(i) = set of positive samples for anchor i (same label, excluding self)
        - A(i) = set of all samples except i (positives + negatives)
        - τ = temperature parameter (default: 0.05)

    Args:
        temperature: Temperature parameter for softmax
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        gather_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            features: L2-normalized embeddings [B, D]
            labels: Integer labels [B] where same label = same class
            gather_fn: Optional function to gather features/labels across GPUs
                       for distributed training. Should preserve gradients.

        Returns:
            Scalar loss value
        """
        device = features.device

        # Gather features and labels from all GPUs if in distributed mode
        if gather_fn is not None:
            all_features = gather_fn(features)
            all_labels = gather_fn(labels)
        else:
            all_features = features
            all_labels = labels

        batch_size = all_features.shape[0]

        # Compute similarity matrix
        # [B, B] where sim[i, j] = z_i · z_j (cosine similarity for L2-normalized features)
        similarity = torch.matmul(all_features, all_features.T) / self.temperature

        # Create mask for positive pairs (same label, excluding self)
        all_labels = all_labels.contiguous().view(-1, 1)
        mask_positives = torch.eq(all_labels, all_labels.T).float().to(device)

        # Create self mask (diagonal)
        mask_self = torch.eye(batch_size, device=device)

        # Remove self from positives
        mask_positives = mask_positives - mask_self

        # For numerical stability, subtract max
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Compute exp(logits) and exclude self from denominator
        exp_logits = torch.exp(logits) * (1 - mask_self)

        # Compute log probability
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Check if there are any positives for each anchor
        num_positives = mask_positives.sum(dim=1)

        # Only compute loss for anchors that have at least one positive
        # Avoid division by zero
        valid_anchors = num_positives > 0

        if valid_anchors.sum() == 0:
            # No valid positive pairs in batch
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask_positives * log_prob).sum(dim=1) / (num_positives + 1e-8)

        # Only average over anchors that have positives
        loss = -mean_log_prob_pos[valid_anchors].mean()

        return loss
