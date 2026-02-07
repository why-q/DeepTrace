"""Evaluation script for TraceDINO."""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.tracedino.config import TraceDINOConfig
from src.tracedino.dataset.datamodule import TraceDINODataModule
from src.tracedino.models.tracedino import TraceDINO


class ContrastiveEvaluator:
    """
    Evaluator for contrastive learning metrics.

    Computes:
    - Triplet Accuracy: P(d(anchor, positive) < d(anchor, hard_negative))
    - Recall@K: Top-K retrieval recall
    - Positive/Negative Similarity: Mean cosine similarity
    - Separation Margin: Difference between positive and negative similarity
    - AUC-ROC: Binary classification AUC
    """

    def __init__(self, device: torch.device):
        self.device = device

    @torch.no_grad()
    def evaluate(self, model: nn.Module, dataloader) -> Dict[str, float]:
        """
        Evaluate model on contrastive learning metrics.

        Args:
            model: TraceDINO model
            dataloader: DataLoader for evaluation

        Returns:
            Dict of metric name → value
        """
        model.eval()

        all_pos_sims = []
        all_neg_sims = []
        all_triplet_accs = []
        all_recall_at_1 = []
        all_recall_at_5 = []
        all_recall_at_10 = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            # Flatten batch: [B, 7, C, H, W] → [B*7, C, H, W]
            images = batch["images"].view(-1, 3, 224, 224).to(self.device)
            labels = batch["labels"].view(-1).to(self.device)

            # Extract embeddings
            embeddings = model(images)  # [B*7, D]

            # Reshape to [B, 7, D]
            batch_size = batch["images"].size(0)
            embeddings = embeddings.view(batch_size, 7, -1)

            # Split into anchors, positives, hard negatives
            # Structure: [anchor, pos1, pos2, pos3, neg1, neg2, neg3]
            anchors = embeddings[:, 0, :]  # [B, D]
            positives = embeddings[:, 1:4, :]  # [B, 3, D]
            hard_negatives = embeddings[:, 4:7, :]  # [B, 3, D]

            # Compute similarities
            # Anchor-positive similarity
            pos_sim = F.cosine_similarity(
                anchors.unsqueeze(1), positives, dim=-1
            )  # [B, 3]
            mean_pos_sim = pos_sim.mean(dim=1)  # [B]

            # Anchor-hard negative similarity
            neg_sim = F.cosine_similarity(
                anchors.unsqueeze(1), hard_negatives, dim=-1
            )  # [B, 3]
            mean_neg_sim = neg_sim.mean(dim=1)  # [B]

            # Triplet accuracy: mean(pos_sim) > mean(neg_sim)
            triplet_acc = (mean_pos_sim > mean_neg_sim).float()

            # Recall@K: For each anchor, check if any positive is in Top-K
            # Compute similarity between anchor and all samples (pos + neg)
            all_samples = torch.cat([positives, hard_negatives], dim=1)  # [B, 6, D]
            all_sims = F.cosine_similarity(
                anchors.unsqueeze(1), all_samples, dim=-1
            )  # [B, 6]

            # Top-K indices
            _, top_1_idx = all_sims.topk(k=1, dim=1)
            _, top_5_idx = all_sims.topk(k=min(5, all_sims.size(1)), dim=1)
            _, top_10_idx = all_sims.topk(k=min(10, all_sims.size(1)), dim=1)

            # Check if any positive (indices 0, 1, 2) is in Top-K
            recall_1 = (top_1_idx < 3).any(dim=1).float()
            recall_5 = (top_5_idx < 3).any(dim=1).float()
            recall_10 = (top_10_idx < 3).any(dim=1).float()

            # Accumulate metrics
            all_pos_sims.extend(mean_pos_sim.cpu().numpy())
            all_neg_sims.extend(mean_neg_sim.cpu().numpy())
            all_triplet_accs.extend(triplet_acc.cpu().numpy())
            all_recall_at_1.extend(recall_1.cpu().numpy())
            all_recall_at_5.extend(recall_5.cpu().numpy())
            all_recall_at_10.extend(recall_10.cpu().numpy())

        # Compute aggregate metrics
        pos_sims = np.array(all_pos_sims)
        neg_sims = np.array(all_neg_sims)

        # AUC-ROC: Classify positive vs negative samples
        y_true = np.concatenate([np.ones_like(pos_sims), np.zeros_like(neg_sims)])
        y_score = np.concatenate([pos_sims, neg_sims])
        auc_roc = roc_auc_score(y_true, y_score)

        metrics = {
            "triplet_accuracy": np.mean(all_triplet_accs) * 100,
            "recall@1": np.mean(all_recall_at_1) * 100,
            "recall@5": np.mean(all_recall_at_5) * 100,
            "recall@10": np.mean(all_recall_at_10) * 100,
            "positive_sim_mean": np.mean(pos_sims),
            "hard_negative_sim_mean": np.mean(neg_sims),
            "separation_margin": np.mean(pos_sims) - np.mean(neg_sims),
            "auc_roc": auc_roc,
        }

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate TraceDINO")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    args = parser.parse_args()

    # Load config
    config = TraceDINOConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data module
    print("Loading datasets...")
    data_module = TraceDINODataModule(
        train_csv=config.train_csv,
        valid_csv=config.valid_csv,
        test_csv=config.test_csv,
        query_video_dir=config.query_video_dir,
        source_frame_dir=config.source_frame_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
    )

    if args.split == "valid":
        dataloader = data_module.valid_dataloader()
    else:
        dataloader = data_module.test_dataloader()

    print(f"Evaluating on {args.split} set with {len(dataloader)} batches")

    # Create model
    print("Creating model...")
    model = TraceDINO(
        backbone_path=config.backbone_path,
        extract_layers=config.extract_layers,
        freeze_backbone=config.freeze_backbone,
        fused_dim=config.fused_dim,
        output_dim=config.output_dim,
        gem_p=config.gem_p,
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Evaluate
    evaluator = ContrastiveEvaluator(device)
    metrics = evaluator.evaluate(model, dataloader)

    # Print results
    print("\n" + "=" * 60)
    print(f"Evaluation Results on {args.split.upper()} Set")
    print("=" * 60)
    print(f"Triplet Accuracy:        {metrics['triplet_accuracy']:.2f}%")
    print(f"Recall@1:                {metrics['recall@1']:.2f}%")
    print(f"Recall@5:                {metrics['recall@5']:.2f}%")
    print(f"Recall@10:               {metrics['recall@10']:.2f}%")
    print(f"Positive Sim (mean):     {metrics['positive_sim_mean']:.4f}")
    print(f"Hard Negative Sim (mean): {metrics['hard_negative_sim_mean']:.4f}")
    print(f"Separation Margin:       {metrics['separation_margin']:.4f}")
    print(f"AUC-ROC:                 {metrics['auc_roc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
