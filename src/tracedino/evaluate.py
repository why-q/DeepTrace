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
    Sample-level metrics (within each sample's 6 candidates):
    - Triplet Accuracy: P(d(anchor, positive) < d(anchor, hard_negative))
    - Recall@K: Top-K retrieval recall within sample
    - Positive/Negative Similarity: Mean cosine similarity
    - Separation Margin: Difference between positive and negative similarity
    - AUC-ROC: Binary classification AUC

    Batch-level metrics (across entire batch):
    - Batch Recall@K: Top-K retrieval recall across all batch samples
    - Batch Triplet Accuracy: P(pos_sim > batch_neg_sim)
    - Hard vs Batch Neg Gap: Difference showing if hard negatives are harder
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

        # Sample-level metrics
        all_pos_sims = []
        all_neg_sims = []
        all_triplet_accs = []
        all_recall_at_1 = []
        all_recall_at_5 = []
        all_recall_at_10 = []

        # Batch-level metrics
        all_batch_neg_sims = []
        all_batch_triplet_accs = []
        all_batch_recall_at_1 = []
        all_batch_recall_at_5 = []
        all_batch_recall_at_10 = []

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

            # ============================================================
            # Batch-level evaluation: test cross-video discrimination
            # ============================================================
            # Compute similarity between each anchor and all other anchors
            # This simulates the batch-level negative samples used in training
            # anchor_sim_matrix[i, j] = cosine_sim(anchor_i, anchor_j)
            anchor_sim_matrix = F.cosine_similarity(
                anchors.unsqueeze(1),  # [B, 1, D]
                anchors.unsqueeze(0),  # [1, B, D]
                dim=-1
            )  # [B, B]

            # For each anchor, compute mean similarity to other anchors (batch negatives)
            # Mask out self-similarity (diagonal)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
            batch_neg_sim = anchor_sim_matrix[mask].view(batch_size, -1).mean(dim=1)  # [B]

            # Batch Triplet Accuracy: mean(pos_sim) > mean(batch_neg_sim)
            batch_triplet_acc = (mean_pos_sim > batch_neg_sim).float()

            # Batch Recall@K: For each anchor, check if positives rank higher than
            # all batch negatives (other anchors) + hard negatives
            # Candidate set: [3 positives, 3 hard negatives, B-1 other anchors]
            if batch_size > 1:
                # Get other anchors for each sample (excluding self)
                # other_anchors[i] = all anchors except anchor_i
                other_anchors_list = []
                for i in range(batch_size):
                    other_idx = torch.cat([
                        torch.arange(0, i, device=self.device),
                        torch.arange(i + 1, batch_size, device=self.device)
                    ])
                    other_anchors_list.append(anchors[other_idx])  # [B-1, D]
                other_anchors = torch.stack(other_anchors_list)  # [B, B-1, D]

                # Compute similarity to other anchors
                other_anchor_sims = F.cosine_similarity(
                    anchors.unsqueeze(1),  # [B, 1, D]
                    other_anchors,  # [B, B-1, D]
                    dim=-1
                )  # [B, B-1]

                # Full candidate set: [3 pos, 3 hard neg, B-1 other anchors]
                batch_all_sims = torch.cat([
                    pos_sim,  # [B, 3] - positives at indices 0, 1, 2
                    neg_sim,  # [B, 3] - hard negatives at indices 3, 4, 5
                    other_anchor_sims  # [B, B-1] - batch negatives at indices 6+
                ], dim=1)  # [B, 6 + B-1]

                # Top-K indices in the full candidate set
                num_candidates = batch_all_sims.size(1)
                _, batch_top_1_idx = batch_all_sims.topk(k=1, dim=1)
                _, batch_top_5_idx = batch_all_sims.topk(k=min(5, num_candidates), dim=1)
                _, batch_top_10_idx = batch_all_sims.topk(k=min(10, num_candidates), dim=1)

                # Check if any positive (indices 0, 1, 2) is in Top-K
                batch_recall_1 = (batch_top_1_idx < 3).any(dim=1).float()
                batch_recall_5 = (batch_top_5_idx < 3).any(dim=1).float()
                batch_recall_10 = (batch_top_10_idx < 3).any(dim=1).float()
            else:
                # Single sample batch: batch metrics equal sample metrics
                batch_recall_1 = recall_1
                batch_recall_5 = recall_5
                batch_recall_10 = recall_10

            # Accumulate sample-level metrics
            all_pos_sims.extend(mean_pos_sim.cpu().numpy())
            all_neg_sims.extend(mean_neg_sim.cpu().numpy())
            all_triplet_accs.extend(triplet_acc.cpu().numpy())
            all_recall_at_1.extend(recall_1.cpu().numpy())
            all_recall_at_5.extend(recall_5.cpu().numpy())
            all_recall_at_10.extend(recall_10.cpu().numpy())

            # Accumulate batch-level metrics
            all_batch_neg_sims.extend(batch_neg_sim.cpu().numpy())
            all_batch_triplet_accs.extend(batch_triplet_acc.cpu().numpy())
            all_batch_recall_at_1.extend(batch_recall_1.cpu().numpy())
            all_batch_recall_at_5.extend(batch_recall_5.cpu().numpy())
            all_batch_recall_at_10.extend(batch_recall_10.cpu().numpy())

        # Compute aggregate metrics
        pos_sims = np.array(all_pos_sims)
        neg_sims = np.array(all_neg_sims)
        batch_neg_sims = np.array(all_batch_neg_sims)

        # AUC-ROC: Classify positive vs negative samples
        y_true = np.concatenate([np.ones_like(pos_sims), np.zeros_like(neg_sims)])
        y_score = np.concatenate([pos_sims, neg_sims])
        auc_roc = roc_auc_score(y_true, y_score)

        metrics = {
            # Sample-level metrics (within 6 candidates)
            "triplet_accuracy": np.mean(all_triplet_accs) * 100,
            "recall@1": np.mean(all_recall_at_1) * 100,
            "recall@5": np.mean(all_recall_at_5) * 100,
            "recall@10": np.mean(all_recall_at_10) * 100,
            "positive_sim_mean": np.mean(pos_sims),
            "hard_negative_sim_mean": np.mean(neg_sims),
            "separation_margin": np.mean(pos_sims) - np.mean(neg_sims),
            "auc_roc": auc_roc,
            # Batch-level metrics (across entire batch)
            "batch_negative_sim_mean": np.mean(batch_neg_sims),
            "batch_triplet_accuracy": np.mean(all_batch_triplet_accs) * 100,
            "batch_recall@1": np.mean(all_batch_recall_at_1) * 100,
            "batch_recall@5": np.mean(all_batch_recall_at_5) * 100,
            "batch_recall@10": np.mean(all_batch_recall_at_10) * 100,
            # Hard vs Batch Neg Gap: positive means hard negatives are harder
            "hard_vs_batch_neg_gap": np.mean(neg_sims) - np.mean(batch_neg_sims),
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

    print("\n--- Sample-Level Metrics (within 6 candidates) ---")
    print(f"Triplet Accuracy:        {metrics['triplet_accuracy']:.2f}%")
    print(f"Recall@1:                {metrics['recall@1']:.2f}%")
    print(f"Recall@5:                {metrics['recall@5']:.2f}%")
    print(f"Recall@10:               {metrics['recall@10']:.2f}%")
    print(f"Positive Sim (mean):     {metrics['positive_sim_mean']:.4f}")
    print(f"Hard Negative Sim (mean): {metrics['hard_negative_sim_mean']:.4f}")
    print(f"Separation Margin:       {metrics['separation_margin']:.4f}")
    print(f"AUC-ROC:                 {metrics['auc_roc']:.4f}")

    print("\n--- Batch-Level Metrics (cross-video discrimination) ---")
    print(f"Batch Triplet Accuracy:  {metrics['batch_triplet_accuracy']:.2f}%")
    print(f"Batch Recall@1:          {metrics['batch_recall@1']:.2f}%")
    print(f"Batch Recall@5:          {metrics['batch_recall@5']:.2f}%")
    print(f"Batch Recall@10:         {metrics['batch_recall@10']:.2f}%")
    print(f"Batch Negative Sim (mean): {metrics['batch_negative_sim_mean']:.4f}")
    print(f"Hard vs Batch Neg Gap:   {metrics['hard_vs_batch_neg_gap']:.4f}")
    print("  (positive = hard negatives are harder than batch negatives)")

    print("=" * 60)


if __name__ == "__main__":
    main()
