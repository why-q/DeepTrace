"""Training script for TraceDINO."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.tracedino.config import TraceDINOConfig
from src.tracedino.dataset.datamodule import TraceDINODataModule
from src.tracedino.losses.combined import TraceDINOLoss
from src.tracedino.models.tracedino import TraceDINO


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
    epoch: int,
    config: TraceDINOConfig,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_supcon = 0.0
    total_koleo = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Flatten batch: [B, 7, C, H, W] → [B*7, C, H, W]
        batch_size = batch["images"].shape[0]
        images = batch["images"].view(-1, 3, config.image_size, config.image_size).to(device)

        # Recompute labels to ensure different samples have unique positive groups
        # Each sample: [anchor, pos1, pos2, pos3, neg1, neg2, neg3]
        # anchor + 3 positives share label i, 3 negatives have unique labels
        labels = []
        for i in range(batch_size):
            sample_labels = [
                i, i, i, i,  # anchor + 3 positives
                batch_size + i * 3,      # hard negative 1
                batch_size + i * 3 + 1,  # hard negative 2
                batch_size + i * 3 + 2,  # hard negative 3
            ]
            labels.extend(sample_labels)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        # Forward pass
        if config.use_amp:
            with torch.cuda.amp.autocast():
                embeddings = model(images)
                losses = criterion(embeddings, labels)
        else:
            embeddings = model(images)
            losses = criterion(embeddings, labels)

        loss = losses["total"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update metrics
        total_loss += loss.item()
        total_supcon += losses["supcon"].item()
        total_koleo += losses["koleo"].item()

        # Update progress bar
        if batch_idx % config.log_interval == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "supcon": f"{losses['supcon'].item():.4f}",
                "koleo": f"{losses['koleo'].item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}",
            })

    avg_loss = total_loss / len(dataloader)
    avg_supcon = total_supcon / len(dataloader)
    avg_koleo = total_koleo / len(dataloader)

    return {"loss": avg_loss, "supcon": avg_supcon, "koleo": avg_koleo}


@torch.no_grad()
def validate(model: nn.Module, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_supcon = 0.0
    total_koleo = 0.0

    for batch in tqdm(dataloader, desc="Validating"):
        batch_size = batch["images"].shape[0]
        images = batch["images"].view(-1, 3, 224, 224).to(device)

        # Recompute labels (same logic as training)
        labels = []
        for i in range(batch_size):
            sample_labels = [
                i, i, i, i,
                batch_size + i * 3,
                batch_size + i * 3 + 1,
                batch_size + i * 3 + 2,
            ]
            labels.extend(sample_labels)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        embeddings = model(images)
        losses = criterion(embeddings, labels)

        total_loss += losses["total"].item()
        total_supcon += losses["supcon"].item()
        total_koleo += losses["koleo"].item()

    avg_loss = total_loss / len(dataloader)
    avg_supcon = total_supcon / len(dataloader)
    avg_koleo = total_koleo / len(dataloader)

    return {"loss": avg_loss, "supcon": avg_supcon, "koleo": avg_koleo}


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, best=False):
    """Save checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }

    if best:
        path = config.checkpoint_dir / "best_model.pth"
    else:
        path = config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train TraceDINO")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    args = parser.parse_args()

    # Load config
    config = TraceDINOConfig()
    print(f"Configuration loaded. Checkpoint dir: {config.checkpoint_dir}")

    # Set device
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
        use_preprocessed=config.use_preprocessed,
        preprocessed_dir=config.preprocessed_dir,
    )

    train_loader = data_module.train_dataloader()
    valid_loader = data_module.valid_dataloader()
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

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

    print(f"Model created. Trainable params: {model.get_num_trainable_params():,}")

    # Create loss function
    criterion = TraceDINOLoss(
        temperature=config.temperature,
        koleo_weight=config.koleo_weight,
    )

    # Create optimizer and scheduler
    optimizer = AdamW(
        model.get_trainable_params(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs * len(train_loader),
        eta_min=config.min_learning_rate,
    )

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'='*50}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, config
        )
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"SupCon: {train_metrics['supcon']:.4f}, "
              f"KoLeo: {train_metrics['koleo']:.4f}")

        # Validate
        if epoch % config.eval_interval == 0:
            val_metrics = validate(model, valid_loader, criterion, device)
            print(f"Valid Loss: {val_metrics['loss']:.4f}, "
                  f"SupCon: {val_metrics['supcon']:.4f}, "
                  f"KoLeo: {val_metrics['koleo']:.4f}")

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, best=True)

        # Save periodic checkpoint
        if epoch % config.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, config, best=False)

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
