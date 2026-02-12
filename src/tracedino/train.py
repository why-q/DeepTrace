"""Training script for TraceDINO with multi-GPU support."""

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.tracedino.config import TraceDINOConfig
from src.tracedino.dataset.datamodule import TraceDINODataModule
from src.tracedino.losses.combined import TraceDINOLoss
from src.tracedino.models.tracedino import TraceDINO
from src.tracedino.utils import (
    cleanup_distributed,
    gather_with_grad,
    gather_without_grad,
    is_main_process,
    setup_distributed,
)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
    epoch: int,
    config: TraceDINOConfig,
    gather_fn=None,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_supcon = 0.0
    total_koleo = 0.0

    # Only show progress bar on main process
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process())
    for batch_idx, batch in enumerate(pbar):
        # Flatten batch: [B, 7, C, H, W] → [B*7, C, H, W]
        local_batch_size = batch["images"].shape[0]
        images = batch["images"].view(-1, 3, config.image_size, config.image_size).to(device)

        # Construct globally unique labels for distributed training
        # Each sample: [anchor, pos1, pos2, pos3, neg1, neg2, neg3]
        # anchor + 3 positives share label, 3 negatives have unique labels
        labels = []
        if config.distributed:
            # Multi-GPU: labels must be globally unique
            # Positive labels: rank * local_batch_size + i
            # Negative labels: world_size * local_batch_size + rank * local_batch_size * 3 + i * 3 + j
            label_offset = config.rank * local_batch_size
            neg_offset = config.world_size * local_batch_size + config.rank * local_batch_size * 3
            for i in range(local_batch_size):
                sample_labels = [
                    label_offset + i,      # anchor
                    label_offset + i,      # positive 1
                    label_offset + i,      # positive 2
                    label_offset + i,      # positive 3
                    neg_offset + i * 3,        # hard negative 1
                    neg_offset + i * 3 + 1,    # hard negative 2
                    neg_offset + i * 3 + 2,    # hard negative 3
                ]
                labels.extend(sample_labels)
        else:
            # Single GPU: original logic
            for i in range(local_batch_size):
                sample_labels = [
                    i, i, i, i,  # anchor + 3 positives
                    local_batch_size + i * 3,      # hard negative 1
                    local_batch_size + i * 3 + 1,  # hard negative 2
                    local_batch_size + i * 3 + 2,  # hard negative 3
                ]
                labels.extend(sample_labels)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        # Forward pass
        if config.use_amp:
            with torch.cuda.amp.autocast():
                embeddings = model(images)
                losses = criterion(embeddings, labels, gather_fn=gather_fn)
        else:
            embeddings = model(images)
            losses = criterion(embeddings, labels, gather_fn=gather_fn)

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

        # Update progress bar (only on main process)
        if batch_idx % config.log_interval == 0 and is_main_process():
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
def validate(model: nn.Module, dataloader, criterion, device, config: TraceDINOConfig):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_supcon = 0.0
    total_koleo = 0.0

    for batch in tqdm(dataloader, desc="Validating", disable=not is_main_process()):
        batch_size = batch["images"].shape[0]
        images = batch["images"].view(-1, 3, 224, 224).to(device)

        # Recompute labels (same logic as training, but no gather needed for validation)
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
        # No gather_fn for validation - each GPU validates independently
        losses = criterion(embeddings, labels)

        total_loss += losses["total"].item()
        total_supcon += losses["supcon"].item()
        total_koleo += losses["koleo"].item()

    avg_loss = total_loss / len(dataloader)
    avg_supcon = total_supcon / len(dataloader)
    avg_koleo = total_koleo / len(dataloader)

    return {"loss": avg_loss, "supcon": avg_supcon, "koleo": avg_koleo}


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, best=False):
    """Save checkpoint (only on main process)."""
    if not is_main_process():
        return

    # Handle DDP wrapper
    model_to_save = model.module if hasattr(model, 'module') else model

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
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

    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    distributed = world_size > 1

    # Load config
    config = TraceDINOConfig()
    config.distributed = distributed
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if is_main_process():
        print(f"Configuration loaded. Checkpoint dir: {config.checkpoint_dir}")
        if distributed:
            print(f"Distributed training enabled: {world_size} GPUs")

    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_process():
        print(f"Using device: {device}")

    # Create data module with distributed settings
    if is_main_process():
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
        distributed=distributed,
        world_size=world_size,
        rank=rank,
    )

    train_loader = data_module.train_dataloader()
    valid_loader = data_module.valid_dataloader()
    if is_main_process():
        print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")
        if distributed:
            print(f"Per-GPU batch size: {data_module.per_gpu_batch_size}")

    # Create model
    if is_main_process():
        print("Creating model...")
    model = TraceDINO(
        backbone_path=config.backbone_path,
        extract_layers=config.extract_layers,
        freeze_backbone=config.freeze_backbone,
        fused_dim=config.fused_dim,
        output_dim=config.output_dim,
        gem_p=config.gem_p,
    ).to(device)

    if is_main_process():
        print(f"Model created. Trainable params: {model.get_num_trainable_params():,}")

    # Wrap model with DDP for distributed training
    if distributed:
        model = DDP(model, device_ids=[local_rank])
        if is_main_process():
            print("Model wrapped with DistributedDataParallel")

    # Create loss function
    criterion = TraceDINOLoss(
        temperature=config.temperature,
        koleo_weight=config.koleo_weight,
    )

    # Create optimizer and scheduler
    # Get trainable params from the underlying model if using DDP
    model_for_params = model.module if hasattr(model, 'module') else model
    optimizer = AdamW(
        model_for_params.get_trainable_params(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs * len(train_loader),
        eta_min=config.min_learning_rate,
    )

    # Setup gather function for distributed training
    gather_fn = gather_with_grad if distributed else None

    # Training loop
    best_val_loss = float("inf")

    try:
        for epoch in range(1, config.num_epochs + 1):
            # Set epoch for distributed sampler (ensures proper shuffling)
            data_module.set_epoch(epoch)

            if is_main_process():
                print(f"\n{'='*50}")
                print(f"Epoch {epoch}/{config.num_epochs}")
                print(f"{'='*50}")

            # Train
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                device, epoch, config, gather_fn=gather_fn
            )

            if is_main_process():
                print(f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"SupCon: {train_metrics['supcon']:.4f}, "
                      f"KoLeo: {train_metrics['koleo']:.4f}")

            # Synchronize before validation
            if distributed:
                dist.barrier()

            # Validate
            if epoch % config.eval_interval == 0:
                val_metrics = validate(model, valid_loader, criterion, device, config)

                if is_main_process():
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

            # Synchronize at end of epoch
            if distributed:
                dist.barrier()

        if is_main_process():
            print("\nTraining completed!")

    finally:
        # Clean up distributed training
        cleanup_distributed()


if __name__ == "__main__":
    main()
