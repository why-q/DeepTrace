"""Debug script to verify TraceDINO training code works correctly.

This script tests:
1. Data loading and sampling
2. Model forward pass
3. Loss computation
4. Label correctness
"""

import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.tracedino.config import TraceDINOConfig
from src.tracedino.dataset.augmentations import TraceDINOTransform
from src.tracedino.dataset.dataset import TraceDINODataset
from src.tracedino.dataset.datamodule import TraceDINODataModule
from src.tracedino.models.tracedino import TraceDINO
from src.tracedino.losses.combined import TraceDINOLoss


def test_dataset():
    """Test dataset loading and sampling."""
    print("\n" + "=" * 60)
    print("Testing Dataset")
    print("=" * 60)

    config = TraceDINOConfig()
    transform = TraceDINOTransform(image_size=config.image_size, is_training=True)

    dataset = TraceDINODataset(
        metadata_csv=config.train_csv,
        query_video_dir=config.query_video_dir,
        source_frame_dir=config.source_frame_dir,
        transform=transform,
        n_anchor_frames=config.n_anchor_frames,
        n_hard_negatives=config.n_hard_negatives,
        safety_radius_sec=config.safety_radius_seconds,
        is_training=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of videos: {len(dataset.metadata)}")
    print(f"Anchor frames per video: {dataset.n_anchor_frames}")

    # Test single sample
    print("\nTesting single sample...")
    sample = dataset[0]
    print(f"  Images shape: {sample['images'].shape}")  # Should be [7, 3, 224, 224]
    print(f"  Labels shape: {sample['labels'].shape}")  # Should be [7]
    print(f"  Labels: {sample['labels'].tolist()}")

    assert sample["images"].shape == (7, 3, 224, 224), "Unexpected image shape"
    assert sample["labels"].shape == (7,), "Unexpected label shape"

    print("Dataset test PASSED")
    return dataset


def test_dataloader():
    """Test DataLoader and batch structure."""
    print("\n" + "=" * 60)
    print("Testing DataLoader")
    print("=" * 60)

    config = TraceDINOConfig()

    # Use smaller batch size for testing
    data_module = TraceDINODataModule(
        train_csv=config.train_csv,
        valid_csv=config.valid_csv,
        test_csv=config.test_csv,
        query_video_dir=config.query_video_dir,
        source_frame_dir=config.source_frame_dir,
        image_size=config.image_size,
        batch_size=4,  # Small batch for testing
        eval_batch_size=4,
        num_workers=0,  # Single worker for debugging
    )

    train_loader = data_module.train_dataloader()
    print(f"Train loader batches: {len(train_loader)}")

    # Get one batch
    print("\nTesting batch loading...")
    batch = next(iter(train_loader))
    print(f"  Batch images shape: {batch['images'].shape}")  # Should be [4, 7, 3, 224, 224]
    print(f"  Batch labels shape: {batch['labels'].shape}")  # Should be [4, 7]

    # Test label recomputation (as done in train.py)
    batch_size = batch["images"].shape[0]
    labels = []
    for i in range(batch_size):
        sample_labels = [
            i, i, i, i,
            batch_size + i * 3,
            batch_size + i * 3 + 1,
            batch_size + i * 3 + 2,
        ]
        labels.extend(sample_labels)
    labels = torch.tensor(labels, dtype=torch.long)

    print(f"\n  Recomputed labels shape: {labels.shape}")  # Should be [28]
    print(f"  Recomputed labels: {labels.tolist()}")

    # Verify label uniqueness for different samples
    unique_positive_labels = set()
    for i in range(batch_size):
        pos_label = labels[i * 7]  # First element of each sample
        unique_positive_labels.add(pos_label.item())

    print(f"\n  Unique positive labels: {unique_positive_labels}")
    assert len(unique_positive_labels) == batch_size, "Positive labels should be unique per sample!"

    print("DataLoader test PASSED")
    return data_module


def test_model():
    """Test model forward pass."""
    print("\n" + "=" * 60)
    print("Testing Model")
    print("=" * 60)

    config = TraceDINOConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading model...")
    model = TraceDINO(
        backbone_path=config.backbone_path,
        extract_layers=config.extract_layers,
        freeze_backbone=config.freeze_backbone,
        fused_dim=config.fused_dim,
        output_dim=config.output_dim,
        gem_p=config.gem_p,
    ).to(device)

    print(f"Total parameters: {model.get_num_total_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")  # Should be [4, 512]

    # Verify L2 normalization
    norms = torch.norm(output, p=2, dim=1)
    print(f"  Output norms: {norms.tolist()}")  # Should all be ~1.0

    assert output.shape == (4, 512), "Unexpected output shape"
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Output should be L2 normalized"

    print("Model test PASSED")
    return model


def test_loss():
    """Test loss computation."""
    print("\n" + "=" * 60)
    print("Testing Loss Function")
    print("=" * 60)

    criterion = TraceDINOLoss(temperature=0.05, koleo_weight=30.0)

    # Create dummy embeddings (L2 normalized)
    batch_size = 4
    embeddings = torch.randn(batch_size * 7, 512)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Create labels (as done in train.py)
    labels = []
    for i in range(batch_size):
        sample_labels = [
            i, i, i, i,
            batch_size + i * 3,
            batch_size + i * 3 + 1,
            batch_size + i * 3 + 2,
        ]
        labels.extend(sample_labels)
    labels = torch.tensor(labels, dtype=torch.long)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    # Compute loss
    losses = criterion(embeddings, labels)

    print(f"\nLoss values:")
    print(f"  Total: {losses['total'].item():.4f}")
    print(f"  SupCon: {losses['supcon'].item():.4f}")
    print(f"  KoLeo: {losses['koleo'].item():.4f}")

    assert not torch.isnan(losses["total"]), "Loss should not be NaN"
    assert not torch.isinf(losses["total"]), "Loss should not be Inf"

    print("Loss test PASSED")
    return criterion


def test_training_step():
    """Test a complete training step."""
    print("\n" + "=" * 60)
    print("Testing Training Step")
    print("=" * 60)

    config = TraceDINOConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Create loss and optimizer
    criterion = TraceDINOLoss(temperature=config.temperature, koleo_weight=config.koleo_weight)
    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=config.learning_rate)

    # Create dummy batch
    batch_size = 2
    images = torch.randn(batch_size, 7, 3, 224, 224).to(device)

    # Flatten images
    images_flat = images.view(-1, 3, 224, 224)

    # Create labels
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

    print(f"Input shape: {images_flat.shape}")
    print(f"Labels shape: {labels.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    model.train()
    embeddings = model(images_flat)
    print(f"Embeddings shape: {embeddings.shape}")

    # Compute loss
    losses = criterion(embeddings, labels)
    print(f"Loss: {losses['total'].item():.4f}")

    # Backward pass
    print("\nRunning backward pass...")
    optimizer.zero_grad()
    losses["total"].backward()

    # Check gradients
    grad_norms = []
    for name, param in model.adapter.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))

    print("Gradient norms (adapter):")
    for name, norm in grad_norms[:5]:
        print(f"  {name}: {norm:.6f}")

    # Optimizer step
    optimizer.step()
    print("\nOptimizer step completed")

    print("Training step test PASSED")


def main():
    print("=" * 60)
    print("TraceDINO Debug Script")
    print("=" * 60)

    try:
        # Run tests
        test_dataset()
        test_dataloader()
        test_model()
        test_loss()
        test_training_step()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
