"""Quick start example for TraceDINO."""

import torch

from src.tracedino import TraceDINO, TraceDINOConfig, TraceDINODataModule, TraceDINOLoss


def example_training():
    """Minimal training example."""

    # 1. Create configuration
    config = TraceDINOConfig()

    # 2. Create data module
    data_module = TraceDINODataModule(
        train_csv=config.train_csv,
        valid_csv=config.valid_csv,
        test_csv=config.test_csv,
        query_video_dir=config.query_video_dir,
        source_frame_dir=config.source_frame_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # 3. Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraceDINO(
        backbone_path=config.backbone_path,
        extract_layers=config.extract_layers,
        output_dim=config.output_dim,
        gem_p=config.gem_p,
    ).to(device)

    print(f"Model has {model.get_num_trainable_params():,} trainable parameters")

    # 4. Create loss function
    criterion = TraceDINOLoss(
        temperature=config.temperature,
        koleo_weight=config.koleo_weight,
    )

    # 5. Get a sample batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    # 6. Forward pass
    images = batch["images"].view(-1, 3, 224, 224).to(device)  # [B*7, C, H, W]
    labels = batch["labels"].view(-1).to(device)  # [B*7]

    embeddings = model(images)  # [B*7, 512]
    losses = criterion(embeddings, labels)

    print(f"Batch shape: {batch['images'].shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Total loss: {losses['total'].item():.4f}")
    print(f"SupCon loss: {losses['supcon'].item():.4f}")
    print(f"KoLeo loss: {losses['koleo'].item():.4f}")


def example_inference():
    """Minimal inference example."""

    config = TraceDINOConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = TraceDINO(
        backbone_path=config.backbone_path,
        output_dim=config.output_dim,
    ).to(device)

    # Load checkpoint (if available)
    checkpoint_path = config.checkpoint_dir / "best_model.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Extract features from dummy images
    model.eval()
    with torch.no_grad():
        dummy_images = torch.randn(4, 3, 224, 224).to(device)
        embeddings = model(dummy_images)
        print(f"Input shape: {dummy_images.shape}")
        print(f"Output embeddings: {embeddings.shape}")
        print(f"Embeddings are L2-normalized: {torch.allclose(embeddings.norm(dim=1), torch.ones(4).to(device))}")


if __name__ == "__main__":
    print("=" * 60)
    print("TraceDINO Example: Training")
    print("=" * 60)
    example_training()

    print("\n" + "=" * 60)
    print("TraceDINO Example: Inference")
    print("=" * 60)
    example_inference()
