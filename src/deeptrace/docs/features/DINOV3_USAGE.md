# DINOv3 Multi-Mode Feature Extraction Usage Examples

## Overview

The DINOv3 feature extractor now supports three output modes:
- **CLS**: CLS token only (default, backward compatible)
- **PATCH**: All 12 transformer layer patch tokens (excludes register tokens)
- **BOTH**: Both CLS and patch tokens saved to separate files

## Output Shapes

For ViT-B/16 with 768-dim features:

| Mode | Output Shape | Files |
|------|--------------|-------|
| CLS | `(num_frames, 768)` | `video.npy` |
| PATCH | `(num_frames, 12, 196, 768)` | `video.npy` |
| BOTH | CLS: `(num_frames, 768)`<br>PATCH: `(num_frames, 12, 196, 768)` | `video_cls.npy`<br>`video_patch.npy` |

## CLI Usage

### CLS Mode (Default)
```bash
python -m deeptrace.features.extract \
  --frames-dir /path/to/frames/ \
  --output-dir /path/to/features/ \
  --extractor dinov3 \
  --model-name pretrained/dinov3/dinov3-vitb16
```

### PATCH Mode
```bash
python -m deeptrace.features.extract \
  --frames-dir /path/to/frames/ \
  --output-dir /path/to/features/ \
  --extractor dinov3 \
  --model-name pretrained/dinov3/dinov3-vitb16 \
  --output-mode patch
```

### BOTH Mode
```bash
python -m deeptrace.features.extract \
  --frames-dir /path/to/frames/ \
  --output-dir /path/to/features/ \
  --extractor dinov3 \
  --model-name pretrained/dinov3/dinov3-vitb16 \
  --output-mode both
```

### With GPU and AMP
```bash
python -m deeptrace.features.extract \
  --frames-dir /path/to/frames/ \
  --output-dir /path/to/features/ \
  --extractor dinov3 \
  --model-name pretrained/dinov3/dinov3-vitb16 \
  --output-mode patch \
  --device cuda \
  --gpu 0 \
  --amp \
  --batch-size 32
```

## Python API Usage

### CLS Mode
```python
from deeptrace.features import DINOv3FeatureExtractor

extractor = DINOv3FeatureExtractor(
    model_name_or_path="pretrained/dinov3/dinov3-vitb16",
    device="cuda",
    batch_size=32,
    output_mode="cls"  # or omit for default
)

# Extract from folder
features = extractor.extract_from_folder(
    folder_path="videos/video1/",
    pattern="*.png",
    save_path="features/video1.npy"
)
print(f"CLS features: {features.shape}")  # (num_frames, 768)
```

### PATCH Mode
```python
from deeptrace.features import DINOv3FeatureExtractor, DINOv3OutputMode

extractor = DINOv3FeatureExtractor(
    model_name_or_path="pretrained/dinov3/dinov3-vitb16",
    device="cuda",
    output_mode=DINOv3OutputMode.PATCH
)

features = extractor.extract_from_folder(
    folder_path="videos/video1/",
    pattern="*.png",
    save_path="features/video1.npy"
)
print(f"PATCH features: {features.shape}")  # (num_frames, 12, 196, 768)
```

### BOTH Mode
```python
from deeptrace.features import DINOv3FeatureExtractor

extractor = DINOv3FeatureExtractor(
    model_name_or_path="pretrained/dinov3/dinov3-vitb16",
    device="cuda",
    output_mode="both"
)

cls_features, patch_features = extractor.extract_from_folder(
    folder_path="videos/video1/",
    pattern="*.png",
    save_path="features/video1.npy"  # Will save as video1_cls.npy and video1_patch.npy
)

print(f"CLS features: {cls_features.shape}")      # (num_frames, 768)
print(f"PATCH features: {patch_features.shape}")  # (num_frames, 12, 196, 768)
```

### Direct Feature Extraction
```python
from PIL import Image
from deeptrace.features import DINOv3FeatureExtractor

extractor = DINOv3FeatureExtractor(
    model_name_or_path="pretrained/dinov3/dinov3-vitb16",
    output_mode="patch"
)

# Load images
images = [Image.open(f"frame_{i:04d}.png") for i in range(10)]

# Extract features
features = extractor.extract_features(images)
print(f"Shape: {features.shape}")  # (10, 12, 196, 768)
```

## Feature Details

### Patch Tokens
- **Number of layers**: 12 (transformer layers, excluding embedding layer)
- **Number of patches**: 196 (14×14 grid for 224×224 images with 16×16 patches)
- **Register tokens removed**: Yes (4 register tokens are excluded)
- **Normalization**: L2-normalized per patch vector

### CLS Token
- **Source**: `pooler_output` from the final transformer layer
- **Normalization**: L2-normalized

## Hugging Face Models

You can also use models directly from Hugging Face Hub:

```bash
python -m deeptrace.features.extract \
  --frames-dir /path/to/frames/ \
  --output-dir /path/to/features/ \
  --extractor dinov3 \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --output-mode patch
```

Supported models:
- `facebook/dinov3-vits16-pretrain-lvd1689m` (384-dim)
- `facebook/dinov3-vitb16-pretrain-lvd1689m` (768-dim)
- `facebook/dinov3-vitl16-pretrain-lvd1689m` (1024-dim)
- `facebook/dinov3-vith16plus-pretrain-lvd1689m` (1280-dim)
- `facebook/dinov3-vit7b16-pretrain-lvd1689m` (1536-dim)

## Loading Extracted Features

```python
import numpy as np

# CLS mode
cls_features = np.load("features/video1.npy")
print(cls_features.shape)  # (num_frames, 768)

# PATCH mode
patch_features = np.load("features/video1.npy")
print(patch_features.shape)  # (num_frames, 12, 196, 768)

# BOTH mode
cls_features = np.load("features/video1_cls.npy")
patch_features = np.load("features/video1_patch.npy")
print(cls_features.shape)    # (num_frames, 768)
print(patch_features.shape)  # (num_frames, 12, 196, 768)

# Access specific layer's patches
layer_5_patches = patch_features[:, 5, :, :]  # (num_frames, 196, 768)

# Access specific patch across all layers
patch_0_all_layers = patch_features[:, :, 0, :]  # (num_frames, 12, 768)
```

## Verification

Run the test script to verify installation:

```bash
python test_output_modes.py
```

Expected output:
- CLS features: (2, 768)
- PATCH features: (2, 12, 196, 768)
- Register tokens correctly removed (196 patches, not 200)
- L2 normalization verified
