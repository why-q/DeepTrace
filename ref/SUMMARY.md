# Summary: Segment-Level Deepfake Video Source Tracing

## Task Definition

Given a forged video and a video library, the goal is to:
1. **Video-level tracing**: Identify the source video from the library
2. **Segment-level tracing**: Precisely locate the original segment (start/end timestamps) in the source video

## DeepTrace Dataset

- **Query set**: 30,000 forged videos (5-20 seconds each)
- **Video library**: 1,500 original YouTube videos
- **Forgery methods**: InsightFace, SimSwap, InfoSwap (face-swap), Wav2Lip (lip-sync), TorToiSe/RVC (audio)
- **Augmentations**: Cropping, rotation, compression, color shift, speed changes
- **Split**: 80%/10%/10% (train/val/test)

## Core Challenges

1. **High visual similarity**: Same person in similar environments creates hard negatives
2. **Low motion**: Talking-head videos have minimal movement, limiting optical flow methods
3. **Generative tampering**: Face regions are unpredictably modified
4. **Feature confusion**: Existing descriptors lack sensitivity to fine-grained non-facial details

## Evaluation Framework

- **Video-level**: Recall@Top-N
- **Segment-level**: Frame Precision, Frame Recall, F1, FRR, FAR, AER
- **Novel metric**: FRAR@Top-N (Frame Alignment Ratio) - measures alignment quality in repetitive scenes

## TraceDINO: Proposed Feature Enhancement Method

### Architecture
- **Backbone**: Frozen DINOv3-ViT-B/14
- **Multi-layer aggregation**: Extract patch tokens from layers {3, 6, 9, 12}
- **Feature fusion**: Linear projection (Kd → d')
- **Spatial pooling**: GeM pooling (robust to cropping)
- **Projection head**: 2-layer MLP → 512-dim embedding

### Training Strategy
- **Anchors**: 32,000 frames from forged videos
- **Positives** (3 per anchor):
  1. Original aligned source frame
  2. Cropped variant (person-aware)
  3. Face-blurred variant (forces non-facial feature learning)
- **Hard negatives**: Same source video, different timestamps (>15s apart)
- **Loss**: Supervised Contrastive Loss + KoLeo Entropy Regularization
