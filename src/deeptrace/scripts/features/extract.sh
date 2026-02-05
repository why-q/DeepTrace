#! /bin/bash

FRAMES_DIR="/home/deepfake/data/frames"
OUTPUT_DIR="/home/deepfake/data/features"
EXTRACTOR="dinov3"
GPU=$1

source .venv/bin/activate

if [ -z "$GPU" ]; then
  echo "Error: GPU is required"
  echo "Usage: $0 <gpu>"
  echo "Example: $0 0"
  exit 1
fi

python -m deeptrace.features.extract \
  --frames-dir $FRAMES_DIR \
  --output-dir $OUTPUT_DIR \
  --extractor $EXTRACTOR \
  --model-name pretrained/dinov3/dinov3-vitb16 \
  --output-mode patch \
  --device cuda \
  --gpu $GPU \
  --amp \
  --batch-size 1500