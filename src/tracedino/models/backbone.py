"""DINOv3 backbone for multi-layer feature extraction."""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor


class DINOv3Backbone(nn.Module):
    """
    DINOv3-ViT-B/16 backbone with multi-layer feature extraction.

    Loads the model from HuggingFace format and extracts patch tokens
    from intermediate layers {3, 6, 9, 12} using output_hidden_states.

    Note: DINOv3 output includes [CLS, reg1-4, patch1-196] tokens (201 total).
    We only extract the patch tokens (skip first 5 tokens).

    Args:
        model_path: Path to the DINOv3 model directory (HuggingFace format)
        extract_layers: Tuple of layer indices to extract features from (1-indexed)
        freeze: Whether to freeze all backbone parameters
    """

    def __init__(
        self,
        model_path: Path,
        extract_layers: Tuple[int, ...] = (3, 6, 9, 12),
        freeze: bool = True,
    ):
        super().__init__()

        self.model_path = Path(model_path)
        self.extract_layers = extract_layers
        self.freeze = freeze
        self.hidden_size = 768  # DINOv3-ViT-B hidden size

        # Load model and processor
        self.model = AutoModel.from_pretrained(str(self.model_path))
        self.processor = AutoImageProcessor.from_pretrained(str(self.model_path))

        # Freeze backbone if specified
        if self.freeze:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all parameters in the backbone."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Forward pass extracting multi-layer patch tokens.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Dict mapping layer index to patch tokens [B, N, D]
            where N = 196 for 224x224 input (14x14 patches with patch_size=16)
            and D = 768 (hidden_size)
        """
        # Forward through backbone with hidden states output
        with torch.set_grad_enabled(not self.freeze):
            outputs = self.model(pixel_values=x, output_hidden_states=True)

        # Extract patch tokens from specified layers
        # hidden_states is a tuple of (embedding_output, layer1_output, ..., layer12_output)
        # So hidden_states[i] corresponds to layer i (0 = embedding, 1-12 = transformer layers)
        multi_layer_features = {}

        for layer_idx in self.extract_layers:
            # hidden_states[layer_idx] gives output after layer `layer_idx`
            hidden_state = outputs.hidden_states[layer_idx]  # [B, 201, 768]

            # Skip CLS token (index 0) and register tokens (indices 1-4)
            # Keep only patch tokens (indices 5:)
            patch_tokens = hidden_state[:, 5:, :]  # [B, 196, 768]
            multi_layer_features[layer_idx] = patch_tokens

        return multi_layer_features

    def get_num_patches(self, image_size: int = 224, patch_size: int = 16) -> int:
        """
        Calculate the number of patches.

        Args:
            image_size: Input image size
            patch_size: Patch size

        Returns:
            Number of patches (e.g., 196 for 224x224 image with patch_size=16)
        """
        return (image_size // patch_size) ** 2

    @property
    def num_layers(self) -> int:
        """Return the number of layers to extract."""
        return len(self.extract_layers)

    @property
    def output_dim(self) -> int:
        """Return the concatenated feature dimension."""
        return self.hidden_size * self.num_layers
