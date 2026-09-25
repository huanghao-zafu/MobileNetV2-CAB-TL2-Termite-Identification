"""Color Attention Block (CAB) — single source of truth.

Moved out of ``train_termite_cls.py`` (original lines 178-199) so that the
training script and the evaluation script share ONE definition.

The forward path is kept byte-for-byte identical to the original
implementation (``torch.mean(..., dim=(2, 3), keepdim=True)``) so that the
already-published 35 checkpoints load with ``strict=True``.
"""
import torch
import torch.nn as nn


class ColorAttentionBlock(nn.Module):
    """Lightweight colour attention.

    Removes the luminance (grey) component before pooling, so the gate is
    driven purely by chroma statistics.

    Args:
        channels: input/output channel count (16 for the MobileNetV2 insertion
            point used in the manuscript).
        reduction: hidden-layer compression ratio; hidden = max(C//reduction, 4).
    """

    def __init__(self, channels: int, reduction: int = 8, min_hidden: int = 4):
        super().__init__()
        hidden = max(channels // reduction, min_hidden)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        color_feat = x - x.mean(dim=1, keepdim=True)      # drop grey component
        y = torch.mean(color_feat, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y


def _last_conv_out_channels(block: nn.Module) -> int:
    for layer in reversed(list(block.modules())):
        if isinstance(layer, nn.Conv2d):
            return layer.out_channels
    raise RuntimeError("cannot infer output channels from block")


def insert_cab(model: nn.Module, position: int = 2, reduction: int = 8,
               channels: int = None) -> nn.Module:
    """Insert a CAB into ``model.features`` at ``position``.

    For MobileNetV2 the manuscript inserts at index 2:
        features[0] = stem conv, features[1] = first InvertedResidual (16 ch),
        features[2] = CAB  <-- inserted here.

    Warning: changing ``position`` renames every downstream ``features.{i}``
    key and therefore breaks all published checkpoints. Keep it at 2.

    Returns:
        the same model object (mutated in place), for chaining.
    """
    if not hasattr(model, "features"):
        raise RuntimeError("model has no .features attribute; CAB insertion "
                           "is only defined for MobileNetV2-style backbones")
    feats = list(model.features)
    if channels is None:
        channels = _last_conv_out_channels(feats[position - 1])
    cab = ColorAttentionBlock(channels, reduction=reduction)
    new_feats = feats[:position] + [cab] + feats[position:]
    model.features = nn.Sequential(*new_feats)
    return model


def cab_parameter_names(model: nn.Module, position: int = 2):
    """Yield parameter names that belong to the inserted CAB."""
    prefix = "features.%d." % position
    for name, _ in model.named_parameters():
        if name.startswith(prefix):
            yield name
