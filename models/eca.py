"""ECA (Efficient Channel Attention) — single source of truth.

Moved out of ``train_termite_cls.py`` (original lines 203-250).
Used by the ``mbv2_eca`` ablation variant (not part of the proposed model).
"""
import torch
import torch.nn as nn


class ECALayer(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                      # [N, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)       # [N, 1, C]
        y = self.conv(y)                          # [N, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)     # [N, C, 1, 1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)


def add_eca_to_mobilenet_v2(model: nn.Module, k_size: int = 3) -> nn.Module:
    """Append an ECA layer after every InvertedResidual block.

    The block class is discovered dynamically so the code keeps working
    across torchvision versions.
    """
    InvertedResidual = None
    for m in model.features:
        if m.__class__.__name__ == "InvertedResidual":
            InvertedResidual = m.__class__
            break
    if InvertedResidual is None:
        raise RuntimeError("no InvertedResidual found in model.features")

    for i, m in enumerate(model.features):
        if isinstance(m, InvertedResidual):
            out_ch = None
            for layer in reversed(list(m.modules())):
                if isinstance(layer, nn.Conv2d):
                    out_ch = layer.out_channels
                    break
            if out_ch is None:
                raise RuntimeError("cannot infer output channels of block %d" % i)
            model.features[i] = nn.Sequential(m, ECALayer(out_ch, k_size))
    return model
