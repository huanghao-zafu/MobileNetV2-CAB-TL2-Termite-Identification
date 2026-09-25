"""Model factory + TL parameter grouping.

Centralises two things that were previously duplicated / inlined:
  1. ``get_model``            <- train_termite_cls.py:422 (and a second copy in
                                 test_termite_cls.py:548)
  2. the CAB-aware parameter  <- train_termite_cls.py:761-787
     grouping for TL training
"""
import torch.optim as optim

from .backbones import (build_mobilenet_v2, build_mobilenet_v2_color,
                        build_mobilenet_v2_eca, build_resnet18,
                        build_efficientnet_b0, build_mobilenet_v1,
                        build_densenet, build_timm, TIMM_ALIASES)

# Models whose *training strategy* is transfer learning (layered LR).
# Their topology is identical to the non-TL twin except for CAB.
TL_MODELS = {"mbv2_tl", "mbv2_tl2"}

# Where the CAB lives inside `features` — MUST stay 2 or every published
# checkpoint becomes unloadable.
CAB_POSITION = 2

MODEL_REGISTRY = {
    "mbv1": build_mobilenet_v1,
    "mbv2": build_mobilenet_v2,
    "mbv2_tl": build_mobilenet_v2,          # same topology, TL training
    "mbv2_color": build_mobilenet_v2_color,
    "mbv2_tl2": build_mobilenet_v2_color,   # same topology, TL training
    "mbv2_eca": build_mobilenet_v2_eca,
    "res18": build_resnet18,
    "effb0": build_efficientnet_b0,
    "mbv3": lambda n: build_timm(TIMM_ALIASES["mbv3"], n),
    "mobilevit": lambda n: build_timm(TIMM_ALIASES["mobilevit"], n),
    "dense121": lambda n: build_densenet("densenet121", n),
    "dense161": lambda n: build_densenet("densenet161", n),
}


def get_model(name: str, num_classes: int):
    """Build a model by name. Case-insensitive."""
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError("unknown model: %s (available: %s)"
                         % (name, ", ".join(sorted(MODEL_REGISTRY))))
    return MODEL_REGISTRY[key](num_classes)


def is_tl_model(name: str) -> bool:
    return name.lower() in TL_MODELS


def build_param_groups(model, lr: float, tl_backbone_lr: float,
                       weight_decay: float = 1e-4, cab_position: int = CAB_POSITION):
    """Split parameters into head / CAB / backbone groups.

    MUST be called AFTER the CAB has been inserted (see models/cab.py),
    because CAB parameters are identified by the substring ``features.{pos}``.
    """
    head_params = list(model.classifier.parameters())
    ca_params, backbone_params = [], []
    cab_prefix = "features.%d." % cab_position

    for name, param in model.named_parameters():
        if name.startswith(cab_prefix):
            ca_params.append(param)
        elif "features" in name:
            backbone_params.append(param)
        # anything else (e.g. timm heads) is intentionally left out of the
        # backbone group; add it to head if you use such a model with TL.

    return [
        {"params": head_params, "lr": lr},
        {"params": ca_params, "lr": lr},
        {"params": backbone_params, "lr": tl_backbone_lr},
    ], weight_decay


def build_optimizer(model, name: str, lr: float = 1e-4, tl_backbone_lr: float = 5e-5,
                    weight_decay: float = 1e-4):
    """AdamW with layered LR for TL models, flat LR otherwise."""
    if is_tl_model(name):
        groups, wd = build_param_groups(model, lr, tl_backbone_lr, weight_decay)
        return optim.AdamW(groups, weight_decay=wd)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
