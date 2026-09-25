"""models — architecture definitions for MobileNetV2-CAB-TL2.

Usage
-----
    from models import get_model, is_tl_model, build_optimizer
    from models.cab import ColorAttentionBlock, insert_cab

    model = get_model("mbv2_tl2", num_classes=8)      # MobileNetV2 + CAB
    opt   = build_optimizer(model, "mbv2_tl2", lr=1e-4, tl_backbone_lr=5e-5)
"""
from .cab import ColorAttentionBlock, insert_cab
from .eca import ECALayer, add_eca_to_mobilenet_v2
from .backbones import (build_mobilenet_v2, build_mobilenet_v2_color,
                        build_mobilenet_v2_eca, build_resnet18,
                        build_efficientnet_b0, build_mobilenet_v1,
                        build_densenet, build_timm)
from .factory import (get_model, is_tl_model, build_param_groups,
                      build_optimizer, MODEL_REGISTRY, TL_MODELS, CAB_POSITION)

__all__ = [
    "ColorAttentionBlock", "insert_cab",
    "ECALayer", "add_eca_to_mobilenet_v2",
    "build_mobilenet_v2", "build_mobilenet_v2_color", "build_mobilenet_v2_eca",
    "build_resnet18", "build_efficientnet_b0", "build_mobilenet_v1",
    "build_densenet", "build_timm",
    "get_model", "is_tl_model", "build_param_groups", "build_optimizer",
    "MODEL_REGISTRY", "TL_MODELS", "CAB_POSITION",
]
