"""models — architecture definitions for MobileNetV2-CAB-TL2.

Layout
------
    cab.py             Color Attention Block (CAB) + insertion helper
    eca.py             ECA control module (ablation variant `mbv2_eca`)
    backbone.py        every backbone constructor used in the manuscript
    mobilenet_v1.py    MobileNetV1 (`mbv1` baseline)
    models_builder.py  DenseNet builders (`dense121` / `dense161`)
    factory.py         name -> model registry (`get_model`)
    tl2.py             TL2 strategy: parameter groups, LR schedule, early stop

Quick start
-----------
    from models import get_model, build_optimizer, build_scheduler, EarlyStopping

    model = get_model("mbv2_tl2", num_classes=8)          # 2,234,224 params
    opt   = build_optimizer(model)                        # head/CAB 1e-4, backbone 5e-5
    sch   = build_scheduler(opt, epochs=80, warmup_epochs=5)
    stop  = EarlyStopping(patience=8)

The CAB is inserted at ``features[2]`` (see ``CAB_POSITION`` in ``tl2.py``).
Moving it renames every downstream ``features.{i}`` key and breaks all 35
published checkpoints — do not change it.
"""
from .cab import ColorAttentionBlock, insert_cab
from .eca import ECALayer, add_eca_to_mobilenet_v2
from .backbone import (build_mobilenet_v2, build_mobilenet_v2_color,
                       build_mobilenet_v2_eca, build_resnet18,
                       build_efficientnet_b0, build_mobilenet_v1,
                       build_densenet, build_timm, TIMM_ALIASES)
from .tl2 import (TL2Config, DEFAULT_TL2, CAB_POSITION, cab_parameter_names,
                  build_param_groups, build_optimizer, build_scheduler,
                  EarlyStopping, describe_groups)
from .factory import (get_model, is_tl_model, build_optimizer_for,
                      MODEL_REGISTRY, TL_MODELS)

# Backwards-compatible alias: earlier drafts exposed the optimizer helper under
# this name in `factory`; keep both importable.
build_optimizer_for_model = build_optimizer_for

__all__ = [
    "ColorAttentionBlock", "insert_cab",
    "ECALayer", "add_eca_to_mobilenet_v2",
    "build_mobilenet_v2", "build_mobilenet_v2_color", "build_mobilenet_v2_eca",
    "build_resnet18", "build_efficientnet_b0", "build_mobilenet_v1",
    "build_densenet", "build_timm", "TIMM_ALIASES",
    "TL2Config", "DEFAULT_TL2", "CAB_POSITION", "cab_parameter_names",
    "build_param_groups", "build_optimizer", "build_scheduler",
    "EarlyStopping", "describe_groups",
    "get_model", "is_tl_model", "build_optimizer_for",
    "build_optimizer_for_model", "MODEL_REGISTRY", "TL_MODELS",
]
