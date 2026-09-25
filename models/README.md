# `models/` — network architectures

This directory holds **architecture code**, not trained weights.
Trained `.pth` files are distributed as GitHub Release assets and are
documented in `weights/` (`weights/README.md`, `weights/weights_manifest.csv`).

## 1. File map

| File | Responsibility | Manuscript role |
|---|---|---|
| `cab.py` | `ColorAttentionBlock` + `insert_cab()` | CAB — the proposed colour-attention module |
| `eca.py` | `ECALayer` + `add_eca_to_mobilenet_v2()` | control module for the `mbv2_eca` ablation |
| `backbone.py` | one constructor per architecture compared in the study | MobileNetV2 / ResNet18 / EfficientNet-B0 / MobileNetV1 / DenseNet / timm models |
| `mobilenet_v1.py` | MobileNetV1 definition (`mbv1` baseline) | Table 3 baseline row |
| `models_builder.py` | DenseNet121 / DenseNet161 builders | Table 3 baseline rows |
| `factory.py` | name → model registry (`get_model`) | single entry point used by train & test |
| `tl2.py` | TL2 strategy: 3 LR groups, warm-up + cosine schedule, early stopping | TL2 / DLR — the second proposed component |
| `__init__.py` | package exports | — |

## 2. Where the CAB goes — and why it must not move

MobileNetV2 exposes its trunk as `model.features`. The manuscript inserts the
CAB **at index 2**:

```
features[0]  stem Conv2d(3 -> 32)
features[1]  first InvertedResidual  (16 output channels)
features[2]  ColorAttentionBlock      <-- inserted here, 16 channels
features[3..] remainder of the trunk
```

Two consequences:

1. CAB sees 16-channel, 112 × 112 feature maps — the cheapest possible place
   to apply a channel gate (128 parameters).
2. `nn.Sequential` indexes by position, so inserting at index 2 renames every
   downstream key from `features.{i}` to `features.{i+1}`. **All 35 published
   checkpoints were saved with CAB at index 2.** Changing `CAB_POSITION`
   (or `insert_cab(position=...)`) makes them unloadable with `strict=True`.
   Keep it at 2.

## 3. CAB forward pass

```python
color_feat = x - x.mean(dim=1, keepdim=True)   # remove the luminance channel
y = torch.mean(color_feat, dim=(2, 3), keepdim=True)   # global avg pool
y = self.sigmoid(self.conv2(self.relu(self.conv1(y)))) # 1x1 -> ReLU -> 1x1 -> sigmoid
return x * y                                            # channel-wise reweight
```

`hidden = max(C // reduction, 4)` with `reduction = 8`; both convolutions are
`bias=False`. The `torch.mean(..., dim=(2, 3), keepdim=True)` form is preserved
byte-for-byte from the original training script so published checkpoints load
unchanged.

## 4. TL2 (two-level transfer learning)

Optimisation only — the topology of `mbv2_tl2` equals that of `mbv2_color`.

| Group | Parameters | LR | Rationale |
|---|---|---|---|
| head | `classifier.*` — 10,248 | 1e-4 | randomly re-initialised for 8 classes |
| CAB | `features.2.*` — 128 | 1e-4 | newly inserted, trained from scratch |
| backbone | remaining `features.*` — 2,223,872 | 5e-5 | ImageNet weights, smaller LR to limit forgetting |

Schedule: linear warm-up 5 epochs (start factor 0.1) → cosine decay to 1e-6
over the remaining 75. Early stopping: patience 8 on validation accuracy; the
best-validation checkpoint is the one reported.

```python
from models import get_model, build_optimizer, build_scheduler, EarlyStopping

model = get_model("mbv2_tl2", num_classes=8)
opt   = build_optimizer(model)                       # 1e-4 / 1e-4 / 5e-5
sch   = build_scheduler(opt, epochs=80, warmup_epochs=5)
stop  = EarlyStopping(patience=8)
```

`build_optimizer(model, flat_lr=True)` gives the single-LR control used by the
"MobileNetV2 + CAB" (no DLR) ablation row.

## 5. Verified numbers (torch 2.7.1 / torchvision 0.22.1, Python 3.10.19)

| Item | Value |
|---|---|
| `mbv2_tl2` total parameters | **2,234,248** (2.2342 M, matches the manuscript) |
| head / CAB / backbone split | 10,248 / 128 / 2,223,872 |
| CAB parameters | 128 |
| CAB position | `features[2]`, 20 blocks in `features` after insertion |
| Input resolution | 224 × 224 (centre-cropped from 256 × 256) |
| Reported single-image latency | 4.37 ms → 228.7 FPS (RTX 3050 Laptop, manuscript Table 3) |

> Parameter counts are reproducible with
> `python -c "from models import get_model; m=get_model('mbv2_tl2',8); print(sum(p.numel() for p in m.parameters()))"`.

## 6. Model registry keys

`get_model(name, num_classes)` accepts (case-insensitive):
`mbv1`, `mbv2`, `mbv2_tl`, `mbv2_color`, `mbv2_tl2`, `mbv2_eca`, `res18`,
`effb0`, `mbv3`, `mobilevit`, `dense121`, `dense161`.

`mbv3` and `mobilevit` need `timm` (`pip install timm`); every other key needs
only `torch` + `torchvision`.

## 7. Reproducibility caveat (read before publishing numbers)

The evaluation geometry must match the training-time validation pipeline:
`Resize(256)` → `CenterCrop(224)`. Evaluating with a direct
`Resize((224, 224))` (no aspect-ratio-preserving crop) is the documented cause
of the 96.25 % vs 98.40 % discrepancy between the two evaluation scripts — see
`weights/002_WEIGHTS_AUDIT.md`.
