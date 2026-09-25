# 05 — Code Manifest

Version: V1.1.0 · 2026-09-24
Migrated from `termit_repo_design/repo_v1.1/data_availability/003_CODE_MANIFEST.md`.

## 1. Environments

| | Reference (manuscript Table 2) | Alternative (tested) |
|---|---|---|
| OS / driver | Windows, NVIDIA driver 512.15 | Windows |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GB VRAM | same class or better |
| CUDA | 11.8 | 12.x |
| Python | 3.10.19 | 3.13 |
| PyTorch | 2.0.1 (`torchvision` 0.15.2) | 2.7.1 (`torchvision` 0.22.1) |
| Pins | `requirements.txt` | `requirements-modern.txt` |

Reported results were obtained with the reference environment. Before tagging, commit a `pip freeze` snapshot of the actual training environment as `environment-freeze.txt`.

## 2. Directory map and entry points

| Directory | Contents | Key entry point |
|---|---|---|
| `models/` | Architecture package: CAB, ECA control, every backbone, MobileNetV1 / DenseNet builders, model registry, TL2 strategy | `models/factory.py::get_model`, `models/tl2.py` |
| `train_termite_cls.py` (root) | Training pipeline: base trainer, CAB (+colour) variant, ECA control, single-model fixed trainer, augmentation mixers | `python train_termite_cls.py --model mbv2_tl2 --data ./TermiteData-256px --seed 0` |
| `test_termite_cls.py` (root) | Evaluation entry point | `python test_termite_cls.py` |
| `evaluation/` | Table 3 (with McNemar), Table 4 ablation (5 seeds), all-model evaluation, aggregated confusion matrix (Fig. 9), parameter counting, latency/FPS benchmark, robustness tests | `evaluation/evaluate_table3_final_lowmem_with_mcnemar.py`, `evaluation/evaluate_table4_ablation_5seeds.py` |
| `interpretability/` | Grad-CAM generation (publication version), batch automation, model comparison, Fig. 10 composition | `interpretability/generate_gradcam_publication_final_v2.py` |
| `figures/` | Curve and figure plotting (ablation convergence, comparative curves, Fig. 3 diagram) | see [`03_FIGURE_TABLE_MAP.md`](03_FIGURE_TABLE_MAP.md) |
| `scripts/` | Release self-verification, weight/dataset packaging | `scripts/verify_release.py` |
| `weights/` | Weight inventory (weights ship as Release assets, not in git) | `weights/README.md` |
| `docs/` | This documentation set (01–08) | `docs/data_description.md` |
| `sample_data_1024/` | Preview subset committed in git (8 × 5 images, 1024 px) | `sample_data_1024/README.md` |

## 3. Architecture package — the single source of truth

`models/` was extracted from `train_termite_cls.py` so that the training script
and the evaluation script share one definition of every module:

| Symbol | Original location in `train_termite_cls.py` |
|---|---|
| `ColorAttentionBlock`, `insert_cab` | lines 178–199 |
| `ECALayer`, `add_eca_to_mobilenet_v2` | lines 203–250 |
| `build_mobilenet_v2` / `_color` / `_eca`, ResNet18, EfficientNet-B0 | lines 333–457 |
| `get_model` | line 422 (duplicated at `test_termite_cls.py:548`) |
| TL2 parameter grouping | lines 761–787 |

Verified: `get_model("mbv2_tl2", 8)` → 2,234,248 parameters, CAB at
`features[2]`, three LR groups (1e-4 / 1e-4 / 5e-5).

## 4. Trained weights (Release assets, not in git)

| Asset | Files | Maps to |
|---|---|---|
| `weights_cab_tl2_5seeds.zip` (~46 MB) | `best_mbv2_tl2.pth` + `best_mbv2_tl2_seed0..4.pth` | Proposed model, Table 3/4, Figs 8–10 |
| `weights_baselines.zip` (optional, ~1.1 GB) | ResNet18 / MobileNetV2(±CAB, ±DLR, ±ECA) / MobileNetV3 / MobileViT / ShuffleNetV2 / EfficientNet-B0, seeds 0–4 | Tables 3–4, Fig. 10 |

See `weights/README.md` for the file → model → manuscript-item mapping and loading instructions.

## 5. Coding conventions kept from the working repository

- No refactoring was performed on the migrated scripts: they run exactly as they did when the reported results were produced. Reorganisation is limited to directory placement, the extraction of `models/`, and this manifest.
- Historical/intermediate script variants (e.g. earlier `eval_*`, `merge_cam_matrix_*` drafts, memory diagnostics) were intentionally **not** migrated; they remain in the authors' working copy.
- `mobilenet_v1.py` and `models_builder.py` were previously imported by `train_termite_cls.py` but never committed. They are now in `models/`; the training script must import them from there (`from models.mobilenet_v1 import mobilenet_v1`).
