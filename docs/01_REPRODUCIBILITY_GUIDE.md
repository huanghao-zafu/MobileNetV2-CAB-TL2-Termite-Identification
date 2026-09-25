# 01 — Reproducibility Guide

Version: V1.1.0 · Applies to manuscript Tables 3–4 and Figures 8–10
Migrated from `termit_repo_design/repo_v1.1/docs/001_REPRODUCIBILITY_GUIDE.md`.

## 1. Environment

Reference environment (manuscript Table 2 — the environment that produced the reported numbers):

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GB VRAM |
| Driver | 512.15 |
| CUDA | 11.8 |
| Python | 3.10.19 |
| PyTorch | 2.0.1 (torchvision 0.15.2) |

```bash
conda create -n termit python=3.10
conda activate termit
pip install -r requirements.txt
```

A newer verified environment (Python 3.13 / PyTorch 2.7.1) is available as `requirements-modern.txt`. Small numerical differences across CUDA versions are expected; with identical seeds and hardware the reported means should reproduce within the stated standard deviations.

## 2. Data

1. Download `TermiteData-256px-V1.1.0.zip` from Release V1.1.0.
2. Unpack to `./TermiteData-256px` (structure: `train/ val/ test/` × 8 category folders — see [`02_DATASET_CATALOG.md`](02_DATASET_CATALOG.md)). Expected: train 3,220 / val 796 / test 400 (4,416 in total; test exactly 50 per category).
3. Verify:

```bash
python scripts/verify_release.py --dataset ./TermiteData-256px --checksums SHA256SUMS.txt
```

The 256 × 256 package reproduces every reported number: the manuscript pipeline resizes to 256 × 256 and centre-crops to 224 × 224 at train/val/test time, so the models never receive information beyond 256 px.

## 3. Training protocol (manuscript §2.3.3)

| Hyperparameter | Value |
|---|---|
| Batch size | 32 |
| Max epochs | 80 |
| Optimizer | AdamW, weight decay 1e-4 |
| LR — pretrained backbone | 5e-5 (linear warm-up 5 epochs → cosine decay) |
| LR — CAB + classification head | 1e-4 (same schedule) |
| Loss | cross-entropy, label smoothing ε = 0.1 |
| Augmentation | training split only; includes intra-class MixUp (α = 0.4) |
| Early stopping | patience 8 on validation accuracy |
| Model selection | best validation accuracy |
| Seeds | 0, 1, 2, 3, 4 (per configuration) |

```bash
for seed in 0 1 2 3 4; do
  python train_termite_cls.py --model mbv2_tl2 --data ./TermiteData-256px --seed $seed
done
```

Model keys accepted by `--model` are the registry keys in `models/factory.py`
(`mbv1`, `mbv2`, `mbv2_tl`, `mbv2_color`, `mbv2_tl2`, `mbv2_eca`, `res18`,
`effb0`, `mbv3`, `mobilevit`, `dense121`, `dense161`). The CAB position and the
TL2 learning-rate groups are defined once in `models/cab.py` and
`models/tl2.py` — do not change `CAB_POSITION`, or every published checkpoint
becomes unloadable.

## 4. Evaluation

```bash
# Table 4 (ablation, 5 seeds) and Table 3 (model comparison + McNemar)
python evaluation/evaluate_table4_ablation_5seeds.py
python evaluation/evaluate_table3_final_lowmem_with_mcnemar.py

# Fig. 9 aggregated confusion matrix
python evaluation/confusion_5seed.py

# Latency / FPS (RTX 3050 Laptop: 50 warm-up + 300 timed runs)
python evaluation/train_test_latency.py
```

Evaluation geometry is **mandatory**: `Resize(256)` → `CenterCrop(224)`. Using a
direct `Resize((224, 224))` instead is the documented cause of the 96.25 % vs
98.40 % discrepancy between the two historical evaluation scripts — see
`weights/002_WEIGHTS_AUDIT.md`.

## 5. Expected results (mean ± SD over 5 seeds)

| Configuration | Acc. (%) | Macro-F1 |
|---|---|---|
| MobileNetV2 | 96.95 ± 0.21 | 0.9697 ± 0.0021 |
| MobileNetV2 + CAB | 97.40 ± 0.42 | 0.9741 ± 0.0042 |
| MobileNetV2 + DLR | 97.20 ± 0.86 | 0.9720 ± 0.0088 |
| MobileNetV2-CAB-TL2 | **98.40 ± 0.38** | **0.9840 ± 0.0038** |

Aggregated confusion matrix (Fig. 9): 1,968 / 2,000 correct across 5 seeds (98.40%); *M. barneyi* soldiers and *R. chinensis* workers correct in 250/250; weakest category *O. formosanus* soldiers 237/250.

Latency: 4.37 ms mean single-image inference → 228.7 FPS (RTX 3050 Laptop).
Parameter count: 2,234,248 (2.2342 M) — reproducible from `models/`.

## 6. Grad-CAM (Fig. 10)

```bash
python interpretability/generate_gradcam_publication_final_v2.py
```

Settings per manuscript §3.4: one held-out test image per category, ground-truth class as target, common target-layer rule, identical visualization settings across models, per-map normalization to [0, 1].

## 7. Reproducibility caveats

- CUDA kernel selection is non-deterministic for some operations; exact per-seed numbers may differ slightly across driver/CUDA versions. The manuscript reports mean ± SD over 5 seeds, which is the reproducible quantity.
- The split is **fixed by the released directory structure** — do not re-split; simply point the trainers at `train/`, `val/`, `test/`.
- The 256 px package mirrors the as-built experimental working set (4,416 images = curated 4,000-pool → cosine near-duplicate exclusion → individual-level split → protocol-matched supplementary captures). The Zenodo master archives the 4,000-image curated pool instead; do not mix the two when reproducing — the release package is checksummed precisely to avoid this.
