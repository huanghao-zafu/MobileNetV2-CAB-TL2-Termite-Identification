# Termite Image Dataset and MobileNetV2-CAB-TL2 Training Code(release V1.1.0).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22710889.svg)](https://doi.org/10.5281/zenodo.22710889)
[![License: MIT](https://img.shields.io/badge/code-MIT-yellow.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-blue.svg)](LICENSE-DATA)

Code, trained weights and image dataset supporting:

> Huang, H., & Wang, H. *A Lightweight Deep Learning Model Based on Improved MobileNetV2 for Fine-Grained Termite Identification With Color Attention* (submitted to *Journal of Applied Entomology*).

**MobileNetV2-CAB-TL2** is a lightweight classifier for eight termite species–caste categories (workers and soldiers of four species). A **Color Attention Block (CAB)** inserted after the first inverted residual block of MobileNetV2 recalibrates channel-centred shallow feature responses, and **differential learning rates (DLR, "TL2")** fine-tune the pretrained backbone more slowly than the new task-specific components.

## Key results (5 random seeds, mean ± SD)

| Model | Acc. (%) | Macro-F1 | Params (M) |
|---|---|---|---|
| ResNet18 | 96.25 ± 0.47 | 0.9631 ± 0.0047 | 11.180 |
| MobileNetV2 | 96.95 ± 0.21 | 0.9697 ± 0.0021 | 2.2341 |
| MobileNetV3-Large | 94.30 ± 0.62 | 0.9438 ± 0.0060 | 4.2122 |
| MobileViT-XXS | 94.75 ± 0.87 | 0.9481 ± 0.0088 | 0.9535 |
| **MobileNetV2-CAB-TL2** | **98.40 ± 0.38** | **0.9840 ± 0.0038** | **2.2342** |

Mean single-image inference latency: 4.37 ms (228.7 FPS) on an NVIDIA GeForce RTX 3050 Laptop GPU. Full ablation (CAB / DLR) in manuscript Table 4.

## Dataset

**4,000 curated RGB images** (4096 × 3072 px), exactly **500 per category** across eight species–caste classes, collected from three sites in Huzhou, Zhejiang Province, China, June–November 2025. Before partitioning, 134 near-duplicate frames (cosine-similarity screening) were excluded, and the training subset was later supplemented with additional captures following the same protocol, giving an as-built experimental working set of **4,416 images**.

| Code | Species | Caste |
|---|---|---|
| `Cf_S` / `Cf_W` | *Coptotermes formosanus* Shiraki | soldier / worker |
| `Rc_S` / `Rc_W` | *Reticulitermes chinensis* Snyder | soldier / worker |
| `Of_S` / `Of_W` | *Odontotermes formosanus* (Shiraki) | soldier / worker |
| `Mb_S` / `Mb_W` | *Macrotermes barneyi* Light | soldier / worker |

- **Split**: individual-level, approximately 70 : 20 : 10 — as-built **3,220 training / 796 validation / 400 test** images. All images of one individual go to a single subset. The test set is fixed at exactly **50 images per category (400 total)**, which is what every reported metric is computed on.
- **Per-category as-built sizes** (train / val / test): Cf_S 361/102/50 · Cf_W 306/88/50 · Mb_S 404/102/50 · Mb_W 413/101/50 · Of_S 459/103/50 · Of_W 418/97/50 · Rc_S 447/102/50 · Rc_W 412/101/50.
- **Imaging**: HONOR 200 smartphone, standard photo mode, ~3× zoom, 10 cm working distance, single-sided frosted PMMA plate background, top LED ring + bottom LED strip illumination, vertical capture angle.
- **Identification**: morphological assignment confirmed by Yongqiang Lu, Huzhou Termite Control Research Institute Co., Ltd.
- **Preprocessing** (as in the manuscript): val/test resized to 256 × 256 then centre-cropped to 224 × 224; training images augmented (including intra-class MixUp, α = 0.4) at the same resolution.

### Getting the data

| Layer | What | Where |
|---|---|---|
| Preview | 5 images per category (1024 px) | [`sample_data/`](sample_data/) in this repository |
| **Reproduction package** | the as-built experimental split (4,416 images) at 256 × 256 — the exact resolution and directory structure consumed by the models — + SHA-256 manifest | GitHub Release **V1.1.0** asset `TermiteData-256px-V1.1.0.zip` |
| Full-resolution master | curated pool: 4,000 images (8 × 500) + `near_duplicates_excluded/` (134 frames rejected by cosine screening) | Zenodo record (linked in [data_availability/004_RELEASE_MANIFEST.md](data_availability/004_RELEASE_MANIFEST.md)) |

Download and verify:

```bash
# from the V1.1.0 release page, then:
python scripts/verify_release.py --dataset path/to/TermiteData-256px
```

## Repository structure

```
├─ training/           # training pipeline (CAB, DLR, all model variants)
├─ evaluation/         # Tables 3–4, Fig. 9 confusion matrix, McNemar tests, latency
├─ interpretability/   # Fig. 10 Grad-CAM generation and composition
├─ figures/            # curve/figure plotting scripts
├─ data_tools/         # dataset auditing and split utilities
├─ deployment/         # ONNX export for edge inference
├─ models/             # weight inventory and download map (weights ship as Release assets)
├─ sample_data/        # 8 × 5 preview images
├─ scripts/            # release self-verification
├─ docs/               # 001 reproducibility guide · 002 figure/table → script map
└─ data_availability/  # statement, dataset catalog, manifests, licences, changelog
```

Every manuscript table and figure maps to a script: see [docs/002_FIGURE_TABLE_MAP.md](docs/002_FIGURE_TABLE_MAP.md).

## Quickstart

```bash
# 1. environment (reference environment of the manuscript, Table 2)
pip install -r requirements.txt        # Python 3.10, PyTorch 2.0.1, CUDA 11.8
# (a newer tested environment is provided in requirements-modern.txt)

# 2. download the dataset (V1.1.0 release asset) and unpack, then verify
python scripts/verify_release.py --dataset ./TermiteData-256px

# 3. train the proposed model (seed 0; repeat with --seed 1..4)
python training/train_termite_cls.py --model mbv2_cab_tl2 --data ./TermiteData-256px --seed 0

# 4. reproduce the evaluation tables
python evaluation/evaluate_table4_ablation_5seeds.py
python evaluation/evaluate_table3_final_lowmem_with_mcnemar.py
```

Exact hyperparameters, seed protocol and expected numbers: [docs/001_REPRODUCIBILITY_GUIDE.md](docs/001_REPRODUCIBILITY_GUIDE.md).

## Training configuration (manuscript §2.3.3)

Batch size 32, ≤ 80 epochs, AdamW (weight decay 1e-4), backbone learning rate 5e-5, CAB + classification head learning rate 1e-4, 5-epoch linear warm-up then cosine decay, cross-entropy with label smoothing ε = 0.1, early stopping patience 8 on validation accuracy, model selection by best validation accuracy. Seeds 0–4.

## Data and code availability

The dataset and source code supporting the findings of this study are publicly available at https://github.com/huanghao-zafu/termit (manuscript link `huanghao3344/termit` redirects to this repository). The reproduction-resolution dataset ships with GitHub Release V1.1.0; the full-resolution master copy is archived on Zenodo. See [data_availability/](data_availability/) for the full availability package.

## Citation

If you use the dataset or code, please cite:

```bibtex
@dataset{huang_wang_2026_termite,
  author       = {Huang, Hao and Wang, Hangjun},
  title        = {Termite image dataset and MobileNetV2-CAB-TL2 training code
                  for fine-grained termite identification},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {V1.1.0},
  doi          = {10.5281/zenodo.22710889},
  url          = {https://doi.org/10.5281/zenodo.22710889}
}
```

GitHub's "Cite this repository" button uses [CITATION.cff](CITATION.cff).

## Licence

- **Code** — MIT, see [LICENSE](LICENSE)
- **Dataset** — Creative Commons Attribution 4.0 International, see [LICENSE-DATA](LICENSE-DATA)

## Funding

Zhejiang Provincial Natural Science Foundation of China (Grant No. LY23C140004).

## Acknowledgements

We thank Yongqiang Lu (Huzhou Termite Control Research Institute Co., Ltd.) for termite species and caste identification.
