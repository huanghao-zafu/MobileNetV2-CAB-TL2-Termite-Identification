# 03 — Manuscript Figure/Table → Script Map

Version: V1.1.0 · Purpose: every reported artefact is one command away for a reviewer.
Migrated from `termit_repo_design/repo_v1.1/docs/002_FIGURE_TABLE_MAP.md`.

Legend: ✅ shipped in this repository · ⏳ to be added with this release (directory not yet in the repository tree) · [[confirm]] = migrated best-guess entry point; verify the exact CLI flags against the script before first run (scripts were migrated verbatim from the working copy without refactoring).

| Manuscript item | Content | Script (this repo) | Input | Expected output |
|---|---|---|---|---|
| Table 1 | Imaging parameters | — (static table, see [`02_DATASET_CATALOG.md`](02_DATASET_CATALOG.md) §3) | — | — |
| Table 2 | Experimental environment | — (static table, see [`01_REPRODUCIBILITY_GUIDE.md`](01_REPRODUCIBILITY_GUIDE.md) §1 + `requirements.txt`) | — | — |
| Table 3 | Model comparison (5 seeds, mean ± SD) | ⏳ `evaluation/evaluate_table3_final_lowmem_with_mcnemar.py` | `test/` split + weight assets | Accuracy / Macro-P / Macro-R / Macro-F1 / params per model + McNemar tests |
| Table 4 | Ablation (CAB, DLR) | ⏳ `evaluation/evaluate_table4_ablation_5seeds.py` | `test/` split + seed 0–4 weights | 4-row ablation table |
| Fig. 1 | Portable imaging platform | — (photograph) | — | — |
| Fig. 2 | Representative images, 8 categories | — (montage; preview available in `sample_data_1024/`) | dataset | — |
| Fig. 3 | Model/architecture diagram | [[confirm]] `figures/generate_fig3.py` | — | diagram |
| Fig. 4–5 | MobileNetV2 building blocks | — (schematic diagrams) | — | — |
| Fig. 6–7 | Training/validation curves | [[confirm]] `figures/generate_comparative_curves.py`, `figures/plot_vertical_curves.py` | training histories | curve figures |
| Fig. 8 | Ablation convergence, 5 seeds (mean ± SD band) | [[confirm]] `figures/plot_ablation_convergence_5seeds.py` (+ `figures/render_curves01_highres.py` for 600 dpi output) | 5-seed training histories | Fig. 8 |
| Fig. 9 | Aggregated confusion matrix (5 seeds) | ⏳ `evaluation/confusion_5seed.py` | `test/` split + seed 0–4 weights | 8 × 8 matrix, 1,968/2,000 correct |
| Fig. 10 | Grad-CAM comparison, 6 models × 8 categories | [[confirm]] `interpretability/generate_gradcam_publication_final_v2.py` + `interpretability/final_merge_gradcam_matrix.py` | `test/` split + weights | publication matrix |

Publication-figure output conventions used by the figure scripts: ≥600 dpi, 170 mm double-column width, PNG + PDF.

## Code → manuscript-item map

The architecture that produces every number above is in `models/`:

| Directory | Contents |
|---|---|
| `models/cab.py` | CAB — Fig. 3/4 component, "MobileNetV2 + CAB" and "MobileNetV2-CAB-TL2" rows |
| `models/tl2.py` | TL2 / DLR — "MobileNetV2 + DLR" and "MobileNetV2-CAB-TL2" rows |
| `models/backbone.py`, `models/factory.py` | every backbone compared in Table 3 |
| `train_termite_cls.py` | training entry point (all model keys) |
| `test_termite_cls.py` | evaluation entry point |

See `models/README.md` for the parameter-count / latency figures quoted in Table 3.

## Weight → manuscript-item map

See `weights/README.md` for the full file-level mapping (which `.pth` file backs which table row).
