# Trained Weights

Weights are **not committed to git** (total ≈ 490 MB for the manuscript-relevant set). They ship as **GitHub Release V1.1.0 assets**; this folder holds the manifest, verification tooling and download helper.

## 1. What to upload — decision table

### Ship to Release V1.1.0 (required, 35 files, 489 MB raw → 452 MB zipped)

Verified by `scripts/build_weights_release.py` on 2026-09-24 (actual byte sizes below).

| `model_key` | Files | Seeds | Size | Manuscript item |
|---|---|---|---|---|
| `mbv2_tl2` | `best_mbv2_tl2_seed{0..4}.pth` | 0–4 | 44 MB | **proposed model** — Table 3 last row, Table 4 last row, Figs. 8, 10 |
| `mbv2_color` | `best_mbv2_color_seed{0..4}.pth` | 0–4 | 44 MB | MobileNetV2 + CAB — Table 4, Fig. 10 |
| `mbv2_tl` | `best_mbv2_tl_seed{0..4}.pth` | 0–4 | 44 MB | MobileNetV2 + DLR — Table 4, Fig. 10 |
| `mbv2` | `best_mbv2_seed{0..4}.pth` | 0–4 | 44 MB | baseline — Tables 3–4, Figs. 8, 10 |
| `res18` | `best_res18_seed{0..4}.pth` | 0–4 | 214 MB | ResNet18 — Table 3 |
| `mbv3` | `best_mbv3_seed{0..4}.pth` | 0–4 | 82 MB | MobileNetV3-Large — Table 3, Fig. 10 |
| `mobilevit` | `best_mobilevit_seed{0..4}.pth` | 0–4 | 19 MB | MobileViT-XXS — Table 3, Fig. 10 |

Pack as two assets so a reviewer can fetch only what they need:

- `weights_proposed_and_ablation.zip` — `mbv2_tl2` + `mbv2_color` + `mbv2_tl` + `mbv2`
  (20 files, **161.1 MB**)
  `d95e5311b75c89d999ae6c0bf4c034ce719d79575d2b77c244af8990128cbfca`
- `weights_baselines.zip` — `res18` + `mbv3` + `mobilevit`
  (15 files, **290.5 MB**)
  `6c5a17ef5283129ebd4960bf4200bf733b32699ce469d2d9c0132e0c85e42f23`

(The two digests above came from a local trial pack on 2026-09-24; re-generate them if any checkpoint changes.)

### Also publish — small and needed for figure reproduction

| Item | Files | Where | Why |
|---|---|---|---|
| Training histories | `history_*_seed{0..4}.npz` (45 files for the 9 released models) | Release asset `training_histories.zip` (≈0.1 MB) | Fig. 8 convergence curves cannot be regenerated without them |
| Class order | generated below into `class_index.json` | this folder | ImageFolder order must match training or metrics are silently wrong |
| ONNX export | `mbv2_tl2.onnx` (8.5 MB) | Release asset | deployment reproducibility; produced by `deployment/export_onnx_mbv2_tl2.py` |

### Do **not** publish

| Item | Reason |
|---|---|
| `best_mbv2.pth`, `best_mbv2_tl.pth`, `best_mbv2_tl2.pth`, `best_mbv2_color.pth`, `best_mbv2_eca.pth`, `best_mbv3.pth`, `best_res18.pth`, `best_mobilevit.pth`, `best_effb0.pth`, `best_mbv1.pth` (no `_seedN`, 2026-08-28~29) | early single-run artefacts superseded by the 5-seed runs; they are what produced `table3_final_metrics.csv` (99.25% single run) and are referenced by `evaluate_table3_final_lowmem_with_mcnemar.py`. Publishing them invites confusion about which run is authoritative. |
| `best_effb0_seed0.pth`, `best_effb0_seed3.pth` | incomplete (seeds 1/2/4 missing) **and seed 3 diverged — test accuracy 25.75%**. A broken checkpoint would be indefensible. Also not in the manuscript. |
| `best_mbv2_eca_seed0.pth` (2026-01-20) | exploratory ECA variant, not in the manuscript; only one seed |
| `best_shufflev2_seed{0..4}.pth` | not in the manuscript (Table 3 compares exactly four architectures) |
| `__pycache__/`, `.npz` for excluded models | noise |

## 2. Files that live in this folder (committed to git)

| File | Purpose |
|---|---|
| `README.md` | this document |
| `002_WEIGHTS_AUDIT.md` | the 96.25 vs 98.40 discrepancy, with the evidence and the decision you need to make |
| `weights_manifest.csv` | one row per released checkpoint: filename, model key, seed, size, SHA-256, manuscript item |
| `download_weights.py` | fetches the Release assets and verifies SHA-256 |
| `class_index.json` | canonical category code → ImageFolder index order |

## 3. Generating the manifest

```bash
python ../scripts/build_weights_release.py \
    --src "E:/imageDesigning/TermiteClassifier" \
    --out ../release \
    --manifest-only        # writes models/weights_manifest.csv, packs nothing
```

Full packaging (two zips + `training_histories.zip` + manifest):

```bash
python ../scripts/build_weights_release.py --src "E:/imageDesigning/TermiteClassifier" --out ../release
```

## 4. Downloading and verifying

```bash
python download_weights.py            # both core assets → ./checkpoints
python download_weights.py --asset proposed   # ablation set only
python download_weights.py --verify-only
```

## 5. Loading example

```python
import json, torch
from training.models_builder import get_model

class_index = json.load(open("class_index.json"))   # {"Cf_S": 0, "Cf_W": 1, ...}
model = get_model("mbv2_tl2", num_classes=8)        # MobileNetV2-CAB-TL2
state = torch.load("checkpoints/best_mbv2_tl2_seed0.pth", map_location="cpu")
model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
model.eval()
```

Each `best_*.pth` holds the epoch with the highest validation accuracy for its seed — model selection per manuscript §2.3.3 (early stopping patience 8 on validation accuracy).

## Licence

Weights inherit the repository licence: **MIT**. Cite the Zenodo DOI (see root `README.md`).
