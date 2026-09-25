# 002 — Weight / Result Consistency Audit

**Status: 🔴 blocking. Do not publish Release V1.1.0 until §4 is decided.**
Evidence date: 2026-09-24

## 1. The problem in one line

The **same checkpoints** (`best_mbv2_tl2_seed{0..4}.pth` etc.) produce **98.40%** with one evaluation script
and **≈96.25%** with another. The manuscript reports 98.40 ± 0.38%. Whichever script ships in the release
decides whether a reviewer reproduces your numbers.

## 2. Evidence

| Source | Date | What it used | MobileNetV2-CAB-TL2 result |
|---|---|---|---|
| `table3_final_evaluation/table3_final_metrics.csv` | 2026-08-25 | `best_*.pth` (no seed suffix), **single** run via `evaluate_table3_final_lowmem_with_mcnemar.py` | 397/400 = **99.25%** (one seed) |
| `table4_results/table4_per_seed.csv` | 2026-09-06 | `best_*_seed{0..4}.pth`, 5 seeds via `evaluate_table4_ablation_5seeds.py` | 98.0 / 98.25 / 99.0 / 98.25 / 98.5 → **98.40 ± 0.38%** ✅ matches manuscript |
| `evaluation_results/per_seed_results.csv` | 2026-09-07 | same `best_*_seed{0..4}.pth`, 5 seeds via `evaluate_table3_models.py` / `evaluate_all_trained_models.py` | 96.0 / 96.0 / 97.0 / 96.25 / 96.0 → **≈96.25%** ✗ |

The last two rows are the crux: identical files, one day apart, **2.15 percentage points apart**.

## 3. Two further inconsistencies to resolve

| # | Issue | Detail |
|---|---|---|
| A | **Baseline MobileNetV2 reported twice, two values** | manuscript Table 3 = 96.95 ± 0.21% (and Abstract uses this to derive "+1.45 pp"); `table4_per_seed.csv` for the same `best_mbv2_seed{0..4}.pth` = **98.40 ± 0.14%**. Both cannot be right. Either Table 3's baseline came from a different (now-overwritten) run, or one of the two tables needs correcting. |
| B | **Table 4 contradicts the manuscript's own prose** | §4.2 states "CAB raised mean accuracy from 96.95% to 97.40% and DLR to 97.20%". `table4_per_seed.csv` gives baseline 98.40, +CAB 98.35, +DLR 98.15 — i.e. **neither component improves over the baseline**, which would undercut the entire contribution claim. The prose and the CSV cannot both stand. |
| C | **EfficientNet-B0 seed 3 diverged** | `per_seed_results.csv`: test accuracy **25.75%**, no training history. Never publish that checkpoint. |

## 4. Decisions you need to make (in this order)

1. **Which pipeline is authoritative?** Re-run one checkpoint through both script families and diff the
   preprocessing (`resize 256 → centre-crop 224` per §2.2.2 vs whatever `evaluate_all_trained_models.py`
   does), normalisation constants and `model.eval()` / dropout handling. Ship only the correct one;
   delete or mark the other `deprecated_/` with a note explaining the 2-point gap.
2. **Resolve the baseline value (issue A).** Pick one number for MobileNetV2 and propagate it to Table 3,
   Table 4 and the Abstract's "+1.45 pp". If the Abstract changes, the whole conclusion sentence changes.
3. **Resolve the ablation story (issue B).** If baseline really is 98.40 and CAB alone gives 98.35, the
   current framing ("CAB helps, DLR helps, together best") is not supported by the CSV. Either the CSV is
   from a different run or §4.2 needs rewriting. This is the single most likely thing a reviewer attacks.
4. Only after 1–3: package checkpoints, because the correct set depends on which run is authoritative.

## 5. Recommended packaging once settled

See `README.md` §1 — 40 checkpoints (7 models × 5 seeds, ≈490 MB) across two Release assets, plus the
45 `history_*.npz` files needed for Fig. 8. Excluded: un-seeded August artefacts, diverged `effb0_seed3`,
`mbv2_eca`, `shufflev2`, `mbv1`.

## 6. Reproducing this audit

```bash
python -c "import pandas as pd; \
print(pd.read_csv(r'E:/imageDesigning/TermiteClassifier/table4_results/table4_per_seed.csv').groupby('model')['accuracy_%'].agg(['mean','std']))"
python -c "import pandas as pd; \
print(pd.read_csv(r'E:/imageDesigning/TermiteClassifier/evaluation_results/per_seed_results.csv').groupby('model')['test_acc_%'].agg(['mean','std']))"
```
