# 04 — Data Availability Statement

Version: V1.1.0 · 2026-09-24
Migrated from `termit_repo_design/repo_v1.1/data_availability/001_DATA_AVAILABILITY_STATEMENT.md`.

## 1. Statement in the manuscript (verbatim)

> **Data Availability Statement:** The dataset and source code supporting the findings of this study are publicly available at https://github.com/huanghao3344/termit.

This file is the repository-side mirror of that statement. Any future revision of the manuscript statement must be reflected here (and vice versa).

Repository status note (2026-09-25): the canonical repository for this manuscript is now `https://github.com/huanghao-zafu/MobileNetV2-CAB-TL2-Termite-Identification`. The URL printed in the manuscript relies on a GitHub account redirect (`huanghao3344` → `huanghao-zafu`) and does not redirect at the repository-name level, so it should be updated to the canonical URL at proof stage.

## 2. How the repository makes the statement true

| Promise in the statement | Repository mechanism | Where |
|---|---|---|
| "The dataset … publicly available" | ① `sample_data_1024/` preview (5 images × 8 categories) committed in git; ② full 4,416-image reproduction-resolution package (256 × 256 px, the exact model input resolution) as a GitHub Release asset with SHA-256 checksums; ③ full-resolution master copy (4096 × 3072 px, 4,000 curated pool) archived on Zenodo, linked from the repository root | `sample_data_1024/`, Release V1.1.0, [`06_RELEASE_MANIFEST.md`](06_RELEASE_MANIFEST.md) |
| "… and source code …" | Architecture package (`models/`), training entry point, evaluation entry point, plus the evaluation / interpretability / figure scripts listed in [`03_FIGURE_TABLE_MAP.md`](03_FIGURE_TABLE_MAP.md) | `models/`, `train_termite_cls.py`, `test_termite_cls.py`, `evaluation/`, `interpretability/`, `figures/` |
| "publicly available" | No access request, no registration, no embargo; permissive licences: MIT (code) + CC BY 4.0 (data) | `LICENSE`, `LICENSE-DATA` |
| Implied: reviewable | Every manuscript table/figure maps to a script with expected output | [`03_FIGURE_TABLE_MAP.md`](03_FIGURE_TABLE_MAP.md) |
| Implied: reproducible | Pinned reference environment (manuscript Table 2), seed protocol 0–4, verification script | [`01_REPRODUCIBILITY_GUIDE.md`](01_REPRODUCIBILITY_GUIDE.md), `scripts/verify_release.py` |

## 3. Pre-release consistency checklist

Tick every box before tagging V1.1.0:

- [ ] Release asset `TermiteData-256px-V1.1.0.zip` uploaded; SHA-256 recorded in [`06_RELEASE_MANIFEST.md`](06_RELEASE_MANIFEST.md)
- [ ] Weight assets uploaded (`weights_cab_tl2_5seeds.zip`, optionally `weights_baselines.zip`)
- [ ] `scripts/verify_release.py` passes against the unpacked release dataset
- [x] Dataset category counts reconciled: curated pool 4,000 (8 × 500, manuscript §2.1); as-built working set 4,416 (3,220 / 796 / 400) — both documented in [`02_DATASET_CATALOG.md`](02_DATASET_CATALOG.md)
- [ ] `LICENSE`, `LICENSE-DATA`, `CITATION.cff`, `.zenodo.json` committed (README references all of them)
- [ ] README contains no `[ TO FILL ]` placeholders
- [ ] Manuscript title page (`Huang_JAE_TitlePage.docx`) and `TermiteData.rar` removed from the repository root
- [ ] Zenodo–GitHub integration archived V1.1.0 with corrected creator metadata (Huang, Hao; Wang, Hangjun)
- [ ] Full-resolution master uploaded to the Zenodo record; DOI recorded in [`06_RELEASE_MANIFEST.md`](06_RELEASE_MANIFEST.md)
- [ ] Manuscript proof stage: update the DAS URL to the canonical repository above
- [ ] Resolve the 96.25 % vs 98.40 % evaluation discrepancy (evaluation geometry; see `weights/002_WEIGHTS_AUDIT.md`) before publishing `results/` CSVs

## 4. Change log for this statement

| Date | Change |
|---|---|
| 2026-09 (submission) | Manuscript DAS points to `https://github.com/huanghao3344/termit`; repository at that time contained only a training script and an 18.9 MB token archive |
| 2026-09-24 | V1.1.0 redesign: full availability package as described in §2 |
| 2026-09-25 | Canonical repository switched to `MobileNetV2-CAB-TL2-Termite-Identification`; `data_availability/` package folded into `docs/` (`04`–`08`); dataset counts reconciled to 4,000 curated pool / 4,416 as-built working set |
