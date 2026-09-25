# 06 — Release Manifest

Version: V1.1.0 · 2026-09-24
Migrated from `termit_repo_design/repo_v1.1/data_availability/004_RELEASE_MANIFEST.md`.

Checksums and DOIs in this file are **produced at packaging time**, not authored
by hand — `scripts/build_dataset_release.py` writes `SHA256SUMS.txt` and
`scripts/build_weights_release.py` writes the weight manifest automatically.
Run them and paste the output into the tables below; nothing else in this file
is a stub.

```bash
python scripts/build_dataset_release.py --build --src ./TermiteData --dest ./release
python scripts/build_weights_release.py
```

## 1. GitHub Release V1.1.0 assets

| Asset | Content | Size (est.) | SHA-256 |
|---|---|---|---|
| `TermiteData-256px-V1.1.0.zip` | as-built experimental split: 4,416 images at 256 × 256 px, `train/ val/ test/` × 8 category folders | ~400–550 MB | written to `SHA256SUMS.txt` by `build_dataset_release.py --build` |
| `weights_cab_tl2_5seeds.zip` | `best_mbv2_tl2.pth` + seeds 0–4 | ~46 MB | written to `weights/weights_manifest.csv` by `build_weights_release.py` |
| `weights_baselines.zip` (optional) | all baseline/ablation weights, seeds 0–4 | ~1.1 GB | same manifest |

Package layout inside the dataset zip:

```
TermiteData-256px/
├─ train/<CODE>/*.jpg   # 3,220 in total (306–459 per category)
├─ val/<CODE>/*.jpg     #   796 in total (88–103 per category)
├─ test/<CODE>/*.jpg    #   400 in total (50 per category, exact)
└─ SHA256SUMS.txt
```

## 2. Zenodo records

| Record | Content | DOI | Status |
|---|---|---|---|
| Concept (all versions) | resolves to latest | 10.5281/zenodo.22710888 | active |
| V1.0.0 (2026-09-11) | snapshot of the initial repository (training script + 18.9 MB token archive) | 10.5281/zenodo.22710889 | superseded by V1.1.0 |
| V1.1.0 (this release) | full repository snapshot via Zenodo–GitHub integration; creator metadata corrected via `.zenodo.json` | assigned by Zenodo when the release is archived; record it here and in `CITATION.cff` | pending |
| Full-resolution master | 4,000 images at 4096 × 3072 px (~9.7 GB) | assigned by Zenodo on upload | pending — upload as a new version or a linked dedicated record |

## 3. Verification

```bash
python scripts/verify_release.py --dataset ./TermiteData-256px --checksums SHA256SUMS.txt
```

Checks performed: directory structure; per-split per-category counts (as-built: 3,220/796/400; `--master` mode: 8 × 500); image readability and dimensions (256 × 256); checksum match.

## 4. Deprecations

| Item | Reason | Replacement |
|---|---|---|
| `TermiteData.rar` (18.9 MB, in V1.0.0 git tree) | non-open format; not the dataset promised by the DAS | `TermiteData-256px-V1.1.0.zip` (Release) + full-res Zenodo master |
| `Huang_JAE_TitlePage_v1.docx` | manuscript material does not belong in the code repository | — |
| `models/modules/01.py` (1-byte placeholder) | empty file, non-descriptive name | `models/cab.py`, `models/backbone.py`, `models/tl2.py` |
| `samples_data/images/` (40 full-resolution images, ~90 MB) | repository bloat; close to the GitHub 100 MB per-file limit | `sample_data_1024/` (1024 px, ≈2.4 MB) |
