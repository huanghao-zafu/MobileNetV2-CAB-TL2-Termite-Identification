# Sample images

Preview only — 5 images per category (40 total), resized to 1024 px on the long edge, JPEG quality 85.

These images are **not sufficient to reproduce any reported result**. The complete dataset is distributed separately:

| Level | Content | Where |
|---|---|---|
| Preview | 8 × 5 images, 1024 px | this folder |
| Reproduction | all images at 256 × 256 (the exact input resolution of the models) + SHA-256 manifest | GitHub Release `V1.1.0` asset `TermiteData-256px-V1.1.0.zip` |
| Full-resolution master | original 4096 × 3072 images | Zenodo (DOI — see `CITATION.cff`) |

## Category codes

| Code | Species | Caste |
|---|---|---|
| `Cf_S` / `Cf_W` | *Coptotermes formosanus* Shiraki | soldier / worker |
| `Rc_S` / `Rc_W` | *Reticulitermes chinensis* Snyder | soldier / worker |
| `Of_S` / `Of_W` | *Odontotermes formosanus* (Shiraki) | soldier / worker |
| `Mb_S` / `Mb_W` | *Macrotermes barneyi* Light | soldier / worker |

Filename convention: `<category index>_<category code>_<sequence>.jpg`.
Species names, per-category counts and the derivation chain (curation →
near-duplicate removal → individual-level split → protocol-added images) are
documented in `docs/02_DATASET_CATALOG.md`.

## Licence

Images are released under **CC BY 4.0**. Specimen identification: Yongqiang Lu
(Huzhou Termite Control Institute).
