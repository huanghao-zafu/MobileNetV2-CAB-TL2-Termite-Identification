# 07 — Licence and Citation

Version: V1.1.0 · 2026-09-24
Migrated from `termit_repo_design/repo_v1.1/data_availability/005_LICENSE_AND_CITATION.md`.

## 1. Licence decisions

| Asset | Licence | File | Rationale |
|---|---|---|---|
| Source code | MIT | `LICENSE` | Permissive, standard for research code; makes "source code publicly available" legally true |
| Dataset (all resolutions) | CC BY 4.0 | `LICENSE-DATA` | Data-appropriate; attribution requirement preserves credit; matches the existing Zenodo record licence |
| Trained weights | CC BY 4.0 | `LICENSE-DATA` | Treated as data-derived artefacts; same attribution requirement |

Dual licensing is scoped by asset type, stated in the README, and mirrored on Zenodo (`.zenodo.json` declares MIT for the software record; the full-resolution dataset record is CC BY 4.0).

## 2. Citation formats

### Dataset / code (current preferred form)

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

### Concept DOI (cites all versions, always resolves to the latest)

`https://doi.org/10.5281/zenodo.22710888`

### Software record (this repository)

`CITATION.cff` at the repository root is the machine-readable form; GitHub
renders it as "Cite this repository". Update the `doi` field there once Zenodo
assigns the V1.1.0 DOI.

### Manuscript (once published)

> Huang, H., & Wang, H. A lightweight deep learning model based on improved MobileNetV2 for fine-grained termite identification with color attention. *Journal of Applied Entomology*.

Volume, issue and page numbers are added when the article is assigned to an issue.

## 3. Attribution line required by CC BY 4.0

> Huang, H., & Wang, H. (2026). Termite image dataset and MobileNetV2-CAB-TL2 training code for fine-grained termite identification (Version V1.1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.22710889

Anyone reusing the dataset — including the 256 px reproduction package and the
preview images in `sample_data_1024/` — must carry this attribution.

## 4. Known metadata issue (V1.0.0)

The Zenodo V1.0.0 record lists the creator as the GitHub account name "huanghao3344" instead of the authors. `.zenodo.json` (added in V1.1.0) fixes this for all future versions; the V1.0.0 record itself is immutable. Cite V1.1.0 or the concept DOI.
