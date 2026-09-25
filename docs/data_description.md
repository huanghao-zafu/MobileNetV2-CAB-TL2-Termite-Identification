# Data description

Short overview of the termite image dataset distributed with this repository.
The full catalogue (counts, provenance chain, imaging protocol, integrity) is in
[`02_DATASET_CATALOG.md`](02_DATASET_CATALOG.md).

## 1. What the dataset is

| Field | Value |
|---|---|
| Content | RGB images of termite **workers and soldiers** on a uniform light background |
| Categories | **8 species–caste categories** (4 species × 2 castes) |
| Curated pool | 4,000 images, exactly balanced (500 per category) |
| As-built working set | 4,416 images — train 3,220 / val 796 / test 400 |
| Native resolution | 4096 × 3072 px (JPEG) |
| Collection | Three sites in Huzhou, Zhejiang Province, China; June–November 2025 |
| Species identification | Yongqiang Lu, Huzhou Termite Control Research Institute Co., Ltd. |
| Licence | CC BY 4.0 (`LICENSE-DATA`) |

## 2. Category dictionary and directory layout

Folder names are the `<Genus><species>_<caste>` codes below. The released
package keeps these exact names so that `torchvision.datasets.ImageFolder`
reproduces the class order used in the manuscript (`0`–`7` in this order).

| Index | Code | Species | Caste | Train | Val | Test | Total |
|---|---|---|---|---|---|---|---|
| 0 | `Cf_S` | *Coptotermes formosanus* Shiraki | soldier | 361 | 102 | 50 | 513 |
| 1 | `Cf_W` | *Coptotermes formosanus* Shiraki | worker | 306 | 88 | 50 | 444 |
| 2 | `Mb_S` | *Macrotermes barneyi* Light | soldier | 404 | 102 | 50 | 556 |
| 3 | `Mb_W` | *Macrotermes barneyi* Light | worker | 413 | 101 | 50 | 564 |
| 4 | `Of_S` | *Odontotermes formosanus* (Shiraki) | soldier | 459 | 103 | 50 | 612 |
| 5 | `Of_W` | *Odontotermes formosanus* (Shiraki) | worker | 418 | 97 | 50 | 565 |
| 6 | `Rc_S` | *Reticulitermes chinensis* Snyder | soldier | 447 | 102 | 50 | 599 |
| 7 | `Rc_W` | *Reticulitermes chinensis* Snyder | worker | 412 | 101 | 50 | 563 |
| | | | **Total** | **3,220** | **796** | **400** | **4,416** |

```
TermiteData-256px/
├─ train/<CODE>/*.jpg     # 3,220 (306–459 per category)
├─ val/<CODE>/*.jpg       #   796 (88–103 per category)
├─ test/<CODE>/*.jpg      #   400 (exactly 50 per category)
└─ SHA256SUMS.txt
```

Filename convention: `<category index>_<category code>_<sequence>.jpg`
(for example `4_Of_S_042.jpg`).

## 3. What is shipped where — three tiers

| Tier | Content | Where | Size |
|---|---|---|---|
| Preview | 8 categories × 5 images, 1024 px long edge, JPEG q85 | `sample_data_1024/` (in git) | ≈ 2.4 MB |
| Reproduction | all 4,416 images at 256 × 256 px (the exact model input resolution) + SHA-256 manifest | GitHub Release `V1.1.0` → `TermiteData-256px-V1.1.0.zip` | ≈ 400–550 MB |
| Full-resolution master | original 4096 × 3072 px images (4,000 curated pool) | Zenodo (DOI in `CITATION.cff`) | ≈ 9.7 GB |

The preview set is **not sufficient to reproduce any reported result** — it is
there so a reviewer can see the data immediately without a large download.
The 256 × 256 package reproduces every reported number, because the pipeline
resizes to 256 × 256 and centre-crops to 224 × 224 at train/val/test time; the
models never see information beyond 256 px.

## 4. How the split was produced

Individual-level partition (all images of one individual stay in one subset),
performed **before** augmentation, realised as ≈ 72.9 / 18.0 / 9.1 %
(train/val/test). Augmentation is applied to the training split only. The test
split is fixed at exactly 50 images per category, which is what makes the
aggregated 5-seed confusion matrix (250 predictions per category) well defined.

Do not re-split: point the trainers at the released `train/`, `val/`, `test/`
directories.

## 5. Known limitations

- Single-region taxa (Huzhou, Zhejiang) and a controlled imaging protocol
  (uniform background, fixed working distance and illumination).
- All reported numbers come from one GPU platform (RTX 3050 Laptop).
- Validation across wider taxa, variable acquisition conditions and edge
  devices is future work.

## 6. Integrity

Every released archive is checksummed (SHA-256). Verify an unpacked dataset
with:

```bash
python scripts/verify_release.py --dataset ./TermiteData-256px --checksums SHA256SUMS.txt
```
