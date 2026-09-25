# 02 — Dataset Catalog

Version: V1.1.0 · 2026-09-24 · Maintains manuscript §2.1–2.2 (Tables 1–2 context)
Migrated from `termit_repo_design/repo_v1.1/data_availability/002_DATASET_CATALOG.md`.

## 1. Identity

| Field | Value |
|---|---|
| Name | Termite species–caste image dataset (Huzhou 2025) |
| Content | 4,000 curated RGB images of worker and soldier termites on a uniform light background (8 × 500) |
| Native resolution | 4096 × 3072 px (JPEG) |
| Categories | 8 species–caste categories, exactly balanced (500 images per category in the curated pool) |
| Collection | Three sites in Huzhou, Zhejiang Province, China; June–November 2025 |
| Released resolutions | 256 × 256 px reproduction package (GitHub Release) · 4096 × 3072 px master (Zenodo) |
| Licences | Data: CC BY 4.0 · Code: MIT |

## 2. Category dictionary

Folder names follow `<Genus><species>_<caste>` codes; the reproduction package keeps these exact names so `torchvision.datasets.ImageFolder` reproduces the class order used in the manuscript.

| Code | Species | Caste | Train | Val | Test | Total |
|---|---|---|---|---|---|---|
| `Cf_S` | *Coptotermes formosanus* Shiraki | soldier | 361 | 102 | 50 | 513 |
| `Cf_W` | *Coptotermes formosanus* Shiraki | worker | 306 | 88 | 50 | 444 |
| `Mb_S` | *Macrotermes barneyi* Light | soldier | 404 | 102 | 50 | 556 |
| `Mb_W` | *Macrotermes barneyi* Light | worker | 413 | 101 | 50 | 564 |
| `Of_S` | *Odontotermes formosanus* (Shiraki) | soldier | 459 | 103 | 50 | 612 |
| `Of_W` | *Odontotermes formosanus* (Shiraki) | worker | 418 | 97 | 50 | 565 |
| `Rc_S` | *Reticulitermes chinensis* Snyder | soldier | 447 | 102 | 50 | 599 |
| `Rc_W` | *Reticulitermes chinensis* Snyder | worker | 412 | 101 | 50 | 563 |
| **Total** | | | **3,220** | **796** | **400** | **4,416** |

Split properties (manuscript §2.2.1): individual-level partition (approximately 70:20:10 → realised as 72.9 / 18.0 / 9.1 %) performed **before** augmentation; all images of one individual belong to a single subset; augmentation applied to the training set only. The test set is fixed at exactly 50 images per category (400 images), consistent with the aggregated confusion matrix (Fig. 9: 250 predictions per category over 5 seeds).

## 2a. From curated pool to experimental working set (provenance chain)

| Stage | Images | Notes |
|---|---|---|
| Curated pool (manuscript §2.1) | **4,000** (8 × 500) | visual quality screening of raw captures; stored as `TermiteData/raw/` |
| Near-duplicate exclusion | −134 | cosine-similarity screening; frames kept in `near_duplicates_excluded/` (published for transparency) |
| Individual-level split | 3,873 → train / val / test | 2025-11-15 |
| Protocol-matched supplementary captures | +509 train (2025-11-17), +10 test (2026-01-16), +24 val (2026-08/09) | self-collected under the same imaging protocol |
| **As-built experimental working set** | **4,416** (3,220 / 796 / 400) | the directory actually consumed by training & evaluation |

The **Zenodo master** archives the 4,000-image curated pool; the **GitHub 256px reproduction package** mirrors the as-built working set exactly — both counts above are reproducible with `scripts/verify_release.py --mode master|repro`.

## 3. Imaging protocol (manuscript Table 1)

| Parameter | Value |
|---|---|
| Device | HONOR 200 smartphone, standard photo mode |
| Resolution | 4096 × 3072 |
| Background | Single-sided frosted PMMA plate |
| Illumination | Top LED ring fill light + bottom LED strip lights |
| Capture angle | Vertical |
| Zoom | ~3× |
| Working distance | 10 cm |

Specimen selection: undamaged workers and soldiers; ~100 individuals per category, each photographed 5–6 times; blurred or mis-positioned shots discarded on visual inspection; 500 retained per category. Near-duplicate frames of the same individual were further excluded by cosine-similarity screening before partitioning.

## 4. Quality and provenance

- Species assignment by morphological comparison (soldier head-capsule shape, mandible structure, species-specific external traits), confirmed by Yongqiang Lu, Huzhou Termite Control Research Institute Co., Ltd.
- No personal data, no GPS coordinates of colonies; site labels at city level only.
- Known limitations (as stated in the manuscript): controlled imaging on one GPU platform; single-region taxa; validation across wider taxa, variable acquisition conditions and edge devices is future work.

## 5. Integrity

SHA-256 checksums for every released archive are recorded in [`06_RELEASE_MANIFEST.md`](06_RELEASE_MANIFEST.md); verify with `scripts/verify_release.py`.
