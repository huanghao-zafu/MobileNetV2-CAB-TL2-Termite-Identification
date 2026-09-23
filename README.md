# MobileNetV2-CAB-TL2-Termite-Identification
A Lightweight Deep Learning Model Based on Improved MobileNetV2 for Fine-Grained Termite Identification With Color Attention
# Termite Image Dataset and MobileNetV2-CAB-TL2 Training Code

Image dataset and PyTorch training code supporting our study on lightweight,
fine-grained identification of termite species and castes using an improved
MobileNetV2 backbone with a **Color-Aware Attention Block (CAB)** and a
hierarchical two-stage transfer learning strategy (**TL2**).

## Repository contents

| Path | Description |
|---|---|
| `README.md` | This file |
| `train_termite_cls.py` | Full training pipeline: model definition, CAB module, TL2 learning-rate schedule, training and evaluation loops |
| `TermiteData.zip` | Termite image dataset (see *Dataset* below) |
| `LICENSE` | MIT for the code |
| `CITATION.cff` | Machine-readable citation metadata |
| `.zenodo.json` | Metadata used to build the Zenodo record |

The archived release is citable via Zenodo:

> Huang, H., & Wang, H. (2026). *Termite image dataset and MobileNetV2-CAB-TL2
> training code for fine-grained termite identification* (Version 1.0.0)
> [Data set]. Zenodo. https://doi.org/10.5281/zenodo.22710889

## Research background

Termites are among the most destructive and highly cryptic structural and
agricultural insect pests worldwide, inflicting irreversible damage on food
crop cultivation, economic forestry production, reservoir dams, water
conservancy infrastructure and building structures. Global direct economic
losses from subterranean and wood-feeding termite infestations are estimated
to exceed US$40 billion annually. Their cryptic foraging behaviour makes
early-stage detection difficult, so field-level surveillance is a matter of
temporal urgency.

Within functional colonies, distinct castes — predominantly soldiers and
workers — execute specialised behavioural roles spanning colony defence,
foraging and nest engineering. From the perspective of Integrated Pest
Management (IPM), termite genera and species occupy highly divergent
ecological niches: subterranean termites (e.g. *Coptotermes*,
*Odontotermes*) construct deep-seated nests threatening water-conservancy
infrastructure, whereas drywood termites (e.g. *Cryptotermes*) are confined
to timber structures without soil contact. These differences imply distinct
damage dynamics, migration patterns and sensitivities to chemical or
bait-based treatments. Tracking caste composition (e.g. the soldier-to-worker
ratio) additionally serves as a biological indicator of the horizontal
transfer efficacy of chronic baiting systems.

Accurate and rapid fine-grained identification of termite species and castes
is therefore a prerequisite for assessing infestation severity and designing
targeted IPM strategies. However, termites are polymorphic insects that are
difficult to collect in adult form and exhibit high morphological similarity
within colonies. Key discriminative features — head capsule proportions,
mandible morphology and subtle body colouration patterns — are closely
similar across taxa, which makes real-time field identification challenging
for non-experts.

## Dataset

```
TermiteData/
├── train/
│   ├── <Genus_species>__<caste>/      # e.g. Coptotermes_formosanus__soldier
│   │   ├── img_0001.jpg
│   │   └── ...
│   └── ...
├── val/
│   └── (same layout)
└── test/
    └── (same layout)
```

| Item | Value |
|---|---|
| Species (classes) | [ TO FILL ] |
| Castes covered | [ e.g. soldier, worker ] |
| Images per class (min / max) | [ TO FILL ] |
| Total images (train / val / test) | [ TO FILL ] |
| Image resolution | [ e.g. 224 x 224 after resize ] |
| Acquisition device | [ camera model / settings ] |
| Collection sites and period | [ TO FILL ] |
| Imbalance handling | [ stratified split / class weights / augmentation ] |
| Specimens identified by | [ e.g. Yongqiang Lu, Huzhou Termite Control Institute ] |
| Voucher specimens deposited at | [ institution, or "not retained" ] |

## Model architecture

Overall architecture of MobileNetV2-CAB-TL2. MobileNetV2 is adopted as the
backbone because of its favourable trade-off between computational
lightweightness and feature representation capability. Two structural
enhancements are integrated into the pipeline:

1. **Feature extraction backbone.** The input image is first processed by the
   preliminary convolutional layers of MobileNetV2 to extract shallow-level
   spatial features.
2. **Colour-Aware Attention Block (CAB).** A custom channel attention
   mechanism is embedded in the shallow layers. The CAB adaptively
   recalibrates feature responses to amplify discriminative colour-sensitive
   patterns and subtle body-surface textures of polymorphic termites.
3. **Hierarchical classification head.** The refined feature maps are
   aggregated by global average pooling and passed to the classification
   head. A hierarchical transfer learning strategy (TL2) assigns
   differentiated learning rates to the backbone and the head, ensuring
   fine-grained adaptation while preserving generic semantic
   representations.

Implementation: [`train_termite_cls.py`](./train_termite_cls.py)

## Requirements

```
python >= 3.8
torch >= 1.10
torchvision
numpy
Pillow
scikit-learn
matplotlib
```

## Usage

Train with the default configuration:

```bash
python train_termite_cls.py \
    --data ./TermiteData \
    --epochs 100 \
    --batch-size 32 \
    --lr-head 1e-3 \
    --lr-backbone 1e-4
```

Evaluate a checkpoint:

```bash
python train_termite_cls.py --data ./TermiteData --eval-only \
    --checkpoint ./checkpoints/best.pth
```

*(Adjust the flags to match the argument names in `train_termite_cls.py`.)*

## Reproducibility notes

Reported numbers can be reproduced with the configuration recorded here:

- framework and CUDA version: [ TO FILL ]
- random seed(s): [ TO FILL ]
- exact train / val / test split: shipped in the dataset archive
- hardware used: [ TO FILL ]

## Data availability

<!--
Choose ONE of the two statements below, then delete the other.
Use the same wording as the manuscript's Data Availability Statement.
Statement B carries real desk-reject risk with the target journal —
only use it if confidentiality genuinely prevents full release.
-->

**Option A — full release (use when everything supporting the findings is
included in this archive).**

The dataset, trained model weights and source code supporting the findings of
this study are openly available in the Zenodo repository at
https://doi.org/10.5281/zenodo.22710889. The dataset is released under the
Creative Commons Attribution 4.0 International licence (CC BY 4.0) and the
code under the MIT licence. All analyses reported in the associated
manuscript can be reproduced from these materials.

**Option B — partial release (only if a confidentiality agreement genuinely
prevents full release).**

The source code, trained model weights and the subset of image data that may
be released under the confidentiality agreement governing this study are
openly available at https://doi.org/10.5281/zenodo.22710889. The remaining
raw images are subject to a confidentiality agreement with the collaborating
local government agency and cannot be made publicly available. The released
materials are sufficient to reproduce all analyses reported in the
associated manuscript.

## Funding

This research was supported by the Zhejiang Provincial Natural Science
Foundation of China under Grant No. LY23C140004.

## Acknowledgements

We thank Yongqiang Lu (Huzhou Termite Control Institute) for termite species
and caste identification.

## Licence

- Code (`train_termite_cls.py`): MIT — see [`LICENSE`](./LICENSE)
- Dataset (`TermiteData`): Creative Commons Attribution 4.0 International
  (CC BY 4.0) — https://creativecommons.org/licenses/by/4.0/

## Citation

```bibtex
@dataset{huang_wang_2026_termite,
  author       = {Huang, Hao and Wang, Hangjun},
  title        = {Termite image dataset and MobileNetV2-CAB-TL2 training code
                  for fine-grained termite identification},
  month        = sep,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {V1.0.0},
  doi          = {10.5281/zenodo.22710889},
  url          = {https://doi.org/10.5281/zenodo.22710889}
}
```
