"""Transforms — single, configurable source of truth.

CRITICAL FIX
------------
The original repository had TWO different evaluation pipelines:

  * train/val : Resize(256) + CenterCrop(224)   <- train_termite_cls.py:146
  * test      : Resize((224, 224))              <- test_termite_cls.py:400

 Feeding the same checkpoint through those two pipelines produces different
 predictions — this is the most likely mechanism behind the 96.25 % vs
 98.40 % discrepancy reported in weights/002_WEIGHTS_AUDIT.md.

 `build_eval_tf()` below is now the ONLY evaluation pipeline:
 Resize(resize) + CenterCrop(crop), matching the training-time validation
 pipeline exactly. `legacy_direct_224=True` reproduces the buggy behaviour
 for auditing purposes only — never use it for reported numbers.
"""
import io
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# custom degradations (moved verbatim from train_termite_cls.py:23-95)
# --------------------------------------------------------------------------
class RandomJPEGCompression(object):
    def __init__(self, quality_range=(40, 80), p=0.3):
        self.quality_range, self.p = quality_range, p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGaussianNoise(object):
    def __init__(self, sigma_range=(2, 8), p=0.3):
        self.sigma_range, self.p = sigma_range, p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        sigma = random.uniform(*self.sigma_range)
        arr = np.array(img).astype(np.float32)
        arr = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255)
        return Image.fromarray(arr.astype(np.uint8))


class RandomSaltPepperNoise(object):
    def __init__(self, noise_ratio_range=(0.01, 0.05), p=0.3):
        self.noise_ratio_range, self.p = noise_ratio_range, p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        ratio = random.uniform(*self.noise_ratio_range)
        arr = np.array(img)
        h, w = arr.shape[0], arr.shape[1]
        n = int(ratio * h * w)
        if n < 2:
            return img
        rh = np.random.randint(0, h, n)
        rw = np.random.randint(0, w, n)
        half = n // 2
        arr[rh[:half], rw[:half], :] = 255
        arr[rh[half:], rw[half:], :] = 0
        return Image.fromarray(arr)


# --------------------------------------------------------------------------
# pipeline builders
# --------------------------------------------------------------------------
def build_train_tf(cfg=None):
    """Training pipeline. Every strength is now configurable (was hardcoded)."""
    cfg = cfg or {}
    a = cfg.get("augment", {})
    e = cfg.get("eval_transform", {})
    size = e.get("resize", 256)
    crop = e.get("crop", 224)

    jpeg = a.get("jpeg_quality", [40, 80])
    sigma = a.get("gauss_sigma", [2, 8])
    sp = a.get("sp_ratio", [0.01, 0.05])
    jitter = a.get("color_jitter", [0.15, 0.15])

    ops = [
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(a.get("hflip", 0.5)),
        transforms.RandomRotation(a.get("rotation", 15)),
        transforms.RandomResizedCrop(crop, scale=tuple(a.get("rrc_scale", [0.8, 1.2]))),
        transforms.ColorJitter(brightness=jitter[0], contrast=jitter[1]),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
            p=a.get("blur_p", 0.3)),
        RandomJPEGCompression(quality_range=tuple(jpeg), p=a.get("jpeg_p", 0.3)),
        RandomGaussianNoise(sigma_range=tuple(sigma), p=a.get("gauss_p", 0.3)),
        RandomSaltPepperNoise(noise_ratio_range=tuple(sp), p=a.get("sp_p", 0.3)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def build_eval_tf(cfg=None, legacy_direct_224=False):
    """Evaluation pipeline. Deterministic — no random ops allowed.

    Matches the training-time validation pipeline: Resize(256) + CenterCrop(224).
    """
    e = (cfg or {}).get("eval_transform", {})
    size = e.get("resize", 256)
    crop = e.get("crop", 224)

    if legacy_direct_224:
        # BUG-PRESERVING MODE: reproduces test_termite_cls.py's non-isotropic
        # squash. Use ONLY to reproduce the 96.25 % numbers during the audit.
        return transforms.Compose([
            transforms.Resize((crop, crop)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_robust_tf(cfg=None):
    """Robustness stress pipeline used by tools/evaluate.py --mode robust.

    Parameters were hardcoded as 8 % salt-and-pepper + RandomErasing(10-20 %)
    in train_termite_cls.py:1064; they are now configurable.
    """
    e = (cfg or {}).get("eval_transform", {})
    r = (cfg or {}).get("eval", {})
    size, crop = e.get("resize", 256), e.get("crop", 224)
    ratio = r.get("robust_sp_ratio", 0.08)
    scale = tuple(r.get("robust_erase_scale", [0.10, 0.20]))

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(crop),
        RandomSaltPepperNoise(noise_ratio_range=(ratio, ratio), p=1.0),
        transforms.ToTensor(),
        transforms.RandomErasing(p=1.0, scale=scale, ratio=(0.5, 2.0), value=0.0),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
