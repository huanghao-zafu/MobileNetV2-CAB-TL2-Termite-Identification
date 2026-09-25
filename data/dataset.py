"""Dataset / DataLoader helpers + class-order persistence.

Why `class_index.json` matters
------------------------------
`test_termite_cls.py` read class names from a user-supplied `labels.txt`.
If that file's order differs from `ImageFolder`'s (alphabetical) order, every
metric is silently mis-assigned with no error. Persisting the order at training
time and loading it at evaluation time removes that failure mode.
"""
import json
import os

from torch.utils.data import DataLoader
from torchvision import datasets

from .transforms import build_train_tf, build_eval_tf


def get_dataloaders(data_root, cfg=None, return_test=True):
    """Build train/val(/test) loaders from ``data_root/{train,val,test}``."""
    cfg = cfg or {}
    d = cfg.get("data", {})
    train_dir = os.path.join(data_root, d.get("train_split", "train"))
    val_dir = os.path.join(data_root, d.get("val_split", "val"))
    test_dir = os.path.join(data_root, d.get("test_split", "test"))

    bs = cfg.get("train", {}).get("batch_size", 32)
    nw = d.get("num_workers", 4)
    pm = d.get("pin_memory", True)

    train_set = datasets.ImageFolder(train_dir, transform=build_train_tf(cfg))
    val_set = datasets.ImageFolder(val_dir, transform=build_eval_tf(cfg))
    class_names = train_set.classes

    loaders = [
        DataLoader(train_set, batch_size=bs, shuffle=True,
                   num_workers=nw, pin_memory=pm),
        DataLoader(val_set, batch_size=bs, shuffle=False,
                   num_workers=nw, pin_memory=pm),
    ]
    if return_test:
        test_set = datasets.ImageFolder(test_dir, transform=build_eval_tf(cfg))
        loaders.append(DataLoader(test_set, batch_size=bs, shuffle=False,
                                  num_workers=nw, pin_memory=pm))
    else:
        loaders.append(None)
    return loaders[0], loaders[1], loaders[2], class_names


def save_class_index(class_names, path="class_index.json"):
    """Persist ImageFolder's class order so evaluation cannot drift."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({name: i for i, name in enumerate(class_names)},
                  fh, ensure_ascii=False, indent=2)
    return path


def load_class_index(path="class_index.json"):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
