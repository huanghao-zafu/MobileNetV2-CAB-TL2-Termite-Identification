#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_release.py — release self-verification for the termit repository.

Checks:
  1. repository structure (training/, evaluation/, ... present)
  2. dataset layout: train/ val/ test/ x 8 category folders
  3. per-split per-category counts — as-built experimental working set by default
     (train 3,220 / val 796 / test 400; census 2026-09-24)
  4. image readability and dimensions (256 x 256)
  5. SHA-256 checksums (optional, --checksums SHA256SUMS.txt)

Modes:
  default        reproduction package: train/ val/ test/ x 8 categories, 4,416 images
  --master       Zenodo master: 8 category folders x 500 + near_duplicates_excluded/
                 (sub-folders allowed inside each category)
  --counts ...   custom per-split totals

Usage:
  python scripts/verify_release.py --dataset ./TermiteData-256px
  python scripts/verify_release.py --dataset ./TermiteData-256px --checksums SHA256SUMS.txt
  python scripts/verify_release.py --dataset ./TermiteData-master-4000 --master

Writes a report to verify_report.txt and prints a PASS/FAIL summary.
"""
import argparse
import hashlib
import os
import sys

CATEGORIES = ["Cf_S", "Cf_W", "Mb_S", "Mb_W", "Of_S", "Of_W", "Rc_S", "Rc_W"]
# as-built experimental working set (census 2026-09-24, recursive) — default
ACTUAL_COUNTS = {
    "train": {"Cf_S": 361, "Cf_W": 306, "Mb_S": 404, "Mb_W": 413,
              "Of_S": 459, "Of_W": 418, "Rc_S": 447, "Rc_W": 412},
    "val":   {"Cf_S": 102, "Cf_W": 88, "Mb_S": 102, "Mb_W": 101,
              "Of_S": 103, "Of_W": 97, "Rc_S": 102, "Rc_W": 101},
    "test":  {c: 50 for c in CATEGORIES},
}
# Zenodo master: curated pool, exactly 500 per category (manuscript §2.1)
MASTER_PER_CATEGORY = 500
MASTER_TOTAL = 4000
REQUIRED_DIRS = [
    "training", "evaluation", "interpretability", "figures",
    "data_tools", "deployment", "models", "sample_data",
    "scripts", "docs", "data_availability",
]
REQUIRED_FILES = [
    "README.md", "LICENSE", "LICENSE-DATA", "CITATION.cff",
    ".zenodo.json", "requirements.txt", ".gitignore",
]
IMG_EXT = (".jpg", ".jpeg", ".png")


def fail(msgs, msg):
    msgs.append("[FAIL] " + msg)


def ok(msgs, msg):
    msgs.append("[ OK ] " + msg)


def check_structure(repo_root, msgs):
    for d in REQUIRED_DIRS:
        if os.path.isdir(os.path.join(repo_root, d)):
            ok(msgs, "directory %s/" % d)
        else:
            fail(msgs, "directory %s/ missing" % d)
    for f in REQUIRED_FILES:
        if os.path.isfile(os.path.join(repo_root, f)):
            ok(msgs, "file %s" % f)
        else:
            fail(msgs, "file %s missing" % f)


def check_dataset(ds_root, msgs, check_dims=True, sample=True, expected=None):
    expected = expected or ACTUAL_COUNTS
    for split in ("train", "val", "test"):
        exp_for_split = expected.get(split)
        if exp_for_split is None:
            continue
        split_dir = os.path.join(ds_root, split)
        if not os.path.isdir(split_dir):
            fail(msgs, "%s/ missing under %s" % (split, ds_root))
            continue
        for cat in CATEGORIES:
            cat_dir = os.path.join(split_dir, cat)
            if not os.path.isdir(cat_dir):
                fail(msgs, "%s/%s/ missing" % (split, cat))
                continue
            files = [f for f in os.listdir(cat_dir) if f.lower().endswith(IMG_EXT)]
            exp = exp_for_split[cat] if isinstance(exp_for_split, dict) else exp_for_split
            if len(files) != exp:
                fail(msgs, "%s/%s: %d images, expected %d" % (split, cat, len(files), exp))
            else:
                ok(msgs, "%s/%s: %d images" % (split, cat, len(files)))
            if check_dims:
                from PIL import Image
                probe = files[:3] if sample else files
                for f in probe:
                    p = os.path.join(cat_dir, f)
                    try:
                        with Image.open(p) as im:
                            if im.size != (256, 256):
                                fail(msgs, "%s: size %s, expected (256, 256)" % (p, im.size))
                    except Exception as e:  # noqa: BLE001
                        fail(msgs, "%s: unreadable (%s)" % (p, e))


def list_images_recursive(root):
    out = []
    for dp, _dn, fn in os.walk(root):
        out.extend(f for f in fn if f.lower().endswith(IMG_EXT))
    return sorted(out)


def check_master(ds_root, msgs, check_dims=True, sample=True):
    """Zenodo master: 8 category folders x 500 images (+ near_duplicates_excluded/)."""
    total = 0
    for cat in CATEGORIES:
        cat_dir = os.path.join(ds_root, cat)
        if not os.path.isdir(cat_dir):
            fail(msgs, "%s/ missing under %s" % (cat, ds_root))
            continue
        files = list_images_recursive(cat_dir)
        total += len(files)
        if len(files) != MASTER_PER_CATEGORY:
            fail(msgs, "%s: %d images, expected %d" % (cat, len(files), MASTER_PER_CATEGORY))
        else:
            ok(msgs, "%s: %d images" % (cat, len(files)))
        if check_dims:
            from PIL import Image
            probe = files[:3] if sample else files
            for f in probe:
                p = os.path.join(cat_dir, f)
                try:
                    with Image.open(p) as im:
                        pass
                except Exception as e:  # noqa: BLE001
                    fail(msgs, "%s: unreadable (%s)" % (p, e))
    if total != MASTER_TOTAL:
        fail(msgs, "master total: %d images, expected %d" % (total, MASTER_TOTAL))
    else:
        ok(msgs, "master total: %d images (8 x %d)" % (total, MASTER_PER_CATEGORY))
    ned = os.path.join(ds_root, "near_duplicates_excluded")
    if os.path.isdir(ned):
        n = len(list_images_recursive(ned))
        ok(msgs, "near_duplicates_excluded/: %d images (informational)" % n)
    else:
        ok(msgs, "near_duplicates_excluded/ not present (optional)")


def check_checksums(ds_root, manifest_path, msgs):
    if not os.path.isfile(manifest_path):
        fail(msgs, "checksum manifest not found: %s" % manifest_path)
        return
    with open(manifest_path, "r", encoding="utf-8") as fh:
        entries = [line.split() for line in fh if line.split()]
    n_checked = 0
    for parts in entries:
        digest, rel = parts[0], parts[1].lstrip("*")
        p = os.path.join(os.path.dirname(manifest_path) or ".", rel)
        if not os.path.isfile(p):
            p = os.path.join(ds_root, rel)
        if not os.path.isfile(p):
            fail(msgs, "checksum target missing: %s" % rel)
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest().lower() == digest.lower():
            n_checked += 1
        else:
            fail(msgs, "checksum mismatch: %s" % rel)
    ok(msgs, "checksums verified: %d files" % n_checked)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="path to the unpacked TermiteData-256px directory")
    ap.add_argument("--checksums", default=None, help="path to SHA256SUMS.txt")
    ap.add_argument("--repo-root", default=".", help="repository root (default: cwd)")
    ap.add_argument("--no-dims", action="store_true", help="skip image dimension checks")
    ap.add_argument("--master", action="store_true",
                    help="verify the Zenodo master layout (8 category folders x 500) "
                         "instead of the as-built reproduction package")
    ap.add_argument("--counts", default=None,
                    help="per-split expected counts, e.g. --counts train=350,val=100,test=50")
    args = ap.parse_args()

    expected = None
    if args.counts:
        expected = {}
        for part in args.counts.split(","):
            k, v = part.split("=")
            expected[k.strip()] = int(v)
    elif not args.master:
        expected = ACTUAL_COUNTS

    msgs = []
    check_structure(args.repo_root, msgs)
    if args.master:
        check_master(args.dataset, msgs, check_dims=not args.no_dims)
    else:
        check_dataset(args.dataset, msgs, check_dims=not args.no_dims, expected=expected)
    if args.checksums:
        check_checksums(args.dataset, args.checksums, msgs)

    report = "\n".join(msgs) + "\n"
    with open("verify_report.txt", "w", encoding="utf-8") as fh:
        fh.write(report)

    n_fail = sum(1 for m in msgs if m.startswith("[FAIL]"))
    print(report)
    print("RESULT: %s (%d checks, %d failures) — report written to verify_report.txt"
          % ("PASS" if n_fail == 0 else "FAIL", len(msgs), n_fail))
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
