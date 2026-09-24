#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_weights.py — fetch trained weights from GitHub Release V1.1.0 and verify SHA-256.

Usage:
  python download_weights.py                     # all core assets -> ./checkpoints
  python download_weights.py --asset proposed    # ablation set only
  python download_weights.py --dest ./models
  python download_weights.py --verify-only       # verify already-downloaded files
"""
import argparse
import csv
import hashlib
import os
import sys
import urllib.request

REPO = "huanghao-zafu/termit"
TAG = "V1.1.0"
BASE = "https://github.com/%s/releases/download/%s" % (REPO, TAG)

ASSETS = {
    "proposed": "weights_proposed_and_ablation.zip",
    "baselines": "weights_baselines.zip",
    "histories": "training_histories.zip",
}
MANIFEST = "weights_manifest.csv"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print("GET %s" % url)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    print("\r  %.1f / %.1f MB" % (done / 1048576, total / 1048576), end="")
            print()
    except Exception as exc:  # noqa: BLE001
        print("  [FAIL] download error: %s" % exc)
        print("  If the asset is not published yet, open the release page and attach it:")
        print("  https://github.com/%s/releases/tag/%s" % (REPO, TAG))
        return False
    return True


def unzip(path, dest_root):
    import zipfile
    name = os.path.splitext(os.path.basename(path))[0]
    target = os.path.join(dest_root, name)
    os.makedirs(target, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(target)
    print("unpacked -> %s" % target)
    # flatten one level if the archive wrapped everything in a single folder
    inner = [d for d in os.listdir(target) if os.path.isdir(os.path.join(target, d))]
    if len(inner) == 1 and not [f for f in os.listdir(target) if f.lower().endswith(".pth")]:
        src = os.path.join(target, inner[0])
        dst = os.path.join(dest_root, inner[0])
        if not os.path.exists(dst):
            os.rename(src, dst)
            os.rmdir(target)
            print("flattened -> %s" % dst)
    return True


def verify(csv_path, search_roots):
    if not os.path.isfile(csv_path):
        print("[FAIL] manifest not found: %s" % csv_path)
        return False
    index = {}
    for root in search_roots:
        for dp, _dn, fn in os.walk(root):
            for f in fn:
                index.setdefault(f, os.path.join(dp, f))
    bad = 0
    checked = 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["filename"]
            if name not in index:
                continue
            checked += 1
            actual = sha256_of(index[name])
            if actual.lower() != row["sha256"].lower():
                print("[FAIL] %s checksum mismatch" % name)
                bad += 1
    print("verified %d checkpoints, %d mismatch(es)" % (checked, bad))
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="./checkpoints", help="download destination root")
    ap.add_argument("--asset", action="append", choices=list(ASSETS), default=None,
                    help="restrict to specific assets (repeatable); default: proposed + baselines")
    ap.add_argument("--keep-zip", action="store_true", help="keep downloaded archives after unpacking")
    ap.add_argument("--verify-only", action="store_true", help="only verify against weights_manifest.csv")
    args = ap.parse_args()

    if args.verify_only:
        here = os.path.dirname(os.path.abspath(__file__))
        ok = verify(os.path.join(here, MANIFEST), [args.dest, here])
        sys.exit(0 if ok else 1)

    wanted = args.asset or ["proposed", "baselines"]
    for key in wanted:
        name = ASSETS[key]
        dest = os.path.join(args.dest, name)
        if not download("%s/%s" % (BASE, name), dest):
            continue
        unzip(dest, args.dest)
        if not args.keep_zip:
            os.remove(dest)

    here = os.path.dirname(os.path.abspath(__file__))
    verify(os.path.join(here, MANIFEST), [args.dest])


if __name__ == "__main__":
    main()
