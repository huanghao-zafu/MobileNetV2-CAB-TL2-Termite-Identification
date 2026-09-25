#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_weights_release.py — 挑选应发布的模型权重并打包 + 生成 manifest.

只发布与稿件对应的"9 月 5-seed"批次:
  - 提出方法 + 消融 (4 个变体 x 5 seeds = 20 个文件, ~176 MB) -> weights_proposed_and_ablation.zip
  - Table 3 基线 (3 个模型 x 5 seeds = 15 个文件, ~315 MB)    -> weights_baselines.zip
  - Fig.8 训练历史 history_*_seed*.npz                        -> training_histories.zip
  - 每个文件的 SHA-256                                         -> models/weights_manifest.csv

明确排除:
  * 8 月底无 _seedN 后缀的单跑权重(早期产物, 且 Table 3 单次评估用它)
  * effb0 (seed 1/2/4 缺失, seed3 训练发散 25.75%)
  * mbv2_eca / shufflev2 / mbv1 (不在稿件 Table 3 的四架构对比中)

用法:
  python scripts/build_weights_release.py --src "E:/imageDesigning/TermiteClassifier" --out ../release
  python scripts/build_weights_release.py --src "E:/imageDesigning/TermiteClassifier" --manifest-only
"""
import argparse
import hashlib
import os
import zipfile

# model_key -> (--model flag, 稿件条目)
CORE = {
    "mbv2_tl2": ("mbv2_tl2", "proposed / Table 3 last row / Table 4 / Figs 8,10"),
    "mbv2_color": ("mbv2_color", "MobileNetV2+CAB / Table 4 / Fig 10"),
    "mbv2_tl": ("mbv2_tl", "MobileNetV2+DLR / Table 4 / Fig 10"),
    "mbv2": ("mbv2", "baseline / Tables 3-4 / Figs 8,10"),
}
BASELINES = {
    "res18": ("res18", "ResNet18 / Table 3"),
    "mbv3": ("mbv3", "MobileNetV3-Large / Table 3 / Fig 10"),
    "mobilevit": ("mobilevit", "MobileViT-XXS / Table 3 / Fig 10"),
}
SEEDS = [0, 1, 2, 3, 4]

# 建议排除但要在报告里说明的(存在于源目录却不打包)
EXCLUDED_NOTE = {
    "best_effb0_seed0.pth": "EfficientNet-B0 不在稿件; seeds 1/2/4 缺失",
    "best_effb0_seed3.pth": "训练发散 (test acc 25.75%) —— 绝不可发布",
    "best_effb0.pth": "8 月单跑产物, 不在稿件",
    "best_mbv2_eca.pth": "探索性 ECA 变体, 不在稿件",
    "best_mbv2_eca_seed0.pth": "探索性 ECA 变体, 仅 1 seed, 不在稿件",
    "best_mbv1.pth": "8 月单跑产物, 不在稿件 Table 3",
    "best_shufflev2_seed0.pth": "ShuffleNetV2 不在稿件 Table 3",
    "best_shufflev2_seed1.pth": "ShuffleNetV2 不在稿件 Table 3",
    "best_shufflev2_seed2.pth": "ShuffleNetV2 不在稿件 Table 3",
    "best_shufflev2_seed3.pth": "ShuffleNetV2 不在稿件 Table 3",
    "best_shufflev2_seed4.pth": "ShuffleNetV2 不在稿件 Table 3",
}

LOG = []


def log(msg):
    LOG.append(msg)
    print(msg)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(src, table, label):
    """返回 [(abs_path, arc_name, model_key, seed, size, sha, note)]"""
    rows = []
    missing = []
    for key, (_flag, note) in table.items():
        for seed in SEEDS:
            name = "best_%s_seed%d.pth" % (key, seed)
            p = os.path.join(src, name)
            if not os.path.isfile(p):
                missing.append(name)
                continue
            rows.append((p, "%s/%s" % (label, name), key, seed,
                         os.path.getsize(p), None, note))
    return rows, missing


def pack(rows, zip_path, arc_root):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p, _arc, key, seed, size, _sha, _note in rows:
            zf.write(p, "%s/%s" % (arc_root, os.path.basename(p)))
    mb = os.path.getsize(zip_path) / 1024 / 1024
    return mb, sha256_of(zip_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="TermiteClassifier 项目根目录")
    ap.add_argument("--out", default="../release", help="输出目录")
    ap.add_argument("--models-dir", default=None, help="models/ 目录(写 manifest). 默认 ../release 同级")
    ap.add_argument("--histories", action="store_true", help="同时打包 history_*_seed*.npz")
    ap.add_argument("--manifest-only", action="store_true", help="只扫描并写 manifest, 不打包")
    args = ap.parse_args()

    core_rows, core_missing = collect(args.src, CORE, "proposed")
    base_rows, base_missing = collect(args.src, BASELINES, "baselines")
    all_rows = core_rows + base_rows

    if core_missing or base_missing:
        log("!! 缺失文件(需在 <src> 下补齐): %s" % ", ".join(core_missing + base_missing))

    if not all_rows:
        log("没有找到任何待发布权重, 检查 --src。")
        return

    log("=== 待发布清单 ===")
    total = 0
    for row in all_rows:
        ok_file = os.path.basename(row[0])
        size = row[4]
        total += size
        log("  %-30s %6.1f MB  seed%d  %s" % (ok_file, size / 1048576, row[3], row[6]))
    log("合计 %d 个文件, %.1f MB" % (len(all_rows), total / 1048576))

    log("")
    log("=== 排除项 ===")
    found_excluded = 0
    for name, why in EXCLUDED_NOTE.items():
        p = os.path.join(args.src, name)
        if os.path.isfile(p):
            found_excluded += 1
            log("  %-30s %s" % (name, why))
    # 8 月无 seed 后缀批次
    aug = [f for f in os.listdir(args.src)
           if f.endswith(".pth") and "_seed" not in f and os.path.isfile(os.path.join(args.src, f))]
    if aug:
        log("  %-30s %s" % ("<%d 个无 _seedN 后缀>" % len(aug), "8 月单跑早期产物(table3_final_metrics 用), 不发布"))

    if args.manifest_only:
        log("\n(--manifest-only) 跳过打包。")
    else:
        os.makedirs(args.out, exist_ok=True)
        assets = []
        if core_rows:
            zp = os.path.join(args.out, "weights_proposed_and_ablation.zip")
            mb, sha = pack(core_rows, zp, "weights_proposed_and_ablation")
            log("\n打包 %s: %d 个, %.1f MB\n  SHA-256: %s" % (os.path.basename(zp), len(core_rows), mb, sha))
            assets.append((os.path.basename(zp), len(core_rows), mb, sha))
        if base_rows:
            zp = os.path.join(args.out, "weights_baselines.zip")
            mb, sha = pack(base_rows, zp, "weights_baselines")
            log("打包 %s: %d 个, %.1f MB\n  SHA-256: %s" % (os.path.basename(zp), len(base_rows), mb, sha))
            assets.append((os.path.basename(zp), len(base_rows), mb, sha))
        if assets:
            log("\n<- 把上面的 SHA-256 填入 data_availability/004_RELEASE_MANIFEST.md")

    # manifest
    models_dir = args.models_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    models_dir = os.path.abspath(models_dir)
    if os.path.isdir(models_dir):
        mpath = os.path.join(models_dir, "weights_manifest.csv")
        with open(mpath, "w", encoding="utf-8") as fh:
            fh.write("filename,model_key,seed,size_bytes,sha256,manuscript_item\n")
            for p, _arc, key, seed, size, _sha, note in all_rows:
                fh.write("%s,%s,%d,%d,%s,%s\n" % (
                    os.path.basename(p), key, seed, size, sha256_of(p), note.replace(",", ";")))
        log("\nmanifest 已写入 %s (%d 行)" % (mpath, len(all_rows)))

    with open("weights_build_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
