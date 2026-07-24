#!/usr/bin/env python
"""
长文本分类实验 —— 数据准备脚本（幂等，可重复运行）

把两个长文本数据集落到 FSKA 共享数据目录
    experiments/text_classification/data/
下，复用既有的 dataset_loader 加载逻辑：

  1. hyperpartisan : 从 RFGPA 项目已下载的本地 jsonl 复制
        RFGPA/code/experiments/hyperpartisan_long_text/hyperpartisan_local/
            {train,dev,test}.jsonl
     →  <DATA>/hyperpartisan/{train,dev,test}.jsonl
     （loader: load_hyperpartisan）

  2. imdb : 下载 Stanford aclImdb_v1.tar.gz (~84MB) 并解析成
        <DATA>/hf_datasets/imdb/data.json
     格式 {"train":{"texts","labels"}, "test":{"texts","labels"}}
     （loader: 复用 load_local_hf_dataset，无需 `datasets` 库）

用法:
    python prepare_data.py
    python prepare_data.py --force        # 重新生成
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FSKA_ROOT = HERE.parent.parent                      # .../FSKA
DATA_DIR = FSKA_ROOT / "experiments" / "text_classification" / "data"
RFGPA_HYP = (
    FSKA_ROOT.parent                                # .../yangzi
    / "RFGPA" / "code" / "experiments"
    / "hyperpartisan_long_text" / "hyperpartisan_local"
)

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


# ────────────────────────────────────────────────────────────────────
# Hyperpartisan: copy local jsonl from RFGPA
# ────────────────────────────────────────────────────────────────────
def prepare_hyperpartisan(force: bool = False):
    dst = DATA_DIR / "hyperpartisan"
    dst.mkdir(parents=True, exist_ok=True)
    files = {"train.jsonl": "train.jsonl",
             "dev.jsonl": "dev.jsonl",
             "test.jsonl": "test.jsonl"}

    print("\n[hyperpartisan]")
    if not RFGPA_HYP.is_dir():
        raise FileNotFoundError(
            f"找不到 RFGPA 的 hyperpartisan_local 目录: {RFGPA_HYP}\n"
            f"请确认 RFGPA 项目存在并已下载该数据。"
        )

    for src_name, dst_name in files.items():
        src = RFGPA_HYP / src_name
        out = dst / dst_name
        if out.exists() and not force:
            n = sum(1 for _ in open(out, encoding="utf-8"))
            print(f"  ✓ {dst_name} 已存在 ({n} 行)，跳过")
            continue
        if not src.is_file():
            raise FileNotFoundError(f"缺少源文件: {src}")
        shutil.copyfile(src, out)
        n = sum(1 for _ in open(out, encoding="utf-8"))
        print(f"  ✓ 复制 {src_name} → {out}  ({n} 行)")

    print(f"  目标目录: {dst}")


# ────────────────────────────────────────────────────────────────────
# IMDB: download Stanford tarball → data.json
# ────────────────────────────────────────────────────────────────────
def _read_imdb_split(root: Path, split: str):
    """读取 aclImdb/<split>/{pos,neg}/*.txt → (texts, labels)。pos=1, neg=0。"""
    texts, labels = [], []
    for label_name, label in (("neg", 0), ("pos", 1)):
        d = root / split / label_name
        files = sorted(d.glob("*.txt"))
        for fp in files:
            texts.append(fp.read_text(encoding="utf-8"))
            labels.append(label)
    return texts, labels


def prepare_imdb(force: bool = False):
    out_dir = DATA_DIR / "hf_datasets" / "imdb"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_json = out_dir / "data.json"
    meta_json = out_dir / "meta.json"

    print("\n[imdb]")
    if data_json.exists() and not force:
        d = json.loads(data_json.read_text(encoding="utf-8"))
        print(f"  ✓ data.json 已存在 (train={len(d['train']['texts'])}, "
              f"test={len(d['test']['texts'])})，跳过")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tar_path = tmp / "aclImdb_v1.tar.gz"
        print(f"  下载 {IMDB_URL} (~84MB) ...")
        urllib.request.urlretrieve(IMDB_URL, tar_path)
        print(f"  解压 {tar_path.name} ...")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tmp)
        root = tmp / "aclImdb"

        train_texts, train_labels = _read_imdb_split(root, "train")
        test_texts, test_labels = _read_imdb_split(root, "test")

    data = {
        "train": {"texts": train_texts, "labels": train_labels},
        "test": {"texts": test_texts, "labels": test_labels},
    }
    data_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    meta_json.write_text(json.dumps({
        "name": "imdb",
        "num_classes": 2,
        "class_names": ["neg", "pos"],
        "source": IMDB_URL,
        "train_size": len(train_texts),
        "test_size": len(test_texts),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ 写入 {data_json}  (train={len(train_texts)}, test={len(test_texts)})")


def main():
    ap = argparse.ArgumentParser(description="准备长文本分类实验数据 (hyperpartisan + imdb)")
    ap.add_argument("--force", action="store_true", help="重新生成/覆盖已有数据")
    ap.add_argument("--only", choices=["hyperpartisan", "imdb"], default=None,
                    help="只准备其中一个数据集")
    args = ap.parse_args()

    print(f"FSKA 数据目录: {DATA_DIR}")
    if args.only in (None, "hyperpartisan"):
        prepare_hyperpartisan(force=args.force)
    if args.only in (None, "imdb"):
        prepare_imdb(force=args.force)
    print("\n完成。可用: load_dataset_by_name('hyperpartisan'/'imdb', <data_dir>)")


if __name__ == "__main__":
    main()
