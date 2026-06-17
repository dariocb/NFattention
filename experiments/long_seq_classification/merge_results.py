#!/usr/bin/env python
"""
合并两张卡 (gpu0 / gpu4) 的长文本实验结果为一张总表。

读取 results/gpu0 和 results/gpu4 下最新 run_<ts>/ 的逐实验结果，
拼接后输出到 results/combined/：
    all_results_combined.csv   —— 全部行（dataset × model × seed）
    pivot_accuracy.csv         —— dataset(行) × model(列) 的 test accuracy 透视表
并在终端打印透视表。

用法:
    python merge_results.py
"""
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
# 模型展示顺序（ours 在前，baseline 在后）
MODEL_ORDER = [
    "ours_hybrid_noprior", "ours_hybrid_ngsm", "ours_hybrid_rbf",
    "ours_hybrid_noprior_deep", "ours_hybrid_ngsm_deep", "ours_hybrid_rbf_deep",
    "transformer", "mikan", "performer", "rka", "gmm_rks", "kpca_scaled", "metala",
]
DATASET_ORDER = ["hyperpartisan", "imdb"]


def _all_run_csvs(card_dir: Path):
    """收集某张卡 **所有** run_* 目录的结果 CSV。

    一张卡可能跨多个 run（例如 metala 单独 capped 跑、中途重启），
    因此读取全部 run，靠 (dataset,model,seed,fold) 去重即可。
    每个 run 优先用最终 all_results.csv，否则用累积的 results_<dataset>.csv。
    """
    runs = sorted(card_dir.glob("run_*"))
    csvs = []
    for run in runs:
        allcsv = run / "all_results.csv"
        if allcsv.is_file():
            csvs.append(allcsv)
        else:
            csvs.extend(sorted(run.glob("results_*.csv")))
    return runs, csvs


def _read_rows(csv_paths):
    rows = []
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def main():
    all_rows = []
    # 主结果目录：gpu0/gpu4 (baseline + 默认早停 ours) + deep_* (5层NF 消融)。
    # 诊断目录 (diag_*, reg_*) 单独处理，不混入自动扫描。
    DIAG_PREFIXES = ("diag_", "reg_", "fair_")
    cards = sorted(
        d.name for d in RESULTS.iterdir()
        if d.is_dir() and d.name != "combined" and any(d.glob("run_*"))
        and not d.name.startswith(DIAG_PREFIXES)
    )
    print(f"主结果目录: {cards}")
    for card in cards:
        card_dir = RESULTS / card
        runs, csvs = _all_run_csvs(card_dir)
        rows = _read_rows(csvs)
        for r in rows:
            r["_card"] = card
        all_rows.extend(rows)
        print(f"  [{card}] {len(runs)} runs, {len(rows)} 行")

    # ── 修正: Hyperpartisan 上全部模型用宽松早停 (公平对比) 的结果 ──
    #   默认早停在 64 条验证集上误触发, 几乎所有模型都被低估。详见 REPORT.md §3。
    #   diag_A_longtrain: ours 3 个 (patience=200)
    #   fair_g0/fair_g3:  6 个 baseline (patience=200)
    #   fair_metala:      metala (patience=30, 递归太慢但已收敛)
    for fair_dir in ("diag_A_longtrain", "fair_g0", "fair_g3", "fair_metala"):
        d = RESULTS / fair_dir
        if d.is_dir():
            _, csvs = _all_run_csvs(d)
            fixed = _read_rows(csvs)
            for r in fixed:
                r["_card"] = f"{fair_dir}(fair_earlystop)"
            all_rows.extend(fixed)
            print(f"  [{fair_dir}] {len(fixed)} 行 (公平早停, 覆盖 @hyperpartisan)")

    # 过滤掉 CSV 拼接时混入的表头噪声行 (model 字段恰为 'model')
    all_rows = [r for r in all_rows if r.get("model") and r.get("model") != "model"]

    # 去重 (同 dataset+model+seed+fold 以最后一条为准; diag_A 在后, 故覆盖旧 ours@hyperpartisan)
    dedup = {}
    for r in all_rows:
        key = (r.get("dataset"), r.get("model"), r.get("seed"), r.get("fold", ""))
        dedup[key] = r
    rows = list(dedup.values())

    out_dir = RESULTS / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 合并明细 CSV
    if rows:
        fields = [k for k in rows[0].keys() if k != "_card"]
        fields = ["_card"] + fields
        with open(out_dir / "all_results_combined.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # 2) 透视表 dataset × model (test accuracy %)
    acc = defaultdict(dict)
    for r in rows:
        try:
            acc[r["dataset"]][r["model"]] = float(r["accuracy"]) * 100
        except (KeyError, ValueError, TypeError):
            pass

    models = [m for m in MODEL_ORDER if any(m in acc[d] for d in acc)]
    datasets = [d for d in DATASET_ORDER if d in acc] + [d for d in acc if d not in DATASET_ORDER]

    with open(out_dir / "pivot_accuracy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset"] + models)
        for d in datasets:
            w.writerow([d] + [f"{acc[d].get(m, float('nan')):.2f}" if m in acc[d] else "" for m in models])

    # 终端打印
    print("\n" + "=" * (16 + 13 * len(models)))
    print("Test Accuracy (%)  —  dataset × model")
    print("=" * (16 + 13 * len(models)))
    print(f"{'dataset':<16}" + "".join(f"{m[:12]:>13}" for m in models))
    for d in datasets:
        print(f"{d:<16}" + "".join(
            (f"{acc[d][m]:>13.2f}" if m in acc[d] else f"{'-':>13}") for m in models))
    print("=" * (16 + 13 * len(models)))
    print(f"\n合并明细: {out_dir/'all_results_combined.csv'}")
    print(f"透视表:   {out_dir/'pivot_accuracy.csv'}")


if __name__ == "__main__":
    main()
