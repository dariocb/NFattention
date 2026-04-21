#!/usr/bin/env python3
import argparse
import csv
import glob
import os


def latest_run_dir(base_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(base_dir, "run_*")))
    if not candidates:
        raise FileNotFoundError(f"No run_* found in {base_dir}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        type=str,
        default="benchmark_results_ours_latest",
        help="Root directory containing run_* outputs.",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Specific run directory. If omitted, use latest run_*.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir(args.results_root)
    summary_csv = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")

    rows = []
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model") == "ours_latest":
                rows.append(row)

    if not rows:
        print("No ours_latest rows found in summary.csv")
        return

    print(f"Run dir: {run_dir}")
    print("-" * 84)
    print(f"{'Dataset':<20} {'Acc(mean±std)':<20} {'F1(mean±std)':<20} {'Time(s)':<12} {'Params':<12}")
    print("-" * 84)
    for r in rows:
        acc = f"{float(r['accuracy_mean'])*100:.2f}±{float(r['accuracy_std'])*100:.2f}"
        f1 = f"{float(r['f1_macro_mean'])*100:.2f}±{float(r['f1_macro_std'])*100:.2f}"
        t = f"{float(r['train_time_mean']):.1f}"
        p = r["num_params"]
        print(f"{r['dataset']:<20} {acc:<20} {f1:<20} {t:<12} {p:<12}")


if __name__ == "__main__":
    main()

