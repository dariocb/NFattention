import argparse
import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from mnist_classification_ablation import (
    load_mnist_data,
    run_experiment_with_data,
)


def build_args(cli_args):
    args = SimpleNamespace()
    args.random_seed = cli_args.seed
    args.N_EPOCHS = cli_args.epochs
    args.LEARNING_RATE = cli_args.lr
    args.HID_DIM = 64
    args.KEY_DIM = 64
    args.ENC_LAYERS = 1
    args.ENC_HEADS = 8
    args.ENC_DROPOUT = 0.0
    args.M = 64
    args.M_frska = 64
    args.prior_var = 0.1
    args.kl_lambda = 0.1
    args.copula_lambda = 1.0
    args.p_norm = 2.0
    args.att_dropout = 0.1
    args.device = cli_args.device
    args.use_full_data = bool(getattr(cli_args, "use_full_data", False))
    args.n_samples = cli_args.n_samples
    args.eval_batch_size = cli_args.eval_batch_size
    args.frska_positive_map = True
    args.dot_qk_mode = "normal"
    return args


def focused_experiments():
    # (name, att_type, freeze_qk, freeze_v, shared_flow, qk_mode, v_mode, overrides)
    return [
        ("Transformer", "dot", False, False, False, "normal", "learnable", {}),
        ("Transformer + NoQK + FixedV", "dot", True, True, False, "no_qk", "learnable", {"dot_qk_mode": "no_qk"}),
        ("IKAN-direct", "ikandirect", False, False, False, "normal", "learnable", {}),
        ("MIKAN", "mikan", False, False, False, "normal", "learnable", {}),
        ("MGK", "mgk", False, False, False, "normal", "learnable", {}),
        (
            "FRSKA-Strict-Base",
            "frska",
            False,
            False,
            True,
            "no_qk",
            "fixed_orth",
            {"kl_lambda_frska": 0.005, "M_frska": 64},
        ),
        (
            "FRSKA-Strict + NormalQK",
            "frska",
            False,
            False,
            True,
            "normal",
            "fixed_orth",
            {"kl_lambda_frska": 0.005, "M_frska": 64, "frska_positive_map": True},
        ),
        (
            "FRSKA-Strict + NormalQK + DirectRFF",
            "frska",
            False,
            False,
            True,
            "normal",
            "fixed_orth",
            {"kl_lambda_frska": 0.005, "M_frska": 64, "frska_positive_map": False},
        ),
        (
            "FRSKA-Strict + LearnableV",
            "frska",
            False,
            False,
            True,
            "no_qk",
            "learnable",
            {"kl_lambda_frska": 0.005, "M_frska": 64},
        ),
        (
            "FRSKA-Strict + PerHeadNF",
            "frska",
            False,
            False,
            False,
            "no_qk",
            "fixed_orth",
            {"kl_lambda_frska": 0.005, "M_frska": 64},
        ),
        (
            "FRSKA-Strict + KL off",
            "frska",
            False,
            False,
            True,
            "no_qk",
            "fixed_orth",
            {"kl_lambda_frska": 0.0, "M_frska": 64},
        ),
        (
            "FRSKA-Strict + KL on",
            "frska",
            False,
            False,
            True,
            "no_qk",
            "fixed_orth",
            {"kl_lambda_frska": 0.005, "M_frska": 64},
        ),
        (
            "FRSKA-Strict + M32",
            "frska",
            False,
            False,
            True,
            "no_qk",
            "fixed_orth",
            {"kl_lambda_frska": 0.005, "M_frska": 32},
        ),
        (
            "FRSKA-Strict + M64",
            "frska",
            False,
            False,
            True,
            "no_qk",
            "fixed_orth",
            {"kl_lambda_frska": 0.005, "M_frska": 64},
        ),
        (
            "FRSKA-Strict + M96",
            "frska",
            False,
            False,
            True,
            "no_qk",
            "fixed_orth",
            {"kl_lambda_frska": 0.005, "M_frska": 96},
        ),
    ]


def write_report(path, seed, settings, results):
    lines = []
    lines.append("=" * 88)
    lines.append("Focused MNIST Ablation Report")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Run setup:")
    lines.append(f"  seed: {seed}")
    if settings.get("use_full_data", False):
        lines.append("  dataset: full MNIST (60000 train + 10000 test)")
    else:
        lines.append(f"  n_samples: {settings['n_samples']} (downsample total)")
    lines.append(f"  epochs: {settings['epochs']}")
    lines.append(f"  learning_rate: {settings['lr']}")
    lines.append(f"  device: {settings['device']}")
    lines.append(f"  eval_batch_size: {settings.get('eval_batch_size', 0)}")
    lines.append(f"  frska_positive_map_default: {settings['frska_positive_map_default']}")
    lines.append("")
    lines.append("-" * 88)
    lines.append(f"{'Model':<36} {'Acc':<10} {'Runtime(s)':<12} {'Params':<12}")
    lines.append("-" * 88)
    for r in results:
        lines.append(
            f"{r['name']:<36} {r['best_test_acc']:<10.4f} {r['runtime']:<12.2f} {r['num_params']:<12,}"
        )
    lines.append("")
    lines.append("=" * 88)
    lines.append("Notes")
    lines.append("=" * 88)
    lines.append("- Baselines are included without extra ablation branches.")
    lines.append("- Detailed predictions are saved for this single-seed run.")
    lines.append("- Update: linear-attention denominator stabilization enabled (ELU+1 positive feature map).")
    lines.append("- Purpose of update: fix abnormal collapse in 'FRSKA-Strict + NormalQK'.")
    name_to_res = {r["name"]: r for r in results}
    stable_name = "FRSKA-Strict + NormalQK"
    direct_name = "FRSKA-Strict + NormalQK + DirectRFF"
    if stable_name in name_to_res and direct_name in name_to_res:
        stable_acc = float(name_to_res[stable_name]["best_test_acc"])
        direct_acc = float(name_to_res[direct_name]["best_test_acc"])
        lines.append("- DirectRFF ablation added:")
        lines.append(f"  {stable_name}: {stable_acc:.4f} (ELU+1 stabilized)")
        lines.append(f"  {direct_name}: {direct_acc:.4f} (raw RFF)")
        lines.append(f"  Accuracy gap (stabilized - raw): {stable_acc - direct_acc:+.4f}")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _to_serializable_result(r):
    def _normalize(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, dict):
            return {kk: _normalize(vv) for kk, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [_normalize(x) for x in v]
        return v

    return {k: _normalize(v) for k, v in r.items()}


def merge_results_with_existing(out_dir, new_results):
    existing_path = os.path.join(out_dir, "results_seed.json")
    if not os.path.exists(existing_path):
        return new_results

    try:
        with open(existing_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        existing_results = payload.get("results", [])
    except Exception:
        existing_results = []

    merged_by_name = {r["name"]: r for r in existing_results if isinstance(r, dict) and "name" in r}
    for r in new_results:
        merged_by_name[r["name"]] = _to_serializable_result(r)

    base_order = [e[0] for e in focused_experiments()]
    merged = []
    for name in base_order:
        if name in merged_by_name:
            merged.append(merged_by_name[name])

    for name, item in merged_by_name.items():
        if name not in base_order:
            merged.append(item)

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--use_full_data", action="store_true", default=False)
    parser.add_argument("--no_full_data", dest="use_full_data", action="store_false")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="mnist_focused_results")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Evaluation batch size for test inference. 1 means per-sample; <=0 means full-batch.",
    )
    parser.add_argument(
        "--only_experiment",
        type=str,
        default=None,
        help="Run only one experiment by exact name in focused_experiments().",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    base_cli_ns = SimpleNamespace(**vars(args), device=str(device))
    run_args = build_args(base_cli_ns)
    tr_data, te_data, num_data, input_dim, output_dim, tr_idx, test_idx, tr_labels, te_labels = load_mnist_data(
        use_full_data=args.use_full_data,
        n_samples=args.n_samples,
        random_seed=args.seed,
        device=device,
    )

    experiments = focused_experiments()
    if args.only_experiment is not None:
        experiments = [e for e in experiments if e[0] == args.only_experiment]
        if not experiments:
            raise ValueError(
                f"Unknown experiment name: {args.only_experiment}. "
                "Please use an exact name from focused_experiments()."
            )
    results = []
    start_all = time.time()
    print(f"Running {len(experiments)} focused experiments on seed={args.seed}")
    for i, (name, att_type, freeze_qk, freeze_v, shared_flow, qk_mode, v_mode, overrides) in enumerate(experiments, 1):
        print(f"[{i}/{len(experiments)}] {name}")
        # 每个实验都从同一默认配置重建，避免上一个实验override串到下一个
        run_args = build_args(base_cli_ns)
        for k, v in overrides.items():
            setattr(run_args, k, v)
        res = run_experiment_with_data(
            name,
            att_type,
            run_args,
            tr_data,
            te_data,
            num_data,
            input_dim,
            output_dim,
            test_idx,
            device,
            freeze_qk=freeze_qk,
            freeze_v=freeze_v,
            shared_flow=shared_flow,
            qk_mode=qk_mode,
            v_mode=v_mode,
        )
        results.append(res)

    elapsed = time.time() - start_all
    settings = {
        "seed": args.seed,
        "n_samples": args.n_samples,
        "use_full_data": args.use_full_data,
        "epochs": args.epochs,
        "lr": args.lr,
        "device": str(device),
        "eval_batch_size": args.eval_batch_size,
        "frska_positive_map_default": getattr(run_args, "frska_positive_map", True),
        "elapsed_s": elapsed,
    }

    serializable_results = [_to_serializable_result(r) for r in results]
    merged_serializable_results = merge_results_with_existing(out_dir, results)
    merged_report_results = []
    for r in merged_serializable_results:
        item = dict(r)
        if isinstance(item.get("predictions"), list):
            item["predictions"] = np.array(item["predictions"])
        merged_report_results.append(item)

    with open(os.path.join(out_dir, "results_seed.json"), "w", encoding="utf-8") as f:
        json.dump({"settings": settings, "results": merged_serializable_results}, f, ensure_ascii=False, indent=2)

    write_report(
        os.path.join(out_dir, "ablation_report.txt"),
        seed=args.seed,
        settings=settings,
        results=merged_report_results,
    )
    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
