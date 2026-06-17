"""Run the WikiText perplexity benchmark across models and seeds."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from .train import train_one_run
    from .utils import ResultLogger, load_config, save_yaml
except ImportError:  # pragma: no cover - script execution fallback
    from train import train_one_run  # type: ignore
    from utils import ResultLogger, load_config, save_yaml  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Run WikiText perplexity benchmark")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config", "default_config.yaml"))
    parser.add_argument("--models", nargs="+", default=["transformer", "mikan", "performer", "rka", "gmm_rks", "ours_shared_ngsm", "ours_hybrid_noprior", "ours_hybrid_ngsm", "ours_hybrid_rbf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--no_cpu_fallback", action="store_true", help="Fail if CUDA is unavailable instead of falling back to CPU")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__),
        "benchmark_results",
        datetime.now().strftime("run_%Y%m%d_%H%M%S"),
    )
    os.makedirs(output_dir, exist_ok=True)
    save_yaml(os.path.join(output_dir, "config.yaml"), config)
    logger = ResultLogger(output_dir)

    for model_name in args.models:
        for seed in args.seeds:
            print(f"running model={model_name} seed={seed}")
            result = train_one_run(
                config,
                model_name,
                seed,
                output_dir,
                args.gpu,
                allow_cpu_fallback=not args.no_cpu_fallback,
                max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
                max_test_batches=args.max_test_batches,
            )
            logger.add_result(result)
            logger.save_csv("results.csv")
            logger.save_results("results.json")
            logger.save_summary_csv("summary.csv")

    print(f"results written to {output_dir}")


if __name__ == "__main__":
    main()
