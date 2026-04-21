#!/usr/bin/env python3
"""
Run all in-repo text models on Quora QQP only.
"""

import os
import argparse
from datetime import datetime

from run_benchmark import run_full_benchmark
from utils import load_config, save_config, get_device, setup_logging


ALL_MODELS = [
    "transformer",
    "mikan",
    "mgk",
    "performer",
    "rka",
    "gmm_rks",
    "kpca_scaled",
    "metala",
    "ours_latest",
    "ours_fixed_qk",
    "ours_trainable_qk",
]


def main():
    parser = argparse.ArgumentParser(description="Run all models on Quora QQP")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Random seeds")
    parser.add_argument("--gpu", type=int, default=None, help="GPU id")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Config path")
    parser.add_argument("--data_dir", type=str, default="../data", help="Data directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results_quora_all_models",
        help="Output root directory",
    )
    parser.add_argument("--no_progress", action="store_true", help="Disable live epoch progress")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = load_config(config_path)

    datasets = ["quora"]
    models = ALL_MODELS
    seeds = args.seeds

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    save_config(config, os.path.join(out_dir, "config.yaml"))

    device = get_device(args.gpu)
    logger = setup_logging(out_dir, "benchmark_quora_all_models")
    logger.info("=" * 60)
    logger.info("Quora QQP: All Models Benchmark")
    logger.info("=" * 60)
    logger.info(f"dataset: {datasets}")
    logger.info(f"models: {models}")
    logger.info(f"seeds: {seeds}")
    logger.info(f"device: {device}")
    logger.info(f"output_dir: {out_dir}")
    logger.info("=" * 60)

    run_full_benchmark(
        datasets=datasets,
        models=models,
        seeds=seeds,
        config=config,
        data_dir=args.data_dir,
        output_dir=out_dir,
        device=device,
        logger=logger,
        enable_progress=(not args.no_progress),
    )


if __name__ == "__main__":
    main()
