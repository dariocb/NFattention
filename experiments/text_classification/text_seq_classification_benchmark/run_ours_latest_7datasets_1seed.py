#!/usr/bin/env python3
"""
Run only ours_latest on 7 datasets with a single seed.
Overrides:
- M = 96
- num_flows = 5
- flow_hidden_dim = 128
"""

import os
import argparse
from datetime import datetime

from run_benchmark import run_full_benchmark
from utils import load_config, save_config, get_device, setup_logging


DATASETS_7 = [
    "rt_polarity",
    "sst2",
    "sst5",
    "trec",
    "ag_news",
    "dbpedia_14",
    "yelp_review_full",
]


def main():
    parser = argparse.ArgumentParser(description="Run ours_latest on 7 datasets with one seed")
    parser.add_argument("--seed", type=int, default=0, help="Single random seed")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Config path")
    parser.add_argument("--data_dir", type=str, default="../data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="benchmark_results_ours_latest_7d_1seed", help="Output root directory")
    parser.add_argument("--no_progress", action="store_true", help="Disable live epoch progress")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = load_config(config_path)

    # Requested latest settings
    config["model_params"]["M"] = 96
    config["model_params"]["num_flows"] = 5
    config["model_params"]["flow_hidden_dim"] = 128
    config["model_params"]["num_mixtures"] = config["model_params"].get("num_mixtures", 10)

    seeds = [args.seed]
    models = ["ours_latest"]
    datasets = DATASETS_7

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    save_config(config, os.path.join(out_dir, "config.yaml"))

    device = get_device(args.gpu)
    logger = setup_logging(out_dir, "benchmark_ours_latest_7d_1seed")
    logger.info("=" * 60)
    logger.info("Ours Latest: 7 Datasets + 1 Seed")
    logger.info("=" * 60)
    logger.info(f"datasets: {datasets}")
    logger.info(f"models: {models}")
    logger.info(f"seed: {seeds}")
    logger.info(f"device: {device}")
    logger.info("override model_params: M=96, num_flows=5, flow_hidden_dim=128")
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

