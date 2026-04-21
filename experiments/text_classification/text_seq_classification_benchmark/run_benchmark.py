#!/usr/bin/env python
"""
Benchmark 运行脚本

运行所有数据集和模型的组合实验，生成论文级别的结果表。
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from itertools import product

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load_dataset_by_name, create_cv_splits, get_available_datasets
from train import run_single_experiment, run_cv_experiment, ProgressReporter
from utils import (
    set_seed, load_config, save_config, get_device,
    ResultLogger, setup_logging, format_time, write_repro_manifest,
)


# 数据集配置
DATASET_CONFIG = {
    "rt_polarity": {
        "use_cv": False,
        "n_folds": 10,
        "description": "RT Polarity (二分类)"
    },
    "sst2": {
        "use_cv": False,
        "description": "SST-2 (二分类)"
    },
    "sst5": {
        "use_cv": False,
        "description": "SST-5 (五分类)"
    },
    "trec": {
        "use_cv": False,
        "description": "TREC (六分类)"
    },
    "ag_news": {
        "use_cv": False,
        "description": "AG News (四分类)"
    },
    "dbpedia_14": {
        "use_cv": False,
        "description": "DBpedia (14分类)"
    },
    "yelp_review_full": {
        "use_cv": False,
        "description": "Yelp Full (五分类)"
    },
    "quora": {
        "use_cv": False,
        "description": "Quora QQP (二分类)"
    }
}

# 模型列表
MODELS = [
    "ours_latest"
]


def run_full_benchmark(
    datasets: List[str],
    models: List[str],
    seeds: List[int],
    config: Dict[str, Any],
    data_dir: str,
    output_dir: str,
    device,
    logger=None,
    enable_progress: bool = True,
    checkpoint_dir: Optional[str] = None,
):
    """
    运行完整 benchmark
    
    Args:
        datasets: 数据集列表
        models: 模型列表
        seeds: 随机种子列表
        config: 配置
        data_dir: 数据目录
        output_dir: 输出目录
        device: 计算设备
        logger: 日志器
        checkpoint_dir: 若设置，则每轮实验后将权重与词表等写入该目录
    """
    result_logger = ResultLogger(output_dir)
    
    total_experiments = 0
    for dataset_name in datasets:
        dataset_config = DATASET_CONFIG.get(dataset_name, {"use_cv": False})
        use_cv = dataset_config.get("use_cv", False)
        if use_cv:
            total_experiments += len(models) * len(seeds) * dataset_config.get("n_folds", 10)
        else:
            total_experiments += len(models) * len(seeds)
    current_exp = 0
    progress_reporter = ProgressReporter(total_experiments=total_experiments) if enable_progress else None
    
    start_time = time.time()
    
    for dataset_name in datasets:
        dataset_config = DATASET_CONFIG.get(dataset_name, {"use_cv": False})
        use_cv = dataset_config.get("use_cv", False)
        
        if logger:
            logger.info(f"\n{'='*60}")
            logger.info(f"数据集: {dataset_name} ({dataset_config.get('description', '')})")
            logger.info(f"{'='*60}")
        
        for model_name in models:
            for seed in seeds:
                current_exp += 1
                
                if logger:
                    logger.info(f"\n[{current_exp}/{total_experiments}] "
                               f"Dataset: {dataset_name}, Model: {model_name}, Seed: {seed}")
                
                try:
                    if use_cv:
                        results = run_cv_experiment(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            config=config,
                            seed=seed,
                            data_dir=data_dir,
                            device=device,
                            n_folds=dataset_config.get("n_folds", 10),
                            logger=logger,
                            progress_reporter=progress_reporter,
                            checkpoint_dir=checkpoint_dir,
                        )
                        for r in results:
                            result_logger.add_result(r)
                    else:
                        result = run_single_experiment(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            config=config,
                            seed=seed,
                            data_dir=data_dir,
                            device=device,
                            logger=logger,
                            progress_reporter=progress_reporter,
                            checkpoint_dir=checkpoint_dir,
                        )
                        result_logger.add_result(result)
                        
                except Exception as e:
                    if logger:
                        logger.error(f"实验失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # 每个数据集完成后保存中间结果
        result_logger.save_csv(f"results_{dataset_name}.csv")
    
    total_time = time.time() - start_time
    
    # 保存所有结果
    result_logger.save_csv("all_results.csv")
    result_logger.save_results("all_results.json")
    result_logger.save_summary_csv("summary.csv")
    
    # 打印汇总
    if logger:
        logger.info(f"\n总运行时间: {format_time(total_time)}")
    
    result_logger.print_summary()
    
    # 生成 LaTeX 表格
    generate_latex_table(result_logger, output_dir)
    
    return result_logger


def generate_latex_table(result_logger: ResultLogger, output_dir: str):
    """生成 LaTeX 格式的结果表"""
    summary = result_logger.generate_summary()
    
    if not summary:
        return
    
    # 按数据集分组
    datasets = sorted(set(s['dataset'] for s in summary.values()))
    models = sorted(set(s['model'] for s in summary.values()))
    
    # 生成 LaTeX 表格
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Text Sequence Classification Results}")
    latex_lines.append("\\label{tab:seq_classification}")
    latex_lines.append("\\begin{tabular}{l" + "c" * len(models) + "}")
    latex_lines.append("\\toprule")
    
    # 表头
    header = "Dataset & " + " & ".join(m.replace("_", "\\_") for m in models) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    
    # 数据行
    for dataset in datasets:
        row = dataset.replace("_", "\\_")
        best_acc = 0
        
        # 找最佳准确率
        for model in models:
            key = f"{dataset}_{model}"
            if key in summary:
                acc = summary[key]['accuracy_mean']
                if acc > best_acc:
                    best_acc = acc
        
        # 生成每个单元格
        for model in models:
            key = f"{dataset}_{model}"
            if key in summary:
                acc = summary[key]['accuracy_mean'] * 100
                std = summary[key]['accuracy_std'] * 100
                
                # 加粗最佳结果
                if abs(summary[key]['accuracy_mean'] - best_acc) < 0.001:
                    row += f" & \\textbf{{{acc:.1f}$\\pm${std:.1f}}}"
                else:
                    row += f" & {acc:.1f}$\\pm${std:.1f}"
            else:
                row += " & -"
        
        row += " \\\\"
        latex_lines.append(row)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    # 保存
    latex_path = os.path.join(output_dir, "results_table.tex")
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_lines))
    
    print(f"\nLaTeX 表格已保存到: {latex_path}")


def main():
    parser = argparse.ArgumentParser(description='Run Text Classification Benchmark')
    
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='要运行的数据集（默认全部）')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='要运行的模型（默认全部）')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                       help='随机种子列表')
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='数据目录')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                       help='输出目录')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（只运行小数据集）')
    parser.add_argument('--no_progress', action='store_true',
                       help='关闭实时 epoch 进度条')
    parser.add_argument('--save_checkpoints', action='store_true',
                       help='将每轮最佳模型与词表等保存到 output_dir/checkpoints/（便于复现）')
    
    args = parser.parse_args()
    
    # 数据集和模型
    datasets = args.datasets or list(DATASET_CONFIG.keys())
    models = args.models or MODELS
    
    if args.quick:
        # 快速模式：只运行 RT Polarity 和 SST-2
        datasets = ["rt_polarity", "sst2"]
        args.seeds = [0]
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = {
            'model_params': {
                'embed_dim': 128,
                'hidden_dim': 128,
                'n_heads': 4,
                'n_layers': 2,
                'pf_dim': 256,
                'dropout': 0.1,
                'max_seq_len': 256,
                'M': 64,
                'kl_lambda': 0.001,
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'n_epochs': 50,
                'patience': 10,
                'scheduler': 'cosine',
                'gradient_clip': 1.0,
            }
        }
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    save_config(config, os.path.join(output_dir, "config.yaml"))

    checkpoint_dir = os.path.join(output_dir, "checkpoints") if args.save_checkpoints else None
    write_repro_manifest(
        output_dir=output_dir,
        config=config,
        datasets=datasets,
        models=models,
        seeds=list(args.seeds),
        data_dir=args.data_dir,
        argv=sys.argv,
        checkpoint_dir=checkpoint_dir,
        benchmark_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    
    # 设置设备
    device = get_device(args.gpu)
    
    # 设置日志
    logger = setup_logging(output_dir, "benchmark")
    
    logger.info("=" * 60)
    logger.info("Text Sequence Classification Benchmark")
    logger.info("=" * 60)
    logger.info(f"数据集: {datasets}")
    logger.info(f"模型: {models}")
    logger.info(f"随机种子: {args.seeds}")
    logger.info(f"设备: {device}")
    logger.info(f"输出目录: {output_dir}")
    if checkpoint_dir:
        logger.info(f"Checkpoint 目录: {checkpoint_dir}")
    logger.info("=" * 60)
    
    # 运行 benchmark
    run_full_benchmark(
        datasets=datasets,
        models=models,
        seeds=args.seeds,
        config=config,
        data_dir=args.data_dir,
        output_dir=output_dir,
        device=device,
        logger=logger,
        enable_progress=(not args.no_progress),
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == '__main__':
    main()

