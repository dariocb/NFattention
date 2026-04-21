"""
工具函数
"""

import os
import sys
import random
import time
import yaml
import json
import shutil
import logging
import subprocess
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """保存配置文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """获取计算设备"""
    if gpu_id is not None and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


@dataclass
class ExperimentResult:
    """实验结果数据类"""
    dataset: str
    model: str
    seed: int
    fold: Optional[int]  # 交叉验证时的折数
    accuracy: float
    f1_macro: Optional[float]
    train_loss: float
    val_loss: Optional[float]
    test_loss: float
    train_time: float  # 秒
    num_params: int
    best_epoch: int
    config: Dict[str, Any]
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResultLogger:
    """结果记录器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results: List[ExperimentResult] = []
        
    def add_result(self, result: ExperimentResult):
        """添加一条结果"""
        self.results.append(result)
        
    def save_results(self, filename: str = "results.json"):
        """保存所有结果到 JSON 文件"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2, ensure_ascii=False)
        return filepath
    
    def save_csv(self, filename: str = "results.csv"):
        """保存结果到 CSV 文件"""
        import csv
        
        filepath = os.path.join(self.output_dir, filename)
        
        if not self.results:
            return filepath
        
        fieldnames = list(self.results[0].to_dict().keys())
        # 移除 config 字段（太复杂）
        fieldnames = [f for f in fieldnames if f != 'config']
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                row = result.to_dict()
                del row['config']
                writer.writerow(row)
        
        return filepath
    
    def generate_summary(self) -> Dict[str, Dict[str, Any]]:
        """生成汇总统计"""
        from collections import defaultdict
        
        # 按 (dataset, model) 分组
        grouped = defaultdict(list)
        for result in self.results:
            key = (result.dataset, result.model)
            grouped[key].append(result)
        
        summary = {}
        for (dataset, model), results in grouped.items():
            accuracies = [r.accuracy for r in results]
            f1_scores = [r.f1_macro for r in results if r.f1_macro is not None]
            times = [r.train_time for r in results]
            params = results[0].num_params  # 参数量相同
            
            key = f"{dataset}_{model}"
            summary[key] = {
                "dataset": dataset,
                "model": model,
                "accuracy_mean": np.mean(accuracies),
                "accuracy_std": np.std(accuracies),
                "f1_macro_mean": np.mean(f1_scores) if f1_scores else None,
                "f1_macro_std": np.std(f1_scores) if f1_scores else None,
                "train_time_mean": np.mean(times),
                "train_time_std": np.std(times),
                "num_params": params,
                "n_runs": len(results)
            }
        
        return summary
    
    def save_summary_csv(self, filename: str = "summary.csv"):
        """保存汇总表到 CSV"""
        import csv
        
        summary = self.generate_summary()
        filepath = os.path.join(self.output_dir, filename)
        
        if not summary:
            return filepath
        
        fieldnames = ["dataset", "model", "accuracy_mean", "accuracy_std", 
                     "f1_macro_mean", "f1_macro_std", "train_time_mean", 
                     "train_time_std", "num_params", "n_runs"]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for key, stats in summary.items():
                writer.writerow(stats)
        
        return filepath
    
    def print_summary(self):
        """打印汇总表"""
        summary = self.generate_summary()
        
        print("\n" + "=" * 100)
        print("实验结果汇总")
        print("=" * 100)
        print(f"{'Dataset':<20} {'Model':<20} {'Accuracy':<20} {'F1-Macro':<20} {'Time':<15} {'#Params':<15}")
        print("-" * 100)
        
        for key, stats in summary.items():
            acc_str = f"{stats['accuracy_mean']*100:.2f} ± {stats['accuracy_std']*100:.2f}"
            f1_str = f"{stats['f1_macro_mean']*100:.2f} ± {stats['f1_macro_std']*100:.2f}" if stats['f1_macro_mean'] else "N/A"
            time_str = format_time(stats['train_time_mean'])
            
            print(f"{stats['dataset']:<20} {stats['model']:<20} {acc_str:<20} {f1_str:<20} {time_str:<15} {stats['num_params']:<15}")
        
        print("=" * 100)


def try_git_commit(start_dir: str, max_up: int = 8) -> Optional[str]:
    """自 start_dir 起向上查找 git 仓库并返回 HEAD commit。"""
    d = os.path.abspath(start_dir)
    for _ in range(max_up):
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=d,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return out.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return None


def write_repro_manifest(
    output_dir: str,
    config: Dict[str, Any],
    datasets: List[str],
    models: List[str],
    seeds: List[int],
    data_dir: str,
    argv: List[str],
    checkpoint_dir: Optional[str] = None,
    benchmark_dir: Optional[str] = None,
) -> None:
    """
    写入复现用元数据：配置快照、命令行、版本号、可选 pip freeze、git commit。
    """
    benchmark_dir = benchmark_dir or os.path.dirname(os.path.abspath(__file__))
    manifest: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "argv": argv,
        "datasets": datasets,
        "models": models,
        "seeds": seeds,
        "data_dir_resolved": os.path.abspath(data_dir),
        "checkpoint_dir": os.path.abspath(checkpoint_dir) if checkpoint_dir else None,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "git_commit": try_git_commit(benchmark_dir),
        "config": config,
        "checkpoint_naming": "{dataset}__{model}__seed{seed}.pt",
        "load_hint": (
            "每个 .pt 含 model_state_dict、model_name、model_ctor_kwargs、词表与 preprocessor_flags。"
            "恢复: TextPreprocessor.from_checkpoint_payload(torch.load(path, map_location='cpu'))；"
            "get_model(ckpt['model_name'], **ckpt['model_ctor_kwargs']).load_state_dict(ckpt['model_state_dict'])"
        ),
    }
    try:
        import sklearn  # noqa: F401
        manifest["sklearn"] = sklearn.__version__
    except Exception:
        manifest["sklearn"] = None

    path = os.path.join(output_dir, "repro_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode == 0 and r.stdout:
            with open(os.path.join(output_dir, "requirements-frozen.txt"), "w", encoding="utf-8") as f:
                f.write(r.stdout)
    except Exception:
        pass

    req_cloud = os.path.join(benchmark_dir, "requirements-cloud.txt")
    if os.path.isfile(req_cloud):
        shutil.copy2(req_cloud, os.path.join(output_dir, "requirements-cloud.txt"))


def setup_logging(log_dir: str, name: str = "benchmark"):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(name)


class EarlyStopping:
    """Early Stopping 类"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: 等待的 epoch 数
            min_delta: 最小改进量
            mode: 'min' 或 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """
        检查是否应该早停
        
        Returns:
            True 如果应该早停
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    n_epochs: int,
    warmup_epochs: int = 5
):
    """获取学习率调度器"""
    if scheduler_type == "cosine":
        T_max = max(1, n_epochs - warmup_epochs)  # 确保至少为 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max
        )
    elif scheduler_type == "step":
        step_size = max(1, n_epochs // 3)  # 确保至少为 1
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=0.1
        )
    elif scheduler_type == "none":
        scheduler = None
    else:
        scheduler = None
    
    return scheduler

