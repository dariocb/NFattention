"""
统一训练入口 - 带实时进度汇报
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataset_by_name, create_cv_splits, TextPreprocessor, get_available_datasets
from models import get_model
from utils import (
    set_seed, load_config, save_config, get_device, count_parameters,
    format_time, ExperimentResult, ResultLogger, EarlyStopping,
    get_lr_scheduler, setup_logging
)


# ============================================================
# 进度汇报工具
# ============================================================

class ProgressReporter:
    """进度汇报器"""
    
    def __init__(self, total_experiments: int = 1):
        self.total_experiments = total_experiments
        self.current_experiment = 0
        self.start_time = time.time()
        self.experiment_times = []
        
    def start_experiment(self, dataset: str, model: str, seed: int, fold: Optional[int] = None):
        """开始新实验"""
        self.current_experiment += 1
        self.exp_start_time = time.time()
        
        fold_str = f", Fold {fold+1}" if fold is not None else ""
        
        print("\n" + "=" * 70)
        print(f"📊 实验进度: [{self.current_experiment}/{self.total_experiments}]")
        print(f"=" * 70)
        print(f"🗂️  数据集: {dataset}")
        print(f"🤖 模型:   {model}")
        print(f"🎲 种子:   {seed}{fold_str}")
        
        # 预计剩余时间
        if self.experiment_times:
            avg_time = np.mean(self.experiment_times)
            remaining = avg_time * (self.total_experiments - self.current_experiment + 1)
            print(f"⏱️  预计剩余时间: {format_time(remaining)}")
        
        print("-" * 70)
        
    def report_epoch(self, epoch: int, total_epochs: int, train_loss: float, 
                     train_acc: float, val_acc: Optional[float] = None,
                     epoch_time: float = 0, is_best: bool = False):
        """汇报 epoch 进度"""
        progress = (epoch + 1) / total_epochs * 100
        bar_len = 30
        filled = int(bar_len * (epoch + 1) / total_epochs)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        best_marker = " ⭐" if is_best else ""
        val_str = f", Val Acc: {val_acc:.4f}" if val_acc is not None else ""
        
        # 预计本次实验剩余时间
        if epoch_time > 0:
            remaining_epochs = total_epochs - epoch - 1
            eta = epoch_time * remaining_epochs
            eta_str = f" | ETA: {format_time(eta)}"
        else:
            eta_str = ""
        
        print(f"\r  [{bar}] {progress:5.1f}% | Epoch {epoch+1}/{total_epochs} | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}{val_str}{best_marker}{eta_str}    ", 
              end="", flush=True)
        
    def finish_experiment(self, result: 'ExperimentResult'):
        """完成实验"""
        exp_time = time.time() - self.exp_start_time
        self.experiment_times.append(exp_time)
        
        print()  # 换行
        print("-" * 70)
        print(f"✅ 训练完成!")
        print(f"   📈 Test Accuracy: {result.accuracy*100:.2f}%")
        print(f"   📊 Test F1-Macro: {result.f1_macro*100:.2f}%")
        print(f"   ⏱️  训练时间: {format_time(exp_time)}")
        print(f"   🔢 参数量: {result.num_params:,}")
        
        # 总进度
        total_elapsed = time.time() - self.start_time
        if self.current_experiment < self.total_experiments:
            avg_time = np.mean(self.experiment_times)
            remaining = avg_time * (self.total_experiments - self.current_experiment)
            print(f"   📍 总进度: {self.current_experiment}/{self.total_experiments} | "
                  f"已用: {format_time(total_elapsed)} | 预计剩余: {format_time(remaining)}")
        else:
            print(f"   🎉 全部完成! 总用时: {format_time(total_elapsed)}")


# ============================================================
# 数据加载
# ============================================================

def create_dataloader(
    texts: List[str],
    labels: List[int],
    preprocessor: TextPreprocessor,
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """创建 DataLoader"""
    input_ids, attention_mask = preprocessor.transform(texts)
    
    dataset = TensorDataset(
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long)
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================
# 训练和评估
# ============================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0
) -> Tuple[float, float]:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        
        logits, kl_div = model(input_ids, attention_mask)
        
        ce_loss = criterion(logits, labels)
        loss = ce_loss + kl_div
        
        loss.backward()
        
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        total_loss += ce_loss.item()
        
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int
) -> Tuple[float, float, float]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            logits, kl_div = model(input_ids, attention_mask)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    if num_classes > 2:
        f1 = f1_score(all_labels, all_preds, average='macro')
    else:
        f1 = f1_score(all_labels, all_preds, average='binary')
    
    return avg_loss, accuracy, f1


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    num_classes: int,
    progress_reporter: Optional[ProgressReporter] = None
) -> Tuple[float, float, float, float, int, float]:
    """训练模型（带进度汇报）"""
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = get_lr_scheduler(
        optimizer,
        config['training']['scheduler'],
        config['training']['n_epochs'],
        config['training'].get('warmup_epochs', 5)
    )
    
    criterion = nn.CrossEntropyLoss()
    
    early_stopping = EarlyStopping(
        patience=config['training']['patience'],
        mode='max'
    )
    
    best_val_acc = 0.0
    best_model_state = None
    n_epochs = config['training']['n_epochs']
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            config['training']['gradient_clip']
        )
        
        # 验证
        val_acc = None
        is_best = False
        if val_loader is not None:
            _, val_acc, _ = evaluate(model, val_loader, criterion, device, num_classes)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                is_best = True
            
            if early_stopping(val_acc, epoch):
                if progress_reporter:
                    print(f"\n  ⚠️  Early stopping at epoch {epoch+1}")
                break
        else:
            if train_acc > best_val_acc:
                best_val_acc = train_acc
                best_model_state = model.state_dict().copy()
                is_best = True
        
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # 进度汇报
        if progress_reporter:
            progress_reporter.report_epoch(
                epoch, n_epochs, train_loss, train_acc, 
                val_acc, epoch_time, is_best
            )
    
    train_time = time.time() - start_time
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 最终测试
    test_loss, test_acc, test_f1 = evaluate(
        model, test_loader, criterion, device, num_classes
    )
    
    return test_acc, test_f1, train_loss, test_loss, early_stopping.best_epoch, train_time


# ============================================================
# 实验运行
# ============================================================

def save_training_checkpoint(
    path: str,
    model: nn.Module,
    model_name: str,
    dataset_name: str,
    seed: int,
    fold: Optional[int],
    num_classes: int,
    model_ctor_kwargs: Dict[str, Any],
    preprocessor: TextPreprocessor,
    best_epoch: int,
    accuracy: float,
    f1_macro: Optional[float],
    config: Dict[str, Any],
) -> None:
    """保存单轮实验的可复现包：权重、构造参数、词表与预处理设置、指标。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "dataset": dataset_name,
        "seed": seed,
        "fold": fold,
        "num_classes": num_classes,
        "model_ctor_kwargs": model_ctor_kwargs,
        "vocab_word2idx": dict(preprocessor.vocab.word2idx),
        "vocab_idx2word": dict(preprocessor.vocab.idx2word),
        "preprocessor_flags": {
            "lowercase": preprocessor.lowercase,
            "remove_punctuation": preprocessor.remove_punctuation,
            "remove_numbers": preprocessor.remove_numbers,
            "max_seq_len": preprocessor.max_seq_len,
            "min_freq": preprocessor.vocab.min_freq,
            "max_vocab_size": preprocessor.vocab.max_size,
        },
        "best_epoch": best_epoch,
        "metrics": {
            "test_accuracy": float(accuracy),
            "test_f1_macro": float(f1_macro) if f1_macro is not None else None,
        },
        "training_config": config,
    }
    torch.save(payload, path)


def run_single_experiment(
    dataset_name: str,
    model_name: str,
    config: Dict[str, Any],
    seed: int,
    data_dir: str,
    device: torch.device,
    logger=None,
    fold: Optional[int] = None,
    train_data: Optional[Tuple] = None,
    val_data: Optional[Tuple] = None,
    test_data: Optional[Tuple] = None,
    progress_reporter: Optional[ProgressReporter] = None,
    checkpoint_dir: Optional[str] = None,
) -> ExperimentResult:
    """运行单次实验"""
    
    set_seed(seed)
    
    # 汇报开始
    if progress_reporter:
        progress_reporter.start_experiment(dataset_name, model_name, seed, fold)
    
    # 加载数据
    if train_data is None:
        print(f"  📂 加载数据集 {dataset_name}...")
        dataset = load_dataset_by_name(dataset_name, data_dir)
        train_texts, train_labels = dataset['train']
        val_texts, val_labels = dataset['val'] if dataset['val'] else (None, None)
        test_texts, test_labels = dataset['test']
        num_classes = dataset['info'].num_classes
        print(f"     训练: {len(train_texts)}, 测试: {len(test_texts)}, 类别: {num_classes}")
    else:
        train_texts, train_labels = train_data
        val_texts, val_labels = val_data if val_data else (None, None)
        test_texts, test_labels = test_data
        num_classes = len(set(train_labels + test_labels))
        print(f"  📂 数据: 训练 {len(train_texts)}, 测试 {len(test_texts)}, 类别 {num_classes}")
    
    # 预处理
    print(f"  🔧 预处理文本...")
    preprocessor = TextPreprocessor(
        max_seq_len=config['model_params']['max_seq_len'],
        min_freq=2,
        max_vocab_size=50000
    )
    preprocessor.fit(train_texts)
    print(f"     词汇表大小: {preprocessor.vocab_size}")
    
    # 创建 DataLoader
    batch_size = config['training']['batch_size']
    train_loader = create_dataloader(train_texts, train_labels, preprocessor, batch_size, shuffle=True)
    
    val_loader = None
    if val_texts is not None:
        val_loader = create_dataloader(val_texts, val_labels, preprocessor, batch_size, shuffle=False)
    
    test_loader = create_dataloader(test_texts, test_labels, preprocessor, batch_size, shuffle=False)
    
    # 创建模型
    print(f"  🏗️  创建模型 {model_name}...")
    model_params = {
        'vocab_size': preprocessor.vocab_size,
        'num_classes': num_classes,
        'embed_dim': config['model_params']['embed_dim'],
        'hidden_dim': config['model_params']['hidden_dim'],
        'n_heads': config['model_params']['n_heads'],
        'n_layers': config['model_params']['n_layers'],
        'pf_dim': config['model_params']['pf_dim'],
        'dropout': config['model_params']['dropout'],
        'max_seq_len': config['model_params']['max_seq_len'],
        'pad_idx': preprocessor.vocab.pad_idx,
        'device': str(device),
        'M': config['model_params']['M'],
        'num_flows': config['model_params'].get('num_flows', 3),
        'flow_hidden_dim': config['model_params'].get('flow_hidden_dim', 64),
        'num_mixtures': config['model_params'].get('num_mixtures', 10),
        'kl_lambda': config['model_params']['kl_lambda'],
    }
    
    model = get_model(model_name, **model_params)
    model = model.to(device)

    if model_name == "ours_latest":
        print(
            "     Ours latest config: "
            f"qk_mode={getattr(model, 'qk_mode', 'unknown')}, "
            f"v_mode={getattr(model, 'v_mode', 'unknown')}, "
            f"shared_flow={getattr(model, 'shared_flow', 'unknown')}, "
            f"M={getattr(model, 'M', 'unknown')}, "
            f"num_flows={getattr(model, 'num_flows', 'unknown')}, "
            f"flow_hidden_dim={getattr(model, 'flow_hidden_dim', 'unknown')}"
        )
    
    num_params = count_parameters(model)
    print(f"     参数量: {num_params:,}")
    
    # 训练
    print(f"  🚀 开始训练 ({config['training']['n_epochs']} epochs)...")
    test_acc, test_f1, train_loss, test_loss, best_epoch, train_time = train_model(
        model, train_loader, val_loader, test_loader,
        config, device, num_classes, progress_reporter
    )

    if checkpoint_dir:
        fname = f"{dataset_name}__{model_name}__seed{seed}.pt"
        if fold is not None:
            fname = f"{dataset_name}__{model_name}__seed{seed}__fold{fold}.pt"
        ck_path = os.path.join(checkpoint_dir, fname)
        save_training_checkpoint(
            ck_path,
            model=model,
            model_name=model_name,
            dataset_name=dataset_name,
            seed=seed,
            fold=fold,
            num_classes=num_classes,
            model_ctor_kwargs=model_params,
            preprocessor=preprocessor,
            best_epoch=best_epoch,
            accuracy=test_acc,
            f1_macro=test_f1,
            config=config,
        )
        print(f"  💾 已保存 checkpoint: {ck_path}")
    
    # 记录结果
    result = ExperimentResult(
        dataset=dataset_name,
        model=model_name,
        seed=seed,
        fold=fold,
        accuracy=test_acc,
        f1_macro=test_f1,
        train_loss=train_loss,
        val_loss=None,
        test_loss=test_loss,
        train_time=train_time,
        num_params=num_params,
        best_epoch=best_epoch,
        config=config
    )
    
    # 汇报完成
    if progress_reporter:
        progress_reporter.finish_experiment(result)
    
    return result


def run_cv_experiment(
    dataset_name: str,
    model_name: str,
    config: Dict[str, Any],
    seed: int,
    data_dir: str,
    device: torch.device,
    logger=None,
    n_folds: int = 10,
    progress_reporter: Optional[ProgressReporter] = None,
    checkpoint_dir: Optional[str] = None,
) -> List[ExperimentResult]:
    """运行交叉验证实验"""
    
    set_seed(seed)
    
    # 加载全部数据
    print(f"\n📂 加载数据集 {dataset_name} (10-fold CV)...")
    dataset = load_dataset_by_name(dataset_name, data_dir)
    texts, labels = dataset['all_data']
    num_classes = dataset['info'].num_classes
    print(f"   总样本: {len(texts)}, 类别: {num_classes}")
    
    # 创建 CV 划分
    cv_splits = create_cv_splits(texts, labels, n_folds=n_folds, random_state=seed)
    
    results = []
    
    for fold_idx, (train_data, test_data) in enumerate(cv_splits):
        result = run_single_experiment(
            dataset_name=dataset_name,
            model_name=model_name,
            config=config,
            seed=seed,
            data_dir=data_dir,
            device=device,
            fold=fold_idx,
            train_data=train_data,
            val_data=None,
            test_data=test_data,
            progress_reporter=progress_reporter,
            checkpoint_dir=checkpoint_dir,
        )
        results.append(result)
    
    # CV 结果汇总
    accuracies = [r.accuracy for r in results]
    print(f"\n📊 CV 汇总: {np.mean(accuracies)*100:.2f}% ± {np.std(accuracies)*100:.2f}%")
    
    return results


def run_all_experiments(
    datasets: List[str],
    models: List[str],
    seeds: List[int],
    config: Dict[str, Any],
    data_dir: str,
    device: torch.device,
    output_dir: str
):
    """运行所有实验组合"""
    
    # 数据集配置（全部使用 train/test split，不用 CV）
    dataset_config = {
        "rt_polarity": {"use_cv": False},  # 改为不用 CV
        "sst2": {"use_cv": False},
        "sst5": {"use_cv": False},
        "trec": {"use_cv": False},
        "ag_news": {"use_cv": False},
        "dbpedia_14": {"use_cv": False},
        "yelp_review_full": {"use_cv": False},
    }
    
    # 计算总实验数
    total = 0
    for ds in datasets:
        ds_config = dataset_config.get(ds, {"use_cv": False})
        if ds_config.get("use_cv", False):
            n_folds = ds_config.get("n_folds", 10)
            total += len(models) * len(seeds) * n_folds
        else:
            total += len(models) * len(seeds)
    
    print(f"\n{'='*70}")
    print(f"🔬 Text Sequence Classification Benchmark")
    print(f"{'='*70}")
    print(f"📋 数据集: {datasets}")
    print(f"🤖 模型:   {models}")
    print(f"🎲 种子:   {seeds}")
    print(f"📊 总实验数: {total}")
    print(f"💻 设备:   {device}")
    print(f"{'='*70}")
    
    progress_reporter = ProgressReporter(total_experiments=total)
    result_logger = ResultLogger(output_dir)
    
    for dataset_name in datasets:
        ds_config = dataset_config.get(dataset_name, {"use_cv": False})
        use_cv = ds_config.get("use_cv", False)
        
        for model_name in models:
            for seed in seeds:
                try:
                    if use_cv:
                        results = run_cv_experiment(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            config=config,
                            seed=seed,
                            data_dir=data_dir,
                            device=device,
                            n_folds=ds_config.get("n_folds", 10),
                            progress_reporter=progress_reporter
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
                            progress_reporter=progress_reporter
                        )
                        result_logger.add_result(result)
                        
                except Exception as e:
                    print(f"\n❌ 实验失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # 每个数据集完成后保存
        result_logger.save_csv(f"results_{dataset_name}.csv")
    
    # 保存所有结果
    result_logger.save_csv("all_results.csv")
    result_logger.save_summary_csv("summary.csv")
    
    print("\n" + "=" * 70)
    result_logger.print_summary()
    
    return result_logger


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Text Sequence Classification Training')
    
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='数据集列表（默认全部）')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='模型列表（默认全部）')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                       help='随机种子列表')
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='数据目录')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID')
    parser.add_argument('--n_epochs', type=int, default=None,
                       help='覆盖配置中的 epoch 数')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='覆盖配置中的 batch size')
    
    args = parser.parse_args()
    
    # 默认配置
    all_datasets = ["rt_polarity", "sst2", "sst5", "trec", "ag_news", "dbpedia_14", "yelp_review_full"]
    all_models = [
        "transformer",      # Standard Transformer baseline
        "mikan",            # MIKAN baseline
        "mgk",              # MGK baseline
        "performer",        # Performer (FAVOR+) baseline
        "rka",              # RKA (Random Kernel Attention) baseline
        "gmm_rks",          # GMM-RKS baseline
        "kpca_scaled",      # KPCA repo scaled attention (Teo & Nguyen, NeurIPS 2024)
        "metala",           # MetaLA-style GLA (Chou et al., NeurIPS 2024)
        "ours_latest",      # 主模型：no_qk + fixed_orth_v + shared_flow
        "ours_fixed_qk",    # Our method (frozen Q,K)
        "ours_trainable_qk" # Our method (trainable Q,K)
    ]
    
    datasets = args.datasets or all_datasets
    models = args.models or all_models
    
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
    
    # 覆盖配置
    if args.n_epochs:
        config['training']['n_epochs'] = args.n_epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # 设置设备
    device = get_device(args.gpu)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    save_config(config, os.path.join(output_dir, "config.yaml"))
    
    # 运行所有实验
    run_all_experiments(
        datasets=datasets,
        models=models,
        seeds=args.seeds,
        config=config,
        data_dir=args.data_dir,
        device=device,
        output_dir=output_dir
    )


if __name__ == '__main__':
    main()
