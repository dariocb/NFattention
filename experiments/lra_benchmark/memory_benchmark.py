"""
LRA 内存基准测试

测量不同模型在不同序列长度下的：
1. 峰值内存消耗（训练 & 推理）
2. 速度（steps/sec）
3. 可处理的最大序列长度
"""

import torch
import torch.nn as nn
import time
import gc
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, asdict

from models import get_model


@dataclass
class BenchmarkResult:
    """单次基准测试结果"""
    model_type: str
    seq_len: int
    batch_size: int
    d_model: int
    n_heads: int
    n_layers: int
    M: int
    
    # 内存指标 (bytes)
    inference_memory: float
    training_memory: float
    peak_memory: float
    
    # 参数统计
    trainable_params: int
    total_params: int
    
    # 速度指标
    inference_time: float  # ms per step
    training_time: float   # ms per step
    inference_throughput: float  # samples/sec
    training_throughput: float   # samples/sec
    
    # 状态
    success: bool
    error_message: str = ""


def clear_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_memory_bytes() -> float:
    """获取当前内存使用（bytes）"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    else:
        # CPU 模式使用 tracemalloc（近似）
        try:
            import tracemalloc
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                return peak
        except:
            pass
        return 0


def reset_tracemalloc_peak():
    """重置 tracemalloc 峰值（兼容旧版本 Python）"""
    try:
        import tracemalloc
        if hasattr(tracemalloc, 'reset_peak'):
            # Python 3.9+
            tracemalloc.reset_peak()
        else:
            # Python 3.8 及更早版本：停止再重新开始
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            tracemalloc.start()
    except:
        pass


def bytes_to_mb(b: float) -> float:
    """转换为 MB"""
    return b / (1024 * 1024)


def benchmark_model(
    model_type: str,
    seq_len: int,
    batch_size: int = 4,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    M: int = 64,
    warmup_steps: int = 3,
    measure_steps: int = 10,
    device: str = "auto"
) -> BenchmarkResult:
    """
    对单个模型进行基准测试
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化结果
    result = BenchmarkResult(
        model_type=model_type,
        seq_len=seq_len,
        batch_size=batch_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        M=M,
        inference_memory=0,
        training_memory=0,
        peak_memory=0,
        trainable_params=0,
        total_params=0,
        inference_time=0,
        training_time=0,
        inference_throughput=0,
        training_throughput=0,
        success=False,
        error_message=""
    )
    
    try:
        clear_memory()
        
        # 创建模型
        model = get_model(
            attention_type=model_type,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=seq_len,
            M=M
        ).to(device)
        
        result.trainable_params = model.count_parameters()
        result.total_params = model.count_total_parameters()
        
        # 创建输入
        x = torch.randn(batch_size, seq_len, d_model).to(device)
        
        # CPU 模式下启用 tracemalloc
        use_tracemalloc = device == "cpu"
        if use_tracemalloc:
            try:
                import tracemalloc
                tracemalloc.start()
            except:
                use_tracemalloc = False
        
        # ==================== 推理测试 ====================
        model.eval()
        clear_memory()
        if use_tracemalloc:
            reset_tracemalloc_peak()
        
        with torch.no_grad():
            # 预热
            for _ in range(warmup_steps):
                _ = model(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 测量内存
            clear_memory()
            if use_tracemalloc:
                reset_tracemalloc_peak()
                
            _ = model(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            result.inference_memory = get_memory_bytes()
            
            # 测量时间
            start_time = time.perf_counter()
            for _ in range(measure_steps):
                _ = model(x)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            result.inference_time = (end_time - start_time) / measure_steps * 1000  # ms
            result.inference_throughput = batch_size * measure_steps / (end_time - start_time)
        
        # ==================== 训练测试 ====================
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        target = torch.randn(batch_size, seq_len, d_model).to(device)
        
        clear_memory()
        if use_tracemalloc:
            reset_tracemalloc_peak()
        
        # 预热
        for _ in range(warmup_steps):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 测量内存
        clear_memory()
        if use_tracemalloc:
            reset_tracemalloc_peak()
            
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        result.training_memory = get_memory_bytes()
        result.peak_memory = result.training_memory
        
        # 测量时间
        start_time = time.perf_counter()
        for _ in range(measure_steps):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        result.training_time = (end_time - start_time) / measure_steps * 1000  # ms
        result.training_throughput = batch_size * measure_steps / (end_time - start_time)
        
        result.success = True
        
        if use_tracemalloc:
            tracemalloc.stop()
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result.error_message = "OOM"
        else:
            result.error_message = str(e)
    except Exception as e:
        result.error_message = str(e)
    finally:
        clear_memory()
        
    return result


def run_memory_vs_seqlen_benchmark(
    model_types: List[str],
    seq_lens: List[int],
    batch_size: int = 4,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    M: int = 64,
    device: str = "auto",
    output_dir: str = "results"
) -> Dict[str, List[BenchmarkResult]]:
    """
    运行 Memory vs Sequence Length 基准测试
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {model_type: [] for model_type in model_types}
    
    print("=" * 70)
    print("🚀 LRA Memory Benchmark")
    print("=" * 70)
    print(f"Models: {model_types}")
    print(f"Sequence lengths: {seq_lens}")
    print(f"Batch size: {batch_size}")
    print(f"d_model: {d_model}, n_heads: {n_heads}, n_layers: {n_layers}, M: {M}")
    print(f"Device: {device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 70)
    
    total_experiments = len(model_types) * len(seq_lens)
    current = 0
    
    for seq_len in seq_lens:
        print(f"\n📏 Sequence Length: {seq_len}")
        print("-" * 50)
        
        for model_type in model_types:
            current += 1
            print(f"  [{current}/{total_experiments}] {model_type}...", end=" ", flush=True)
            
            result = benchmark_model(
                model_type=model_type,
                seq_len=seq_len,
                batch_size=batch_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                M=M,
                device=device
            )
            
            results[model_type].append(result)
            
            if result.success:
                print(f"✅ Inf: {bytes_to_mb(result.inference_memory):.1f}MB, "
                      f"Train: {bytes_to_mb(result.training_memory):.1f}MB, "
                      f"Speed: {result.training_throughput:.1f} samples/s")
            else:
                print(f"❌ {result.error_message}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"memory_benchmark_{timestamp}.json")
    
    # 转换为可序列化格式
    serializable_results = {}
    for model_type, result_list in results.items():
        serializable_results[model_type] = [asdict(r) for r in result_list]
    
    with open(results_file, 'w') as f:
        json.dump({
            "config": {
                "model_types": model_types,
                "seq_lens": seq_lens,
                "batch_size": batch_size,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "M": M,
                "timestamp": timestamp
            },
            "results": serializable_results
        }, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")
    
    return results


def print_summary_table(results: Dict[str, List[BenchmarkResult]]):
    """打印结果汇总表"""
    print("\n" + "=" * 100)
    print("📊 Summary Table")
    print("=" * 100)
    
    # 获取序列长度
    first_results = list(results.values())[0]
    seq_lens = [r.seq_len for r in first_results]
    
    # 表头
    header = f"{'Model':<20} | " + " | ".join([f"Len={s:<5}" for s in seq_lens])
    print(header)
    print("-" * len(header))
    
    # 训练内存
    print("\n📦 Training Memory (MB):")
    print("-" * 80)
    for model_type, result_list in results.items():
        row = f"{model_type:<20} | "
        for r in result_list:
            if r.success:
                row += f"{bytes_to_mb(r.training_memory):>8.1f} | "
            else:
                row += f"{'OOM':>8} | "
        print(row)
    
    # 推理内存
    print("\n🔮 Inference Memory (MB):")
    print("-" * 80)
    for model_type, result_list in results.items():
        row = f"{model_type:<20} | "
        for r in result_list:
            if r.success:
                row += f"{bytes_to_mb(r.inference_memory):>8.1f} | "
            else:
                row += f"{'OOM':>8} | "
        print(row)
    
    # 训练速度
    print("\n⚡ Training Throughput (samples/sec):")
    print("-" * 80)
    for model_type, result_list in results.items():
        row = f"{model_type:<20} | "
        for r in result_list:
            if r.success:
                row += f"{r.training_throughput:>8.1f} | "
            else:
                row += f"{'N/A':>8} | "
        print(row)
    
    # 可训练参数
    print("\n📐 Trainable Parameters:")
    print("-" * 80)
    for model_type, result_list in results.items():
        if result_list and result_list[0].success:
            params = result_list[0].trainable_params
            total = result_list[0].total_params
            print(f"{model_type:<20} | Trainable: {params:>10,} | Total: {total:>10,} | Ratio: {params/total*100:.1f}%")


def find_max_seq_len(
    model_type: str,
    batch_size: int = 4,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    M: int = 64,
    max_test_len: int = 16384,
    device: str = "auto"
) -> int:
    """二分搜索找到模型能处理的最大序列长度"""
    low, high = 256, max_test_len
    max_working = 0
    
    while low <= high:
        mid = (low + high) // 2
        
        result = benchmark_model(
            model_type=model_type,
            seq_len=mid,
            batch_size=batch_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            M=M,
            warmup_steps=1,
            measure_steps=1,
            device=device
        )
        
        if result.success:
            max_working = mid
            low = mid + 256
        else:
            high = mid - 256
    
    return max_working


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LRA Memory Benchmark")
    parser.add_argument("--models", nargs="+", 
                        default=[
                            "transformer", "mikan", "performer", "rka", "gmm_rks",
                            "kpca_scaled", "metala", "ours_latest",
                        ],
                        help="Models to benchmark")
    parser.add_argument("--seq_lens", nargs="+", type=int,
                        default=[256, 512, 1024, 2048],
                        help="Sequence lengths to test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--M", type=int, default=64, help="Number of random features")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results")
    
    args = parser.parse_args()
    
    # 运行基准测试
    results = run_memory_vs_seqlen_benchmark(
        model_types=args.models,
        seq_lens=args.seq_lens,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        M=args.M,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # 打印汇总
    print_summary_table(results)

