"""
完整 LRA 基准测试运行脚本

云端运行命令：
python run_full_benchmark.py --device cuda

本地测试命令（CPU，小规模）：
python run_full_benchmark.py --device cpu --quick_test
"""

import os
import sys
import argparse
from datetime import datetime

from memory_benchmark import run_memory_vs_seqlen_benchmark, print_summary_table
from visualize import generate_all_figures
from export_data import export_to_csv


def run_full_benchmark(args):
    """运行完整基准测试"""
    
    # 配置
    if args.quick_test:
        # 快速测试模式（本地 CPU 验证）
        model_types = [
            "transformer", "mikan", "performer", "rka", "gmm_rks", "mgk",
            "kpca_scaled", "metala", "ours_latest",
        ]
        seq_lens = [128, 256, 512]
        batch_size = 2
        d_model = 64
        n_heads = 2
        n_layers = 1
        M = 32
    else:
        # 完整测试模式（云端 GPU）
        model_types = args.models if args.models else [
            "transformer", "mikan", "performer", "rka", "gmm_rks", "mgk",
            "kpca_scaled", "metala", "ours_latest",
        ]
        seq_lens = args.seq_lens if args.seq_lens else [256, 512, 1024, 2048, 4096]
        batch_size = args.batch_size
        d_model = args.d_model
        n_heads = args.n_heads
        n_layers = args.n_layers
        M = args.M
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("🚀 LRA Full Benchmark")
    print("=" * 70)
    print(f"Mode: {'Quick Test' if args.quick_test else 'Full Benchmark'}")
    print(f"Device: {args.device}")
    print(f"Models: {model_types}")
    print(f"Sequence Lengths: {seq_lens}")
    print(f"Config: batch_size={batch_size}, d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, M={M}")
    print("=" * 70)
    
    # 运行内存基准测试
    print("\n📊 Phase 1: Memory Benchmark")
    print("-" * 50)
    
    results = run_memory_vs_seqlen_benchmark(
        model_types=model_types,
        seq_lens=seq_lens,
        batch_size=batch_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        M=M,
        device=args.device,
        output_dir=output_dir
    )
    
    # 打印汇总表
    print_summary_table(results)
    
    # 生成图表
    print("\n📈 Phase 2: Generate Figures")
    print("-" * 50)
    
    # 找到最新的结果文件
    result_files = sorted([f for f in os.listdir(output_dir) if f.startswith("memory_benchmark_") and f.endswith(".json")])
    if result_files:
        latest_results = os.path.join(output_dir, result_files[-1])
        figures_dir = os.path.join(output_dir, "figures")
        
        try:
            generate_all_figures(latest_results, figures_dir)
        except Exception as e:
            print(f"⚠️ 图表生成失败: {e}")
            print("可以稍后手动运行: python visualize.py --results_file <results.json>")

        # 导出结构化数据（含组件参数与复杂度拟合）
        try:
            export_to_csv(latest_results, output_dir)
        except Exception as e:
            print(f"⚠️ 数据导出失败: {e}")
            print("可以稍后手动运行: python export_data.py --results_file <results.json>")
    
    print("\n" + "=" * 70)
    print("✅ Benchmark Complete!")
    print("=" * 70)
    print(f"📁 Results: {output_dir}")
    
    # 生成报告摘要
    generate_report(results, output_dir)


def generate_report(results, output_dir):
    """生成文本报告"""
    report_path = os.path.join(output_dir, "report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LRA Memory Benchmark Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 分析 Fixed QK 的优势
        f.write("## Key Findings\n\n")
        
        # 获取最大序列长度的数据
        first_results = list(results.values())[0]
        if first_results:
            max_seq_len = max(r.seq_len for r in first_results)
            
            f.write(f"### At Sequence Length = {max_seq_len}\n\n")
            
            # 收集数据
            data = {}
            for model_type, result_list in results.items():
                for r in result_list:
                    if r.seq_len == max_seq_len and r.success:
                        data[model_type] = {
                            'train_memory': r.training_memory / (1024*1024),
                            'inference_memory': r.inference_memory / (1024*1024),
                            'throughput': r.training_throughput,
                            'trainable_params': r.trainable_params
                        }
            
            if data:
                # 内存对比
                f.write("| Model | Train Memory (MB) | Inference Memory (MB) | Ratio | Throughput |\n")
                f.write("|-------|------------------|----------------------|-------|------------|\n")
                
                for model_type, d in data.items():
                    ratio = d['train_memory'] / d['inference_memory'] if d['inference_memory'] > 0 else 0
                    f.write(f"| {model_type:15s} | {d['train_memory']:16.1f} | {d['inference_memory']:20.1f} | {ratio:5.2f} | {d['throughput']:10.1f} |\n")
                
                # Ours(latest) 优势分析（相对 Performer）
                if 'ours_latest' in data and 'performer' in data:
                    ours = data['ours_latest']
                    perf = data['performer']
                    mem_delta = (perf['train_memory'] - ours['train_memory']) / max(perf['train_memory'], 1e-8) * 100
                    speed_delta = (ours['throughput'] - perf['throughput']) / max(perf['throughput'], 1e-8) * 100
                    f.write(f"\n### Ours (Latest) vs Performer\n")
                    f.write(f"- Training Memory Delta: {mem_delta:+.1f}% (positive means ours uses less)\n")
                    f.write(f"- Throughput Delta: {speed_delta:+.1f}% (positive means ours is faster)\n")
                    f.write(f"- Ours Trainable Params: {ours['trainable_params']:,}\n")
                    f.write(f"- Performer Trainable Params: {perf['trainable_params']:,}\n")
    
    print(f"📄 Report saved: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full LRA Benchmark")
    
    # 基本参数
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to run on")
    parser.add_argument("--quick_test", action="store_true",
                        help="Run quick test with small models")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory")
    
    # 模型参数
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to benchmark")
    parser.add_argument("--seq_lens", nargs="+", type=int, default=None,
                        help="Sequence lengths to test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--M", type=int, default=64)
    
    args = parser.parse_args()
    
    run_full_benchmark(args)
