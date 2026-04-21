"""
导出实验数据为多种格式，方便重新绘图
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, Any
from models import get_model


def load_results(results_file: str) -> Dict[str, Any]:
    """加载 JSON 结果"""
    with open(results_file, 'r') as f:
        return json.load(f)


def bytes_to_mb(b: float) -> float:
    """转换为 MB"""
    return b / (1024 * 1024)


def _count_params_by_components(model) -> Dict[str, int]:
    """按组件统计可训练参数（真实计数）"""
    groups = {
        "Q_proj": 0,
        "K_proj": 0,
        "V_proj": 0,
        "O_proj": 0,
        "QKV": 0,
        "Spectral": 0,
        "FFN": 0,
        "NormPosInputOther": 0,
    }
    spectral_keywords = ("spectral", "generator", "mean", "sigma", "w1", "w2")

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        lname = name.lower()
        if "qkv" in lname:
            groups["QKV"] += n
        elif "q_proj" in lname:
            groups["Q_proj"] += n
        elif "k_gate" in lname:
            groups["K_proj"] += n
        elif "v_proj" in lname:
            groups["V_proj"] += n
        elif "out_proj" in lname and ("mixer" in lname or "attention" in lname):
            groups["O_proj"] += n
        elif "attention.proj" in lname and "qkv" not in lname:
            groups["O_proj"] += n
        elif "w_q" in lname:
            groups["Q_proj"] += n
        elif "w_k" in lname:
            groups["K_proj"] += n
        elif "w_v" in lname:
            groups["V_proj"] += n
        elif "w_o" in lname:
            groups["O_proj"] += n
        elif "ff." in lname or "linear1" in lname or "linear2" in lname:
            groups["FFN"] += n
        elif any(k in lname for k in spectral_keywords):
            groups["Spectral"] += n
        else:
            groups["NormPosInputOther"] += n
    return groups


def export_to_csv(results_file: str, output_dir: str = None):
    """
    导出数据为 CSV 格式
    
    生成文件:
    - summary.csv: 汇总表
    - memory_vs_seqlen.csv: 内存 vs 序列长度
    - throughput_vs_seqlen.csv: 吞吐量 vs 序列长度
    - parameters.csv: 参数统计
    """
    data = load_results(results_file)
    config = data['config']
    results = data['results']
    
    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== 1. 完整数据表 ====================
    all_rows = []
    for model_type, result_list in results.items():
        for r in result_list:
            all_rows.append({
                'model': model_type,
                'seq_len': r['seq_len'],
                'batch_size': r['batch_size'],
                'd_model': r['d_model'],
                'n_heads': r['n_heads'],
                'n_layers': r['n_layers'],
                'M': r['M'],
                'inference_memory_bytes': r['inference_memory'],
                'training_memory_bytes': r['training_memory'],
                'inference_memory_mb': bytes_to_mb(r['inference_memory']),
                'training_memory_mb': bytes_to_mb(r['training_memory']),
                'peak_memory_mb': bytes_to_mb(r['peak_memory']),
                'trainable_params': r['trainable_params'],
                'total_params': r['total_params'],
                'param_ratio': r['trainable_params'] / r['total_params'] * 100,
                'inference_time_ms': r['inference_time'],
                'training_time_ms': r['training_time'],
                'inference_throughput': r['inference_throughput'],
                'training_throughput': r['training_throughput'],
                'success': r['success']
            })
    
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(output_dir, 'full_data.csv'), index=False)
    print(f"✅ Saved: {os.path.join(output_dir, 'full_data.csv')}")
    
    # ==================== 2. Memory vs Seq Len (宽表) ====================
    pivot_train_mem = df_all.pivot(index='model', columns='seq_len', values='training_memory_mb')
    pivot_train_mem.columns = [f'train_mem_seq{c}' for c in pivot_train_mem.columns]
    pivot_train_mem.to_csv(os.path.join(output_dir, 'memory_vs_seqlen_train.csv'))
    print(f"✅ Saved: {os.path.join(output_dir, 'memory_vs_seqlen_train.csv')}")
    
    pivot_inf_mem = df_all.pivot(index='model', columns='seq_len', values='inference_memory_mb')
    pivot_inf_mem.columns = [f'inf_mem_seq{c}' for c in pivot_inf_mem.columns]
    pivot_inf_mem.to_csv(os.path.join(output_dir, 'memory_vs_seqlen_inference.csv'))
    print(f"✅ Saved: {os.path.join(output_dir, 'memory_vs_seqlen_inference.csv')}")
    
    # ==================== 3. Throughput vs Seq Len (宽表) ====================
    pivot_throughput = df_all.pivot(index='model', columns='seq_len', values='training_throughput')
    pivot_throughput.columns = [f'throughput_seq{c}' for c in pivot_throughput.columns]
    pivot_throughput.to_csv(os.path.join(output_dir, 'throughput_vs_seqlen.csv'))
    print(f"✅ Saved: {os.path.join(output_dir, 'throughput_vs_seqlen.csv')}")
    
    # ==================== 4. 参数统计 ====================
    params_data = []
    for model_type, result_list in results.items():
        if result_list:
            r = result_list[0]
            params_data.append({
                'model': model_type,
                'trainable_params': r['trainable_params'],
                'total_params': r['total_params'],
                'param_ratio_percent': r['trainable_params'] / r['total_params'] * 100,
                'frozen_params': r['total_params'] - r['trainable_params']
            })
    
    df_params = pd.DataFrame(params_data)
    df_params.to_csv(os.path.join(output_dir, 'parameters.csv'), index=False)
    print(f"✅ Saved: {os.path.join(output_dir, 'parameters.csv')}")
    
    # ==================== 5. 汇总表 (最大序列长度) ====================
    max_seq_len = max(config['seq_lens'])
    summary_rows = []
    
    for model_type, result_list in results.items():
        for r in result_list:
            if r['seq_len'] == max_seq_len and r['success']:
                summary_rows.append({
                    'model': model_type,
                    'seq_len': max_seq_len,
                    'training_memory_mb': bytes_to_mb(r['training_memory']),
                    'inference_memory_mb': bytes_to_mb(r['inference_memory']),
                    'memory_ratio': r['training_memory'] / r['inference_memory'] if r['inference_memory'] > 0 else 0,
                    'training_throughput': r['training_throughput'],
                    'trainable_params': r['trainable_params'],
                    'total_params': r['total_params'],
                    'param_ratio_percent': r['trainable_params'] / r['total_params'] * 100
                })
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    print(f"✅ Saved: {os.path.join(output_dir, 'summary.csv')}")
    
    # ==================== 6. 配置信息 ====================
    config_df = pd.DataFrame([config])
    config_df.to_csv(os.path.join(output_dir, 'config.csv'), index=False)
    print(f"✅ Saved: {os.path.join(output_dir, 'config.csv')}")

    # ==================== 7. 组件参数分解 ====================
    d_model = config.get("d_model", 128)
    n_heads = config.get("n_heads", 4)
    n_layers = config.get("n_layers", 2)
    M = config.get("M", 64)
    max_seq_len = max(config.get("seq_lens", [4096]))

    comp_rows = []
    for model_type in results.keys():
        model = get_model(
            attention_type=model_type,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            M=M
        )
        comps = _count_params_by_components(model)
        total_trainable = sum(comps.values())
        row = {"model": model_type, "trainable_total": total_trainable}
        row.update(comps)
        row["QK_total"] = comps["Q_proj"] + comps["K_proj"]
        row["QK_share_percent"] = (row["QK_total"] / total_trainable * 100) if total_trainable > 0 else 0.0
        row["Spectral_share_percent"] = (comps["Spectral"] / total_trainable * 100) if total_trainable > 0 else 0.0
        comp_rows.append(row)

    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(os.path.join(output_dir, 'component_params.csv'), index=False)
    print(f"✅ Saved: {os.path.join(output_dir, 'component_params.csv')}")

    # ==================== 8. 复杂度拟合 ====================
    # 在 log-log 空间拟合: metric ~= c * seq_len^alpha
    complexity_rows = []
    for model_type in df_all["model"].unique():
        sub = df_all[(df_all["model"] == model_type) & (df_all["success"] == True)].copy()
        sub = sub.sort_values("seq_len")
        if len(sub) < 2:
            continue

        x = np.log(sub["seq_len"].to_numpy(dtype=float))
        y_mem = np.log(sub["training_memory_mb"].clip(lower=1e-8).to_numpy(dtype=float))
        y_time = np.log(sub["training_time_ms"].clip(lower=1e-8).to_numpy(dtype=float))

        beta = np.polyfit(x, y_mem, 1)[0]
        alpha = np.polyfit(x, y_time, 1)[0]
        complexity_rows.append({
            "model": model_type,
            "memory_exponent_beta": beta,
            "time_exponent_alpha": alpha,
            "num_points": len(sub)
        })

    df_complexity = pd.DataFrame(complexity_rows)
    df_complexity.to_csv(os.path.join(output_dir, 'complexity_fit.csv'), index=False)
    print(f"✅ Saved: {os.path.join(output_dir, 'complexity_fit.csv')}")
    
    print(f"\n📁 All data exported to: {output_dir}")
    
    return df_all


def print_latex_table(results_file: str):
    """生成 LaTeX 表格"""
    data = load_results(results_file)
    config = data['config']
    results = data['results']
    max_seq_len = max(config['seq_lens'])
    
    print("\n% LaTeX Table - 可直接复制到论文中")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Memory and Speed Comparison}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Model & Train Mem (MB) & Inf Mem (MB) & Throughput & Params \\\\")
    print("\\midrule")
    
    for model_type, result_list in results.items():
        for r in result_list:
            if r['seq_len'] == max_seq_len and r['success']:
                train_mem = bytes_to_mb(r['training_memory'])
                inf_mem = bytes_to_mb(r['inference_memory'])
                throughput = r['training_throughput']
                params = r['trainable_params']
                total = r['total_params']
                ratio = params / total * 100
                
                # 格式化模型名称
                model_name = model_type.replace('_', '\\_')
                
                print(f"{model_name} & {train_mem:.1f} & {inf_mem:.1f} & {throughput:.1f} & {params:,} ({ratio:.0f}\\%) \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="Export LRA benchmark data")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Path to results JSON file (default: latest)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as results file)")
    parser.add_argument("--latex", action="store_true",
                        help="Also print LaTeX table")
    
    args = parser.parse_args()
    
    # 如果没有指定文件，使用最新的
    if args.results_file is None:
        result_files = glob.glob("results/memory_benchmark_*.json")
        if not result_files:
            print("❌ No results files found!")
            exit(1)
        args.results_file = sorted(result_files)[-1]
        print(f"📂 Using latest results: {args.results_file}")
    
    # 导出 CSV
    export_to_csv(args.results_file, args.output_dir)
    
    # 打印 LaTeX 表格
    if args.latex:
        print_latex_table(args.results_file)

