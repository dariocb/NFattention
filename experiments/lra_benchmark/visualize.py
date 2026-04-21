"""
LRA 实验可视化

生成类似论文中的 Figure 3 和 Figure 4 风格的图表
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Any
from pathlib import Path
from models import get_model


# 设置中文字体和样式
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 颜色方案
COLORS = {
    'transformer': '#E74C3C',     # 红色
    'mikan': '#1ABC9C',           # 青色
    'performer': '#3498DB',       # 蓝色
    'rka': '#8E44AD',             # 深紫色
    'gmm_rks': '#2ECC71',         # 绿色
    'mgk': '#27AE60',             # 深绿
    'kpca_scaled': '#16A085',     # 深青绿
    'metala': '#D35400',          # 南瓜橙
    'ours_latest': '#F39C12',        # 橙色/金色
    'ours_trainable_qk': '#9B59B6',  # 兼容旧结果
    'ours_fixed_qk': '#F5B041',      # 兼容旧结果
}

MARKERS = {
    'transformer': 'o',
    'mikan': 'h',
    'performer': 's',
    'rka': 'p',
    'gmm_rks': '^',
    'mgk': 'd',
    'kpca_scaled': 'v',
    'metala': 'P',
    'ours_latest': '*',
    'ours_trainable_qk': 'D',
    'ours_fixed_qk': 'X',
}

MODEL_NAMES = {
    'transformer': 'Transformer',
    'mikan': 'MIKAN',
    'performer': 'Performer',
    'rka': 'RKA',
    'gmm_rks': 'GMM-RKS',
    'mgk': 'MGK',
    'kpca_scaled': 'KPCA-Scaled',
    'metala': 'MetaLA',
    'ours_latest': 'Ours (Latest)',
    'ours_trainable_qk': 'Ours (Trainable)',
    'ours_fixed_qk': 'Ours (Fixed QK)',
}


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


def _get_param_breakdown_for_models(results_data: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """根据 benchmark 配置构建模型并统计组件参数"""
    config = results_data["config"]
    d_model = config.get("d_model", 128)
    n_heads = config.get("n_heads", 4)
    n_layers = config.get("n_layers", 2)
    M = config.get("M", 64)
    max_seq_len = max(config.get("seq_lens", [4096]))

    breakdown = {}
    for model_type in results_data["results"].keys():
        model = get_model(
            attention_type=model_type,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            M=M
        )
        breakdown[model_type] = _count_params_by_components(model)
    return breakdown


def load_results(results_file: str) -> Dict[str, Any]:
    """加载基准测试结果"""
    with open(results_file, 'r') as f:
        return json.load(f)


def bytes_to_mb(b: float) -> float:
    """转换为 MB"""
    return b / (1024 * 1024)


def plot_figure3_style(
    results_data: Dict[str, Any],
    output_path: str = "figure3_memory_vs_accuracy_speed.pdf",
    title: str = "Memory vs Performance vs Speed Comparison"
):
    """
    生成类似 Figure 3 的图表
    X轴: Performance (Accuracy) - 这里用 throughput 代替
    Y轴: Peak Memory Consumption
    圆圈大小: Speed (相对于 Softmax)
    """
    results = results_data['results']
    config = results_data['config']
    
    # 选择一个序列长度进行展示（取最大的）
    seq_lens = config['seq_lens']
    target_seq_len = max(seq_lens)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 获取 Transformer 的速度作为基准
    softmax_speed = None
    for r in results.get('transformer', []):
        if r['seq_len'] == target_seq_len and r['success']:
            softmax_speed = r['training_throughput']
            break
    
    if softmax_speed is None:
        softmax_speed = 1.0  # 默认值
    
    # 绘制每个模型
    legend_handles = []
    
    for model_type, result_list in results.items():
        for r in result_list:
            if r['seq_len'] == target_seq_len and r['success']:
                memory = bytes_to_mb(r['training_memory'])
                throughput = r['training_throughput']
                
                # 相对速度（相对于 Softmax）
                relative_speed = throughput / softmax_speed if softmax_speed > 0 else 1.0
                
                # 圆圈大小基于相对速度
                size = max(100, min(2000, relative_speed * 300))
                
                scatter = ax.scatter(
                    throughput, memory,
                    s=size,
                    c=COLORS.get(model_type, '#888888'),
                    marker=MARKERS.get(model_type, 'o'),
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=1.5,
                    label=MODEL_NAMES.get(model_type, model_type)
                )
                
                # 添加模型名称标注
                ax.annotate(
                    MODEL_NAMES.get(model_type, model_type),
                    (throughput, memory),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.8
                )
                
                legend_handles.append(scatter)
                break
    
    ax.set_xlabel('Throughput (samples/sec)', fontsize=12)
    ax.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax.set_title(f'{title}\n(Sequence Length = {target_seq_len})', fontsize=14)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加注释说明
    ax.text(
        0.02, 0.98,
        'Circle size ∝ Relative Speed (vs Softmax)\nLarger = Faster',
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_figure4_memory_vs_seqlen(
    results_data: Dict[str, Any],
    output_path: str = "figure4_memory_vs_seqlen.pdf",
    mode: str = "training",  # "training" or "inference"
    title: str = "Memory vs Sequence Length"
):
    """
    生成类似 Figure 4 的图表
    X轴: Sequence Length
    Y轴: Memory Consumption
    """
    results = results_data['results']
    config = results_data['config']
    seq_lens = config['seq_lens']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    memory_key = 'training_memory' if mode == "training" else 'inference_memory'
    
    for model_type, result_list in results.items():
        x_data = []
        y_data = []
        
        for r in result_list:
            if r['success']:
                x_data.append(r['seq_len'])
                y_data.append(bytes_to_mb(r[memory_key]))
        
        if x_data:
            ax.plot(
                x_data, y_data,
                marker=MARKERS.get(model_type, 'o'),
                color=COLORS.get(model_type, '#888888'),
                linewidth=2,
                markersize=10,
                label=MODEL_NAMES.get(model_type, model_type)
            )
            
            # 标记 OOM 点
            for r in result_list:
                if not r['success'] and 'OOM' in r.get('error_message', ''):
                    ax.scatter(
                        [r['seq_len']], [ax.get_ylim()[1] * 0.95],
                        marker='x',
                        color=COLORS.get(model_type, '#888888'),
                        s=200,
                        linewidths=3
                    )
                    ax.annotate(
                        'OOM',
                        (r['seq_len'], ax.get_ylim()[1] * 0.95),
                        xytext=(0, 10),
                        textcoords='offset points',
                        fontsize=9,
                        ha='center',
                        color=COLORS.get(model_type, '#888888')
                    )
    
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel(f'{"Training" if mode == "training" else "Inference"} Memory (MB)', fontsize=12)
    ax.set_title(f'{title} ({mode.capitalize()} Mode)', fontsize=14)
    
    # 对数刻度可选
    # ax.set_xscale('log', base=2)
    # ax.set_yscale('log')
    
    ax.set_xticks(seq_lens)
    ax.set_xticklabels([str(s) for s in seq_lens])
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10)
    
    # 添加 O(n²) vs O(n) 参考线
    if len(seq_lens) >= 2:
        base_len = seq_lens[0]
        base_memory = None
        
        for model_type in ['transformer']:
            for r in results.get(model_type, []):
                if r['seq_len'] == base_len and r['success']:
                    base_memory = bytes_to_mb(r[memory_key])
                    break
        
        if base_memory:
            # O(n²) 参考线
            ref_y_n2 = [base_memory * (s/base_len)**2 for s in seq_lens]
            ax.plot(seq_lens, ref_y_n2, '--', color='gray', alpha=0.5, linewidth=1, label='O(n²) reference')
            
            # O(n) 参考线
            ref_y_n = [base_memory * (s/base_len) for s in seq_lens]
            ax.plot(seq_lens, ref_y_n, ':', color='gray', alpha=0.5, linewidth=1, label='O(n) reference')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_training_vs_inference_memory(
    results_data: Dict[str, Any],
    output_path: str = "figure_train_vs_inference.pdf",
    title: str = "Training vs Inference Memory Comparison"
):
    """
    对比训练和推理内存消耗 - 突出 Fixed QK 的优势
    """
    results = results_data['results']
    config = results_data['config']
    
    # 选择最大序列长度
    target_seq_len = max(config['seq_lens'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    model_types = list(results.keys())
    x = np.arange(len(model_types))
    width = 0.35
    
    train_memory = []
    inference_memory = []
    model_names = []
    colors = []
    
    for model_type in model_types:
        for r in results[model_type]:
            if r['seq_len'] == target_seq_len:
                if r['success']:
                    train_memory.append(bytes_to_mb(r['training_memory']))
                    inference_memory.append(bytes_to_mb(r['inference_memory']))
                else:
                    train_memory.append(0)
                    inference_memory.append(0)
                model_names.append(MODEL_NAMES.get(model_type, model_type))
                colors.append(COLORS.get(model_type, '#888888'))
                break
    
    # 左图：内存对比柱状图
    bars1 = ax1.bar(x - width/2, train_memory, width, label='Training', color=[c for c in colors], alpha=0.8)
    bars2 = ax1.bar(x + width/2, inference_memory, width, label='Inference', color=[c for c in colors], alpha=0.5, hatch='//')
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Memory (MB)', fontsize=12)
    ax1.set_title(f'Memory Comparison (Seq Len = {target_seq_len})', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars1, train_memory):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 右图：训练/推理内存比率
    ratios = [t/i if i > 0 else 0 for t, i in zip(train_memory, inference_memory)]
    bars3 = ax2.bar(x, ratios, color=colors, alpha=0.8)
    
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Training / Inference Memory Ratio', fontsize=12)
    ax2.set_title('Memory Efficiency (Lower is Better)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars3, ratios):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{val:.2f}x', ha='center', va='bottom', fontsize=9)
    
    # 标记 Ours (Latest)
    if 'ours_latest' in model_types:
        idx = model_types.index('ours_latest')
        ax2.annotate(
            '✓ Ours\n(Latest)',
            (idx, ratios[idx]),
            xytext=(idx + 0.5, ratios[idx] + 0.5),
            fontsize=10,
            color=COLORS['ours_latest'],
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORS['ours_latest'])
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_speed_comparison(
    results_data: Dict[str, Any],
    output_path: str = "figure_speed_comparison.pdf"
):
    """
    速度对比图
    """
    results = results_data['results']
    config = results_data['config']
    seq_lens = config['seq_lens']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：Throughput vs Sequence Length
    for model_type, result_list in results.items():
        x_data = []
        y_data = []
        
        for r in result_list:
            if r['success']:
                x_data.append(r['seq_len'])
                y_data.append(r['training_throughput'])
        
        if x_data:
            ax1.plot(
                x_data, y_data,
                marker=MARKERS.get(model_type, 'o'),
                color=COLORS.get(model_type, '#888888'),
                linewidth=2,
                markersize=8,
                label=MODEL_NAMES.get(model_type, model_type)
            )
    
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Throughput (samples/sec)', fontsize=12)
    ax1.set_title('Training Throughput vs Sequence Length', fontsize=14)
    ax1.set_xticks(seq_lens)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=9)
    
    # 右图：相对速度（相对于最大序列长度下的 Transformer）
    target_seq_len = max(seq_lens)
    
    # 获取基准速度
    softmax_speed = 1.0
    for r in results.get('transformer', []):
        if r['seq_len'] == target_seq_len and r['success']:
            softmax_speed = r['training_throughput']
            break
    
    model_names = []
    relative_speeds = []
    colors = []
    
    for model_type, result_list in results.items():
        for r in result_list:
            if r['seq_len'] == target_seq_len:
                if r['success']:
                    relative_speeds.append(r['training_throughput'] / softmax_speed)
                else:
                    relative_speeds.append(0)
                model_names.append(MODEL_NAMES.get(model_type, model_type))
                colors.append(COLORS.get(model_type, '#888888'))
                break
    
    x = np.arange(len(model_names))
    bars = ax2.bar(x, relative_speeds, color=colors, alpha=0.8)
    
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Relative Speed (vs Softmax)', fontsize=12)
    ax2.set_title(f'Relative Speed Comparison (Seq Len = {target_seq_len})', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Softmax baseline')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars, relative_speeds):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{val:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_parameter_breakdown(
    results_data: Dict[str, Any],
    output_path: str = "figure_parameter_breakdown.pdf"
):
    """
    参数分解图（真实统计） - 展示 Fixed QK 节省的部分
    """
    breakdown = _get_param_breakdown_for_models(results_data)
    model_types = list(breakdown.keys())
    components = ["Q_proj", "K_proj", "V_proj", "O_proj", "QKV", "Spectral", "FFN", "NormPosInputOther"]
    comp_colors = ['#3498DB', '#5DADE2', '#2ECC71', '#27AE60', '#1ABC9C', '#9B59B6', '#E74C3C', '#95A5A6']

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(model_types))
    width = 0.6
    bottom = np.zeros(len(model_types))

    for comp, color in zip(components, comp_colors):
        heights = []
        for model_type in model_types:
            heights.append(breakdown[model_type][comp])

        bars = ax.bar(x, heights, width, bottom=bottom, label=comp, color=color, alpha=0.8)
        bottom += heights

    # 标注 latest 模型中 Q/K 被移除（若配置为 no_qk）
    if "ours_latest" in model_types:
        idx_latest = model_types.index("ours_latest")
        latest_qk = breakdown["ours_latest"]["Q_proj"] + breakdown["ours_latest"]["K_proj"]
        if latest_qk == 0:
            ax.annotate(
                "Q/K removed\n(trainable=0)",
                (idx_latest, bottom[idx_latest]),
                xytext=(idx_latest + 0.3, bottom[idx_latest] * 1.05),
                fontsize=9,
                fontweight='bold',
                color=COLORS['ours_latest'],
                arrowprops=dict(arrowstyle='->', color=COLORS['ours_latest'])
            )

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Trainable Parameters', fontsize=12)
    ax.set_title('Trainable Parameter Breakdown by Component', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES.get(m, m) for m in model_types], rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加注释说明 fixed qk 的优势
    ax.text(
        0.02, 0.98,
        '★ Ours (Latest) saves trainable parameters by:\n'
        '  • Removing Q/K trainable projections\n'
        '  • Keeping kernel spectrum trainable',
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def generate_all_figures(
    results_file: str,
    output_dir: str = "figures"
):
    """生成所有图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("📊 Generating LRA Figures")
    print("=" * 60)
    
    data = load_results(results_file)
    
    # Figure 3 风格
    plot_figure3_style(
        data,
        output_path=os.path.join(output_dir, "figure3_memory_vs_perf_speed.pdf")
    )
    
    # Figure 4 风格 - 训练内存
    plot_figure4_memory_vs_seqlen(
        data,
        output_path=os.path.join(output_dir, "figure4_train_memory_vs_seqlen.pdf"),
        mode="training"
    )
    
    # Figure 4 风格 - 推理内存
    plot_figure4_memory_vs_seqlen(
        data,
        output_path=os.path.join(output_dir, "figure4_inference_memory_vs_seqlen.pdf"),
        mode="inference"
    )
    
    # 训练 vs 推理内存对比
    plot_training_vs_inference_memory(
        data,
        output_path=os.path.join(output_dir, "figure_train_vs_inference.pdf")
    )
    
    # 速度对比
    plot_speed_comparison(
        data,
        output_path=os.path.join(output_dir, "figure_speed_comparison.pdf")
    )
    
    # 参数分解（真实统计）
    plot_parameter_breakdown(
        data,
        output_path=os.path.join(output_dir, "figure_parameter_breakdown.pdf")
    )
    
    print("\n✅ All figures generated!")
    print(f"📁 Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LRA Figures")
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to benchmark results JSON file")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Output directory for figures")
    
    args = parser.parse_args()
    
    generate_all_figures(args.results_file, args.output_dir)
