# LRA (Long Range Arena) Style Memory Benchmark

这个模块用于生成类似论文 Figure 3 和 Figure 4 风格的内存/性能对比图。

## 🎯 核心功能

1. **Memory vs Sequence Length**: 测量不同序列长度下的内存消耗
2. **Training vs Inference Memory**: 对比训练和推理时的内存使用
3. **Speed Comparison**: 速度（throughput）对比
4. **Fixed QK Advantage**: 突出展示 Fixed QK 方法的内存优势

## 📁 文件结构

```
LRA/
├── __init__.py
├── models.py              # 模型定义（各种注意力机制）
├── memory_benchmark.py    # 内存基准测试
├── visualize.py           # 可视化（生成论文风格图表）
├── run_full_benchmark.py  # 完整基准测试入口
├── quick_test.py          # 快速验证脚本
└── results/               # 结果输出目录
    ├── figures/           # 生成的图表
    └── *.json             # 原始数据
```

## 🚀 使用方法

### 本地快速测试（CPU）

```bash
# 验证代码可以运行
python quick_test.py

# 运行快速测试版基准测试
python run_full_benchmark.py --device cpu --quick_test
```

### 云端完整测试（GPU）

```bash
# 默认配置
python run_full_benchmark.py --device cuda

# 自定义配置
python run_full_benchmark.py \
    --device cuda \
    --models softmax performer gmm_rks ours_trainable ours_fixed \
    --seq_lens 256 512 1024 2048 4096 8192 \
    --batch_size 4 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 2 \
    --M 64
```

### 仅生成图表

如果已有 benchmark 结果 JSON 文件：

```bash
python visualize.py --results_file results/memory_benchmark_xxx.json --output_dir figures
```

## 📊 生成的图表

1. **figure3_memory_vs_perf_speed.pdf** - 类似论文 Figure 3
   - X轴: Throughput
   - Y轴: Peak Memory
   - 圆圈大小: 相对速度

2. **figure4_train_memory_vs_seqlen.pdf** - 类似论文 Figure 4
   - X轴: Sequence Length  
   - Y轴: Training Memory

3. **figure_train_vs_inference.pdf** - 训练/推理内存对比
   - 突出 Fixed QK 的优势

4. **figure_speed_comparison.pdf** - 速度对比

5. **figure_memory_breakdown.pdf** - 内存分解图

## 🔬 测试的模型 (与 seq_classification 一致)

| 模型 | 注意力复杂度 | 特点 |
|------|------------|------|
| `transformer` | O(n²) | 标准 Softmax Transformer |
| `mikan` | O(n²) | MIKAN (Implicit Kernel Attention) |
| `performer` | O(n) | Performer (FAVOR+) |
| `rka` | O(n) | RKA (Random Kernel Attention) |
| `gmm_rks` | O(n) | GMM-RKS (Gaussian Mixture) |
| `ours_trainable_qk` | O(n) | 我们的方法 (可训练 QK) |
| `ours_fixed_qk` | O(n) | **我们的方法 (固定 QK)** - 节省内存！|

## ⭐ Fixed QK 的优势

**核心发现**: Fixed QK 通过冻结 Q 和 K 的投影层：

1. **减少可训练参数**: ~15% 参数节省
2. **减少训练内存**: 不需要存储 Q/K 激活值用于反向传播
3. **更快训练**: 减少梯度计算
4. **性能不降**: 通过学习谱分布补偿

```
Standard Method:
├── Store Q activations for backward
├── Store K activations for backward
└── Compute Q/K gradients

Fixed QK Method:
├── Q computed with no_grad (no storage)
├── K computed with no_grad (no storage)
└── Only spectral parameters need gradients
```

## 📝 云端部署

上传整个 `LRA/` 文件夹到云端：

```bash
scp -P <port> -r LRA/ root@<server>:/mnt/NFAttention/
```

在云端运行：

```bash
cd /mnt/NFAttention/LRA
python run_full_benchmark.py --device cuda
```

