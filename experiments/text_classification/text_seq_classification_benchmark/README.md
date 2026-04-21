# Text Sequence Classification Benchmark

用于在多个文本序列分类数据集上统一跑 benchmark 的实验系统。

## 目录结构

```
text_seq_classification_benchmark/
├── __init__.py                 # 包初始化
├── config/
│   └── default_config.yaml     # 默认配置文件
├── data/
│   ├── __init__.py
│   ├── dataset_loader.py       # 数据集加载器
│   └── preprocessing.py        # 文本预处理
├── models/
│   ├── __init__.py
│   ├── base.py                 # 基础模型类
│   ├── transformer.py          # 标准 Transformer
│   ├── mikan.py                # MIKAN (Implicit Kernel Attention)
│   ├── performer.py            # Performer (FAVOR+ linear attention)
│   ├── rka.py                  # RKA (Random Kernel Attention)
│   ├── gmm_rks.py              # GMM-RKS (Gaussian Mixture RKS)
│   └── ours.py                 # 我们的方法 (FRSKA-based)
├── train.py                    # 训练脚本
├── run_benchmark.py            # Benchmark 运行脚本
├── utils.py                    # 工具函数
└── README.md                   # 本文件
```

## 支持的数据集

### 本地数据集

1. **RT Polarity** (`rt_polarity`)
   - 句子级二分类（正面/负面）
   - 默认使用 10-fold 交叉验证

2. **SST-2** (`sst2`)
   - Stanford Sentiment Treebank 二分类
   - 使用官方 train/dev/test 划分

3. **SST-5** (`sst5`)
   - Stanford Sentiment Treebank 五分类
   - 使用官方划分

### HuggingFace 数据集

4. **TREC** (`trec`)
   - 问题分类，6 类

5. **AG News** (`ag_news`)
   - 新闻分类，4 类

6. **DBpedia** (`dbpedia_14`)
   - 本体分类，14 类

7. **Yelp Review Full** (`yelp_review_full`)
   - 评论评分，5 类

## 支持的模型

| 模型名称 | 描述 | 参考 |
|---------|------|------|
| `transformer` | 标准 Transformer (dot-product attention) | Vaswani et al., 2017 |
| `mikan` | MIKAN (Implicit Kernel Attention, IKAN-direct) | - |
| `performer` | Performer (FAVOR+ linear attention) | Choromanski et al., 2020 |
| `rka` | RKA (Random Kernel Attention with learned features) | - |
| `gmm_rks` | GMM-RKS (Gaussian Mixture Random Kitchen Sinks) | - |
| `ours_fixed_qk` | 我们的方法，W_Q 和 W_K 固定 | - |
| `ours_trainable_qk` | 我们的方法，W_Q 和 W_K 可训练 | - |

### 模型对比

| 模型 | 注意力复杂度 | 特征学习 | 特点 |
|------|-------------|---------|------|
| Transformer | O(N²) | - | 标准 softmax attention |
| MIKAN | O(N·M) | 固定/可学习 | Implicit Kernel Attention |
| Performer | O(N·D) | 固定 | 正交随机特征近似 softmax |
| RKA | O(N·M) | 生成器网络 | 学习 random features |
| GMM-RKS | O(N·D) | GMM 参数 | 学习频谱分布 |
| Ours | O(N·M) | Normalizing Flow | 学习复杂频谱分布 |

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy scikit-learn pyyaml datasets
# 可选：安装 normflows（用于 FRSKA）
pip install normflows
```

### 2. 运行单个实验

```bash
# 在 RT Polarity 上训练 Transformer
python train.py --dataset rt_polarity --model transformer --seed 0 --use_cv

# 在 SST-2 上训练 MIKAN
python train.py --dataset sst2 --model mikan --seed 0

# 在 AG News 上训练 Performer
python train.py --dataset ag_news --model performer --seed 0

# 在 TREC 上训练 GMM-RKS
python train.py --dataset trec --model gmm_rks --seed 0
```

### 3. 运行完整 Benchmark

```bash
# 运行所有数据集和模型（包括新增的 baseline）
python run_benchmark.py --seeds 0 1 2 3 4

# 快速模式（只运行小数据集）
python run_benchmark.py --quick

# 指定数据集和模型
python run_benchmark.py --datasets rt_polarity sst2 --models transformer mikan performer

# 只运行新增的 baseline
python run_benchmark.py --datasets sst2 trec --models performer rka gmm_rks
```

## 配置文件

编辑 `config/default_config.yaml` 来修改默认参数：

```yaml
# 模型超参数
model_params:
  embed_dim: 128
  hidden_dim: 128
  n_heads: 4
  n_layers: 2
  pf_dim: 256
  dropout: 0.1
  max_seq_len: 256
  M: 64                 # 随机特征数量
  kl_lambda: 0.001      # KL 散度权重

# 训练超参数
training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  n_epochs: 50
  patience: 10          # Early stopping
  scheduler: cosine
  gradient_clip: 1.0
```

## 输出结果

运行完成后，结果保存在 `benchmark_results/run_YYYYMMDD_HHMMSS/` 目录：

- `all_results.csv`: 所有实验的详细结果
- `summary.csv`: 汇总统计（mean ± std）
- `results_table.tex`: LaTeX 格式的论文表格
- `config.yaml`: 使用的配置文件
- `benchmark_*.log`: 运行日志

### 示例输出表格

```
======================================================================================================
实验结果汇总
======================================================================================================
Dataset              Model                Accuracy             F1-Macro             Time            #Params        
------------------------------------------------------------------------------------------------------
rt_polarity          transformer          76.32 ± 1.23         76.30 ± 1.25         45s             125,440        
rt_polarity          mikan                77.15 ± 1.18         77.12 ± 1.20         52s             158,720        
rt_polarity          performer            75.89 ± 1.35         75.85 ± 1.38         38s             122,880        
rt_polarity          rka                  76.78 ± 1.22         76.74 ± 1.25         48s             145,920        
rt_polarity          gmm_rks              77.02 ± 1.15         76.98 ± 1.18         50s             152,320        
rt_polarity          ours_fixed_qk        77.89 ± 1.05         77.86 ± 1.08         68s             185,600        
rt_polarity          ours_trainable_qk    78.21 ± 0.98         78.18 ± 1.01         72s             201,984        
======================================================================================================
```

## 注意事项

1. **数据目录**: 本地数据集（RT Polarity, SST）需要放在 `../data/` 目录下
2. **HuggingFace 数据集**: 首次运行会自动下载并缓存
3. **GPU 支持**: 使用 `--gpu 0` 指定 GPU
4. **内存**: 大数据集（如 Yelp）可能需要减小 batch_size

## 扩展

### 添加新数据集

在 `data/dataset_loader.py` 中添加新的加载函数，并更新 `DATASET_CONFIG`。

### 添加新模型

1. 在 `models/` 目录下创建新的模型文件
2. 继承 `BaseClassifier` 类
3. 在 `models/__init__.py` 中注册模型

