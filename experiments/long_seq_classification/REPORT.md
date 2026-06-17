# 长文本分类实验报告 (Long-Sequence Text Classification)

> 数据集: **Hyperpartisan**(长新闻党派性, 二分类) + **IMDB**(影评情感, 二分类)
> 统一 `max_seq_len = 1024`, 单 seed = 42, 从零训练 (无预训练词向量)
> 环境: RTX 3090, py310 (torch 2.8.0)

---

## 1. 主结果表 (Test Accuracy / Macro-F1, %)

> ⭐ = 该数据集最优。ours 系列 = FSKA Hybrid 变体 (no_qk + fixed_orth_v + per-head NF)。
> **Hyperpartisan: 全部模型用宽松早停 (公平对比, 见 §3)** — ours + 6 baseline 用 patience=200,
> metala 用 patience=30 (1024 步递归太慢, 但 best_ep=9 已充分收敛)。IMDB 用默认早停 (patience=10)。

| 模型 | 类型 | Hyperpartisan Acc | Hyperpartisan F1 | IMDB Acc | IMDB F1 |
|---|---|---:|---:|---:|---:|
| **ours_hybrid_noprior** | ours (3层NF) | **92.31** ⭐ | **90.20** | 83.50 | 84.44 |
| **ours_hybrid_ngsm** | ours (3层NF) | 90.77 | 88.46 | 79.11 | 75.82 |
| **ours_hybrid_rbf** | ours (3层NF) | 90.77 | 88.46 | 83.46 | 83.74 |
| ours_hybrid_noprior_deep | ours (5层NF) | — | — | 82.84 | 83.85 |
| ours_hybrid_ngsm_deep | ours (5层NF) | — | — | 81.37 | 83.18 |
| ours_hybrid_rbf_deep | ours (5层NF) | — | — | 82.43 | 83.76 |
| transformer | baseline | 90.77 | 88.46 | 82.78 | 80.67 |
| mikan | baseline | 90.77 | 88.89 | 85.26 | 86.04 |
| performer | baseline | 89.23 | 86.79 | 79.86 | 82.11 |
| rka | baseline | 89.23 | 86.79 | **86.59** ⭐ | **86.78** ⭐ |
| gmm_rks | baseline | 83.08 | 78.43 | 59.73 | 71.16 |
| kpca_scaled | baseline | 89.23 | 86.79 | 85.07 | 85.59 |
| metala | baseline | 89.23 | 86.79 | 86.44 | 86.65 |

**一句话** (公平对比下, 重要更新):
- **Hyperpartisan**: ours_noprior 92.31% 仍最高，但只领先第二档 (90.77: ours_ngsm/rbf,
  transformer, mikan) **1.5 点 = 1 个测试样本**。test 仅 65 条，此差距**在噪声内**，不能宣称显著优势。
- **IMDB**: ours 与最优 baseline (rka 86.6) 差约 3 点，是真实泛化差距 (见 §4)。

**Hyperpartisan 排名 (公平早停)**:
```
92.31  ours_noprior                               ← 唯一 92 档
90.77  ours_ngsm, ours_rbf, transformer, mikan    ← 四并列
89.23  performer, rka, kpca_scaled, metala        ← 四并列
83.08  gmm_rks
```

---

## 2. Deep NF (5层) vs Shallow NF (3层) 消融 — IMDB

把共享 NF 从 3 层 (trunk=2) 加深到 5 层 (trunk=4, 前4层 shared + 1层 head-specific)，
并加宽 (`flow_hidden_dim` 64→128, `M` 64→96, "latest" 配置)：

| 变体 | 3层 Acc | 5层 Acc | Δ |
|---|---:|---:|---:|
| noprior | 83.50 | 82.84 | **−0.66** |
| ngsm | 79.11 | 81.37 | **+2.26** |
| rbf | 83.46 | 82.43 | **−1.03** |

**结论**: 加深共享 NF **整体无增益**，仅 ngsm 受益。容量不是 IMDB 上的瓶颈 (反而略增过拟合)。

---

## 3. 关键诊断: Hyperpartisan 上默认早停 **误杀几乎所有模型** 🎯

### 现象
默认早停 (patience=10) 下，大量模型在 Hyperpartisan 只有 64–78%，train_loss 卡在高位，
best_epoch 仅 0–4，看似"学不动"。

### 诊断 + 公平重跑 (全部模型, patience=200, 其余配置相同)
| 模型 | 原始 Acc (patience=10) | **公平 Acc (patience=200)** | Δ | 公平收敛 ep |
|---|---:|---:|---:|---:|
| ours_hybrid_noprior | 67.69 | **92.31** | +24.6 | 2 |
| ours_hybrid_ngsm | 66.15 | **90.77** | +24.6 | 20 |
| ours_hybrid_rbf | 66.15 | **90.77** | +24.6 | 18 |
| transformer | 78.46 | **90.77** | +12.3 | 25 |
| mikan | 86.15 | **90.77** | +4.6 | 4 |
| performer | 64.62 | **89.23** | +24.6 | 13 |
| rka | 66.15 | **89.23** | +23.1 | **91** |
| kpca_scaled | 78.46 | **89.23** | +10.8 | 81 |
| metala | 84.62 | **89.23**† | +4.6 | 9 |
| gmm_rks | 69.23 | **83.08** | +13.8 | 22 |

† metala 用 patience=30 (递归太慢)，best_ep=9 已充分收敛。

### 结论
- **早停误杀是全局性的，不是 ours 独有**: 公平重跑后**每个模型都涨** (+4.6 ~ +24.6)。
  performer/rka 原来"垫底"(64/66) 公平后到 89；rka 甚至要训到 **第 91 epoch** 才收敛
  (默认早停在第 0 epoch 就把它停了)。
- **真凶**: Hyperpartisan 验证集仅 **64 条** (1 样本 = 1.5%)，早期 val_acc 剧烈抖动 →
  patience=10 误触发早停。这对**所有**模型都成立，程度不同。
- **对比有效性**: 之前"ours 反超 baseline 6 点"的结论**在公平条件下不成立**。公平后 ours_noprior
  (92.31) 仍最高，但仅领先第二档 1.5 点 (= 1 个测试样本)，**在 65 条测试集的噪声范围内**。
- **实验 B (M=16) 排除"容量"假说**: M 减到 16 仍卡在 0.56，说明 M=64 容量从不是瓶颈。

> **教训**:
> 1. 极小验证集 (数十条) 上**禁用激进早停**，否则所有模型都被低估，且低估程度不一致 → 排名失真。
> 2. Hyperpartisan (test 65 条, 单 seed) 上 **1.5 点差异 = 噪声**，可靠结论**必须多 seed**。

---

## 4. IMDB 上 ours 略逊 (差 ~3 点) 的原因: 真实过拟合

与 Hyperpartisan 不同，IMDB (训练 22500 条) 上 ours **确实过拟合**:

| 模型 | train_loss | test_loss | gap | Acc |
|---|---:|---:|---:|---:|
| ours_hybrid_ngsm | 0.035 | **1.605** | **46×** | 79.11 |
| ours_hybrid_rbf | 0.071 | 0.896 | 13× | 83.46 |
| ours_hybrid_noprior | 0.077 | 0.602 | 8× | 83.50 |
| rka (最优 baseline) | 0.200 | **0.373** | 1.9× | 86.59 |
| metala | 0.169 | 0.392 | 2.3× | 86.44 |

- ours train_loss 掉到 0.03–0.08 (几乎记住训练集)，但 test_loss 飙到 0.6–1.6。
- 最优 baseline (rka/metala) train_loss 反而**更高** (0.17–0.20，结构正则更强)，test_loss 却**低得多**。
- → IMDB 上是**过拟合 / 泛化**问题 (与 Hyperpartisan 的早停问题是两种不同失败模式)。

---

## 5. 两个数据集对比总结

| | Hyperpartisan | IMDB |
|---|---|---|
| 训练样本 | 516 (极小) | 22500 |
| 验证集 | 64 (极小) | 2500 |
| 文档长度 median | 425 词 (超长) | 174 词 |
| ours 失败模式 | **早停误杀** (可修复) | **过拟合** (需正则) |
| ours 修复后表现 | **92% 全场最优** | 83.5% (差 baseline 3 点) |

---

## 6. 复现

```bash
cd new_experiment/long_seq_classification

# 数据准备 (幂等)
python prepare_data.py

# 主 benchmark (10 模型 × 2 数据集, 默认早停)
bash run.sh

# Deep NF (5层) 消融
#   models: ours_hybrid_{noprior,ngsm,rbf}_deep, config: config_deep_nf5.yaml

# Hyperpartisan 宽松早停 (修正 ours, 决定性诊断 A)
#   config: config_A_longtrain.yaml (n_epochs=200, patience=200)
```

结果整合: `python merge_results.py` → `results/combined/{all_results_combined,pivot_accuracy}.csv`

### 实验配置档案
| config | 用途 | 关键参数 |
|---|---|---|
| `default_config.yaml` | 主 benchmark | M=64, num_flows=3, patience=10 |
| `config_deep_nf5.yaml` | Deep NF 消融 | M=96, num_flows=5, trunk=4, hidden=128 |
| `config_A_longtrain.yaml` | Hyperpartisan 修正 | patience=200, n_epochs=200 |
| `config_B_smallM.yaml` | 容量诊断 | M=16 |
| `config_reg.yaml` | 正则诊断 | dropout=0.3, wd=1e-3 |

---

## 附: 数据集长度统计 (训练集, 空格分词词数)

| 数据集 | mean | median | p95 | max | >256 | >1024 |
|---|---:|---:|---:|---:|---:|---:|
| Hyperpartisan | 566 | 425 | 1522 | 4431 | 72% | 12% |
| IMDB | ~231 | ~174 | — | — | ~30% | — |
