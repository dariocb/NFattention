# 项目设置与改动记录 (SETUP / 交接文档)

> 本文件记录「长文本分类实验」(`long_seq_classification`) 的全部代码改动、新增文件、
> 实验设计与配置，方便后续接手与进一步修改。
> 实验科学结论见同目录 **REPORT.md**；快速上手见 **README.md**。

---

## 0. 一句话背景

给 FSKA 文本分类 benchmark 加了 **2 个长文本数据集** (Hyperpartisan + IMDB)，统一在
`max_seq_len=1024` 下评估 FSKA Hybrid 模型 vs 全部 baseline，并做了 Deep-NF 消融与
**早停公平性修正** (关键发现：默认早停在小验证集上误杀几乎所有模型，见 REPORT.md §3)。

环境固定为 conda env **`py310`**：`/home/kyzhang/miniconda3/envs/py310/bin/python`
(torch 2.8.0 + sklearn，**无 `datasets` 库** → 数据全部走本地文件，不依赖 HF 下载)。

---

## 1. 改动过的「共享代码」文件 (会影响整个 benchmark)

路径前缀: `experiments/text_classification/text_seq_classification_benchmark/`

### 1.1 `data/dataset_loader.py` — 注册两个数据集
| 行 | 改动 |
|---|---|
| `_normalize_label()` (~L184) | 标签归一化 (`"true"/"false"/0/1/bool → int`)，移植自 RFGPA |
| `load_hyperpartisan()` (~L201) | 读本地 jsonl `<data_dir>/hyperpartisan/{train,dev,test}.jsonl`，dev=val |
| `dataset_configs` 加 `"imdb"` (~L355) | 让 imdb 走既有 HF-local 路径 (`text`/`label`, 2类) |
| `load_dataset_by_name` 加 `elif "hyperpartisan"` (~L510) | 构造 DatasetInfo (2类, binary) |
| `elif ... in [..., "imdb"]` (~L526) | imdb 复用 `load_local_hf_dataset` + 从 train 切 val |
| `dataset_info_map` 加 `"imdb"` (~L583) | IMDB DatasetInfo (`["neg","pos"]`) |
| `get_available_datasets()` (~L612) | 追加 `"hyperpartisan"`, `"imdb"` |

> imdb 不需要新 loader 函数：staging 成 `data/hf_datasets/imdb/data.json` 后，
> 既有的 `load_local_hf_dataset` (L184 附近) 直接读 → **运行期零新依赖**。

### 1.2 `run_benchmark.py` — 多模型驱动 (主用脚本)
| 行 | 改动 |
|---|---|
| `DATASET_CONFIG` 加 hyperpartisan / imdb (L58, L62) | 登记两个数据集 (use_cv=False) |
| `--max_seq_len` 参数 (L276) | CLI 覆盖 max_seq_len (长文本传 1024) |
| 应用覆盖 (L324-326) | 加载 config 后写入 `config['model_params']['max_seq_len']` |

### 1.3 `models/__init__.py` — 新增 3 个 Deep-NF 模型
在 `model_map` 里新增 (L115-145，与原 hybrid 模型并列，**没改原有模型**)：
- `ours_hybrid_noprior_deep` / `ours_hybrid_ngsm_deep` / `ours_hybrid_rbf_deep`
- 与对应非-deep 版唯一区别：**`trunk_layers=2 → 4`** (5层NF = 前4层 shared + 1层 head-specific)
- 其余超参 (num_flows/flow_hidden_dim/M) 由 config 传入

### 1.4 `new_experiment/text_classification_hybrid_noprior/run_text_hybrid.py` — ours 单模型快测
| 行 | 改动 |
|---|---|
| `--datasets` choices 加 hyperpartisan/imdb (L390) | |
| `--max_seq_len` 参数 + 应用 (L403, L419) | |

> 此脚本只跑单个 ours 模型，本项目主用的是 `run_benchmark.py`。

---

## 2. 新建文件 (全部在 `new_experiment/long_seq_classification/`)

### 脚本
| 文件 | 作用 |
|---|---|
| `prepare_data.py` | **幂等** 数据准备：复制 RFGPA 的 hyperpartisan jsonl + 下载解析 Stanford IMDB → `data/hf_datasets/imdb/data.json`。`--force` 重生成，`--only X` 单独准备 |
| `run.sh` | 主 benchmark：10 模型 × 2 数据集 (默认早停)。`GPU=/SEED=/MAX_SEQ_LEN=` 可覆盖 |
| `gpu4_relaunch.sh` | (历史) 把 metala 单独 capped 跑的脚本，主流程不再用 |
| `merge_results.py` | **整合所有结果** → `results/combined/{all_results_combined,pivot_accuracy}.csv`。自动扫描 results 子目录，并对 **Hyperpartisan 用公平早停结果覆盖** (见 §4) |

### 配置档案 (yaml)
| config | 用途 | 关键参数 |
|---|---|---|
| (基线) `BENCH/config/default_config.yaml` | 主 benchmark | M=64, num_flows=3, **patience=10** |
| `config_deep_nf5.yaml` | Deep-NF 消融 | M=96, num_flows=5, trunk=4, hidden=128, max_seq_len=1024 |
| `config_A_longtrain.yaml` | **公平早停** (ours+6 baseline) | **patience=200, n_epochs=200**, M=64 |
| `config_metala_fair.yaml` | metala 公平早停 (递归慢) | **patience=30**, n_epochs=100 |
| `config_B_smallM.yaml` | 容量诊断 | M=16 (验证容量非瓶颈) |
| `config_reg.yaml` | 正则诊断 | dropout=0.3, wd=1e-3 |
| `config_metala_cap10.yaml` | (历史) metala capped | n_epochs=10 |

### 文档
- `README.md` — 数据集说明 + 快速运行
- `REPORT.md` — 科学结论 (主表/Deep消融/早停诊断/过拟合分析)
- `SETUP.md` — 本文件

---

## 3. 数据落地位置 (已 staging，勿删)

```
experiments/text_classification/data/
├── hyperpartisan/{train,dev,test}.jsonl      ← 来自 RFGPA (516/64/65)
└── hf_datasets/imdb/{data.json,meta.json}    ← Stanford aclImdb (25k/25k, ~63MB)
```
重新生成：`python prepare_data.py --force`

---

## 4. ⚠️ 结果整合的「公平早停」逻辑 (重要，改动前必读)

`merge_results.py` 对 Hyperpartisan 做了**特殊覆盖**，因为默认早停 (patience=10) 在
64 条验证集上误杀模型 (REPORT.md §3)。整合时：

- **主结果目录** (`gpu0/gpu4/deep_*`)：自动扫描，提供 IMDB 全部 + Deep 消融
- **公平早停目录** (`diag_A_longtrain` + `fair_g0/fair_g3/fair_metala`)：**覆盖** Hyperpartisan
  上的 ours + baseline 结果 (这些是 patience=200/30 重跑的公平值)
- 代码里以 `DIAG_PREFIXES = ("diag_","reg_","fair_")` 把这些目录排除出主扫描，再在 fair 块显式加入并靠 dedup 覆盖

→ **若以后重跑 Hyperpartisan，务必用宽松早停** (config_A)，否则结果失真。

---

## 5. 结果目录速查 (`results/`)

| 目录 | 内容 | 早停 |
|---|---|---|
| `gpu0/` `gpu4/` | 初版 10 模型 × 2 数据集 (含 IMDB 权威结果) | 默认 p=10 |
| `deep_noprior/` `deep_ngsm/` `deep_rbf/` | Deep-NF 5层 (IMDB 消融) | 默认 |
| `diag_A_longtrain/` | ours 3个 @ Hyperpartisan **公平** | p=200 |
| `fair_g0/` `fair_g3/` `fair_metala/` | 6+1 baseline @ Hyperpartisan **公平** | p=200 / p=30 |
| `diag_B_smallM/` `reg_hyperpartisan/` | 诊断实验 (容量/正则) | — |
| `combined/` | **最终整合表** (merge_results.py 产出) | 混合 |

每个 `run_<ts>/` 内含：`all_results.csv`、`results_<dataset>.csv`、`config.yaml`、
`results_table.tex`、`repro_manifest.json`。

---

## 6. 常见后续改动怎么做

| 想做的事 | 怎么改 |
|---|---|
| **多 seed** (最该做，小数据集必需) | `run_benchmark.py --seeds 0 1 2 ...`；merge 已按 (ds,model,seed,fold) 去重，需扩展透视表做 mean±std |
| 加新数据集 | 仿 §1.1 在 `dataset_loader.py` 加 loader + 注册；本地文件优先 (避开无 `datasets` 库) |
| 加新模型 | 在 `models/__init__.py` 的 `model_map` 加一项；模型类需吃 `**kwargs` |
| 调 NF 深度/宽度 | 改 config 的 `num_flows`/`flow_hidden_dim`/`M`；改 `trunk_layers` 需在 model_map 新建模型名 (像 `_deep`) |
| 公平重跑全部模型 | 用 `config_A_longtrain.yaml` (慢模型如 metala 用 `config_metala_fair.yaml`)，再 `merge_results.py` |
| 换 GPU | `run.sh` 用 `GPU=N`；手动跑用 `CUDA_VISIBLE_DEVICES=N ... --gpu 0`。注意 GPU6/7 曾掉卡，用前先 `nvidia-smi -i N` 探活 |

---

## 7. 关键科学结论 (摘要，详见 REPORT.md)

1. **Hyperpartisan 上默认早停误杀几乎所有模型** (验证集仅 64 条)。公平重跑后每个模型 +4.6~+24.6 点。
2. **公平条件下** ours_noprior 92.31% 仍最高，但仅领先第二档 (90.77) 1.5 点 = 1 个测试样本 → **噪声内，需多 seed**。
3. **IMDB** (test 25k 可信)：ours 与最优 baseline (rka 86.6) 差 ~3 点，是**真实过拟合** (train_loss 0.07 vs test 0.6)。
4. **Deep-NF (5层) 在 IMDB 上整体无增益** → 容量不是瓶颈 (M=16 诊断佐证)。
