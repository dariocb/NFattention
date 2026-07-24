# 长文本分类实验 (Long-Sequence Text Classification)

在 **两个长文本数据集** 上评估 FSKA 模型与论文全部 baseline，统一使用
`max_seq_len = 1024`（FSKA 短文本实验默认 256）。

## 数据集

| 数据集 | 任务 | 类别 | train / val / test | median 词数 | 来源 |
|---|---|---|---:|---:|---|
| **hyperpartisan** | 长新闻党派性判断 | 2 (binary) | 516 / 64 / 65 | **425** | RFGPA 已下载的本地 jsonl (`jonathanli/hyperpartisan-longformer-split`) |
| **imdb** | 影评情感 | 2 (binary) | ~22.5k / ~2.5k / 25k | ~174 | Stanford `aclImdb_v1.tar.gz` |

> dev.jsonl 即 hyperpartisan 的验证集；IMDB 无官方验证集，由 `dataset_loader`
> 从 train 按 `val_ratio` 分层切出（与其它 HF 数据集一致）。

### 为什么是这两个 / 为什么 1024

各数据集训练集 **按空格分词的词数** 实测对比：

| 数据集 | mean | median | p95 | max | >256 | >1024 |
|---|---:|---:|---:|---:|---:|---:|
| 短文本 (trec…ag_news…dbpedia) | 10–46 | 10–46 | ≤80 | ≤1484 | 0% | 0% |
| yelp_review_full | 134 | 99 | 370 | 1052 | 12% | 0% |
| **hyperpartisan** | **566** | **425** | **1522** | 4431 | **72%** | **12%** |
| **imdb** | ~231 | ~174 | — | — | ~30% | — |
| ~~20news (去泄露)~~ | 191 | 86 | 584 | 11765 | 15% | 2% |

- hyperpartisan 是真·长文本（72% 超过 256 token，是唯一在 >1024 仍有显著占比的数据集）。
- imdb 干净、中长且均匀；优于去泄露后 median 仅 86 词的 20news。
- 取 `max_seq_len=1024` 覆盖 hyperpartisan 的主体长度。

## 模型 (10 个)

- **ours (hybrid)**：`ours_hybrid_noprior`、`ours_hybrid_ngsm`、`ours_hybrid_rbf`
- **baseline (论文全部)**：`transformer`、`mikan`、`performer`、`rka`、`gmm_rks`、
  `kpca_scaled`、`metala`

所有模型经由共享的 `text_seq_classification_benchmark/run_benchmark.py` 运行，
按模型自动构造超参（hybrid 的 `num_flows/flow_hidden_dim/num_mixtures` 取默认 3/64/10）。

## 运行

```bash
cd new_experiment/long_seq_classification

# 1) 仅准备数据（幂等；复制 hyperpartisan jsonl + 下载解析 IMDB）
/home/kyzhang/miniconda3/envs/py310/bin/python prepare_data.py

# 2) 全量实验（10 模型 × 2 数据集，seed 42, max_seq_len 1024）
bash run.sh
# 可选: GPU=1 SEED=0 MAX_SEQ_LEN=512 bash run.sh
```

结果写入 `results/run_<timestamp>/`（含 `results_table.tex`、各实验 json、`config.yaml`、
复现 manifest）。

## 实现要点

- 数据加载统一加在共享的
  `experiments/text_classification/text_seq_classification_benchmark/data/dataset_loader.py`：
  - `load_hyperpartisan()` 读本地 jsonl（标签 `"true"/"false"` → 1/0）。
  - `imdb` 复用既有 `load_local_hf_dataset`（读 `data/hf_datasets/imdb/data.json`），
    **运行期不依赖 `datasets` 库**。
- `run_benchmark.py` 新增 `--max_seq_len` 覆盖项，并把两个数据集登记进 `DATASET_CONFIG`。
- 模型位置编码为正弦式 `PositionalEncoding(embed_dim, max_seq_len)`，1024 无需改结构。
