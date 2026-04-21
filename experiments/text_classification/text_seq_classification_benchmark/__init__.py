"""
Text Sequence Classification Benchmark
=====================================
用于在多个文本序列分类数据集上统一跑 benchmark 的实验系统。

支持的数据集：
- RT Polarity (二分类)
- SST-2 (二分类)
- SST-5 (五分类)
- TREC (六分类)
- AG News (四分类)
- DBpedia (14分类)
- Yelp Review Full (五分类)

支持的模型：
- Transformer (标准多头自注意力)
- MIKAN (Implicit Kernel Attention)
- Ours (fixed_qk / trainable_qk)
"""

__version__ = "1.0.0"

