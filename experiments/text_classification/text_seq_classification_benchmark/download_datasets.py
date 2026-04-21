#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载所有 HuggingFace 数据集到本地

运行此脚本后，将 data 文件夹上传到云端服务器即可
"""

import os
import json
import sys
from tqdm import tqdm

def download_all_datasets():
    """下载所有数据集"""
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("请先安装 datasets: pip install datasets")
        return False
    
    # 输出目录
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "hf_datasets")
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据集配置
    datasets_config = {
        "trec": {
            "path": "trec",
            "text_field": "text",
            "label_field": "coarse_label",
            "num_classes": 6,
            "class_names": ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]
        },
        "ag_news": {
            "path": "ag_news",
            "text_field": "text",
            "label_field": "label",
            "num_classes": 4,
            "class_names": ["World", "Sports", "Business", "Sci/Tech"]
        },
        "dbpedia_14": {
            "path": "dbpedia_14",
            "text_field": "content",
            "label_field": "label",
            "num_classes": 14,
            "class_names": [f"class_{i}" for i in range(14)]
        },
        "yelp_review_full": {
            "path": "yelp_review_full",
            "text_field": "text",
            "label_field": "label",
            "num_classes": 5,
            "class_names": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
        }
    }
    
    print("=" * 70)
    print("📥 下载 HuggingFace 数据集到本地")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    print("=" * 70)
    
    for ds_name, config in datasets_config.items():
        print(f"\n📂 下载 {ds_name}...")
        
        ds_dir = os.path.join(output_dir, ds_name)
        os.makedirs(ds_dir, exist_ok=True)
        
        try:
            # 加载数据集
            print(f"   从 HuggingFace 加载...")
            try:
                dataset = load_dataset(config["path"], trust_remote_code=True)
            except TypeError:
                # 旧版本 datasets 不支持 trust_remote_code 参数
                dataset = load_dataset(config["path"])
            
            # 提取训练集
            print(f"   处理训练集...")
            train_texts = []
            train_labels = []
            for example in tqdm(dataset["train"], desc="   Train"):
                train_texts.append(example[config["text_field"]])
                train_labels.append(example[config["label_field"]])
            
            # 提取测试集
            print(f"   处理测试集...")
            test_texts = []
            test_labels = []
            for example in tqdm(dataset["test"], desc="   Test"):
                test_texts.append(example[config["text_field"]])
                test_labels.append(example[config["label_field"]])
            
            # 保存为 JSON
            data = {
                "name": ds_name,
                "num_classes": config["num_classes"],
                "class_names": config["class_names"],
                "train": {
                    "texts": train_texts,
                    "labels": train_labels
                },
                "test": {
                    "texts": test_texts,
                    "labels": test_labels
                }
            }
            
            output_file = os.path.join(ds_dir, "data.json")
            print(f"   保存到 {output_file}...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            
            # 计算文件大小
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            
            print(f"   ✅ 完成! 训练: {len(train_texts)}, 测试: {len(test_texts)}, 文件大小: {file_size:.1f} MB")
            
            # 保存元信息
            meta = {
                "name": ds_name,
                "num_classes": config["num_classes"],
                "class_names": config["class_names"],
                "train_size": len(train_texts),
                "test_size": len(test_texts),
                "text_field": config["text_field"],
                "label_field": config["label_field"]
            }
            
            with open(os.path.join(ds_dir, "meta.json"), 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 70)
    print("✅ 所有数据集下载完成!")
    print(f"📁 数据保存在: {output_dir}")
    print("\n请将以下目录上传到云端服务器:")
    print(f"   {output_dir}")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = download_all_datasets()
    sys.exit(0 if success else 1)

