#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CPU 训练快速测试脚本

用于在没有 GPU 的机器上快速验证训练流程
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import numpy as np

from data import load_dataset_by_name, TextPreprocessor
from models import get_model
from utils import set_seed, format_time


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        logits, kl_div = model(input_ids, attention_mask)
        
        ce_loss = criterion(logits, labels)
        loss = ce_loss + kl_div
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += ce_loss.item()
        
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            logits, kl_div = model(input_ids, attention_mask)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='CPU 训练快速测试')
    parser.add_argument('--dataset', type=str, default='trec', 
                       choices=['rt_polarity', 'sst2', 'sst5', 'trec', 'ag_news', 'dbpedia_14', 'yelp_review_full'],
                       help='数据集')
    parser.add_argument('--model', type=str, default='transformer',
                       choices=['transformer', 'mikan', 'mgk', 'performer', 'rka', 'gmm_rks', 'ours_fixed_qk', 'ours_trainable_qk'],
                       help='模型')
    parser.add_argument('--epochs', type=int, default=3, help='训练 epoch 数')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_samples', type=int, default=1000, help='最大样本数 (用于快速测试)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cpu')
    
    print("=" * 70)
    print("🧪 CPU 训练快速测试")
    print("=" * 70)
    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"最大样本数: {args.max_samples}")
    print(f"设备: {device}")
    print("=" * 70)
    
    # 加载数据
    print("\n📂 加载数据...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data_dir = os.path.abspath(data_dir)
    
    dataset = load_dataset_by_name(args.dataset, data_dir)
    train_texts, train_labels = dataset['train']
    test_texts, test_labels = dataset['test']
    num_classes = dataset['info'].num_classes
    
    # 截断数据用于快速测试
    if args.max_samples and len(train_texts) > args.max_samples:
        indices = np.random.permutation(len(train_texts))[:args.max_samples]
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
    
    if args.max_samples and len(test_texts) > args.max_samples // 5:
        indices = np.random.permutation(len(test_texts))[:args.max_samples // 5]
        test_texts = [test_texts[i] for i in indices]
        test_labels = [test_labels[i] for i in indices]
    
    print(f"   训练样本: {len(train_texts)}")
    print(f"   测试样本: {len(test_texts)}")
    print(f"   类别数: {num_classes}")
    
    # 预处理
    print("\n🔧 预处理文本...")
    preprocessor = TextPreprocessor(
        max_seq_len=128,
        min_freq=2,
        max_vocab_size=20000
    )
    preprocessor.fit(train_texts)
    print(f"   词汇表大小: {preprocessor.vocab_size}")
    
    train_ids, train_mask = preprocessor.transform(train_texts)
    test_ids, test_mask = preprocessor.transform(test_texts)
    
    # 创建 DataLoader
    train_dataset = TensorDataset(
        torch.tensor(train_ids, dtype=torch.long),
        torch.tensor(train_mask, dtype=torch.long),
        torch.tensor(train_labels, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(test_ids, dtype=torch.long),
        torch.tensor(test_mask, dtype=torch.long),
        torch.tensor(test_labels, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    print(f"\n🏗️ 创建模型 {args.model}...")
    model_params = {
        'vocab_size': preprocessor.vocab_size,
        'num_classes': num_classes,
        'embed_dim': 64,
        'hidden_dim': 64,
        'n_heads': 2,
        'n_layers': 1,
        'pf_dim': 128,
        'dropout': 0.1,
        'max_seq_len': 128,
        'pad_idx': preprocessor.vocab.pad_idx,
        'device': 'cpu',
        'M': 32,
        'kl_lambda': 0.001,
    }
    
    model = get_model(args.model, **model_params)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   参数量: {num_params:,}")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    print(f"\n🚀 开始训练...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        
        epoch_time = time.time() - epoch_start
        
        print(f"   Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}% | "
              f"Time: {format_time(epoch_time)}")
    
    total_time = time.time() - start_time
    
    # 最终评估
    print("\n📊 最终评估...")
    final_loss, final_acc = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "=" * 70)
    print("✅ 训练完成!")
    print(f"   最终测试准确率: {final_acc*100:.2f}%")
    print(f"   总训练时间: {format_time(total_time)}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
