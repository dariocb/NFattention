"""
MNIST分类任务消融实验：所有模型对比
使用完整MNIST数据集（60000训练 + 10000测试）
包含固定QK、固定V、固定QKV的对比

包含以下模型：
1. Transformer (dot)
2. IKAN-direct
3. MIKAN
4. FRSKA (独立Flow)
5. FRSKA (共享NF)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import os
import time
import matplotlib.pyplot as plt
import sys

# 进度条
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("警告: tqdm未安装，将使用简单进度显示")

try:
    from torchvision import datasets, transforms
    HAS_TORCHVISION = True
except:
    HAS_TORCHVISION = False
    try:
        from sklearn.datasets import fetch_openml
        print("使用sklearn加载MNIST数据...")
    except:
        print("警告: 无法加载torchvision或sklearn，将使用随机数据作为测试")

from models.baseTransformer import baseTransformer

def load_mnist_data(use_full_data=True, n_samples=10000, random_seed=42, device='cpu'):
    """
    加载MNIST数据集
    use_full_data: 如果为True，使用完整数据集（60000训练 + 10000测试）
    n_samples: 如果use_full_data为False，从完整数据集中采样的样本数（70%训练/30%测试）
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    if HAS_TORCHVISION:
        # 使用torchvision加载MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
        ])
        
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        if use_full_data:
            # 使用完整数据集
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
            
            train_data, train_labels = next(iter(train_loader))
            test_data, test_labels = next(iter(test_loader))
            
            # 展平图像: [batch, 1, 28, 28] -> [batch, 784]
            train_data = train_data.squeeze(1).view(len(train_dataset), -1)
            test_data = test_data.squeeze(1).view(len(test_dataset), -1)
        else:
            # 合并训练集和测试集用于downsample
            full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
            
            # 随机选择n_samples个样本（用于downsample）
            downsample_indices = np.random.choice(len(full_dataset), min(n_samples, len(full_dataset)), replace=False)
            downsample_subset = torch.utils.data.Subset(full_dataset, downsample_indices)
            
            # 创建数据加载器获取所有数据
            downsample_loader = torch.utils.data.DataLoader(downsample_subset, batch_size=len(downsample_subset), shuffle=False)
            all_data, all_labels = next(iter(downsample_loader))
            
            # 展平图像: [batch, 1, 28, 28] -> [batch, 784]
            all_data_flat = all_data.squeeze(1).view(len(downsample_subset), -1)
            
            # 按70%/30%分割训练集和测试集
            n_train = int(len(downsample_subset) * 0.7)
            
            # 随机打乱
            shuffle_indices = np.random.permutation(len(downsample_subset))
            train_indices = shuffle_indices[:n_train]
            test_indices = shuffle_indices[n_train:]
            
            train_data = all_data_flat[train_indices]
            train_labels = all_labels[train_indices]
            test_data = all_data_flat[test_indices]
            test_labels = all_labels[test_indices]
    else:
        try:
            # 使用sklearn加载MNIST
            from sklearn.datasets import fetch_openml
            print("正在从OpenML下载MNIST数据...")
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
            X, y = mnist.data / 255.0, mnist.target.astype(int)
            
            # 归一化
            X = (X - 0.1307) / 0.3081
            
            if use_full_data:
                # 使用完整数据集（sklearn版本是70000个样本）
                train_data = torch.FloatTensor(X[:60000])
                train_labels = torch.LongTensor(y[:60000])
                test_data = torch.FloatTensor(X[60000:])
                test_labels = torch.LongTensor(y[60000:])
            else:
                # 先从完整数据集中随机选择n_samples个样本（用于downsample）
                downsample_indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
                X_downsampled = X[downsample_indices]
                y_downsampled = y[downsample_indices]
                
                # 从downsample的样本中按70%/30%分配训练集和测试集
                n_train = int(len(X_downsampled) * 0.7)
                
                # 随机打乱并分配
                shuffle_indices = np.random.permutation(len(X_downsampled))
                train_indices = shuffle_indices[:n_train]
                test_indices = shuffle_indices[n_train:]
                
                train_data = torch.FloatTensor(X_downsampled[train_indices])
                train_labels = torch.LongTensor(y_downsampled[train_indices])
                test_data = torch.FloatTensor(X_downsampled[test_indices])
                test_labels = torch.LongTensor(y_downsampled[test_indices])
        except Exception as e:
            print(f"加载MNIST失败: {e}")
            print("使用随机数据作为测试...")
            # 生成随机数据作为fallback
            train_data = torch.randn(60000 if use_full_data else int(n_samples * 0.7), 784)
            train_labels = torch.randint(0, 10, (60000 if use_full_data else int(n_samples * 0.7),))
            test_data = torch.randn(10000 if use_full_data else int(n_samples * 0.3), 784)
            test_labels = torch.randint(0, 10, (10000 if use_full_data else int(n_samples * 0.3),))
    
    # 转换标签
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)
    
    # 获取数据维度
    n_train = train_data.shape[0]
    n_test = test_data.shape[0]
    
    # 创建mask
    train_mask = torch.ones(n_train, 1, dtype=torch.float32).to(device)
    test_mask = torch.ones(n_test, 1, dtype=torch.float32).to(device)
    
    # 创建索引
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_test)
    
    # 转换为tensor
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    
    # 对于分类任务，标签需要是one-hot格式
    train_labels_onehot = F.one_hot(train_labels, num_classes=10).float().to(device)
    test_labels_onehot = F.one_hot(test_labels, num_classes=10).float().to(device)
    
    tr_data = (train_data, train_labels_onehot, train_mask, train_idx)
    te_data = (test_data, test_labels_onehot, test_mask, test_idx)
    
    return tr_data, te_data, n_train, 784, 10, train_idx, test_idx, train_labels, test_labels

class ClassificationTransformer(baseTransformer):
    """扩展baseTransformer以支持分类任务"""
    def __init__(self, args, num_data, input_dim, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, freeze_qk=False, freeze_v=False, num_classes=10):
        # 将output_dim设为num_classes
        super().__init__(args, num_data, input_dim, num_classes, hid_dim, n_layers, n_heads, pf_dim, dropout, device, freeze_qk=freeze_qk, freeze_v=freeze_v)
        
    def forward(self, x):
        # x: [seq_len, feature_dim] 或 [batch, seq_len, feature_dim]
        output, KLD = super().forward(x)
        return output, KLD

def count_parameters(model):
    """统计可学习参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_classification(model, data, optimizer):
    """分类任务训练函数"""
    model.train()
    
    x, y, mask, idx = data
    optimizer.zero_grad()
    
    output, KLD = model(x)
    output = output.squeeze(0)
    
    target = y.argmax(dim=-1)
    loss = F.cross_entropy(output, target)
    
    # 添加KL散度
    if torch.isnan(KLD) or torch.isinf(KLD):
        KLD = torch.tensor(0.0, device=KLD.device)
    KLD = torch.clamp(KLD, min=0.0, max=1000.0)
    
    loss = loss + KLD / torch.sum(mask)
    
    # 检查NaN
    if torch.isnan(loss) or torch.isinf(loss):
        return output, 0.0, 0.0, 0.0
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # 计算准确率
    with torch.no_grad():
        pred = output.argmax(dim=-1)
        acc = (pred == target).float().mean().item()
    
    return output, loss.item(), acc, KLD.item()

def evaluate_classification(model, data, eval_batch_size=0):
    """分类任务评估函数

    Args:
        eval_batch_size:
            - <=0: 维持原行为（整批一次前向，transductive 风格）
            - >0: 以小 batch 评估，避免大规模 test 样本互相注意力
              其中 1 表示逐样本推理
    """
    model.eval()
    with torch.no_grad():
        x, y, mask, idx = data
        target = y.argmax(dim=-1)

        if eval_batch_size is None or eval_batch_size <= 0:
            output, _ = model(x)
            output = output.squeeze(0)
        else:
            outputs = []
            n = x.shape[0]
            for start in range(0, n, eval_batch_size):
                end = min(start + eval_batch_size, n)
                batch_x = x[start:end]
                batch_output, _ = model(batch_x)
                outputs.append(batch_output.squeeze(0))
            output = torch.cat(outputs, dim=0)

        loss = F.cross_entropy(output, target)
        
        # 计算准确率
        pred = output.argmax(dim=-1)
        acc = (pred == target).float().mean().item()
        
    return output, loss.item(), acc

def plot_confusion_matrices(results_list, save_path, true_labels_dict):
    """绘制混淆矩阵"""
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
    except ImportError:
        print("警告: sklearn或seaborn未安装，跳过混淆矩阵绘制")
        return
    
    n_experiments = len(results_list)
    cols = 5
    rows = (n_experiments + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1) if n_experiments > 1 else [[axes]]
    
    for idx, results in enumerate(results_list):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col] if rows > 1 else axes[0][col]
        
        test_idx = results['te_idx']
        predictions = results['predictions'][test_idx] if results['predictions'].shape[0] > max(test_idx) else results['predictions']
        pred_classes = predictions.argmax(axis=-1) if len(predictions.shape) > 1 else predictions
        true_classes = true_labels_dict.get(results['name'], np.zeros_like(pred_classes))
        
        cm = confusion_matrix(true_classes, pred_classes)
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f"{results['name'][:20]}\nAcc: {results['best_test_acc']:.2%}", fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(n_experiments, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off') if rows > 1 else axes[0][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_training_curves(results_list, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(results_list)))
    
    for i, results in enumerate(results_list):
        test_acc = results['test_acc_list']
        epochs = range(1, len(test_acc) + 1)
        
        plt.plot(epochs, test_acc, '-', alpha=0.9, label=f"{results['name']}", linewidth=1.5, color=colors[i])
    
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def generate_report(results_list, save_path, args):
    """生成实验报告"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MNIST分类任务消融实验报告: 所有模型对比\n")
        f.write("="*80 + "\n\n")
        
        f.write("实验设置:\n")
        if args.use_full_data:
            f.write("  数据集: MNIST完整数据集 (60000训练 + 10000测试)\n")
        else:
            f.write(f"  数据集: MNIST (降采样到{args.n_samples}个样本)\n")
        f.write("  任务: 10类分类\n")
        f.write("  输入: 28x28图像展平为784维\n")
        f.write(f"  训练轮数: {args.N_EPOCHS}\n")
        f.write(f"  学习率: {args.LEARNING_RATE}\n")
        f.write(f"  设备: {args.device}\n\n")
        
        f.write("="*80 + "\n")
        f.write("实验结果对比\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'实验名称':<40} {'Test Acc':<12} {'Runtime (s)':<12} {'参数数量':<15}\n")
        f.write("-" * 80 + "\n")
        
        for results in results_list:
            f.write(f"{results['name']:<40} {results['best_test_acc']:<12.4f} "
                   f"{results['runtime']:<12.2f} {results['num_params']:<15,}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("详细结果\n")
        f.write("="*80 + "\n\n")
        
        for results in results_list:
            f.write(f"\n{results['name']}:\n")
            f.write(f"  最佳测试准确率: {results['best_test_acc']:.4f} ({results['best_test_acc']*100:.2f}%)\n")
            f.write(f"  最佳 epoch: {results['best_epoch']}\n")
            f.write(f"  运行时间: {results['runtime']:.2f} 秒\n")
            f.write(f"  可学习参数数量: {results['num_params']:,}\n")
        
        # 固定QK/V对比分析
        f.write("\n" + "="*80 + "\n")
        f.write("固定参数影响分析\n")
        f.write("="*80 + "\n\n")
        
        # 分组对比
        model_groups = {
            'Transformer': ['Transformer', 'Transformer + 固定QK', 'Transformer + 固定V', 'Transformer + 固定QKV'],
            'MIKAN': ['MIKAN', 'MIKAN + 固定QK', 'MIKAN + 固定V', 'MIKAN + 固定QKV'],
            'FRSKA (共享NF)': ['FRSKA (共享NF)', 'FRSKA (共享NF) + 固定QK', 'FRSKA (共享NF) + 固定V', 'FRSKA (共享NF) + 固定QKV']
        }
        
        for model_name, exp_names in model_groups.items():
            f.write(f"\n{model_name}:\n")
            base_result = next((r for r in results_list if r['name'] == exp_names[0]), None)
            if base_result:
                f.write(f"  基础: Acc={base_result['best_test_acc']:.4f}, Params={base_result['num_params']:,}\n")
                for exp_name in exp_names[1:]:
                    exp_result = next((r for r in results_list if r['name'] == exp_name), None)
                    if exp_result:
                        acc_diff = exp_result['best_test_acc'] - base_result['best_test_acc']
                        param_diff = exp_result['num_params'] - base_result['num_params']
                        f.write(f"  {exp_name}: Acc变化={acc_diff:+.4f}, Params变化={param_diff:+,}\n")
    
    print(f"\n实验报告已保存到: {save_path}")

def run_experiment_with_data(exp_name, att_type, args, tr_data, te_data, num_data, input_dim, output_dim, test_idx, device, freeze_qk=False, freeze_v=False, shared_flow=False, qk_mode='normal', v_mode='learnable'):
    """使用预加载的数据运行实验
    
    Args:
        qk_mode: 'normal' (使用Q/K投影) 或 'no_qk' (Q=X, K=X)
        v_mode: 'learnable' (可学习V投影) 或 'fixed_orth' (固定正交V矩阵)
    """
    print(f"\n{'='*60}")
    print(f"实验: {exp_name}")
    print(f"  注意力类型: {att_type}")
    print(f"  固定QK: {freeze_qk}, 固定V: {freeze_v}")
    if att_type == 'frska':
        print(f"  共享Flow: {shared_flow}")
        print(f"  QK模式: {qk_mode}, V模式: {v_mode}")
        print(f"  正特征映射稳定化: {getattr(args, 'frska_positive_map', True)}")
    print(f"{'='*60}")
    
    # 设置注意力类型
    args.att_type = att_type
    
    # 设置FRSKA特定参数
    if att_type == 'frska':
        args.frska_shared_flow = shared_flow
        args.frska_qk_mode = qk_mode
        args.frska_v_mode = v_mode
    
    # 为FRSKA使用更多的随机特征
    original_M = args.M
    if att_type == 'frska':
        args.M = args.M_frska
    
    # 创建模型（分类版本）
    model = ClassificationTransformer(
        args, num_data, input_dim, output_dim,
        args.HID_DIM, args.ENC_LAYERS, args.ENC_HEADS,
        args.HID_DIM, args.ENC_DROPOUT, device,
        freeze_qk=freeze_qk,
        freeze_v=freeze_v,
        num_classes=output_dim
    ).to(device)
    
    # 恢复原始M
    if att_type == 'frska':
        args.M = original_M
    
    # 统计参数数量
    num_params = count_parameters(model)
    print(f"可学习参数数量: {num_params:,}")
    
    # 优化器设置
    if att_type == 'frska':
        lr = args.LEARNING_RATE * 0.5
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        if not hasattr(args, 'kl_lambda_frska'):
            args.kl_lambda_frska = 0.005
    else:
        lr = args.LEARNING_RATE
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练
    start_time = time.time()
    best_test_acc = 0.0
    best_epoch = 0
    train_acc_list = []
    test_acc_list = []
    best_predictions = None
    
    # 进度条
    if HAS_TQDM:
        epoch_iterator = tqdm(range(args.N_EPOCHS), desc=f"{exp_name[:25]}", ncols=100)
    else:
        epoch_iterator = range(args.N_EPOCHS)
    
    for epoch in epoch_iterator:
        tr_pred, tr_loss, tr_acc, _ = train_classification(model, tr_data, optimizer)
        te_pred, te_loss, te_acc = evaluate_classification(
            model,
            te_data,
            eval_batch_size=getattr(args, "eval_batch_size", 0),
        )
        
        train_acc_list.append(tr_acc)
        test_acc_list.append(te_acc)
        
        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_epoch = epoch + 1
            best_predictions = te_pred.cpu().numpy()
        
        # 更新进度条
        if HAS_TQDM:
            epoch_iterator.set_postfix({
                'tr_acc': f'{tr_acc:.3f}',
                'te_acc': f'{te_acc:.3f}',
                'best': f'{best_test_acc:.3f}'
            })
        elif (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d}/{args.N_EPOCHS} | Train: {tr_acc:.4f} | Test: {te_acc:.4f} | Best: {best_test_acc:.4f}")
    
    runtime = time.time() - start_time
    
    print(f"\n结果: 最佳测试准确率 = {best_test_acc:.4f} ({best_test_acc*100:.2f}%) @ Epoch {best_epoch}")
    print(f"      运行时间 = {runtime:.2f}秒, 参数数量 = {num_params:,}")
    
    return {
        'name': exp_name,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'runtime': runtime,
        'num_params': num_params,
        'predictions': best_predictions if best_predictions is not None else np.zeros((len(test_idx), 10)),
        'te_idx': test_idx,
        'train_acc_list': train_acc_list,
        'test_acc_list': test_acc_list
    }

def main():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--N_EPOCHS', type=int, default=50, help='Number of epochs')
    parser.add_argument('--LEARNING_RATE', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--HID_DIM', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--KEY_DIM', type=int, default=64, help='Key dimension')
    parser.add_argument('--ENC_LAYERS', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--ENC_HEADS', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ENC_DROPOUT', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--M', type=int, default=64, help='Number of random features')
    parser.add_argument('--M_frska', type=int, default=96, help='Number of random features for FRSKA')
    parser.add_argument('--prior_var', type=float, default=0.1, help='Prior variance')
    parser.add_argument('--kl_lambda', type=float, default=0.1, help='KL loss lambda')
    parser.add_argument('--copula_lambda', type=float, default=1.0, help='Copula loss lambda')
    parser.add_argument('--p_norm', type=float, default=2.0, help='p-norm for IKA')
    parser.add_argument('--att_dropout', type=float, default=0.1, help='Attention dropout')
    parser.add_argument('--n_samples', type=int, default=10000, help='Total number of samples when not using full data')
    parser.add_argument('--use_full_data', action='store_true', default=False, help='Use full MNIST dataset')
    parser.add_argument('--no_full_data', dest='use_full_data', action='store_false', help='Disable full data')
    parser.add_argument('--output_dir', type=str, default='mnist_results', help='Output directory')
    parser.add_argument('--quick_test', action='store_true', help='Quick test with fewer epochs')

    args = parser.parse_args()
    
    # 快速测试模式
    if args.quick_test:
        args.N_EPOCHS = 5
        args.use_full_data = False
        args.n_samples = 1000
    
    # 设置随机种子
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = str(device)
    print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载MNIST数据
    print("="*60)
    print("加载MNIST数据")
    print("="*60)
    tr_data, te_data, num_data, input_dim, output_dim, tr_idx, test_idx, tr_labels, te_labels = load_mnist_data(
        use_full_data=args.use_full_data,
        n_samples=args.n_samples,
        random_seed=args.random_seed,
        device=device
    )
    print(f"数据加载完成:")
    print(f"  训练样本数: {len(tr_idx)}")
    print(f"  测试样本数: {len(test_idx)}")
    print(f"  输入维度: {input_dim} (28x28图像展平)")
    print(f"  输出维度: {output_dim} (10类)")

    # 定义实验列表
    # (实验名称, 注意力类型, freeze_qk, freeze_v, shared_flow, qk_mode, v_mode)
    experiments = [
        # 1. Transformer
        ("Transformer", "dot", False, False, False, "normal", "learnable"),
        
        # 2-4. IKAN-direct
        ("IKAN-direct", "ikandirect", False, False, False, "normal", "learnable"),
        ("IKAN-direct + 固定QK", "ikandirect", True, False, False, "normal", "learnable"),
        ("IKAN-direct + 固定QKV", "ikandirect", True, True, False, "normal", "learnable"),
        
        # 5-7. MIKAN
        ("MIKAN", "mikan", False, False, False, "normal", "learnable"),
        ("MIKAN + 固定QK", "mikan", True, False, False, "normal", "learnable"),
        ("MIKAN + 固定QKV", "mikan", True, True, False, "normal", "learnable"),
        
        # 8-10. FRSKA (独立Flow)
        ("FRSKA", "frska", False, False, False, "normal", "learnable"),
        ("FRSKA + 固定QK", "frska", True, False, False, "normal", "learnable"),
        ("FRSKA + 固定QKV", "frska", True, True, False, "normal", "learnable"),
        
        # 11-13. FRSKA (共享NF)
        ("FRSKA (共享NF)", "frska", False, False, True, "normal", "learnable"),
        ("FRSKA (共享NF) + 固定QK", "frska", True, False, True, "normal", "learnable"),
        ("FRSKA (共享NF) + 固定QKV", "frska", True, True, True, "normal", "learnable"),
        
        # 14-15. FRSKA + NoQK + FixedOrthV (独立Flow)
        ("FRSKA + NoQK + FixedOrthV", "frska", False, False, False, "no_qk", "fixed_orth"),
        
        # 16-17. FRSKA + NoQK + FixedOrthV (共享NF)
        ("FRSKA (共享NF) + NoQK + FixedOrthV", "frska", False, False, True, "no_qk", "fixed_orth"),
        
        # 18-19. FRSKA + NoQK + LearnableV (独立Flow)
        ("FRSKA + NoQK + LearnableV", "frska", False, False, False, "no_qk", "learnable"),
        
        # 20-21. FRSKA + NoQK + LearnableV (共享NF)
        ("FRSKA (共享NF) + NoQK + LearnableV", "frska", False, False, True, "no_qk", "learnable"),
        
        # # 22. MGK
        # ("MGK", "mgk", False, False, False, "normal", "learnable"),

    ]

    # 运行实验
    results_list = []
    all_true_labels = {}

    print("\n" + "="*60)
    print(f"开始MNIST分类消融实验: 共{len(experiments)}个实验")
    print("="*60)

    total_start_time = time.time()
    
    for i, (exp_name, att_type, freeze_qk, freeze_v, shared_flow, qk_mode, v_mode) in enumerate(experiments):
        print(f"\n>>> 实验 {i+1}/{len(experiments)}: {exp_name}")
        
        result = run_experiment_with_data(
            exp_name, att_type, args, tr_data, te_data, 
            num_data, input_dim, output_dim, test_idx, device,
            freeze_qk=freeze_qk, freeze_v=freeze_v, shared_flow=shared_flow,
            qk_mode=qk_mode, v_mode=v_mode
        )
        results_list.append(result)
        
        # 保存真实标签
        _, y_te, _, _ = te_data
        all_true_labels[exp_name] = y_te.argmax(dim=-1).cpu().numpy()
    
    total_time = time.time() - total_start_time
    print(f"\n总运行时间: {total_time/60:.2f} 分钟")

    # 生成图表和报告
    try:
        plot_confusion_matrices(results_list, os.path.join(args.output_dir, 'confusion_matrices.png'), all_true_labels)
        print("混淆矩阵已保存")
    except Exception as e:
        print(f"绘制混淆矩阵时出错: {e}")

    try:
        plot_training_curves(results_list, os.path.join(args.output_dir, 'training_curves.png'))
        print("训练曲线已保存")
    except Exception as e:
        print(f"绘制训练曲线时出错: {e}")
    
    generate_report(results_list, os.path.join(args.output_dir, 'ablation_report.txt'), args)

    # 打印最终结果摘要
    print("\n" + "="*60)
    print("实验结果摘要")
    print("="*60)
    print(f"{'实验名称':<40} {'Acc':<8} {'Time(s)':<10} {'Params':<12}")
    print("-"*70)
    for r in results_list:
        print(f"{r['name']:<40} {r['best_test_acc']:.4f}   {r['runtime']:<10.2f} {r['num_params']:,}")

    print("\n" + "="*60)
    print(f"所有实验完成！结果保存在: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
