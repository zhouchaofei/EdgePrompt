"""
GNN特征选择实验
测试不同的功能图和节点特征组合
"""

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import numpy as np
import pandas as pd
from datetime import datetime
import argparse

from gnn_dataset import load_brain_graph_dataset, split_dataset
from gnn_models import SimpleGCN, LinearProbe


def train_epoch(model, loader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)

        # 检查输出是否包含 NaN
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("⚠️  Warning: Model output contains NaN/Inf during training, skipping batch")
            continue

        loss = F.cross_entropy(out, data.y.view(-1))

        # 检查损失是否为 NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️  Warning: Loss is NaN/Inf, skipping batch")
            continue

        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """评估模型"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for data in loader:
        data = data.to(device)
        out = model(data)

        # 检查输出是否包含 NaN
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("⚠️  Warning: Model output contains NaN/Inf, replacing with zeros")
            out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

        pred = out.argmax(dim=1)
        prob = F.softmax(out, dim=1)[:, 1]

        # 再次检查概率是否包含 NaN
        if torch.isnan(prob).any() or torch.isinf(prob).any():
            print("⚠️  Warning: Softmax output contains NaN/Inf, replacing with 0.5")
            prob = torch.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)

        all_preds.append(pred.cpu())
        all_labels.append(data.y.view(-1).cpu())
        all_probs.append(prob.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    # 最后检查一次
    if np.isnan(all_probs).any() or np.isinf(all_probs).any():
        print("⚠️  Warning: Final probabilities contain NaN/Inf, replacing with 0.5")
        all_probs = np.nan_to_num(all_probs, nan=0.5, posinf=1.0, neginf=0.0)

    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    # 安全的 AUC 计算
    try:
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.0
    except ValueError as e:
        print(f"⚠️  Warning: Cannot compute AUC: {e}")
        auc = 0.0

    return balanced_acc, auc


def run_single_experiment(
    dataset_name,
    fc_method,
    node_feature_type,
    model_type='gcn',
    data_folder='./data',
    temporal_method='pca',
    num_epochs=100,
    batch_size=32,
    lr=0.001,
    hidden_dim=64,
    seed=42
):
    """
    运行单次实验

    Args:
        dataset_name: 数据集名称
        fc_method: FC方法
        node_feature_type: 节点特征类型
        model_type: 模型类型
        data_folder: 数据文件夹
        temporal_method: 时序编码方法（仅当node_feature_type='temporal'时使用）
        num_epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        hidden_dim: 隐藏层维度
        seed: 随机种子
    """

    print(f"\n{'='*80}")
    print(f"Experiment: {fc_method} + {node_feature_type} + {model_type}")
    if node_feature_type == 'temporal':
        print(f"  Temporal method: {temporal_method}")
    print(f"{'='*80}")

    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载数据集
    print("\n1. Loading dataset...")
    try:
        dataset = load_brain_graph_dataset(
            dataset_name=dataset_name,
            data_folder=data_folder,
            fc_method=fc_method,
            node_feature_type=node_feature_type,
            threshold=None,  # 全连接图
            temporal_method=temporal_method
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None

    # 划分数据集
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, seed=seed)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 创建模型
    print("\n2. Creating model...")
    input_dim = dataset.feature_dim

    if model_type == 'gcn':
        model = SimpleGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_layers=2,
            dropout=0.5
        )
    elif model_type == 'linear':
        model = LinearProbe(
            input_dim=input_dim,
            num_classes=2
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    print(f"  Model: {model_type}")
    print(f"  Input dim: {input_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")

    # 优化器
    # 使用更保守的学习率和更强的正则化
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)

    # 训练
    print("\n3. Training...")
    best_val_acc = 0
    best_epoch = 0
    patience = 20
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc, val_auc = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}: Loss={train_loss:.4f}, "
                  f"Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # 加载最佳模型并测试
    print("\n4. Testing...")
    model.load_state_dict(best_model_state)
    test_acc, test_auc = evaluate(model, test_loader, device)

    print(f"  Best epoch: {best_epoch}")
    print(f"  Test Balanced Acc: {test_acc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")

    results = {
        'fc_method': fc_method,
        'node_feature_type': node_feature_type,
        'temporal_method': temporal_method if node_feature_type == 'temporal' else 'N/A',
        'model_type': model_type,
        'input_dim': input_dim,
        'best_epoch': best_epoch,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_auc': test_auc
    }

    return results


def run_all_experiments(
    dataset_name,
    data_folder='./data',
    temporal_method='pca',
    save_dir='./results/gnn_feature_selection',
    num_repeats=5
):
    """
    运行所有特征选择实验

    Args:
        dataset_name: 数据集名称
        data_folder: 数据文件夹
        temporal_method: 时序编码方法
        save_dir: 结果保存目录
        num_repeats: 重复次数
    """

    print(f"\n{'='*80}")
    print(f"GNN FEATURE SELECTION EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Temporal method: {temporal_method}")
    print(f"Repeats: {num_repeats}")
    print(f"{'='*80}\n")

    os.makedirs(save_dir, exist_ok=True)

    # 定义实验配置
    fc_methods = ['pearson', 'ledoit_wolf']
    node_feature_types = ['statistical', 'temporal']
    model_types = ['linear', 'gcn']

    all_results = []

    # 运行所有组合
    for fc_method in fc_methods:
        for node_feature_type in node_feature_types:
            for model_type in model_types:

                print(f"\n{'='*80}")
                print(f"Configuration: FC={fc_method}, Features={node_feature_type}, Model={model_type}")
                print(f"{'='*80}")

                # 多次重复实验
                config_results = []

                for repeat in range(num_repeats):
                    print(f"\n--- Repeat {repeat + 1}/{num_repeats} ---")

                    result = run_single_experiment(
                        dataset_name=dataset_name,
                        fc_method=fc_method,
                        node_feature_type=node_feature_type,
                        model_type=model_type,
                        data_folder=data_folder,
                        temporal_method=temporal_method,
                        seed=42 + repeat
                    )

                    if result is None:
                        print(f"⚠️  Skipping this configuration due to missing data")
                        break

                    result['repeat'] = repeat + 1
                    config_results.append(result)
                    all_results.append(result)

                # 打印该配置的平均结果
                if config_results:
                    test_accs = [r['test_acc'] for r in config_results]
                    test_aucs = [r['test_auc'] for r in config_results]

                    print(f"\n{'='*60}")
                    print(f"Summary for {fc_method} + {node_feature_type} + {model_type}:")
                    print(f"  Test Acc: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
                    print(f"  Test AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
                    print(f"{'='*60}\n")

    if not all_results:
        print("❌ No experiments completed successfully!")
        return None

    # 保存所有结果
    results_df = pd.DataFrame(all_results)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(
        save_dir,
        f'{dataset_name.lower()}_feature_selection_{temporal_method}_{timestamp}.csv'
    )
    results_df.to_csv(results_file, index=False)

    print(f"\n✅ All results saved to: {results_file}")

    # 生成汇总报告
    summary_file = os.path.join(
        save_dir,
        f'{dataset_name.lower()}_feature_selection_summary_{temporal_method}_{timestamp}.txt'
    )

    with open(summary_file, 'w') as f:
        f.write(f"{dataset_name} GNN Feature Selection Experiments\n")
        f.write(f"="*80 + "\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Temporal method: {temporal_method}\n")
        f.write(f"Repeats: {num_repeats}\n\n")

        # 按配置分组统计
        grouped = results_df.groupby(['fc_method', 'node_feature_type', 'model_type'])

        f.write("Summary by Configuration:\n")
        f.write("-"*80 + "\n\n")

        for (fc, feature, model), group in grouped:
            f.write(f"FC: {fc}, Features: {feature}, Model: {model}\n")
            f.write(f"  Input dim: {group['input_dim'].iloc[0]}\n")
            f.write(f"  Test Acc: {group['test_acc'].mean():.4f} ± {group['test_acc'].std():.4f}\n")
            f.write(f"  Test AUC: {group['test_auc'].mean():.4f} ± {group['test_auc'].std():.4f}\n")
            f.write(f"  Val Acc:  {group['val_acc'].mean():.4f} ± {group['val_acc'].std():.4f}\n\n")

        # 找出最佳配置
        f.write("\n" + "="*80 + "\n")
        f.write("Best Configurations:\n")
        f.write("-"*80 + "\n\n")

        # 按Test Acc排序
        best_by_acc = grouped['test_acc'].mean().sort_values(ascending=False)
        f.write("Top 3 by Test Accuracy:\n")
        for i, (config, acc) in enumerate(best_by_acc.head(3).items(), 1):
            fc, feature, model = config
            auc = grouped.get_group(config)['test_auc'].mean()
            f.write(f"  {i}. FC={fc}, Features={feature}, Model={model}\n")
            f.write(f"     Test Acc: {acc:.4f}, Test AUC: {auc:.4f}\n")

        f.write("\n")

        # 按Test AUC排序
        best_by_auc = grouped['test_auc'].mean().sort_values(ascending=False)
        f.write("Top 3 by Test AUC:\n")
        for i, (config, auc) in enumerate(best_by_auc.head(3).items(), 1):
            fc, feature, model = config
            acc = grouped.get_group(config)['test_acc'].mean()
            f.write(f"  {i}. FC={fc}, Features={feature}, Model={model}\n")
            f.write(f"     Test AUC: {auc:.4f}, Test Acc: {acc:.4f}\n")

    print(f"✅ Summary saved to: {summary_file}")

    # 可视化结果
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. FC方法对比
        ax = axes[0, 0]
        fc_comparison = results_df.groupby('fc_method')['test_acc'].mean()
        fc_comparison.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral'])
        ax.set_title('FC Method Comparison')
        ax.set_ylabel('Test Accuracy')
        ax.set_xlabel('FC Method')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        # 2. 节点特征对比
        ax = axes[0, 1]
        feature_comparison = results_df.groupby('node_feature_type')['test_acc'].mean()
        feature_comparison.plot(kind='bar', ax=ax, color=['lightgreen', 'orange'])
        ax.set_title('Node Feature Comparison')
        ax.set_ylabel('Test Accuracy')
        ax.set_xlabel('Feature Type')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        # 3. 模型对比
        ax = axes[1, 0]
        model_comparison = results_df.groupby('model_type')['test_acc'].mean()
        model_comparison.plot(kind='bar', ax=ax, color=['plum', 'khaki'])
        ax.set_title('Model Type Comparison')
        ax.set_ylabel('Test Accuracy')
        ax.set_xlabel('Model Type')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        # 4. 热力图：FC × Feature
        ax = axes[1, 1]
        pivot_data = results_df.pivot_table(
            values='test_acc',
            index='fc_method',
            columns='node_feature_type',
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax)
        ax.set_title('FC Method × Node Feature')

        plt.tight_layout()

        plot_file = os.path.join(
            save_dir,
            f'{dataset_name.lower()}_feature_selection_{temporal_method}_{timestamp}.png'
        )
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Visualization saved to: {plot_file}")

    except ImportError:
        print("⚠️  Matplotlib not available, skipping visualization")

    print(f"\n{'='*80}")
    print(f"✅ All experiments completed!")
    print(f"{'='*80}\n")

    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN Feature Selection Experiments')

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['ABIDE', 'MDD'],
        help='Dataset name'
    )

    parser.add_argument(
        '--data_folder',
        type=str,
        default='./data',
        help='Data root folder'
    )

    parser.add_argument(
        '--temporal_method',
        type=str,
        default='pca',
        choices=['pca', 'cnn', 'transformer'],
        help='Temporal encoding method'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='./results/gnn_feature_selection',
        help='Results save directory'
    )

    parser.add_argument(
        '--num_repeats',
        type=int,
        default=5,
        help='Number of repeated experiments'
    )

    args = parser.parse_args()

    run_all_experiments(
        dataset_name=args.dataset,
        data_folder=args.data_folder,
        temporal_method=args.temporal_method,
        save_dir=args.save_dir,
        num_repeats=args.num_repeats
    )
