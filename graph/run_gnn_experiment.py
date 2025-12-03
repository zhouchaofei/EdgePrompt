"""
GNN实验脚本 - 混合特征版
目标：验证功能图和节点特征的组合
- FC methods: Pearson vs LedoitWolf
- Node features: Hybrid (Deep + Statistical)
- Models: Linear, GNN
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, f1_score, confusion_matrix
)
import pandas as pd
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from prepare_gnn_data import load_gnn_dataset
from simple_gnn import create_model


def setup_logger(save_dir='./results/gnn_experiments'):
    """设置日志"""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'gnn_experiment_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__), timestamp


def train_epoch(model, loader, optimizer, device):
    """训练一个epoch（增强NaN检查）"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)

        # 检查输入数据
        if torch.isnan(data.x).any():
            print("  Warning: NaN detected in input features, skipping batch")
            continue

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if torch.isnan(data.edge_attr).any():
                print("  Warning: NaN detected in edge attributes, skipping batch")
                continue

        optimizer.zero_grad()

        try:
            output = model(data)

            # 检查输出
            if torch.isnan(output).any():
                print("  Warning: NaN in model output, skipping batch")
                continue

            loss = F.cross_entropy(output, data.y)

            # 检查损失
            if torch.isnan(loss):
                print("  Warning: NaN loss, skipping batch")
                continue

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            pred = output.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.num_graphs

        except Exception as e:
            print(f"  Error in training: {e}")
            continue

    if total == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, loader, device):
    """评估模型"""
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)

            prob = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)

            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob[:, 1].cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='binary'),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    }

    return metrics, y_true, y_pred


def run_single_fold(graph_list, labels, train_idx, test_idx,
                    model_config, device, epochs=100,
                    batch_size=32, lr=0.001, patience=20):
    """运行单个fold"""

    # 准备数据
    train_graphs = [graph_list[i] for i in train_idx]
    test_graphs = [graph_list[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # 创建模型
    input_dim = train_graphs[0].x.shape[1]
    model = create_model(
        model_type=model_config['model_type'],
        input_dim=input_dim,
        hidden_dim=model_config.get('hidden_dim', 64),
        output_dim=2,
        num_layers=model_config.get('num_layers', 2),
        gnn_type=model_config.get('gnn_type', 'gcn'),
        dropout=model_config.get('dropout', 0.5),
        pooling=model_config.get('pooling', 'flatten')
    ).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 早停
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None

    # 训练
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)

        # 早停验证（这里简单用训练集，实际应该有验证集）
        if train_acc > best_val_acc:
            best_val_acc = train_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 评估
    test_metrics, y_true, y_pred = evaluate(model, test_loader, device)

    return test_metrics


def run_cross_validation(graph_list, labels, model_config, device,
                         n_folds=5, n_repeats=10, epochs=100,
                         batch_size=32, lr=0.001):
    """运行交叉验证"""

    all_results = {
        'accuracy': [],
        'balanced_accuracy': [],
        'f1': [],
        'auc': []
    }

    for repeat in range(n_repeats):
        print(f"\n  Repeat {repeat + 1}/{n_repeats}")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42 + repeat)

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            print(f"    Fold {fold + 1}/{n_folds}...", end=' ')

            metrics = run_single_fold(
                graph_list=graph_list,
                labels=labels,
                train_idx=train_idx,
                test_idx=test_idx,
                model_config=model_config,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr
            )

            for key in all_results.keys():
                all_results[key].append(metrics[key])

            print(f"Bal_Acc={metrics['balanced_accuracy']:.4f}, AUC={metrics['auc']:.4f}")

    # 计算统计
    final_results = {}
    for key in all_results.keys():
        final_results[f'{key}_mean'] = np.mean(all_results[key])
        final_results[f'{key}_std'] = np.std(all_results[key])

    return final_results


def run_experiment(data_folder='./data/gnn_datasets',
                   save_dir='./results/gnn_experiments',
                   dataset='ABIDE',
                   device='cuda',
                   n_folds=5,
                   n_repeats=10,
                   epochs=100,
                   batch_size=32,
                   lr=0.001):
    """运行完整实验"""

    logger, timestamp = setup_logger(save_dir)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"GNN EXPERIMENT: Hybrid Features (Deep + Statistical)")
    logger.info(f"{'=' * 80}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Device: {device}")
    logger.info(f"Cross-validation: {n_folds}-fold × {n_repeats} repeats")
    logger.info(f"{'=' * 80}\n")

    # 实验配置
    fc_methods = ['pearson', 'ledoit_wolf']
    feature_types = ['hybrid']  # 🔥 使用混合特征

    model_configs = [
        {'name': 'Linear', 'model_type': 'linear', 'pooling': 'flatten'},
        {'name': 'MLP', 'model_type': 'mlp', 'hidden_dim': 128, 'pooling': 'flatten'},
        {'name': 'GCN', 'model_type': 'gnn', 'gnn_type': 'gcn', 'hidden_dim': 64,
         'num_layers': 2, 'pooling': 'flatten'},
        {'name': 'GAT', 'model_type': 'gnn', 'gnn_type': 'gat', 'hidden_dim': 64,
         'num_layers': 2, 'pooling': 'flatten'},
        {'name': 'Linear', 'model_type': 'linear', 'pooling': 'mean'},
        {'name': 'MLP', 'model_type': 'mlp', 'hidden_dim': 128, 'pooling': 'mean'},
        {'name': 'GCN', 'model_type': 'gnn', 'gnn_type': 'gcn', 'hidden_dim': 64,
         'num_layers': 2, 'pooling': 'mean'},
        {'name': 'GAT', 'model_type': 'gnn', 'gnn_type': 'gat', 'hidden_dim': 64,
         'num_layers': 2, 'pooling': 'mean'},
        {'name': 'Linear', 'model_type': 'linear', 'pooling': 'max'},
        {'name': 'MLP', 'model_type': 'mlp', 'hidden_dim': 128, 'pooling': 'max'},
        {'name': 'GCN', 'model_type': 'gnn', 'gnn_type': 'gcn', 'hidden_dim': 64,
         'num_layers': 2, 'pooling': 'max'},
        {'name': 'GAT', 'model_type': 'gnn', 'gnn_type': 'gat', 'hidden_dim': 64,
         'num_layers': 2, 'pooling': 'max'},
        {'name': 'Linear', 'model_type': 'linear', 'pooling': 'mean_max'},
        {'name': 'MLP', 'model_type': 'mlp', 'hidden_dim': 128, 'pooling': 'mean_max'},
        {'name': 'GCN', 'model_type': 'gnn', 'gnn_type': 'gcn', 'hidden_dim': 64,
         'num_layers': 2, 'pooling': 'mean_max'},
        {'name': 'GAT', 'model_type': 'gnn', 'gnn_type': 'gat', 'hidden_dim': 64,
         'num_layers': 2, 'pooling': 'mean_max'},
    ]

    all_results = []

    # 遍历所有组合
    for fc_method in fc_methods:
        for feature_type in feature_types:

            # 加载数据
            data_file = os.path.join(
                data_folder,
                f"{dataset}_{fc_method}_{feature_type}.pkl"
            )

            if not os.path.exists(data_file):
                logger.warning(f"Data file not found: {data_file}, skipping...")
                continue

            logger.info(f"\n{'=' * 80}")
            logger.info(f"Configuration: {fc_method} + {feature_type}")
            logger.info(f"{'=' * 80}")

            graph_list, labels, metadata = load_gnn_dataset(data_file)
            labels = np.array(labels)

            logger.info(f"Loaded {len(graph_list)} graphs")
            logger.info(f"Node feature dim: {metadata['node_feature_dim']}")
            logger.info(f"Label distribution: {np.bincount(labels)}")

            # 测试不同模型
            for model_config in model_configs:
                model_name = model_config['name']

                logger.info(f"\nTesting {model_name}...")

                try:
                    results = run_cross_validation(
                        graph_list=graph_list,
                        labels=labels,
                        model_config=model_config,
                        device=device,
                        n_folds=n_folds,
                        n_repeats=n_repeats,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr
                    )

                    # 记录结果
                    result_entry = {
                        'fc_method': fc_method,
                        'feature_type': feature_type,
                        'model': model_name,
                        'node_feature_dim': metadata['node_feature_dim']
                    }
                    result_entry.update(results)
                    all_results.append(result_entry)

                    # 打印结果
                    logger.info(f"\n  Results:")
                    logger.info(f"    Balanced Accuracy: {results['balanced_accuracy_mean']:.4f} ± "
                                f"{results['balanced_accuracy_std']:.4f}")
                    logger.info(f"    AUC: {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
                    logger.info(f"    F1: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")

                except Exception as e:
                    logger.error(f"Error testing {model_name}: {e}")
                    continue

    # 保存结果
    df_results = pd.DataFrame(all_results)
    csv_file = os.path.join(save_dir, f'{dataset}_gnn_results_{timestamp}.csv')
    df_results.to_csv(csv_file, index=False, float_format='%.4f')
    logger.info(f"\n✓ Results saved to: {csv_file}")

    # 打印最佳结果
    print_best_results(df_results, logger)

    return df_results


def print_best_results(df_results, logger):
    """打印最佳结果摘要"""
    logger.info(f"\n{'=' * 80}")
    logger.info("BEST RESULTS SUMMARY")
    logger.info(f"{'=' * 80}\n")

    # 按balanced_accuracy排序
    df_sorted = df_results.sort_values('balanced_accuracy_mean', ascending=False)

    logger.info("Top configurations:")
    for i, row in df_sorted.head(5).iterrows():
        logger.info(f"\n{i + 1}. {row['fc_method']} + {row['feature_type']} + {row['model']}")
        logger.info(f"   Balanced Acc: {row['balanced_accuracy_mean']:.4f} ± "
                    f"{row['balanced_accuracy_std']:.4f}")
        logger.info(f"   AUC: {row['auc_mean']:.4f} ± {row['auc_std']:.4f}")
        logger.info(f"   F1: {row['f1_mean']:.4f} ± {row['f1_std']:.4f}")
        logger.info(f"   Feature Dim: {row['node_feature_dim']}")

    # 比较不同因素的影响
    logger.info(f"\n{'=' * 60}")
    logger.info("Factor Analysis:")
    logger.info(f"{'=' * 60}")

    # FC method比较
    logger.info("\nFC Method comparison:")
    for fc in df_results['fc_method'].unique():
        mean_acc = df_results[df_results['fc_method'] == fc]['balanced_accuracy_mean'].mean()
        logger.info(f"  {fc}: {mean_acc:.4f}")

    # Model比较
    logger.info("\nModel comparison:")
    for model in df_results['model'].unique():
        mean_acc = df_results[df_results['model'] == model]['balanced_accuracy_mean'].mean()
        logger.info(f"  {model}: {mean_acc:.4f}")

    logger.info(f"\n{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description='Run GNN experiments with hybrid features')

    parser.add_argument('--dataset', type=str, default='ABIDE',
                        choices=['ABIDE', 'MDD'],
                        help='Dataset name')
    parser.add_argument('--data_folder', type=str, default='./data/gnn_datasets',
                        help='Folder containing prepared GNN data')
    parser.add_argument('--save_dir', type=str, default='./results/gnn_experiments',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--n_repeats', type=int, default=10,
                        help='Number of CV repeats')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')

    args = parser.parse_args()

    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # 运行实验
    results = run_experiment(
        data_folder=args.data_folder,
        save_dir=args.save_dir,
        dataset=args.dataset,
        device=args.device,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    print("\n✅ Experiment completed successfully!")


if __name__ == '__main__':
    main()