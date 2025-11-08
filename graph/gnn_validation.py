"""
GNN模型验证脚本
用于测试不同的功能图 × 节点特征组合
使用 5-fold CV × 10 repeats
"""

import os
import numpy as np
import pickle
import argparse
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from gnn_models import get_model

import warnings

warnings.filterwarnings('ignore')


class BrainGraphDataset(Dataset):
    """
    脑功能图数据集
    """

    def __init__(self, fc_matrices, node_features, labels):
        """
        Args:
            fc_matrices: [N, ROI, ROI] 功能连接矩阵
            node_features: [N, ROI, F] 节点特征
            labels: [N] 标签
        """
        self.fc_matrices = fc_matrices
        self.node_features = node_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        返回PyG Data对象
        """
        fc = self.fc_matrices[idx]
        x = self.node_features[idx]
        y = self.labels[idx]

        # 构建边索引和边权重（全连接图）
        n_nodes = fc.shape[0]
        edge_index = []
        edge_weight = []

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:  # 排除自环
                    edge_index.append([i, j])
                    edge_weight.append(fc[i, j])

        edge_index = torch.LongTensor(edge_index).t()  # [2, E]
        edge_weight = torch.FloatTensor(edge_weight)  # [E]

        # 创建Data对象
        data = Data(
            x=torch.FloatTensor(x),
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=torch.LongTensor([y])
        )

        return data


def collate_fn(batch):
    """
    自定义collate函数用于DataLoader
    """
    return Batch.from_data_list(batch)


def train_epoch(model, loader, criterion, optimizer, device):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        # 前向传播
        out = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )

        loss = criterion(out, batch.y)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    """
    评估模型
    """
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )

            # 检查是否有NaN
            if torch.isnan(out).any():
                print("警告：模型输出中存在NaN值!")
                continue  # 跳过这次batch的评估

            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs[:, 1].cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)

    if len(np.unique(all_labels)) == 2:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.0

    f1 = f1_score(all_labels, all_preds, average='weighted')

    return acc, auc, f1, all_preds, all_labels


def run_cv_experiment(
        fc_matrices,
        node_features,
        labels,
        model_name='gcn',
        n_folds=5,
        n_repeats=10,
        hidden_dim=64,
        lr=0.0001,
        epochs=100,
        batch_size=32,
        device='cuda',
        verbose=True
):
    """
    运行交叉验证实验

    Args:
        fc_matrices: [N, ROI, ROI]
        node_features: [N, ROI, F]
        labels: [N]
        model_name: 模型名称
        n_folds: 折数
        n_repeats: 重复次数
        hidden_dim: 隐藏层维度
        lr: 学习率
        epochs: 训练轮数
        batch_size: 批大小
        device: 设备
        verbose: 是否打印详细信息

    Returns:
        results: 结果字典
    """
    n_subjects = len(labels)
    in_dim = node_features.shape[-1]
    num_classes = len(np.unique(labels))

    all_results = []

    for repeat in range(n_repeats):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Repeat {repeat + 1}/{n_repeats}")
            print(f"{'=' * 60}")

        # StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=repeat)

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_subjects), labels)):
            if verbose:
                print(f"\nFold {fold + 1}/{n_folds}")

            # 准备数据
            train_dataset = BrainGraphDataset(
                fc_matrices[train_idx],
                node_features[train_idx],
                labels[train_idx]
            )

            val_dataset = BrainGraphDataset(
                fc_matrices[val_idx],
                node_features[val_idx],
                labels[val_idx]
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )

            # 初始化模型
            model = get_model(
                model_name,
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes
            )
            model = model.to(device)

            # 优化器和损失函数
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            # 训练
            best_val_acc = 0
            patience = 20
            patience_counter = 0

            for epoch in range(epochs):
                train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                val_acc, val_auc, val_f1, _, _ = evaluate(model, val_loader, device)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 1 == 0:
                    print(f"  Epoch {epoch + 1:3d}: Loss={train_loss:.4f}, "
                          f"Val Acc={val_acc:.4f}, AUC={val_auc:.4f}, F1={val_f1:.4f}")

                # Early stopping
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

            # 最终评估
            val_acc, val_auc, val_f1, preds, true_labels = evaluate(model, val_loader, device)

            if verbose:
                print(f"  Final: Acc={val_acc:.4f}, AUC={val_auc:.4f}, F1={val_f1:.4f}")

            fold_results.append({
                'fold': fold,
                'acc': val_acc,
                'auc': val_auc,
                'f1': val_f1,
                'confusion_matrix': confusion_matrix(true_labels, preds).tolist()
            })

        all_results.append(fold_results)

    # 汇总结果
    all_accs = []
    all_aucs = []
    all_f1s = []

    for repeat_results in all_results:
        for fold_result in repeat_results:
            all_accs.append(fold_result['acc'])
            all_aucs.append(fold_result['auc'])
            all_f1s.append(fold_result['f1'])

    summary = {
        'model': model_name,
        'n_folds': n_folds,
        'n_repeats': n_repeats,
        'mean_acc': np.mean(all_accs),
        'std_acc': np.std(all_accs),
        'mean_auc': np.mean(all_aucs),
        'std_auc': np.std(all_aucs),
        'mean_f1': np.mean(all_f1s),
        'std_f1': np.std(all_f1s),
        'all_results': all_results
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Validate GNN models with different graph & feature combinations'
    )

    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to preprocessed data file (.pkl)'
    )

    parser.add_argument(
        '--fc_method',
        type=str,
        required=True,
        choices=['pearson', 'ledoit_wolf'],
        help='FC construction method'
    )

    parser.add_argument(
        '--feature_method',
        type=str,
        required=True,
        choices=['statistical', 'temporal'],
        help='Node feature method'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gcn',
        choices=['gcn', 'gat', 'linear'],
        help='GNN model type'
    )

    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of CV folds'
    )

    parser.add_argument(
        '--n_repeats',
        type=int,
        default=10,
        help='Number of CV repeats'
    )

    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=64,
        help='Hidden dimension'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Learning rate'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Training epochs'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/gnn_validation',
        help='Output directory'
    )

    args = parser.parse_args()

    # 加载数据
    print(f"\n{'=' * 60}")
    print(f"Loading data from: {args.data_file}")
    print(f"{'=' * 60}")

    with open(args.data_file, 'rb') as f:
        data = pickle.load(f)

    fc_matrices = data['fc_matrices'][args.fc_method]
    node_features = data['node_features'][args.feature_method]
    labels = data['labels']
    # 检查labels是否包含NaN
    if np.any(np.isnan(labels)):
        print("警告：标签中存在NaN值!")
        labels = np.nan_to_num(labels)  # 替换NaN值为0或其他默认值
    # 检查节点特征是否有NaN
    if np.any(np.isnan(node_features)):
        print("警告：节点特征中存在NaN值!")
        node_features = np.nan_to_num(node_features)  # 替换NaN值为0

    print(f"FC method: {args.fc_method}")
    print(f"Feature method: {args.feature_method}")
    print(f"FC shape: {fc_matrices.shape}")
    print(f"Features shape: {node_features.shape}")
    print(f"Labels: {labels.shape}, distribution: {np.bincount(labels)}")

    # 标准化节点特征
    print(f"\nStandardizing node features...")
    n_subjects, n_rois, n_features = node_features.shape
    node_features_flat = node_features.reshape(-1, n_features)
    scaler = StandardScaler()
    node_features_flat = scaler.fit_transform(node_features_flat)
    node_features = node_features_flat.reshape(n_subjects, n_rois, n_features)

    # 运行实验
    print(f"\n{'=' * 60}")
    print(f"Running {args.model.upper()} validation...")
    print(f"{'=' * 60}")

    results = run_cv_experiment(
        fc_matrices,
        node_features,
        labels,
        model_name=args.model,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        verbose=True
    )

    # 打印结果
    print(f"\n{'=' * 60}")
    print(f"Final Results")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {results['mean_acc']:.4f} ± {results['std_acc']:.4f}")
    print(f"AUC:       {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
    print(f"F1 Score:  {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    print(f"{'=' * 60}")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(
        args.output_dir,
        f'{args.fc_method}_{args.feature_method}_{args.model}_{timestamp}.json'
    )

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {result_file}")


if __name__ == '__main__':
    main()
