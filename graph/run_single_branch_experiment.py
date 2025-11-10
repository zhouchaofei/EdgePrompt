"""
单分支实验脚本
目标：验证双分支的提升是否来自融合

实验配置：
1. Single-branch Functional: temporal + LedoitWolf + GAT
2. Single-branch Anatomical: temporal + anatomical prior + GAT/GCN

评估：5-fold × 10 repeats CV
比较：与dual-branch baseline对比
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, f1_score
)
import pandas as pd
import argparse
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from prepare_gnn_data import load_gnn_dataset
from simple_gnn import create_model


def setup_logger(save_dir, branch_type):
    """设置日志"""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'single_branch_{branch_type}_{timestamp}.log')

    # 创建新的logger
    logger = logging.getLogger(f'single_branch_{branch_type}_{timestamp}')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清除已有的handlers

    # 文件handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # 控制台handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, timestamp


def train_epoch(model, loader, optimizer, device, clip_grad=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)

        if torch.isnan(data.x).any():
            continue

        optimizer.zero_grad()

        try:
            output = model(data)

            if torch.isnan(output).any():
                continue

            loss = F.cross_entropy(output, data.y)

            if torch.isnan(loss):
                continue

            loss.backward()

            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            pred = output.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.num_graphs

        except Exception as e:
            continue

    if total == 0:
        return 0.0, 0.0

    return total_loss / total, correct / total


def evaluate(model, loader, device):
    """评估模型"""
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            try:
                output = model(data)
                prob = F.softmax(output, dim=1)
                pred = output.argmax(dim=1)

                y_true.extend(data.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                y_prob.extend(prob[:, 1].cpu().numpy())
            except:
                continue

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    }

    return metrics


def run_single_fold(graph_list, labels, train_idx, test_idx,
                    model_config, device, epochs=100,
                    batch_size=32, lr=0.001, weight_decay=1e-5,
                    patience=20):
    """运行单个fold"""

    train_graphs = [graph_list[i] for i in train_idx]
    test_graphs = [graph_list[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # 创建模型
    input_dim = train_graphs[0].x.shape[1]
    model = create_model(
        model_type='gnn',
        input_dim=input_dim,
        hidden_dim=model_config.get('hidden_dim', 64),
        output_dim=2,
        num_layers=model_config.get('num_layers', 2),
        gnn_type=model_config.get('gnn_type', 'gat'),
        dropout=model_config.get('dropout', 0.5),
        pooling=model_config.get('pooling', 'mean')
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False
    )

    # 早停
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)

        if epoch % 5 == 0:
            val_metrics = evaluate(model, train_loader, device)
            val_acc = val_metrics['balanced_accuracy']

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 测试
    test_metrics = evaluate(model, test_loader, device)

    return test_metrics


def run_cross_validation(graph_list, labels, model_config, device,
                         n_folds=5, n_repeats=10, epochs=100,
                         batch_size=32, lr=0.001, weight_decay=1e-5):
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
                lr=lr,
                weight_decay=weight_decay
            )

            for key in all_results.keys():
                all_results[key].append(metrics[key])

            print(f"BAcc={metrics['balanced_accuracy']:.4f}, AUC={metrics['auc']:.4f}")

    # 计算统计
    final_results = {}
    for key in all_results.keys():
        final_results[f'{key}_mean'] = np.mean(all_results[key])
        final_results[f'{key}_std'] = np.std(all_results[key])
        final_results[f'{key}_all'] = all_results[key]

    return final_results


def run_experiment(
        dataset='ABIDE',
        branch_type='functional',
        data_folder='./data',
        save_dir='./results/single_branch',
        device='cuda',
        # Model config
        gnn_type='gat',
        hidden_dim=64,
        num_layers=2,
        dropout=0.5,
        pooling='mean',
        # Training config
        n_folds=5,
        n_repeats=10,
        epochs=100,
        batch_size=32,
        lr=0.001,
        weight_decay=1e-5
):
    """运行完整实验"""

    logger, timestamp = setup_logger(save_dir, branch_type)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"SINGLE-BRANCH EXPERIMENT: {branch_type.upper()}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Branch: {branch_type}")
    logger.info(f"Device: {device}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Model Config:")
    logger.info(f"  GNN Type: {gnn_type}")
    logger.info(f"  Hidden Dim: {hidden_dim}")
    logger.info(f"  Num Layers: {num_layers}")
    logger.info(f"  Dropout: {dropout}")
    logger.info(f"  Pooling: {pooling}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Training Config:")
    logger.info(f"  CV: {n_folds}-fold × {n_repeats} repeats")
    logger.info(f"  Epochs: {epochs}, Batch: {batch_size}")
    logger.info(f"  LR: {lr}, Weight Decay: {weight_decay}")
    logger.info(f"{'=' * 80}\n")

    # 加载数据
    if branch_type == 'functional':
        data_file = os.path.join(
            data_folder, 'preprocessed_gnn',
            f"{dataset}_ledoit_wolf_temporal.pkl"
        )
    elif branch_type == 'anatomical':
        data_file = os.path.join(
            data_folder, 'single_branch_datasets',
            f"{dataset}_anatomical_temporal.pkl"
        )
    else:
        raise ValueError(f"Unknown branch type: {branch_type}")

    logger.info(f"Loading data from: {data_file}")

    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        logger.error("Please run data preparation first!")
        return None

    graph_list, labels, metadata = load_gnn_dataset(data_file)

    logger.info(f"Loaded {len(graph_list)} graphs")
    logger.info(f"Node feature dim: {metadata['node_feature_dim']}")
    logger.info(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # 模型配置
    model_config = {
        'gnn_type': gnn_type,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'pooling': pooling
    }

    # 运行交叉验证
    logger.info(f"\nStarting cross-validation...")
    results = run_cross_validation(
        graph_list=graph_list,
        labels=labels,
        model_config=model_config,
        device=device,
        n_folds=n_folds,
        n_repeats=n_repeats,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay
    )

    # 打印结果
    logger.info(f"\n{'=' * 80}")
    logger.info(f"FINAL RESULTS - {branch_type.upper()} BRANCH")
    logger.info(f"{'=' * 80}")
    logger.info(f"Balanced Accuracy: {results['balanced_accuracy_mean']:.4f} ± "
                f"{results['balanced_accuracy_std']:.4f}")
    logger.info(f"Accuracy: {results['accuracy_mean']:.4f} ± "
                f"{results['accuracy_std']:.4f}")
    logger.info(f"AUC: {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
    logger.info(f"F1: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    logger.info(f"{'=' * 80}\n")

    # 保存结果
    result_dict = {
        'dataset': dataset,
        'branch_type': branch_type,
        'model_config': model_config,
        'training_config': {
            'n_folds': n_folds,
            'n_repeats': n_repeats,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay
        },
        'results': results,
        'metadata': metadata
    }

    # 保存为CSV
    df_results = pd.DataFrame({
        'metric': ['balanced_accuracy', 'accuracy', 'auc', 'f1'],
        'mean': [results[f'{m}_mean'] for m in ['balanced_accuracy', 'accuracy', 'auc', 'f1']],
        'std': [results[f'{m}_std'] for m in ['balanced_accuracy', 'accuracy', 'auc', 'f1']]
    })

    csv_file = os.path.join(save_dir, f'{dataset}_{branch_type}_results_{timestamp}.csv')
    df_results.to_csv(csv_file, index=False, float_format='%.4f')
    logger.info(f"Results saved to: {csv_file}")

    # 保存详细结果
    import pickle
    pickle_file = os.path.join(save_dir, f'{dataset}_{branch_type}_detailed_{timestamp}.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(result_dict, f)
    logger.info(f"Detailed results saved to: {pickle_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run single-branch experiments')

    # Data
    parser.add_argument('--dataset', type=str, default='ABIDE',
                        choices=['ABIDE', 'MDD'],
                        help='Dataset name')
    parser.add_argument('--branch_type', type=str, required=True,
                        choices=['functional', 'anatomical'],
                        help='Branch type to test')
    parser.add_argument('--data_folder', type=str, default='./data',
                        help='Root data folder')
    parser.add_argument('--save_dir', type=str, default='./results/single_branch',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')

    # Model
    parser.add_argument('--gnn_type', type=str, default='gat',
                        choices=['gat', 'gcn'],
                        help='GNN type')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'max', 'mean_max'],
                        help='Graph pooling method')

    # Training
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--n_repeats', type=int, default=10,
                        help='Number of CV repeats')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # 运行实验
    results = run_experiment(
        dataset=args.dataset,
        branch_type=args.branch_type,
        data_folder=args.data_folder,
        save_dir=args.save_dir,
        device=args.device,
        gnn_type=args.gnn_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    if results is not None:
        print("\n✅ Experiment completed successfully!")


if __name__ == '__main__':
    main()