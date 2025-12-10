import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dual_model_final import DualBranchGNN, compute_total_loss
import pickle
import argparse
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class EarlyStopping:
    """Early Stopping工具类"""

    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


def train_epoch(model, loader, optimizer, device, lambda_orth=0.1, lambda_cons=0.05, use_consistency=False):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    loss_dict_sum = {'cls': 0, 'orth': 0, 'cons': 0, 'total': 0}

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # 前向传播
        if use_consistency:
            logits, z_struct, z_func, logits_struct, logits_func = model(batch)
        else:
            logits, z_struct, z_func = model(batch)
            logits_struct, logits_func = None, None

        # 计算损失
        loss, loss_dict = compute_total_loss(
            logits, batch.y, z_struct, z_func,
            logits_struct, logits_func,
            lambda_orth=lambda_orth,
            lambda_cons=lambda_cons
        )

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        total_loss += loss.item()
        for k in loss_dict:
            loss_dict_sum[k] += loss_dict[k]

    # 计算平均损失
    n_batches = len(loader)
    avg_loss_dict = {k: v / n_batches for k, v in loss_dict_sum.items()}

    return total_loss / n_batches, avg_loss_dict


@torch.no_grad()
def evaluate(model, loader, device, use_consistency=False):
    """评估模型"""
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for batch in loader:
        batch = batch.to(device)

        if use_consistency:
            logits, _, _, _, _ = model(batch)
        else:
            logits, _, _ = model(batch)

        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs[:, 1].cpu().numpy())  # 类别1的概率

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # 计算指标
    metrics = {
        'acc': np.mean(y_true == y_pred),
        'bacc': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    }

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics


def train_fold(train_loader, val_loader, test_loader, args, device):
    """训练单个fold"""

    # 初始化模型
    model = DualBranchGNN(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        pooling_ratio=args.pooling_ratio,
        dropout=args.dropout,
        num_anchors=args.num_anchors,
        use_consistency=args.use_consistency
    ).to(device)

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False
    )

    # Early Stopping
    early_stopping = EarlyStopping(patience=args.patience)

    best_val_bacc = 0
    best_test_metrics = None

    for epoch in range(args.epochs):
        # 训练
        train_loss, train_loss_dict = train_epoch(
            model, train_loader, optimizer, device,
            lambda_orth=args.lambda_orth,
            lambda_cons=args.lambda_cons,
            use_consistency=args.use_consistency
        )

        # 验证
        val_metrics = evaluate(model, val_loader, device, args.use_consistency)
        test_metrics = evaluate(model, test_loader, device, args.use_consistency)

        # 学习率调度
        scheduler.step(val_metrics['bacc'])

        # 保存最佳模型
        if val_metrics['bacc'] > best_val_bacc:
            best_val_bacc = val_metrics['bacc']
            best_test_metrics = test_metrics.copy()

            if args.save_model:
                torch.save(model.state_dict(), args.model_save_path)

        # Early Stopping
        early_stopping(val_metrics['bacc'])
        if early_stopping.early_stop:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}/{args.epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Val BACC: {val_metrics['bacc']:.4f} | "
                  f"Test BACC: {test_metrics['bacc']:.4f}")

    return best_test_metrics


def k_fold_train(dataset_path, args, device):
    """K-Fold交叉验证训练"""

    print("=" * 80)
    print("加载数据...")
    print("=" * 80)

    # 加载数据
    with open(dataset_path, 'rb') as f:
        data_pack = pickle.load(f)

    graph_list = data_pack['graph_list']
    labels = data_pack['labels']

    print(f"数据集: {data_pack['metadata']['dataset']}")
    print(f"被试数: {len(graph_list)}")
    print(f"标签分布: {np.bincount(labels)}")
    print(f"节点特征维度: {graph_list[0].x.shape[1]}")

    # K-Fold分割
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    all_fold_metrics = []

    print("\n" + "=" * 80)
    print("开始K-Fold训练...")
    print("=" * 80)

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(graph_list, labels)):
        print(f"\n{'=' * 80}")
        print(f"Fold {fold + 1}/{args.n_folds}")
        print(f"{'=' * 80}")

        # 进一步划分训练集和验证集
        trainval_labels = labels[trainval_idx]
        train_idx, val_idx = [], []

        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        for tr_idx, vl_idx in skf_inner.split(trainval_idx, trainval_labels):
            train_idx = trainval_idx[tr_idx]
            val_idx = trainval_idx[vl_idx]
            break  # 只取第一个split

        # 创建DataLoader
        train_loader = DataLoader(
            [graph_list[i] for i in train_idx],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            [graph_list[i] for i in val_idx],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        test_loader = DataLoader(
            [graph_list[i] for i in test_idx],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        print(f"训练集: {len(train_idx)} | 验证集: {len(val_idx)} | 测试集: {len(test_idx)}")

        # 训练
        fold_metrics = train_fold(train_loader, val_loader, test_loader, args, device)
        all_fold_metrics.append(fold_metrics)

        # 打印fold结果
        print(f"\nFold {fold + 1} 测试结果:")
        for k, v in fold_metrics.items():
            print(f"  {k.upper()}: {v:.4f}")

    # 汇总结果
    print("\n" + "=" * 80)
    print("K-Fold 汇总结果")
    print("=" * 80)

    metrics_summary = {}
    for key in all_fold_metrics[0].keys():
        values = [m[key] for m in all_fold_metrics]
        metrics_summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }

    for metric, stats in metrics_summary.items():
        print(f"{metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  各Fold: {[f'{v:.4f}' for v in stats['values']]}")

    # 保存结果
    if args.save_results:
        results = {
            'args': vars(args),
            'fold_metrics': all_fold_metrics,
            'summary': metrics_summary
        }
        result_path = os.path.join(args.result_dir, f'{args.dataset}_results.pkl')
        os.makedirs(args.result_dir, exist_ok=True)
        with open(result_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n结果已保存至: {result_path}")


def main():
    parser = argparse.ArgumentParser(description='双分支GNN训练脚本')

    # 数据参数
    parser.add_argument('--dataset', type=str, default='ABIDE', choices=['ABIDE', 'MDD'])
    parser.add_argument('--data_path', type=str, default='./data/gnn_datasets/ABIDE_DualBranch.pkl')
    parser.add_argument('--input_dim', type=int, default=308, help='输入特征维度')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=3, help='GNN层数')
    parser.add_argument('--pooling_ratio', type=float, default=0.5, help='结构分支pooling比例')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_anchors', type=int, default=5, help='Prompt锚点数')

    # 损失函数参数
    parser.add_argument('--lambda_orth', type=float, default=0.1, help='正交损失权重')
    parser.add_argument('--lambda_cons', type=float, default=0.05, help='一致性损失权重')
    parser.add_argument('--use_consistency', action='store_true', help='是否使用一致性损失')

    # 训练参数
    parser.add_argument('--n_folds', type=int, default=5, help='K-Fold数量')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)

    # 保存参数
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints/best_model.pth')
    parser.add_argument('--save_results', action='store_true', default=True)
    parser.add_argument('--result_dir', type=str, default='./results')

    # 设备
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 打印配置
    print("=" * 80)
    print("训练配置")
    print("=" * 80)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # 开始训练
    device = torch.device(args.device)
    k_fold_train(args.data_path, args, device)


if __name__ == '__main__':
    main()