"""
run_baseline.py - 单分支基线实验
支持结构分支/功能分支的独立训练
支持开启/关闭Prompt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool
import pickle
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
import warnings

# 导入模型组件
from model import GIN
from prompt import NodePromptplus, EdgePromptplus
from layers import HGPSLPool

warnings.filterwarnings('ignore')


class SingleBranchGNN(nn.Module):
    """
    单分支GNN模型

    支持两种模式：
    1. struct: 结构分支（固定图 + 可选NodePrompt）
    2. func: 功能分支（加权图 + 可选EdgePrompt）
    """

    def __init__(self, mode, input_dim, hidden_dim=64, num_classes=2,
                 use_prompt=False, num_anchors=5, dropout=0.5):
        super(SingleBranchGNN, self).__init__()
        self.mode = mode
        self.use_prompt = use_prompt

        if mode == 'struct':
            # ===== 结构分支 =====
            # 策略：固定图 -> 用Node Prompt调整特征
            if use_prompt:
                self.prompt = NodePromptplus(input_dim, num_anchors=num_anchors)

            # GIN 1层 + HGPSL Pooling + GIN 2层
            self.gin1 = GIN(num_layer=1, input_dim=input_dim, hidden_dim=hidden_dim, drop_ratio=dropout)
            self.pool = HGPSLPool(hidden_dim, ratio=0.5, sample=True, sparse=True, sl=True)
            self.gin2 = GIN(num_layer=2, input_dim=hidden_dim, hidden_dim=hidden_dim, drop_ratio=dropout)

        elif mode == 'func':
            # ===== 功能分支 =====
            # 策略：加权图 -> 用Edge Prompt调整边权重
            if use_prompt:
                # 3层GIN，每层输入维度
                self.prompt = EdgePromptplus(
                    [input_dim, hidden_dim, hidden_dim],
                    num_anchors=num_anchors
                )

            # 3层GIN
            self.gin = GIN(num_layer=3, input_dim=input_dim, hidden_dim=hidden_dim, drop_ratio=dropout)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, batch = data.x, data.batch

        # 运行时NaN防御
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.mode == 'struct':
            # ===== 结构分支处理 =====
            edge_index = data.edge_index_struct

            # 1. Node Prompt
            if self.use_prompt:
                x = self.prompt.add(x)

            # 2. GIN第1层
            x = self.gin1.convs[0](x, edge_index, edge_prompt=False)
            x = F.relu(self.gin1.batch_norms[0](x))

            # 3. HGPSL Pooling（修复：创建1D伪edge_attr）
            pseudo_edge_attr = x.new_ones(edge_index.size(1))
            x, edge_index, edge_attr, batch = self.pool(x, edge_index, pseudo_edge_attr, batch)

            # 4. GIN第2层（2个子层）
            for i in range(len(self.gin2.convs)):
                x = self.gin2.convs[i](x, edge_index, edge_prompt=False)
                x = F.relu(self.gin2.batch_norms[i](x))

        elif self.mode == 'func':
            # ===== 功能分支处理 =====
            # 只使用ROI节点（去除Global节点）
            mask = data.roi_mask.bool()

            # 创建节点索引映射：旧索引 -> 新索引
            # 例如：[0,1,2,...,115, 116, 117,...,231, 232] -> [0,1,...,115, 116,...,231]
            old_to_new = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
            old_to_new[mask] = torch.arange(mask.sum(), device=x.device)

            # 过滤节点和batch
            x = x[mask]
            batch = batch[mask]

            # 重新映射edge_index
            edge_index = data.edge_index_func
            row, col = edge_index[0], edge_index[1]

            # 只保留两端节点都在ROI中的边
            valid_edges = mask[row] & mask[col]
            row = old_to_new[row[valid_edges]]
            col = old_to_new[col[valid_edges]]
            edge_index = torch.stack([row, col], dim=0)

            # 逐层GIN（支持EdgePrompt）
            for i in range(self.gin.num_layer):
                # 获取edge prompt（如果使用）
                if self.use_prompt:
                    edge_prompt = self.prompt.get_prompt(x, edge_index, layer=i)
                else:
                    edge_prompt = False

                # GIN卷积
                x = self.gin.convs[i](x, edge_index, edge_prompt=edge_prompt)
                x = self.gin.batch_norms[i](x)

                # 激活函数（最后一层不用ReLU）
                if i < self.gin.num_layer - 1:
                    x = F.relu(x)

        # Readout（Mean + Max Pooling）
        x = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch)
        ], dim=1)

        # 分类
        return self.classifier(x)


def train_epoch(model, loader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        loss = F.cross_entropy(out, batch.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        probs = F.softmax(out, dim=1)
        preds = out.argmax(dim=1)

        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs[:, 1].cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = {
        'acc': np.mean(y_true == y_pred),
        'bacc': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    }

    return metrics


def run_experiment(args):
    """运行完整实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 80}")
    print(f"实验配置: Mode={args.mode}, Prompt={args.use_prompt}")
    print(f"{'=' * 80}")

    # 加载数据
    with open(args.data_path, 'rb') as f:
        data_pack = pickle.load(f)

    graphs = data_pack['graph_list']
    labels = data_pack['labels']

    print(f"数据集: {data_pack['metadata']['dataset']}")
    print(f"被试数: {len(graphs)}")
    print(f"标签分布: {np.bincount(labels)}")
    print(f"节点特征维度: {graphs[0].x.shape[1]}")

    # K-Fold交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(graphs, labels)):
        print(f"\n{'=' * 80}")
        print(f"Fold {fold + 1}/5")
        print(f"{'=' * 80}")

        # 创建DataLoader
        train_loader = DataLoader(
            [graphs[i] for i in train_idx],
            batch_size=args.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            [graphs[i] for i in val_idx],
            batch_size=args.batch_size,
            shuffle=False
        )

        # 初始化模型
        model = SingleBranchGNN(
            mode=args.mode,
            input_dim=graphs[0].x.shape[1],
            hidden_dim=args.hidden_dim,
            num_classes=2,
            use_prompt=args.use_prompt,
            num_anchors=args.num_anchors,
            dropout=args.dropout
        ).to(device)

        # 优化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # 训练
        best_val_bacc = 0
        best_val_metrics = None
        patience_counter = 0

        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)

            # 保存最佳模型
            if val_metrics['bacc'] > best_val_bacc:
                best_val_bacc = val_metrics['bacc']
                best_val_metrics = val_metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}: Loss={train_loss:.4f}, Val BACC={val_metrics['bacc']:.4f}")

        # 保存fold结果
        fold_results.append(best_val_metrics)

        print(f"\nFold {fold + 1} 最佳结果:")
        for k, v in best_val_metrics.items():
            print(f"  {k.upper()}: {v:.4f}")

    # 汇总结果
    print(f"\n{'=' * 80}")
    print("5-Fold 汇总结果")
    print(f"{'=' * 80}")

    for metric in ['acc', 'bacc', 'f1', 'auc']:
        values = [r[metric] for r in fold_results]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric.upper()}: {mean:.4f} ± {std:.4f}")

    return fold_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='单分支GNN基线实验')

    # 数据参数
    parser.add_argument('--data_path', type=str,
                        default='./data/gnn_datasets/ABIDE_DualBranch.pkl')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['struct', 'func'],
                        help='分支模式: struct(结构) 或 func(功能)')
    parser.add_argument('--use_prompt', action='store_true',
                        help='是否使用Prompt')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_anchors', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)

    args = parser.parse_args()

    # 打印实验类型
    prompt_status = "有Prompt" if args.use_prompt else "无Prompt"
    branch_name = "结构分支" if args.mode == 'struct' else "功能分支"

    print(f"\n{'=' * 80}")
    print(f"实验类型: {branch_name} - {prompt_status}")
    print(f"{'=' * 80}")

    if args.mode == 'struct' and args.use_prompt:
        print("策略: 使用NodePromptplus调整节点特征")
    elif args.mode == 'func' and args.use_prompt:
        print("策略: 使用EdgePromptplus调整边权重")

    # 运行实验
    run_experiment(args)

    print(f"\n{'=' * 80}")
    print("✅ 实验完成！")
    print(f"{'=' * 80}")