"""
简单的GNN模型用于验证功能图和节点特征组合
包括GCN和GAT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class SimpleGCN(nn.Module):
    """
    简单的GCN分类器
    """

    def __init__(self, in_dim, hidden_dim=64, num_classes=2, dropout=0.5):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        Args:
            x: [N, in_dim] 节点特征
            edge_index: [2, E] 边索引
            edge_weight: [E] 边权重
            batch: [N] batch索引（用于图级别任务）

        Returns:
            out: [batch_size, num_classes] 分类logits
        """
        # GCN layers
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        # Graph-level pooling
        if batch is None:
            # 单图情况：全局平均池化
            x = x.mean(dim=0, keepdim=True)
        else:
            # 多图情况：按batch池化
            x = global_mean_pool(x, batch)

        # 分类
        out = self.classifier(x)

        return out


class SimpleGAT(nn.Module):
    """
    简单的GAT分类器
    """

    def __init__(self, in_dim, hidden_dim=64, num_classes=2, heads=4, dropout=0.5):
        super().__init__()

        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        Args:
            x: [N, in_dim] 节点特征
            edge_index: [2, E] 边索引
            edge_weight: [E] 边权重（GAT会忽略，使用注意力机制）
            batch: [N] batch索引

        Returns:
            out: [batch_size, num_classes] 分类logits
        """
        # GAT layers
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # Graph-level pooling
        if batch is None:
            x = x.mean(dim=0, keepdim=True)
        else:
            x = global_mean_pool(x, batch)

        # 分类
        out = self.classifier(x)

        return out


class LinearProbe(nn.Module):
    """
    线性探针（用于快速基线测试）
    """

    def __init__(self, in_dim, num_classes=2):
        super().__init__()

        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x, edge_index=None, edge_weight=None, batch=None):
        """
        Args:
            x: [N, in_dim] 节点特征
            其他参数不使用，保持接口一致

        Returns:
            out: [batch_size, num_classes]
        """
        # 全局平均池化
        if batch is None:
            x = x.mean(dim=0, keepdim=True)
        else:
            x = global_mean_pool(x, batch)

        out = self.classifier(x)

        return out


def get_model(model_name, in_dim, hidden_dim=64, num_classes=2, **kwargs):
    """
    获取模型实例

    Args:
        model_name: 'gcn', 'gat', 或 'linear'
        in_dim: 输入维度
        hidden_dim: 隐藏层维度
        num_classes: 类别数
        **kwargs: 额外参数

    Returns:
        model: 模型实例
    """
    if model_name.lower() == 'gcn':
        dropout = kwargs.get('dropout', 0.5)
        return SimpleGCN(in_dim, hidden_dim, num_classes, dropout)

    elif model_name.lower() == 'gat':
        heads = kwargs.get('heads', 4)
        dropout = kwargs.get('dropout', 0.5)
        return SimpleGAT(in_dim, hidden_dim, num_classes, heads, dropout)

    elif model_name.lower() == 'linear':
        return LinearProbe(in_dim, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # 测试模型
    print("Testing GNN models...")

    # 模拟数据
    n_nodes = 116
    in_dim = 7  # 统计特征维度
    num_graphs = 4

    # 节点特征
    x = torch.randn(n_nodes * num_graphs, in_dim)

    # 边（全连接图示例）
    edge_index = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            edge_index.append([i, j])
            edge_index.append([j, i])
    edge_index = torch.LongTensor(edge_index).t()

    # 扩展到多图
    full_edge_index = []
    for g in range(num_graphs):
        offset = g * n_nodes
        full_edge_index.append(edge_index + offset)
    full_edge_index = torch.cat(full_edge_index, dim=1)

    # Batch索引
    batch = torch.cat([torch.full((n_nodes,), i) for i in range(num_graphs)])

    # 测试GCN
    print("\n1. Testing GCN:")
    model_gcn = SimpleGCN(in_dim=in_dim, hidden_dim=64, num_classes=2)
    out_gcn = model_gcn(x, full_edge_index, batch=batch)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out_gcn.shape}")
    print(f"   Logits: {out_gcn}")

    # 测试GAT
    print("\n2. Testing GAT:")
    model_gat = SimpleGAT(in_dim=in_dim, hidden_dim=64, num_classes=2)
    out_gat = model_gat(x, full_edge_index, batch=batch)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out_gat.shape}")
    print(f"   Logits: {out_gat}")

    # 测试Linear
    print("\n3. Testing Linear Probe:")
    model_linear = LinearProbe(in_dim=in_dim, num_classes=2)
    out_linear = model_linear(x, batch=batch)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out_linear.shape}")
    print(f"   Logits: {out_linear}")

    print("\n✅ All models tested successfully!")

