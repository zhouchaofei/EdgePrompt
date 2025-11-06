"""
简单的GNN baseline模型
用于特征选择实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool


class SimpleGCN(nn.Module):
    """
    简单的GCN模型
    用于验证功能图和节点特征的有效性
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_classes=2,
        num_layers=2,
        dropout=0.5,
        pooling='mean'
    ):
        """
        Args:
            input_dim: 输入特征维度（自动适配）
            hidden_dim: 隐藏层维度
            num_classes: 类别数
            num_layers: GCN层数
            dropout: Dropout比例
            pooling: 图池化方式 ('mean' 或 'add')
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # GCN层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 第一层：input_dim -> hidden_dim
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # 中间层：hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        """
        Args:
            data: PyG Data对象
                - x: [N, input_dim]
                - edge_index: [2, E]
                - edge_attr: [E, 1]
                - batch: [N]

        Returns:
            out: [batch_size, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = data.edge_attr.squeeze() if hasattr(data, 'edge_attr') else None

        # GCN层
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 图级池化
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)

        # 分类
        out = self.classifier(x)

        return out


class LinearProbe(nn.Module):
    """
    线性probe模型
    用于快速测试特征质量
    """

    def __init__(self, input_dim, num_classes=2, pooling='mean'):
        """
        Args:
            input_dim: 输入特征维度（自动适配）
            num_classes: 类别数
            pooling: 图池化方式
        """
        super().__init__()

        self.pooling = pooling
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, data):
        x, batch = data.x, data.batch

        # 图级池化
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)

        # 分类
        out = self.classifier(x)

        return out
