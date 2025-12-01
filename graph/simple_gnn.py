"""
简单GNN模型用于验证功能图和节点特征组合
支持多种GNN架构：GCN, GAT, GraphSAGE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool


class SimpleGNN(nn.Module):
    """简单的GNN分类器"""

    def __init__(self, input_dim, hidden_dim=64, output_dim=2,
                 num_layers=2, gnn_type='gcn', dropout=0.5,
                 pooling='mean', num_rois=116):  # 🔥 新增 num_rois 参数
        """
        Args:
            input_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出类别数
            num_layers: GNN层数
            gnn_type: GNN类型 ('gcn', 'gat', 'sage')
            dropout: dropout比率
            pooling: 图池化方式 ('mean', 'max', 'mean_max', 'flatten')
            num_rois: ROI数量（用于flatten pooling）
        """
        super().__init__()

        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.pooling = pooling
        self.dropout = dropout
        self.num_rois = num_rois  # 🔥 保存 num_rois

        # GNN层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim

            if gnn_type == 'gcn':
                self.convs.append(GCNConv(in_dim, out_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(in_dim, out_dim // 4, heads=4, concat=True))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(in_dim, out_dim))
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")

            self.batch_norms.append(nn.BatchNorm1d(out_dim))

        # 🔥 根据pooling方式确定分类器输入维度
        if pooling == 'flatten':
            classifier_input_dim = hidden_dim * num_rois  # 例如: 64 * 116 = 7424
        elif pooling == 'mean_max':
            classifier_input_dim = hidden_dim * 2
        else:  # 'mean' or 'max'
            classifier_input_dim = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        """
        Args:
            data: PyG Data对象

        Returns:
            logits: [batch_size, output_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 使用边权重
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weight = data.edge_attr.squeeze()
            if torch.isnan(edge_weight).any() or torch.isinf(edge_weight).any():
                print("⚠️ 检测到NaN/Inf边权重，替换为0")
                edge_weight = torch.nan_to_num(edge_weight, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            edge_weight = None

        # GNN layers
        for i in range(self.num_layers):
            if self.gnn_type == 'gat':
                x = self.convs[i](x, edge_index)
            else:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight)

            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 🔥 Graph pooling/readout
        if self.pooling == 'flatten':
            # 将每个图的所有节点特征展平
            batch_size = int(batch.max().item() + 1)
            # 确保节点按batch顺序排列（PyG标准行为）
            x = x.view(batch_size, self.num_rois * x.size(1))  # [batch_size, num_rois * hidden_dim]

        elif self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'mean_max':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)

        # Classification
        logits = self.classifier(x)

        return logits


class LinearProbe(nn.Module):
    """线性探针模型（用于快速验证特征质量）"""

    def __init__(self, input_dim, output_dim=2, pooling='mean', num_rois=116):  # 🔥 新增 num_rois
        """
        Args:
            input_dim: 节点特征维度
            output_dim: 输出类别数
            pooling: 图池化方式
            num_rois: ROI数量（用于flatten pooling）
        """
        super().__init__()

        self.pooling = pooling
        self.num_rois = num_rois  # 🔥 保存 num_rois

        # 🔥 根据pooling方式确定分类器输入维度
        if pooling == 'flatten':
            classifier_input_dim = input_dim * num_rois
        elif pooling == 'mean_max':
            classifier_input_dim = input_dim * 2
        else:
            classifier_input_dim = input_dim

        self.classifier = nn.Linear(classifier_input_dim, output_dim)

    def forward(self, data):
        """
        Args:
            data: PyG Data对象

        Returns:
            logits: [batch_size, output_dim]
        """
        x, batch = data.x, data.batch

        # 🔥 Graph pooling
        if self.pooling == 'flatten':
            batch_size = int(batch.max().item() + 1)
            x = x.view(batch_size, self.num_rois * x.size(1))

        elif self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'mean_max':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)

        # Classification
        logits = self.classifier(x)

        return logits


class MLPProbe(nn.Module):
    """MLP探针模型（稍复杂的baseline）"""

    def __init__(self, input_dim, hidden_dim=128, output_dim=2,
                 dropout=0.5, pooling='mean', num_rois=116):  # 🔥 新增 num_rois
        """
        Args:
            input_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出类别数
            dropout: dropout比率
            pooling: 图池化方式
            num_rois: ROI数量（用于flatten pooling）
        """
        super().__init__()

        self.pooling = pooling
        self.num_rois = num_rois  # 🔥 保存 num_rois

        # 🔥 根据pooling方式确定分类器输入维度
        if pooling == 'flatten':
            classifier_input_dim = input_dim * num_rois
        elif pooling == 'mean_max':
            classifier_input_dim = input_dim * 2
        else:
            classifier_input_dim = input_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data):
        """
        Args:
            data: PyG Data对象

        Returns:
            logits: [batch_size, output_dim]
        """
        x, batch = data.x, data.batch

        # 🔥 Graph pooling
        if self.pooling == 'flatten':
            batch_size = int(batch.max().item() + 1)
            x = x.view(batch_size, self.num_rois * x.size(1))

        elif self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'mean_max':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)

        # Classification
        logits = self.classifier(x)

        return logits


def create_model(model_type, input_dim, hidden_dim=64, output_dim=2,
                 num_layers=2, gnn_type='gcn', dropout=0.5, pooling='mean',
                 num_rois=116):  # 🔥 新增 num_rois 参数
    """
    创建模型的工厂函数

    Args:
        model_type: 'linear', 'mlp', 'gnn'
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        num_layers: GNN层数
        gnn_type: GNN类型
        dropout: dropout比率
        pooling: 池化方式 ('mean', 'max', 'mean_max', 'flatten')
        num_rois: ROI数量

    Returns:
        model: PyTorch模型
    """
    if model_type == 'linear':
        model = LinearProbe(
            input_dim=input_dim,
            output_dim=output_dim,
            pooling=pooling,
            num_rois=num_rois  # 🔥 传递参数
        )
    elif model_type == 'mlp':
        model = MLPProbe(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            pooling=pooling,
            num_rois=num_rois  # 🔥 传递参数
        )
    elif model_type == 'gnn':
        model = SimpleGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=dropout,
            pooling=pooling,
            num_rois=num_rois  # 🔥 传递参数
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model