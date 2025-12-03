"""
简单GNN模型 - 修复版
主要修改：添加输入层BatchNorm，提高收敛性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool


# class SimpleGNN(nn.Module):
#     """简单的GNN分类器（增强版）"""
#
#     def __init__(self, input_dim, hidden_dim=64, output_dim=2,
#                  num_layers=2, gnn_type='gcn', dropout=0.5,
#                  pooling='mean', num_rois=116):
#         super().__init__()
#
#         self.num_layers = num_layers
#         self.gnn_type = gnn_type
#         self.pooling = pooling
#         self.dropout = dropout
#         self.num_rois = num_rois
#
#         # 🔥 关键修改：输入层BatchNorm
#         self.input_bn = nn.BatchNorm1d(input_dim)
#
#         # GNN层
#         self.convs = nn.ModuleList()
#         self.batch_norms = nn.ModuleList()
#
#         for i in range(num_layers):
#             in_dim = input_dim if i == 0 else hidden_dim
#             out_dim = hidden_dim
#
#             if gnn_type == 'gcn':
#                 self.convs.append(GCNConv(in_dim, out_dim))
#             elif gnn_type == 'gat':
#                 self.convs.append(GATConv(in_dim, out_dim // 4, heads=4, concat=True))
#             elif gnn_type == 'sage':
#                 self.convs.append(SAGEConv(in_dim, out_dim))
#             else:
#                 raise ValueError(f"Unknown gnn_type: {gnn_type}")
#
#             self.batch_norms.append(nn.BatchNorm1d(out_dim))
#
#         # 根据pooling方式确定分类器输入维度
#         if pooling == 'flatten':
#             classifier_input_dim = hidden_dim * num_rois
#         elif pooling == 'mean_max':
#             classifier_input_dim = hidden_dim * 2
#         else:
#             classifier_input_dim = hidden_dim
#
#         self.classifier = nn.Sequential(
#             nn.Linear(classifier_input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, output_dim)
#         )
#
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#
#         # 🔥 输入归一化
#         x = self.input_bn(x)
#
#         # 使用边权重
#         if hasattr(data, 'edge_attr') and data.edge_attr is not None:
#             edge_weight = data.edge_attr.squeeze()
#             if torch.isnan(edge_weight).any() or torch.isinf(edge_weight).any():
#                 edge_weight = torch.nan_to_num(edge_weight, nan=0.0, posinf=0.0, neginf=0.0)
#         else:
#             edge_weight = None
#
#         # GNN layers
#         for i in range(self.num_layers):
#             if self.gnn_type == 'gat':
#                 x = self.convs[i](x, edge_index)
#             else:
#                 x = self.convs[i](x, edge_index, edge_weight=edge_weight)
#
#             x = self.batch_norms[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # Graph pooling/readout
#         if self.pooling == 'flatten':
#             batch_size = int(batch.max().item() + 1)
#             x = x.view(batch_size, self.num_rois * x.size(1))
#         elif self.pooling == 'mean':
#             x = global_mean_pool(x, batch)
#         elif self.pooling == 'max':
#             x = global_max_pool(x, batch)
#         elif self.pooling == 'mean_max':
#             x_mean = global_mean_pool(x, batch)
#             x_max = global_max_pool(x, batch)
#             x = torch.cat([x_mean, x_max], dim=1)
#
#         # Classification
#         logits = self.classifier(x)
#
#         return logits
# simple_gnn.py 修改后的 SimpleGNN 类

class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2,
                 gnn_type='gcn', dropout=0.5, pooling='mean', num_rois=116):  # <--- 🔥 加上 num_rois
        super(SimpleGNN, self).__init__()

        self.pooling = pooling
        self.num_rois = num_rois  # 记录节点数，用于 flatten

        # 特征投影
        self.feature_projector = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GNN Layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # ... (中间 GNN 层定义保持不变) ...
        # 注意：如果 input_dim 很大，投影后的 hidden_dim 最好大一点，比如 128

        if gnn_type == 'gcn':
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif gnn_type == 'gat':
            self.convs.append(GATConv(hidden_dim, hidden_dim))

        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # 🔥 计算分类器的输入维度
        if self.pooling == 'flatten':
            # Flatten 模式下，输入维度是 hidden_dim * 节点数
            clf_input_dim = hidden_dim * self.num_rois
        elif self.pooling == 'mean_max':
            clf_input_dim = hidden_dim * 2
        else:
            clf_input_dim = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(clf_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Project
        x = self.feature_projector(x)

        # 2. GNN
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)

        # 3. Pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling == 'mean_max':
            x1 = global_mean_pool(x, batch)
            x2 = global_max_pool(x, batch)
            x = torch.cat([x1, x2], dim=1)
        elif self.pooling == 'flatten':
            # 🔥 这里的 reshape 逻辑必须正确
            # x shape: [Batch_size * Num_Nodes, Hidden]
            # 变成: [Batch_size, Num_Nodes * Hidden]
            batch_size = int(batch.max().item() + 1)
            # 确保 x 可以被整除 (处理 drop_last=False 可能导致的边缘情况，但通常 PyG 数据加载器 OK)
            x = x.view(batch_size, -1)

            # 4. Classify
        x = self.classifier(x)
        return x


class LinearProbe(nn.Module):
    """线性探针模型（增强版）"""

    def __init__(self, input_dim, output_dim=2, pooling='mean', num_rois=116):
        super().__init__()

        self.pooling = pooling
        self.num_rois = num_rois

        # 🔥 关键修改：输入层BatchNorm
        self.input_bn = nn.BatchNorm1d(input_dim)

        # 根据pooling方式确定分类器输入维度
        if pooling == 'flatten':
            classifier_input_dim = input_dim * num_rois
        elif pooling == 'mean_max':
            classifier_input_dim = input_dim * 2
        else:
            classifier_input_dim = input_dim

        self.classifier = nn.Linear(classifier_input_dim, output_dim)

    def forward(self, data):
        x, batch = data.x, data.batch

        # 🔥 输入归一化
        x = self.input_bn(x)

        # Graph pooling
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
    """MLP探针模型（增强版）"""

    def __init__(self, input_dim, hidden_dim=128, output_dim=2,
                 dropout=0.5, pooling='mean', num_rois=116):
        super().__init__()

        self.pooling = pooling
        self.num_rois = num_rois

        # 🔥 关键修改：输入层BatchNorm
        self.input_bn = nn.BatchNorm1d(input_dim)

        # 根据pooling方式确定分类器输入维度
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
        x, batch = data.x, data.batch

        # 🔥 输入归一化
        x = self.input_bn(x)

        # Graph pooling
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
                 num_rois=116):
    """
    创建模型的工厂函数（增强版）

    所有模型都包含输入层BatchNorm
    """
    if model_type == 'linear':
        model = LinearProbe(
            input_dim=input_dim,
            output_dim=output_dim,
            pooling=pooling,
            num_rois=num_rois
        )
    elif model_type == 'mlp':
        model = MLPProbe(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            pooling=pooling,
            num_rois=num_rois
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
            num_rois=num_rois
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model