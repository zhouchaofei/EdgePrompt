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
                 pooling='mean'):
        """
        Args:
            input_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出类别数
            num_layers: GNN层数
            gnn_type: GNN类型 ('gcn', 'gat', 'sage')
            dropout: dropout比率
            pooling: 图池化方式 ('mean', 'max', 'mean_max')
        """
        super().__init__()

        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.pooling = pooling
        self.dropout = dropout

        # GNN层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim

            if gnn_type == 'gcn':
                self.convs.append(GCNConv(in_dim, out_dim))
            elif gnn_type == 'gat':
                # GAT with 4 attention heads
                self.convs.append(GATConv(in_dim, out_dim // 4, heads=4, concat=True))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(in_dim, out_dim))
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")

            self.batch_norms.append(nn.BatchNorm1d(out_dim))

        # 分类器
        classifier_input_dim = hidden_dim
        if pooling == 'mean_max':
            classifier_input_dim = hidden_dim * 2

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

        # 如果有边权重，使用它
        edge_weight = data.edge_attr.squeeze() if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

        # GNN layers
        for i in range(self.num_layers):
            if self.gnn_type == 'gat':
                # GAT不支持edge_weight
                x = self.convs[i](x, edge_index)
            else:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight)

            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph pooling
        if self.pooling == 'mean':
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

    def __init__(self, input_dim, output_dim=2, pooling='mean'):
        """
        Args:
            input_dim: 节点特征维度
            output_dim: 输出类别数
            pooling: 图池化方式
        """
        super().__init__()

        self.pooling = pooling

        classifier_input_dim = input_dim
        if pooling == 'mean_max':
            classifier_input_dim = input_dim * 2

        self.classifier = nn.Linear(classifier_input_dim, output_dim)

    def forward(self, data):
        """
        Args:
            data: PyG Data对象

        Returns:
            logits: [batch_size, output_dim]
        """
        x, batch = data.x, data.batch

        # Graph pooling
        if self.pooling == 'mean':
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
                 dropout=0.5, pooling='mean'):
        """
        Args:
            input_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出类别数
            dropout: dropout比率
            pooling: 图池化方式
        """
        super().__init__()

        self.pooling = pooling

        classifier_input_dim = input_dim
        if pooling == 'mean_max':
            classifier_input_dim = input_dim * 2

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

        # Graph pooling
        if self.pooling == 'mean':
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
                 num_layers=2, gnn_type='gcn', dropout=0.5, pooling='mean'):
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
        pooling: 池化方式

    Returns:
        model: PyTorch模型
    """
    if model_type == 'linear':
        model = LinearProbe(
            input_dim=input_dim,
            output_dim=output_dim,
            pooling=pooling
        )
    elif model_type == 'mlp':
        model = MLPProbe(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            pooling=pooling
        )
    elif model_type == 'gnn':
        model = SimpleGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=dropout,
            pooling=pooling
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


# if __name__ == '__main__':
#     # 测试代码
#     from torch_geometric.data import Batch
#
#     # 创建假数据
#     data1 = Data(
#         x=torch.randn(116, 11),  # 116个ROI，11维统计特征
#         edge_index=torch.randint(0, 116, (2, 500)),
#         edge_attr=torch.randn(500, 1),
#         y=torch.LongTensor([0])
#     )
#
#     data2 = Data(
#         x=torch.randn(116, 11),
#         edge_index=torch.randint(0, 116, (2, 500)),
#         edge_attr=torch.randn(500, 1),
#         y=torch.LongTensor([1])
#     )
#
#     batch = Batch.from_data_list([data1, data2])
#
#     # 测试不同模型
#     print("Testing models...")
#
#     for model_type in ['linear', 'mlp', 'gnn']:
#         print(f"\n{model_type.upper()}:")
#         model = create_model(
#             model_type=model_type,
#             input_dim=11,
#             hidden_dim=64,
#             output_dim=2
#         )
#
#         output = model(batch)
#         print(f"  Output shape: {output.shape}")
#         print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")