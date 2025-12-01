"""
SF-DPL Model: Structure-Functional Decoupling with Prompt Learning
解决GNN在脑网络分析中的位置丢失问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool


class SF_DPL_Model(nn.Module):
    """
    带有 Node Prompt 的 GNN 模型

    核心创新：
    1. Node Prompt: 给每个ROI一个可学习的位置编码
    2. Enhanced Pooling: Mean + Max 结合
    3. 更强的正则化
    """

    def __init__(self, input_dim, hidden_dim=64, output_dim=2,
                 num_layers=2, dropout=0.5,
                 use_node_prompt=True, use_edge_prompt=False,
                 num_rois=116, gnn_type='gcn'):
        """
        Args:
            input_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出类别数
            num_layers: GNN层数
            dropout: dropout比率
            use_node_prompt: 是否使用节点提示
            use_edge_prompt: 是否使用边提示（暂未实现）
            num_rois: ROI数量
            gnn_type: GNN类型 ('gcn' or 'gat')
        """
        super().__init__()

        self.use_node_prompt = use_node_prompt
        self.use_edge_prompt = use_edge_prompt
        self.num_rois = num_rois
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout

        print(f"\n{'='*60}")
        print(f"初始化 SF-DPL 模型")
        print(f"{'='*60}")
        print(f"  Node Prompt: {use_node_prompt}")
        print(f"  GNN类型: {gnn_type}")
        print(f"  ROI数量: {num_rois}")
        print(f"  隐藏维度: {hidden_dim}")
        print(f"{'='*60}\n")

        # ===== 1. Node Prompt =====
        # 核心创新：给每个ROI一个独特的可学习向量
        # 形状: [1, num_rois, input_dim]
        if use_node_prompt:
            self.node_prompt = nn.Parameter(
                torch.randn(1, num_rois, input_dim) * 0.02
            )
            print(f"  ✓ Node Prompt 参数: {self.node_prompt.numel():,}")

        # ===== 2. GNN Layers =====
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim

            if gnn_type == 'gcn':
                self.convs.append(GCNConv(in_dim, out_dim))
            elif gnn_type == 'gat':
                # GAT with 4 heads
                self.convs.append(
                    GATConv(in_dim, out_dim // 4, heads=4, concat=True)
                )
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")

            self.batch_norms.append(nn.BatchNorm1d(out_dim))

        # ===== 3. Readout & Classifier =====
        # 使用 Mean + Max pooling
        classifier_input_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  ✓ 总参数量: {total_params:,}\n")

    def apply_node_prompt(self, x, batch):
        """
        应用 Node Prompt

        Args:
            x: [num_total_nodes, feature_dim]
            batch: [num_total_nodes]

        Returns:
            x: [num_total_nodes, feature_dim] (加上了prompt)
        """
        if not self.use_node_prompt:
            return x

        try:
            batch_size = batch.max().item() + 1

            # 将 x 重塑为 [batch_size, num_rois, feature_dim]
            x_reshaped = x.view(batch_size, self.num_rois, -1)

            # 广播加法: [B, N, D] + [1, N, D]
            x_reshaped = x_reshaped + self.node_prompt

            # 变回 PyG 格式: [B*N, D]
            x = x_reshaped.view(-1, x.shape[1])

            return x

        except Exception as e:
            # 如果形状不匹配，跳过 prompt
            print(f"  ⚠️  Node Prompt 应用失败: {e}")
            return x

    def forward(self, data):
        """
        前向传播

        Args:
            data: PyG Data对象

        Returns:
            logits: [batch_size, output_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 获取边权重
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weight = data.edge_attr.squeeze()

            # 清理无效值
            if torch.isnan(edge_weight).any() or torch.isinf(edge_weight).any():
                edge_weight = torch.nan_to_num(
                    edge_weight, nan=0.0, posinf=0.0, neginf=0.0
                )
        else:
            edge_weight = None

        # ===== Step 1: Apply Node Prompt =====
        x = self.apply_node_prompt(x, batch)

        # ===== Step 2: GNN Propagation =====
        for i in range(self.num_layers):
            if self.gnn_type == 'gat':
                # GAT 不支持 edge_weight
                x = self.convs[i](x, edge_index)
            else:
                # GCN 支持 edge_weight
                x = self.convs[i](x, edge_index, edge_weight=edge_weight)

            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # ===== Step 3: Graph Pooling =====
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        # 拼接
        x = torch.cat([x_mean, x_max], dim=1)

        # ===== Step 4: Classification =====
        logits = self.classifier(x)

        return logits


class LinearProbe(nn.Module):
    """线性探针（用于对比）"""

    def __init__(self, input_dim, output_dim=2, pooling='mean_max', num_rois=116):
        super().__init__()

        self.pooling = pooling

        classifier_input_dim = input_dim
        if pooling == 'mean_max':
            classifier_input_dim = input_dim * 2

        self.classifier = nn.Linear(classifier_input_dim, output_dim)

    def forward(self, data):
        x, batch = data.x, data.batch

        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'mean_max':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)

        return self.classifier(x)


class MLPProbe(nn.Module):
    """MLP探针（用于对比）"""

    def __init__(self, input_dim, hidden_dim=128, output_dim=2,
                 dropout=0.5, pooling='mean_max', num_rois=116):
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
        x, batch = data.x, data.batch

        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'mean_max':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)

        return self.classifier(x)


def create_model(model_type, input_dim, hidden_dim=64, output_dim=2,
                 num_layers=2, gnn_type='gcn', dropout=0.5,
                 pooling='mean_max', num_rois=116,
                 use_node_prompt=True, use_edge_prompt=False):
    """
    模型工厂函数

    Args:
        model_type: 'sf_dpl', 'linear', 'mlp'
        其他参数同上

    Returns:
        model: PyTorch模型
    """
    if model_type == 'sf_dpl':
        model = SF_DPL_Model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_node_prompt=use_node_prompt,
            use_edge_prompt=use_edge_prompt,
            num_rois=num_rois,
            gnn_type=gnn_type
        )
    elif model_type == 'linear':
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
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


if __name__ == '__main__':
    # 测试代码
    from torch_geometric.data import Data, Batch

    print("Testing SF-DPL Model...")

    # 创建假数据
    data1 = Data(
        x=torch.randn(116, 64),  # 116个ROI，64维特征
        edge_index=torch.randint(0, 116, (2, 500)),
        edge_attr=torch.randn(500, 1),
        y=torch.LongTensor([0])
    )

    data2 = Data(
        x=torch.randn(116, 64),
        edge_index=torch.randint(0, 116, (2, 500)),
        edge_attr=torch.randn(500, 1),
        y=torch.LongTensor([1])
    )

    batch = Batch.from_data_list([data1, data2])

    # 测试模型
    model = create_model(
        model_type='sf_dpl',
        input_dim=64,
        hidden_dim=64,
        num_rois=116,
        use_node_prompt=True
    )

    output = model(batch)
    print(f"\n✓ 输出形状: {output.shape}")
    print(f"✓ 参数量: {sum(p.numel() for p in model.parameters()):,}")