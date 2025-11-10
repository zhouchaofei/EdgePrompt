"""
双分支模型: Shared + Private Encoders
- Anatomical branch: temporal features + anatomical adjacency
- Functional branch: temporal features + functional adjacency (LedoitWolf)
- Encoders: SharedEncoder + PrivateEncoder_a + PrivateEncoder_f
- Pooling: mean pooling → concat → FC classifier

不包含prompt和解耦loss (baseline版本)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool


class SharedEncoder(nn.Module):
    """共享编码器 (用于两个分支)"""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2,
                 gnn_type='gat', dropout=0.5, heads=4):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            gnn_type: GNN类型 ('gat', 'gcn')
            dropout: dropout比率
            heads: GAT注意力头数 (仅当gnn_type='gat'时使用)
        """
        super().__init__()

        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim

            if gnn_type == 'gat':
                # GAT: output_dim = hidden_dim // heads * heads
                self.convs.append(
                    GATConv(in_dim, hidden_dim // heads, heads=heads, concat=True, dropout=dropout)
                )
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(in_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [N, input_dim] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: [E, 1] 边权重 (可选)

        Returns:
            h: [N, hidden_dim] 节点嵌入
        """
        h = x

        for i in range(self.num_layers):
            if self.gnn_type == 'gat':
                # GAT不直接支持edge_attr作为权重，这里简化处理
                h = self.convs[i](h, edge_index)
            elif self.gnn_type == 'gcn':
                # GCN支持edge_weight
                edge_weight = edge_attr.squeeze() if edge_attr is not None else None
                h = self.convs[i](h, edge_index, edge_weight=edge_weight)

            h = self.batch_norms[i](h)

            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h


class PrivateEncoder(nn.Module):
    """私有编码器 (每个分支独立)"""

    def __init__(self, input_dim, hidden_dim=64, num_layers=1,
                 gnn_type='gat', dropout=0.5, heads=4):
        """
        Args:
            input_dim: 输入特征维度 (应该等于SharedEncoder的输出维度)
            hidden_dim: 输出维度
            num_layers: GNN层数 (通常1-2层)
            gnn_type: GNN类型
            dropout: dropout比率
            heads: GAT注意力头数
        """
        super().__init__()

        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim

            if gnn_type == 'gat':
                self.convs.append(
                    GATConv(in_dim, hidden_dim // heads, heads=heads, concat=True, dropout=dropout)
                )
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(in_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [N, input_dim] 来自SharedEncoder的特征
            edge_index: [2, E] 边索引
            edge_attr: [E, 1] 边权重

        Returns:
            h: [N, hidden_dim] 私有嵌入
        """
        h = x

        for i in range(self.num_layers):
            if self.gnn_type == 'gat':
                h = self.convs[i](h, edge_index)
            elif self.gnn_type == 'gcn':
                edge_weight = edge_attr.squeeze() if edge_attr is not None else None
                h = self.convs[i](h, edge_index, edge_weight=edge_weight)

            h = self.batch_norms[i](h)

            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h


class DualBranchModel(nn.Module):
    """
    双分支模型 (Baseline版本 - 无prompt, 无解耦loss)

    架构:
        Input → SharedEncoder → [PrivateEncoder_a, PrivateEncoder_f] → Pooling → Concat → Classifier
    """

    def __init__(self,
                 input_dim,
                 shared_hidden_dim=64,
                 private_hidden_dim=64,
                 output_dim=2,
                 shared_layers=2,
                 private_layers=1,
                 gnn_type='gat',
                 dropout=0.5,
                 pooling='mean',
                 heads=4):
        """
        Args:
            input_dim: 节点特征维度
            shared_hidden_dim: 共享编码器输出维度
            private_hidden_dim: 私有编码器输出维度
            output_dim: 分类类别数
            shared_layers: 共享编码器层数
            private_layers: 私有编码器层数
            gnn_type: GNN类型 ('gat', 'gcn')
            dropout: dropout比率
            pooling: 图池化方式 ('mean', 'max', 'mean_max')
            heads: GAT注意力头数
        """
        super().__init__()

        self.pooling = pooling
        self.gnn_type = gnn_type

        # ============ Shared Encoder ============
        self.shared_encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dim=shared_hidden_dim,
            num_layers=shared_layers,
            gnn_type=gnn_type,
            dropout=dropout,
            heads=heads
        )

        # ============ Private Encoders ============
        self.private_encoder_anat = PrivateEncoder(
            input_dim=shared_hidden_dim,
            hidden_dim=private_hidden_dim,
            num_layers=private_layers,
            gnn_type=gnn_type,
            dropout=dropout,
            heads=heads
        )

        self.private_encoder_func = PrivateEncoder(
            input_dim=shared_hidden_dim,
            hidden_dim=private_hidden_dim,
            num_layers=private_layers,
            gnn_type=gnn_type,
            dropout=dropout,
            heads=heads
        )

        # ============ Classifier ============
        # 输入: [batch, private_hidden_dim * 2] (concat两个分支)
        if pooling == 'mean_max':
            classifier_input_dim = private_hidden_dim * 2 * 2  # 两分支 * 两种pooling
        else:
            classifier_input_dim = private_hidden_dim * 2  # 两分支

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden_dim, output_dim)
        )

        print(f"DualBranchModel initialized:")
        print(f"  GNN type: {gnn_type}")
        print(f"  Shared encoder: {input_dim} → {shared_hidden_dim} ({shared_layers} layers)")
        print(f"  Private encoders: {shared_hidden_dim} → {private_hidden_dim} ({private_layers} layers)")
        print(f"  Classifier input: {classifier_input_dim}")
        print(f"  Output: {output_dim} classes")

    def forward(self, data):
        """
        Args:
            data: PyG Data对象，包含:
                - x: [N, input_dim] 节点特征
                - func_edge_index, func_edge_attr: 功能边
                - anat_edge_index, anat_edge_attr: 解剖边
                - batch: [N] batch索引

        Returns:
            logits: [batch_size, output_dim] 分类logits
        """
        x = data.x
        batch = data.batch

        # ===========================
        # 1. Shared Encoding (使用功能图 - 也可以选择解剖图或两者都用)
        # ===========================
        # 这里选择用功能图做shared encoding,因为功能连接包含更多信息
        shared_feat = self.shared_encoder(
            x,
            data.func_edge_index,
            data.func_edge_attr
        )

        # ===========================
        # 2. Private Encoding
        # ===========================
        # Anatomical branch
        anat_feat = self.private_encoder_anat(
            shared_feat,
            data.anat_edge_index,
            data.anat_edge_attr
        )

        # Functional branch
        func_feat = self.private_encoder_func(
            shared_feat,
            data.func_edge_index,
            data.func_edge_attr
        )

        # ===========================
        # 3. Graph Pooling
        # ===========================
        if self.pooling == 'mean':
            anat_graph = global_mean_pool(anat_feat, batch)
            func_graph = global_mean_pool(func_feat, batch)
        elif self.pooling == 'max':
            anat_graph = global_max_pool(anat_feat, batch)
            func_graph = global_max_pool(func_feat, batch)
        elif self.pooling == 'mean_max':
            anat_mean = global_mean_pool(anat_feat, batch)
            anat_max = global_max_pool(anat_feat, batch)
            func_mean = global_mean_pool(func_feat, batch)
            func_max = global_max_pool(func_feat, batch)
            anat_graph = torch.cat([anat_mean, anat_max], dim=1)
            func_graph = torch.cat([func_mean, func_max], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # ===========================
        # 4. Concatenate and Classify
        # ===========================
        combined = torch.cat([anat_graph, func_graph], dim=1)
        logits = self.classifier(combined)

        return logits

    def get_embeddings(self, data):
        """
        获取中间嵌入表示 (用于可视化或分析)

        Returns:
            dict: 包含shared, anat, func, combined等嵌入
        """
        x = data.x
        batch = data.batch

        # Shared encoding
        shared_feat = self.shared_encoder(
            x, data.func_edge_index, data.func_edge_attr
        )

        # Private encoding
        anat_feat = self.private_encoder_anat(
            shared_feat, data.anat_edge_index, data.anat_edge_attr
        )
        func_feat = self.private_encoder_func(
            shared_feat, data.func_edge_index, data.func_edge_attr
        )

        # Pooling
        if self.pooling == 'mean':
            anat_graph = global_mean_pool(anat_feat, batch)
            func_graph = global_mean_pool(func_feat, batch)
        else:
            anat_graph = global_mean_pool(anat_feat, batch)
            func_graph = global_mean_pool(func_feat, batch)

        combined = torch.cat([anat_graph, func_graph], dim=1)

        return {
            'shared': shared_feat,
            'anat': anat_feat,
            'func': func_feat,
            'anat_graph': anat_graph,
            'func_graph': func_graph,
            'combined': combined
        }


if __name__ == '__main__':
    # 测试代码
    from torch_geometric.data import Data, Batch

    print("=" * 80)
    print("Testing DualBranchModel")
    print("=" * 80)

    # 创建模拟数据
    n_nodes = 116
    input_dim = 64  # temporal embedding dim

    # Data 1
    data1 = Data(
        x=torch.randn(n_nodes, input_dim),
        func_edge_index=torch.randint(0, n_nodes, (2, 1000)),
        func_edge_attr=torch.randn(1000, 1),
        anat_edge_index=torch.randint(0, n_nodes, (2, 500)),
        anat_edge_attr=torch.ones(500, 1),
        y=torch.LongTensor([0])
    )

    # Data 2
    data2 = Data(
        x=torch.randn(n_nodes, input_dim),
        func_edge_index=torch.randint(0, n_nodes, (2, 1000)),
        func_edge_attr=torch.randn(1000, 1),
        anat_edge_index=torch.randint(0, n_nodes, (2, 500)),
        anat_edge_attr=torch.ones(500, 1),
        y=torch.LongTensor([1])
    )

    # Batch
    batch = Batch.from_data_list([data1, data2])

    # 创建模型
    model = DualBranchModel(
        input_dim=input_dim,
        shared_hidden_dim=64,
        private_hidden_dim=64,
        output_dim=2,
        shared_layers=2,
        private_layers=1,
        gnn_type='gat',
        pooling='mean'
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")

    # Forward pass
    print(f"\nForward pass...")
    logits = model(batch)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits: {logits}")

    # Get embeddings
    print(f"\nGetting embeddings...")
    embeddings = model.get_embeddings(batch)
    for key, value in embeddings.items():
        if value.dim() == 2:
            print(f"  {key}: {value.shape}")

    print(f"\n✓ Model test passed!")