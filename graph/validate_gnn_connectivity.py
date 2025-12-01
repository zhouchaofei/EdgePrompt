"""
验证3：GNN连通性验证
目的：检查GNN是否会出现NaN
"""

import torch
import numpy as np
from torch_geometric.data import Data, Batch

from simple_gnn import create_model


def gnn_connectivity_validation():
    """
    GNN连通性验证

    期望：梯度中无NaN
    """
    print("\n" + "=" * 80)
    print("验证3：GNN连通性验证")
    print("=" * 80)

    # 1. 创建随机图数据
    n_graphs = 10
    n_nodes = 116
    n_features = 64
    n_edges_per_graph = 500

    print(f"\n生成随机测试数据:")
    print(f"  图数量: {n_graphs}")
    print(f"  节点数: {n_nodes}")
    print(f"  特征维度: {n_features}")

    data_list = []
    for i in range(n_graphs):
        x = torch.randn(n_nodes, n_features)
        edge_index = torch.randint(0, n_nodes, (2, n_edges_per_graph))
        edge_attr = torch.rand(n_edges_per_graph, 1)
        y = torch.LongTensor([i % 2])

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data_list.append(data)

    batch = Batch.from_data_list(data_list)

    # 2. 测试不同GNN类型
    # gnn_types = ['gcn', 'sage']
    gnn_types = ['gcn']

    for gnn_type in gnn_types:
        print(f"\n测试 {gnn_type.upper()}...")

        model = create_model(
            model_type='gnn',
            input_dim=n_features,
            hidden_dim=64,
            output_dim=2,
            num_layers=2,
            gnn_type=gnn_type,
            dropout=0.5,
            pooling='mean'
        )

        # Forward
        output = model(batch)
        loss = torch.nn.functional.cross_entropy(output, batch.y)

        # Backward
        loss.backward()

        # 检查梯度
        has_nan = False
        has_inf = False

        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  ❌ {name} 的梯度包含NaN")
                    has_nan = True
                if torch.isinf(param.grad).any():
                    print(f"  ❌ {name} 的梯度包含Inf")
                    has_inf = True

        if not has_nan and not has_inf:
            print(f"  ✅ {gnn_type.upper()} 通过测试，无NaN/Inf")

        # 检查输出
        if torch.isnan(output).any():
            print(f"  ❌ 输出包含NaN")
        elif torch.isinf(output).any():
            print(f"  ❌ 输出包含Inf")
        else:
            print(f"  ✅ 输出正常")

    print(f"\n{'=' * 60}")
    print("连通性验证完成")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    gnn_connectivity_validation()