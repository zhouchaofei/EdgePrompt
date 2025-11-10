"""
准备单分支解剖图数据
使用解剖邻接矩阵 + temporal/statistical 节点特征
"""

import os
import numpy as np
import torch
import pickle
from datetime import datetime
import argparse

from anatomical_graph import get_anatomical_constructor
from prepare_gnn_data import load_gnn_dataset


def prepare_anatomical_branch_data(
        dataset_name='ABIDE',
        data_folder='./data/gnn_datasets',
        save_dir='./data/single_branch_datasets',
        feature_type='temporal',
        atlas='aal116'
):
    """
    准备单分支解剖图数据

    Args:
        dataset_name: 数据集名称
        data_folder: 包含节点特征的数据路径
        save_dir: 保存路径
        feature_type: 节点特征类型 ('temporal' or 'statistical')
        atlas: 脑图谱
    """

    print(f"\n{'=' * 80}")
    print(f"Preparing Single-Branch ANATOMICAL Data")
    print(f"{'=' * 80}")
    print(f"Dataset: {dataset_name}")
    print(f"Feature Type: {feature_type}")
    print(f"Atlas: {atlas}")
    print(f"{'=' * 80}\n")

    # 1. 加载节点特征（从功能图数据中提取）
    # 我们只需要节点特征，不需要功能边
    func_data_file = os.path.join(
        data_folder,
        f"{dataset_name}_ledoit_wolf_{feature_type}.pkl"
    )

    if not os.path.exists(func_data_file):
        raise FileNotFoundError(f"Functional data file not found: {func_data_file}")

    print(f"Loading node features from: {func_data_file}")
    func_graphs, labels, metadata = load_gnn_dataset(func_data_file)

    print(f"Loaded {len(func_graphs)} graphs")
    print(f"Node feature dim: {metadata['node_feature_dim']}")

    # 2. 获取解剖图构建器
    anat_constructor = get_anatomical_constructor(atlas)
    anat_edge_index, anat_edge_attr = anat_constructor.create_anatomical_edge_index()

    print(f"\nAnatomical graph:")
    print(f"  Edges: {anat_edge_index.shape[1]}")
    print(f"  Edge attr shape: {anat_edge_attr.shape}")

    # 3. 创建解剖图数据（使用相同的节点特征，但替换为解剖边）
    from torch_geometric.data import Data

    anat_graphs = []
    for i, func_graph in enumerate(func_graphs):
        # 创建新的Data对象，只使用解剖边
        anat_data = Data(
            x=func_graph.x,  # 保持相同的节点特征
            edge_index=torch.LongTensor(anat_edge_index),
            edge_attr=torch.FloatTensor(anat_edge_attr),
            y=func_graph.y
        )
        anat_graphs.append(anat_data)

        if (i + 1) % 100 == 0:
            print(f"  Processed: {i + 1}/{len(func_graphs)}")

    print(f"  ✓ Created {len(anat_graphs)} anatomical graphs")

    # 4. 保存
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{dataset_name}_anatomical_{feature_type}.pkl"
    filepath = os.path.join(save_dir, filename)

    save_data = {
        'graph_list': anat_graphs,
        'labels': labels,
        'metadata': {
            'dataset': dataset_name,
            'branch': 'anatomical',
            'feature_type': feature_type,
            'atlas': atlas,
            'n_subjects': len(labels),
            'n_nodes': anat_graphs[0].x.shape[0],
            'node_feature_dim': anat_graphs[0].x.shape[1],
            'n_edges': anat_graphs[0].edge_index.shape[1],
            'timestamp': timestamp
        }
    }

    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\n✓ Data saved to: {filepath}")

    # 打印统计信息
    print(f"\nDataset Statistics:")
    print(f"  Number of subjects: {len(labels)}")
    print(f"  Nodes per graph: {save_data['metadata']['n_nodes']}")
    print(f"  Node feature dim: {save_data['metadata']['node_feature_dim']}")
    print(f"  Edges per graph: {save_data['metadata']['n_edges']}")
    print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    return filepath


def main():
    parser = argparse.ArgumentParser(description='Prepare single-branch anatomical data')

    parser.add_argument('--dataset', type=str, default='ABIDE',
                        choices=['ABIDE', 'MDD'],
                        help='Dataset name')
    parser.add_argument('--data_folder', type=str, default='./data/gnn_datasets',
                        help='Folder containing node features')
    parser.add_argument('--save_dir', type=str, default='./data/single_branch_datasets',
                        help='Directory to save single-branch data')
    parser.add_argument('--feature_type', type=str, default='temporal',
                        choices=['statistical', 'temporal'],
                        help='Node feature type')
    parser.add_argument('--atlas', type=str, default='aal116',
                        help='Brain atlas')

    args = parser.parse_args()

    # 准备数据
    filepath = prepare_anatomical_branch_data(
        dataset_name=args.dataset,
        data_folder=args.data_folder,
        save_dir=args.save_dir,
        feature_type=args.feature_type,
        atlas=args.atlas
    )

    print(f"\n{'=' * 80}")
    print(f"✅ Anatomical branch data preparation completed!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()