"""
准备双分支模型数据
- 加载temporal node features (已训练好的1D-CNN embeddings)
- 加载LedoitWolf功能邻接矩阵
- 构建解剖邻接矩阵
- 保存为PyG格式
"""

import os
import numpy as np
import torch
import pickle
from datetime import datetime
import argparse

from anatomical_graph import create_dual_branch_data, get_anatomical_constructor
from prepare_gnn_data import load_gnn_dataset


def convert_to_dual_branch_format(
        dataset_name='ABIDE',
        data_folder='./data/gnn_datasets',
        save_dir='./data/dual_branch_datasets',
        fc_method='ledoit_wolf',
        feature_type='temporal',
        atlas='aal116'
):
    """
    将单分支GNN数据转换为双分支格式

    Args:
        dataset_name: 数据集名称
        data_folder: 单分支数据路径
        save_dir: 保存路径
        fc_method: 功能连接方法 ('ledoit_wolf' 推荐)
        feature_type: 节点特征类型 ('temporal' 推荐)
        atlas: 脑图谱
    """

    print(f"\n{'=' * 80}")
    print(f"Converting to Dual-Branch Format")
    print(f"{'=' * 80}")
    print(f"Dataset: {dataset_name}")
    print(f"FC Method: {fc_method}")
    print(f"Feature Type: {feature_type}")
    print(f"Atlas: {atlas}")
    print(f"{'=' * 80}\n")

    # 1. 加载单分支数据
    data_file = os.path.join(
        data_folder,
        f"{dataset_name}_{fc_method}_{feature_type}.pkl"
    )

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    print(f"Loading data from: {data_file}")
    graph_list, labels, metadata = load_gnn_dataset(data_file)

    print(f"Loaded {len(graph_list)} graphs")
    print(f"Node feature dim: {metadata['node_feature_dim']}")
    print(f"Label distribution: {np.bincount(labels)}")

    # 2. 获取解剖图构建器
    anat_constructor = get_anatomical_constructor(atlas)

    # 3. 转换每个图
    print(f"\nConverting to dual-branch format...")
    dual_branch_graphs = []

    for i, graph in enumerate(graph_list):
        # 提取功能边
        func_edge_index = graph.edge_index.numpy()
        func_edge_attr = graph.edge_attr.numpy() if hasattr(graph,
                                                            'edge_attr') and graph.edge_attr is not None else None

        # 如果没有边权重，使用默认值
        if func_edge_attr is None:
            func_edge_attr = np.ones((func_edge_index.shape[1], 1), dtype=np.float32)

        # 创建双分支数据
        dual_data = create_dual_branch_data(
            node_features=graph.x.numpy(),
            func_edge_index=func_edge_index,
            func_edge_attr=func_edge_attr,
            label=labels[i],
            atlas=atlas
        )

        dual_branch_graphs.append(dual_data)

        if (i + 1) % 100 == 0:
            print(f"  Converted: {i + 1}/{len(graph_list)}")

    print(f"  ✓ Conversion completed")

    # 4. 保存
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{dataset_name}_{fc_method}_{feature_type}_dual_branch.pkl"
    filepath = os.path.join(save_dir, filename)

    save_data = {
        'graph_list': dual_branch_graphs,
        'labels': labels,
        'metadata': {
            'dataset': dataset_name,
            'fc_method': fc_method,
            'feature_type': feature_type,
            'atlas': atlas,
            'n_subjects': len(labels),
            'n_nodes': dual_branch_graphs[0].x.shape[0],
            'node_feature_dim': dual_branch_graphs[0].x.shape[1],
            'n_func_edges': dual_branch_graphs[0].func_edge_index.shape[1],
            'n_anat_edges': dual_branch_graphs[0].anat_edge_index.shape[1],
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
    print(f"  Functional edges: {save_data['metadata']['n_func_edges']}")
    print(f"  Anatomical edges: {save_data['metadata']['n_anat_edges']}")
    print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # 保存元信息
    meta_file = os.path.join(save_dir, f"{dataset_name}_dual_branch_meta.txt")
    with open(meta_file, 'w') as f:
        f.write(f"Dual-Branch Dataset Metadata\n")
        f.write(f"{'=' * 60}\n")
        for key, value in save_data['metadata'].items():
            f.write(f"{key}: {value}\n")

    print(f"✓ Metadata saved to: {meta_file}")

    return filepath


def load_dual_branch_dataset(filepath):
    """
    加载双分支数据集

    Args:
        filepath: 数据文件路径

    Returns:
        graph_list: PyG图列表
        labels: 标签
        metadata: 元信息
    """
    with open(filepath, 'rb') as f:
        data_dict = pickle.load(f)

    print(f"Loaded dual-branch dataset: {filepath}")
    print(f"  Subjects: {data_dict['metadata']['n_subjects']}")
    print(f"  Nodes: {data_dict['metadata']['n_nodes']}")
    print(f"  Node features: {data_dict['metadata']['node_feature_dim']}")
    print(f"  Func edges: {data_dict['metadata']['n_func_edges']}")
    print(f"  Anat edges: {data_dict['metadata']['n_anat_edges']}")

    return data_dict['graph_list'], data_dict['labels'], data_dict['metadata']


def main():
    parser = argparse.ArgumentParser(description='Prepare dual-branch data')

    parser.add_argument('--dataset', type=str, default='ABIDE',
                        choices=['ABIDE', 'MDD'],
                        help='Dataset name')
    parser.add_argument('--data_folder', type=str, default='./data/gnn_datasets',
                        help='Folder containing single-branch GNN data')
    parser.add_argument('--save_dir', type=str, default='./data/dual_branch_datasets',
                        help='Directory to save dual-branch data')
    parser.add_argument('--fc_method', type=str, default='ledoit_wolf',
                        choices=['pearson', 'ledoit_wolf'],
                        help='Functional connectivity method')
    parser.add_argument('--feature_type', type=str, default='temporal',
                        choices=['statistical', 'temporal'],
                        help='Node feature type')
    parser.add_argument('--atlas', type=str, default='aal116',
                        help='Brain atlas')

    args = parser.parse_args()

    # 转换数据
    filepath = convert_to_dual_branch_format(
        dataset_name=args.dataset,
        data_folder=args.data_folder,
        save_dir=args.save_dir,
        fc_method=args.fc_method,
        feature_type=args.feature_type,
        atlas=args.atlas
    )

    print(f"\n{'=' * 80}")
    print(f"✅ Data preparation completed!")
    print(f"{'=' * 80}")
    print(f"\nTo use this dataset:")
    print(f"  from prepare_dual_branch_data import load_dual_branch_dataset")
    print(f"  graphs, labels, meta = load_dual_branch_dataset('{filepath}')")


if __name__ == '__main__':
    main()