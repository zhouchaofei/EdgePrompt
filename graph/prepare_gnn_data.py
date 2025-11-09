"""
准备GNN实验数据
生成：
1. 两种功能图：Pearson_none + LedoitWolf_none
2. 两种节点特征：统计特征 + 时序编码
3. 保存为PyTorch Geometric格式
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
from datetime import datetime
import argparse

from fc_construction import FCConstructor, ThresholdStrategy
from node_features import extract_all_features


def load_timeseries_data(dataset_name, data_folder='./data'):
    """加载时间序列数据"""
    print(f"\n{'='*80}")
    print(f"Loading {dataset_name} time series data...")
    print(f"{'='*80}")

    if dataset_name == 'ABIDE':
        from abide_data_baseline import ABIDEBaselineProcessor
        processor = ABIDEBaselineProcessor(data_folder=data_folder)
        timeseries_list, labels, subject_ids, site_ids = processor.download_and_extract(
            n_subjects=None, apply_zscore=True
        )

    elif dataset_name == 'MDD':
        from mdd_data_baseline import MDDBaselineProcessor
        processor = MDDBaselineProcessor(data_folder=data_folder)
        timeseries_list, labels, subject_ids, site_ids = processor.load_roi_signals(
            apply_zscore=True
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\n✓ Loaded {len(labels)} subjects")
    print(f"  ROIs: {timeseries_list[0].shape[1]}")
    print(f"  Label distribution: {np.bincount(labels)}")

    return timeseries_list, labels, subject_ids, site_ids


def construct_functional_graphs(timeseries_list, methods=['pearson', 'ledoit_wolf']):
    """
    构建功能连接图

    Args:
        timeseries_list: 时间序列列表
        methods: FC构建方法列表

    Returns:
        fc_dict: {method: fc_matrices}
    """
    print(f"\n{'='*80}")
    print(f"Constructing functional connectivity graphs...")
    print(f"{'='*80}")

    fc_dict = {}

    for method in methods:
        print(f"\nMethod: {method}")
        constructor = FCConstructor(method=method)

        fc_matrices = []
        for i, ts in enumerate(timeseries_list):
            fc = constructor.compute_fc_matrix(ts)
            fc_matrices.append(fc)

            if (i + 1) % 100 == 0:
                print(f"  Processed: {i+1}/{len(timeseries_list)}")

        fc_matrices = np.array(fc_matrices)
        fc_dict[method] = fc_matrices

        print(f"  ✓ {method}: shape={fc_matrices.shape}")
        print(f"    Stats: mean={fc_matrices.mean():.4f}, "
              f"std={fc_matrices.std():.4f}")

    return fc_dict


def create_pyg_graphs(fc_matrices, node_features, labels, threshold=0.0):
    """
    创建PyTorch Geometric图对象

    Args:
        fc_matrices: [N, N_ROI, N_ROI] FC矩阵
        node_features: [N, N_ROI, feature_dim] 节点特征
        labels: [N] 标签
        threshold: 边权重阈值（默认保留所有边）

    Returns:
        graph_list: PyG Data对象列表
    """
    print(f"\nCreating PyG graphs...")
    print(f"  Threshold: {threshold}")

    graph_list = []
    n_subjects = len(fc_matrices)
    invalid_count = 0

    for i in range(n_subjects):
        fc = fc_matrices[i]
        x = node_features[i]
        y = labels[i]

        # ===== 关键：检查并清理数据 =====
        # 1. 清理 FC 矩阵中的 NaN/Inf
        if np.any(np.isnan(fc)) or np.any(np.isinf(fc)):
            print(f"  Warning: Subject {i} has NaN/Inf in FC matrix, cleaning...")
            fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
            invalid_count += 1

        # 2. 清理节点特征中的 NaN/Inf
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print(f"  Warning: Subject {i} has NaN/Inf in node features, cleaning...")
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            invalid_count += 1

        # 构建边
        adj = np.abs(fc) > threshold
        np.fill_diagonal(adj, False)

        edge_index = np.array(np.where(adj))
        edge_attr = fc[adj]

        # 3. 再次检查边权重
        edge_attr = np.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

        # 转换为PyG格式
        data = Data(
            x=torch.FloatTensor(x),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr).unsqueeze(1),
            y=torch.LongTensor([y])
        )

        # 4. 最终验证
        assert not torch.isnan(data.x).any(), f"Subject {i} still has NaN in x"
        assert not torch.isnan(data.edge_attr).any(), f"Subject {i} still has NaN in edge_attr"

        graph_list.append(data)

        if (i + 1) % 100 == 0:
            print(f"  Created: {i + 1}/{n_subjects}")

    if invalid_count > 0:
        print(f"  ⚠️  Cleaned {invalid_count} subjects with invalid values")

    print(f"  ✓ Created {len(graph_list)} graphs")
    print(f"    Nodes: {graph_list[0].x.shape[0]}")
    print(f"    Node features: {graph_list[0].x.shape[1]}")
    print(f"    Avg edges: {np.mean([g.edge_index.shape[1] for g in graph_list]):.1f}")

    return graph_list


def save_gnn_dataset(save_dir, dataset_name, fc_dict, features_dict,
                     labels, subject_ids, site_ids):
    """
    保存GNN数据集

    保存格式：
    {dataset}_{fc_method}_{feature_type}.pkl

    每个文件包含：
    - graph_list: PyG图列表
    - labels: 标签
    - subject_ids: 被试ID
    - site_ids: 站点ID
    - metadata: 元信息
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Saving GNN datasets...")
    print(f"{'='*80}")

    saved_files = []

    for fc_method, fc_matrices in fc_dict.items():
        for feature_type, node_features in features_dict.items():
            print(f"\nProcessing: {fc_method} + {feature_type}")

            # 创建图
            graph_list = create_pyg_graphs(
                fc_matrices=fc_matrices,
                node_features=node_features,
                labels=labels,
                threshold=0.0  # 保留所有边
            )

            # 保存
            filename = f"{dataset_name}_{fc_method}_{feature_type}.pkl"
            filepath = os.path.join(save_dir, filename)

            data_dict = {
                'graph_list': graph_list,
                'labels': labels,
                'subject_ids': subject_ids,
                'site_ids': site_ids,
                'metadata': {
                    'dataset': dataset_name,
                    'fc_method': fc_method,
                    'feature_type': feature_type,
                    'n_subjects': len(labels),
                    'n_nodes': graph_list[0].x.shape[0],
                    'node_feature_dim': graph_list[0].x.shape[1],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }

            with open(filepath, 'wb') as f:
                pickle.dump(data_dict, f)

            print(f"  ✓ Saved: {filepath}")
            saved_files.append(filepath)

    # 保存元信息文件
    meta_file = os.path.join(save_dir, f"{dataset_name}_metadata.txt")
    with open(meta_file, 'w') as f:
        f.write(f"GNN Dataset Preparation Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of subjects: {len(labels)}\n")
        f.write(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}\n")
        f.write(f"\nFC methods: {list(fc_dict.keys())}\n")
        f.write(f"Feature types: {list(features_dict.keys())}\n")
        f.write(f"\nSaved files:\n")
        for filepath in saved_files:
            f.write(f"  - {os.path.basename(filepath)}\n")
        f.write(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\n✓ Metadata saved: {meta_file}")

    return saved_files


def main():
    parser = argparse.ArgumentParser(description='Prepare GNN experiment data')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ABIDE', 'MDD'],
                        help='Dataset name')
    parser.add_argument('--data_folder', type=str, default='./data',
                        help='Root folder for datasets')
    parser.add_argument('--save_dir', type=str, default='./data/gnn_datasets',
                        help='Directory to save prepared data')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Dimension for temporal encoding')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for temporal encoder training')
    parser.add_argument('--skip_temporal', action='store_true',
                        help='Skip temporal encoding (only use statistical features)')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"GNN DATA PREPARATION")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Save directory: {args.save_dir}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")

    # 1. 加载时间序列数据
    timeseries_list, labels, subject_ids, site_ids = load_timeseries_data(
        args.dataset, args.data_folder
    )

    # 2. 构建功能图（Pearson + LedoitWolf）
    fc_dict = construct_functional_graphs(
        timeseries_list,
        methods=['pearson', 'ledoit_wolf']
    )

    # 3. 提取节点特征
    features_dict = {}

    # 统计特征
    stat_features, stat_dim = extract_all_features(
        timeseries_list, labels,
        feature_type='statistical'
    )
    features_dict['statistical'] = np.array(stat_features)

    # 时序编码特征（可选）
    if not args.skip_temporal:
        temporal_features, temporal_dim = extract_all_features(
            timeseries_list, labels,
            feature_type='temporal',
            embedding_dim=args.embedding_dim,
            device=args.device
        )
        features_dict['temporal'] = np.array(temporal_features)

    # 4. 保存数据
    saved_files = save_gnn_dataset(
        save_dir=args.save_dir,
        dataset_name=args.dataset,
        fc_dict=fc_dict,
        features_dict=features_dict,
        labels=labels,
        subject_ids=subject_ids,
        site_ids=site_ids
    )

    print(f"\n{'='*80}")
    print(f"✅ Data preparation completed!")
    print(f"{'='*80}")
    print(f"\nGenerated {len(saved_files)} dataset files")
    print(f"\nTo use these datasets:")
    print(f"  from prepare_gnn_data import load_gnn_dataset")
    print(f"  graphs, labels, meta = load_gnn_dataset('{args.dataset}_pearson_statistical.pkl')")


def load_gnn_dataset(filepath):
    """
    加载准备好的GNN数据集

    Args:
        filepath: 数据文件路径

    Returns:
        graph_list: PyG图列表
        labels: 标签
        metadata: 元信息
    """
    with open(filepath, 'rb') as f:
        data_dict = pickle.load(f)

    print(f"Loaded GNN dataset: {filepath}")
    print(f"  Subjects: {data_dict['metadata']['n_subjects']}")
    print(f"  Nodes: {data_dict['metadata']['n_nodes']}")
    print(f"  Node features: {data_dict['metadata']['node_feature_dim']}")

    return data_dict['graph_list'], data_dict['labels'], data_dict['metadata']


if __name__ == '__main__':
    main()