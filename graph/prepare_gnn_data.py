"""
GNN数据预处理脚本
生成功能图（Pearson & Ledoit-Wolf）和节点特征（统计 & 时序编码）
保存为标准格式供后续实验使用
"""

import os
import numpy as np
import argparse
from datetime import datetime
import pickle

# 导入FC构建模块
from fc_construction import compute_fc_matrices_enhanced

# 导入节点特征提取模块
from node_features import extract_node_features_batch


def load_timeseries(dataset_name, data_folder='./data'):
    """
    加载时间序列数据

    Returns:
        timeseries_list: 时间序列列表
        labels: 标签
        subject_ids: 被试ID
        site_ids: 站点ID
        meta: 元信息
    """
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name} time series data...")
    print(f"{'='*60}")

    if dataset_name == 'ABIDE':
        from abide_data_baseline import ABIDEBaselineProcessor
        processor = ABIDEBaselineProcessor(data_folder=data_folder)
        timeseries_list, labels, subject_ids, site_ids = processor.download_and_extract(
            n_subjects=None,
            apply_zscore=True  # 确保时间序列已标准化
        )
        meta = {
            'dataset': 'ABIDE',
            'n_subjects': len(labels),
            'n_rois': timeseries_list[0].shape[1]
        }

    elif dataset_name == 'MDD':
        from mdd_data_baseline import MDDBaselineProcessor
        processor = MDDBaselineProcessor(data_folder=data_folder)
        timeseries_list, labels, subject_ids, site_ids = processor.load_roi_signals(
            apply_zscore=True
        )
        meta = {
            'dataset': 'MDD',
            'n_subjects': len(labels),
            'n_rois': 116
        }

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Loaded {len(labels)} subjects")
    print(f"ROIs: {meta['n_rois']}")
    print(f"Label distribution: {np.bincount(labels)}")

    return timeseries_list, labels, subject_ids, site_ids, meta


def build_functional_graphs(timeseries_list, methods=['pearson', 'ledoit_wolf']):
    """
    构建功能连接图

    Args:
        timeseries_list: 时间序列列表
        methods: FC构建方法列表

    Returns:
        fc_dict: {method: fc_matrices}
    """
    print(f"\n{'='*60}")
    print(f"Building functional connectivity graphs...")
    print(f"{'='*60}")

    fc_dict = {}

    for method in methods:
        print(f"\nMethod: {method}")

        fc_matrices, sparsity_rates = compute_fc_matrices_enhanced(
            timeseries_list=timeseries_list,
            method=method,
            threshold_strategy='none',  # 保留所有权重
            threshold_k=None
        )

        fc_dict[method] = fc_matrices

        print(f"  FC shape: {fc_matrices.shape}")
        print(f"  Mean: {fc_matrices.mean():.4f}")
        print(f"  Std: {fc_matrices.std():.4f}")
        print(f"  Sparsity: {sparsity_rates.mean():.4f}")

    return fc_dict


def extract_node_features_all(
    timeseries_list,
    methods=['statistical', 'temporal'],
    temporal_dim=64,
    device='cuda'
):
    """
    提取所有类型的节点特征

    Args:
        timeseries_list: 时间序列列表
        methods: 特征提取方法列表
        temporal_dim: 时序编码维度
        device: 计算设备

    Returns:
        features_dict: {method: features}
    """
    print(f"\n{'='*60}")
    print(f"Extracting node features...")
    print(f"{'='*60}")

    features_dict = {}

    for method in methods:
        print(f"\nMethod: {method}")

        if method == 'statistical':
            features = extract_node_features_batch(
                timeseries_list,
                method='statistical'
            )
        elif method == 'temporal':
            features = extract_node_features_batch(
                timeseries_list,
                method='temporal',
                out_dim=temporal_dim,
                device=device
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        features_dict[method] = features

        print(f"  Feature shape: {features.shape}")
        print(f"  Feature stats: mean={features.mean():.4f}, std={features.std():.4f}")

    return features_dict


def save_preprocessed_data(
    dataset_name,
    fc_dict,
    features_dict,
    labels,
    subject_ids,
    site_ids,
    meta,
    save_dir='./data/gnn_preprocessed'
):
    """
    保存预处理后的数据

    Args:
        dataset_name: 数据集名称
        fc_dict: 功能连接字典
        features_dict: 节点特征字典
        labels: 标签
        subject_ids: 被试ID
        site_ids: 站点ID
        meta: 元信息
        save_dir: 保存目录
    """
    print(f"\n{'='*60}")
    print(f"Saving preprocessed data...")
    print(f"{'='*60}")

    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{dataset_name}_gnn_data_{timestamp}.pkl'
    filepath = os.path.join(save_dir, filename)

    data = {
        'fc_matrices': fc_dict,
        'node_features': features_dict,
        'labels': labels,
        'subject_ids': subject_ids,
        'site_ids': site_ids,
        'meta': meta,
        'timestamp': timestamp
    }

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"✅ Data saved to: {filepath}")

    # 保存元信息
    meta_file = os.path.join(save_dir, f'{dataset_name}_gnn_meta.txt')
    with open(meta_file, 'w') as f:
        f.write(f"GNN Preprocessed Data - {dataset_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Generated: {timestamp}\n\n")

        f.write(f"Functional Connectivity Methods:\n")
        for method, fc in fc_dict.items():
            f.write(f"  {method}: {fc.shape}\n")

        f.write(f"\nNode Feature Methods:\n")
        for method, feat in features_dict.items():
            f.write(f"  {method}: {feat.shape}\n")

        f.write(f"\nDataset Info:\n")
        for key, value in meta.items():
            f.write(f"  {key}: {value}\n")

        f.write(f"\nLabel Distribution:\n")
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            f.write(f"  Class {u}: {c} samples\n")

    print(f"✅ Meta info saved to: {meta_file}")

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description='Prepare GNN data with functional graphs and node features'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['ABIDE', 'MDD'],
        help='Dataset name'
    )

    parser.add_argument(
        '--data_folder',
        type=str,
        default='./data',
        help='Root folder for datasets'
    )

    parser.add_argument(
        '--fc_methods',
        type=str,
        nargs='+',
        default=['pearson', 'ledoit_wolf'],
        help='FC construction methods'
    )

    parser.add_argument(
        '--feature_methods',
        type=str,
        nargs='+',
        default=['statistical', 'temporal'],
        help='Node feature extraction methods'
    )

    parser.add_argument(
        '--temporal_dim',
        type=int,
        default=64,
        help='Temporal embedding dimension'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for temporal feature extraction'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='./data/gnn_preprocessed',
        help='Directory to save preprocessed data'
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"GNN Data Preparation")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"FC methods: {args.fc_methods}")
    print(f"Feature methods: {args.feature_methods}")
    print(f"Temporal dim: {args.temporal_dim}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    # 1. 加载时间序列
    timeseries_list, labels, subject_ids, site_ids, meta = load_timeseries(
        args.dataset,
        args.data_folder
    )

    # 2. 构建功能图
    fc_dict = build_functional_graphs(
        timeseries_list,
        methods=args.fc_methods
    )

    # 3. 提取节点特征
    features_dict = extract_node_features_all(
        timeseries_list,
        methods=args.feature_methods,
        temporal_dim=args.temporal_dim,
        device=args.device
    )

    # 4. 保存数据
    filepath = save_preprocessed_data(
        args.dataset,
        fc_dict,
        features_dict,
        labels,
        subject_ids,
        site_ids,
        meta,
        save_dir=args.save_dir
    )

    print(f"\n{'='*60}")
    print(f"✅ Data preparation completed!")
    print(f"{'='*60}")
    print(f"Saved to: {filepath}")
    print(f"\nTo load the data:")
    print(f"  import pickle")
    print(f"  with open('{filepath}', 'rb') as f:")
    print(f"      data = pickle.load(f)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
