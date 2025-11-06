"""
GNN数据准备脚本
生成功能图（Pearson + Ledoit-Wolf）和节点特征（统计+时序编码）
"""

import os
import numpy as np
import argparse
from datetime import datetime

# 导入现有模块
from abide_data_baseline import ABIDEBaselineProcessor
from mdd_data_baseline import MDDBaselineProcessor
from fc_construction import FCConstructor
from node_features import extract_node_features_batch


def prepare_gnn_dataset(
    dataset_name,
    data_folder='./data',
    fs=0.5,
    temporal_dim=64,
    temporal_method='pca'
):
    """
    准备GNN数据集

    生成内容：
    1. Pearson功能图（保留权重）
    2. Ledoit-Wolf功能图（保留权重）
    3. 统计节点特征（12维）
    4. 时序编码节点特征（可选PCA/CNN/Transformer）

    Args:
        dataset_name: 'ABIDE' 或 'MDD'
        data_folder: 数据根目录
        fs: 采样频率（Hz）
        temporal_dim: 时序编码维度
        temporal_method: 时序编码方法 ('pca' / 'cnn' / 'transformer')
    """

    print(f"\n{'='*80}")
    print(f"GNN DATASET PREPARATION")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Temporal encoding method: {temporal_method}")
    print(f"Temporal encoding dim: {temporal_dim}")
    print(f"{'='*80}\n")

    # 1. 加载时间序列数据
    print("Step 1: Loading time series data...")

    if dataset_name == 'ABIDE':
        processor = ABIDEBaselineProcessor(data_folder=data_folder)
        timeseries_list, labels, subject_ids, site_ids = processor.download_and_extract(
            n_subjects=None,
            apply_zscore=True
        )
        save_path = os.path.join(data_folder, 'ABIDE', 'gnn_data')

    elif dataset_name == 'MDD':
        processor = MDDBaselineProcessor(data_folder=data_folder)
        timeseries_list, labels, subject_ids, site_ids = processor.load_roi_signals(
            apply_zscore=True
        )
        save_path = os.path.join(data_folder, 'REST-meta-MDD', 'gnn_data')

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    os.makedirs(save_path, exist_ok=True)

    print(f"  Loaded {len(labels)} subjects")
    print(f"  Number of ROIs: {timeseries_list[0].shape[1]}")

    # 打印时间点统计
    lengths = [ts.shape[0] for ts in timeseries_list]
    print(f"  Time points statistics:")
    print(f"    Min: {min(lengths)}")
    print(f"    Max: {max(lengths)}")
    print(f"    Mean: {np.mean(lengths):.1f}")
    print(f"    Median: {np.median(lengths):.1f}")

    # 2. 构建两种功能图
    print("\nStep 2: Building functional connectivity graphs...")

    # 2.1 Pearson功能图
    print("\n  2.1 Pearson correlation...")
    fc_constructor_pearson = FCConstructor(method='pearson')
    fc_matrices_pearson = []

    for i, ts in enumerate(timeseries_list):
        fc = fc_constructor_pearson.compute_fc_matrix(ts)
        fc_matrices_pearson.append(fc)

        if (i + 1) % 100 == 0:
            print(f"    Processed: {i + 1}/{len(timeseries_list)}")

    fc_matrices_pearson = np.array(fc_matrices_pearson)

    print(f"    Shape: {fc_matrices_pearson.shape}")
    print(f"    Mean: {fc_matrices_pearson.mean():.4f}")
    print(f"    Std: {fc_matrices_pearson.std():.4f}")

    # 2.2 Ledoit-Wolf功能图
    print("\n  2.2 Ledoit-Wolf shrinkage...")
    fc_constructor_lw = FCConstructor(method='ledoit_wolf')
    fc_matrices_lw = []

    for i, ts in enumerate(timeseries_list):
        fc = fc_constructor_lw.compute_fc_matrix(ts)
        fc_matrices_lw.append(fc)

        if (i + 1) % 100 == 0:
            print(f"    Processed: {i + 1}/{len(timeseries_list)}")

    fc_matrices_lw = np.array(fc_matrices_lw)

    print(f"    Shape: {fc_matrices_lw.shape}")
    print(f"    Mean: {fc_matrices_lw.mean():.4f}")
    print(f"    Std: {fc_matrices_lw.std():.4f}")

    # 3. 提取节点特征
    print("\nStep 3: Extracting node features...")

    # 3.1 统计特征
    print("\n  3.1 Statistical features...")
    node_features_stat, stat_dim = extract_node_features_batch(
        timeseries_list,
        method='statistical',
        fs=fs
    )

    # 3.2 时序编码特征
    print(f"\n  3.2 Temporal encoding features ({temporal_method})...")
    node_features_temporal, temporal_dim_actual = extract_node_features_batch(
        timeseries_list,
        method='temporal',
        temporal_method=temporal_method,
        output_dim=temporal_dim
    )

    # 4. 保存所有数据
    print("\nStep 4: Saving GNN dataset...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{dataset_name.lower()}_gnn_dataset_{temporal_method}_{timestamp}.npz'
    filepath = os.path.join(save_path, filename)

    np.savez_compressed(
        filepath,
        # 功能图
        fc_pearson=fc_matrices_pearson,
        fc_ledoit_wolf=fc_matrices_lw,
        # 节点特征
        node_features_statistical=np.array(node_features_stat),
        node_features_temporal=np.array(node_features_temporal),
        # 标签和ID
        labels=labels,
        subject_ids=subject_ids,
        site_ids=site_ids,
        # 元信息
        dataset=dataset_name,
        n_subjects=len(labels),
        n_rois=timeseries_list[0].shape[1],
        stat_feature_dim=stat_dim,
        temporal_feature_dim=temporal_dim_actual,
        temporal_method=temporal_method,
        fs=fs
    )

    print(f"  ✅ Saved to: {filepath}")

    # 保存元信息
    meta_file = os.path.join(save_path, f'{dataset_name.lower()}_gnn_meta_{temporal_method}_{timestamp}.txt')
    with open(meta_file, 'w') as f:
        f.write(f"{dataset_name} GNN Dataset\n")
        f.write(f"="*60 + "\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(f"Functional Graphs:\n")
        f.write(f"  - Pearson correlation (weighted, no threshold)\n")
        f.write(f"  - Ledoit-Wolf shrinkage (weighted, no threshold)\n\n")
        f.write(f"Node Features:\n")
        f.write(f"  - Statistical: {stat_dim} dimensions\n")
        f.write(f"  - Temporal encoding ({temporal_method}): {temporal_dim_actual} dimensions\n\n")
        f.write(f"Dataset Statistics:\n")
        f.write(f"  Subjects: {len(labels)}\n")
        f.write(f"  ROIs: {timeseries_list[0].shape[1]}\n")
        f.write(f"  Time points: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}\n")

        unique, counts = np.unique(labels, return_counts=True)
        f.write(f"\nLabel Distribution:\n")
        for u, c in zip(unique, counts):
            f.write(f"  Class {u}: {c} ({c/len(labels)*100:.1f}%)\n")

    print(f"  ✅ Meta info saved to: {meta_file}")

    print(f"\n{'='*80}")
    print(f"✅ GNN dataset preparation completed!")
    print(f"{'='*80}\n")

    return filepath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare GNN dataset')

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
        help='Data root folder'
    )

    parser.add_argument(
        '--fs',
        type=float,
        default=0.5,
        help='Sampling frequency (Hz)'
    )

    parser.add_argument(
        '--temporal_dim',
        type=int,
        default=64,
        help='Temporal encoding dimension'
    )

    parser.add_argument(
        '--temporal_method',
        type=str,
        default='pca',
        choices=['pca', 'cnn', 'transformer'],
        help='Temporal encoding method'
    )

    args = parser.parse_args()

    prepare_gnn_dataset(
        dataset_name=args.dataset,
        data_folder=args.data_folder,
        fs=args.fs,
        temporal_dim=args.temporal_dim,
        temporal_method=args.temporal_method
    )
