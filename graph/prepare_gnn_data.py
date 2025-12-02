"""
GNN数据准备 - 终极修复版
主要修改：
1. 强制ID对齐，防止特征-标签错位
2. 支持统计特征回退（Statistical Fallback）
3. 增强数据质量检查
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
from datetime import datetime
import argparse

from fc_construction import FCConstructor
from extract_node_features import extract_node_features_pretrained


def extract_statistical_features(timeseries):
    """
    提取鲁棒的统计特征

    Args:
        timeseries: [T, N_ROI] 时间序列

    Returns:
        features: [N_ROI, feature_dim] 统计特征
    """
    T, N_ROI = timeseries.shape

    # 1. 基础统计量 (Mean, Std)
    mean = np.mean(timeseries, axis=0)  # [N_ROI]
    std = np.std(timeseries, axis=0)    # [N_ROI]

    # 2. FC Profile (每个节点与其他节点的连接强度分布)
    fc = np.corrcoef(timeseries.T)      # [N_ROI, N_ROI]
    np.fill_diagonal(fc, 0)

    # 拼接特征: [Mean(1), Std(1), FC(N_ROI)] = (2 + N_ROI) dim
    features = np.column_stack([mean, std, fc])

    return features


def load_data_as_dict(dataset_name, data_folder='./data'):
    """
    加载数据并返回字典格式 {subject_id: {ts, label, site}}
    确保ID对齐
    """
    print(f"\n{'='*80}")
    print(f"Loading {dataset_name} data as dictionary (ID-aligned)...")
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

    # 构建字典 {subject_id: data}
    data_dict = {}
    for i, sub_id in enumerate(subject_ids):
        data_dict[str(sub_id)] = {
            'ts': timeseries_list[i],
            'label': labels[i],
            'site': site_ids[i] if site_ids is not None else 'unknown'
        }

    print(f"✓ Loaded {len(data_dict)} subjects as dictionary")
    print(f"  Keys (first 3): {list(data_dict.keys())[:3]}")

    return data_dict


def construct_functional_graphs(timeseries_list, methods=['pearson', 'ledoit_wolf']):
    """构建功能连接图（保持原有逻辑）"""
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


def create_pyg_graphs(fc_matrices, node_features, labels, top_k=20):
    """
    创建PyTorch Geometric图对象（Top-K稀疏化）
    """
    print(f"\n创建PyG图对象（Top-K={top_k}）...")

    graph_list = []
    n_subjects = len(fc_matrices)
    invalid_count = 0

    for i in range(n_subjects):
        fc = fc_matrices[i].copy()
        x = node_features[i]
        y = labels[i]

        # 清理无效值
        if np.any(np.isnan(fc)) or np.any(np.isinf(fc)):
            fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
            invalid_count += 1

        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            invalid_count += 1

        # 处理FC矩阵
        fc_abs = np.abs(fc)
        np.fill_diagonal(fc_abs, 0)

        # Top-K 稀疏化
        num_nodes = fc_abs.shape[0]
        k = min(top_k, num_nodes - 1)

        topk_indices = np.argsort(fc_abs, axis=1)[:, -k:]

        edge_index_list = []
        edge_attr_list = []

        for row in range(num_nodes):
            for col in topk_indices[row]:
                if fc_abs[row, col] > 0:
                    edge_index_list.append([row, col])
                    edge_attr_list.append(fc_abs[row, col])

        if len(edge_index_list) == 0:
            edge_index = torch.arange(num_nodes).repeat(2, 1)
            edge_attr = torch.ones(num_nodes, 1) * 0.01
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).unsqueeze(1)

        # 创建PyG Data对象
        data = Data(
            x=torch.FloatTensor(x),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.LongTensor([y])
        )

        # 最终验证
        assert not torch.isnan(data.x).any(), f"被试 {i} 的x包含NaN"
        assert not torch.isnan(data.edge_attr).any(), f"被试 {i} 的edge_attr包含NaN"

        graph_list.append(data)

        if (i + 1) % 100 == 0:
            print(f"  创建进度: {i + 1}/{n_subjects}")

    if invalid_count > 0:
        print(f"  ⚠️  清理了 {invalid_count} 个被试的无效值")

    # 统计信息
    avg_edges = np.mean([g.edge_index.shape[1] for g in graph_list])
    num_nodes = graph_list[0].x.shape[0]
    sparsity = avg_edges / (num_nodes * (num_nodes - 1)) * 100

    print(f"  ✓ 创建了 {len(graph_list)} 个图")
    print(f"    节点数: {num_nodes}")
    print(f"    节点特征维度: {graph_list[0].x.shape[1]}")
    print(f"    平均边数: {avg_edges:.1f}")
    print(f"    稀疏度: {sparsity:.2f}% (Top-K={top_k})")

    return graph_list


def save_gnn_dataset(save_dir, dataset_name, fc_dict, features_dict,
                     labels, subject_ids, site_ids, top_k=20):
    """保存GNN数据集"""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"保存GNN数据集（Top-K={top_k}）...")
    print(f"{'='*80}")

    saved_files = []

    for fc_method, fc_matrices in fc_dict.items():
        for feature_type, node_features in features_dict.items():
            print(f"\n处理: {fc_method} + {feature_type}")

            graph_list = create_pyg_graphs(
                fc_matrices=fc_matrices,
                node_features=node_features,
                labels=labels,
                top_k=top_k
            )

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
                    'top_k': top_k,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }

            with open(filepath, 'wb') as f:
                pickle.dump(data_dict, f)

            print(f"  ✓ 保存至: {filepath}")
            saved_files.append(filepath)

    return saved_files


def load_gnn_dataset(filepath):
    """加载已保存的GNN数据集"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    print(f"Loading dataset from: {filepath}")

    with open(filepath, 'rb') as f:
        data_dict = pickle.load(f)

    graph_list = data_dict['graph_list']
    labels = data_dict['labels']
    metadata = data_dict.get('metadata', {})

    return graph_list, labels, metadata


def main():
    parser = argparse.ArgumentParser(description='Prepare GNN data (Fixed Version)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ABIDE', 'MDD'])
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./data/gnn_datasets')
    parser.add_argument('--encoder_path', type=str,
                        default='./pretrained_models/node_encoder_best.pth',
                        help='Path to pretrained encoder')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Keep top-k strongest connections per node')
    parser.add_argument('--use_statistical', action='store_true',
                        help='Force use of statistical features (for debugging)')

    args = parser.parse_args()

    print(f"\n{'=' * 80}")
    print(f"GNN数据准备（修复版）")
    print(f"{'=' * 80}")
    print(f"数据集: {args.dataset}")
    print(f"Top-K: {args.top_k}")
    print(f"强制使用统计特征: {args.use_statistical}")
    print(f"{'=' * 80}\n")

    # 1. 加载数据（字典格式，确保ID对齐）
    data_dict = load_data_as_dict(args.dataset, args.data_folder)
    subject_ids = list(data_dict.keys())

    # 2. 提取时间序列列表
    timeseries_list = [data_dict[sid]['ts'] for sid in subject_ids]
    labels = np.array([data_dict[sid]['label'] for sid in subject_ids])
    site_ids = np.array([data_dict[sid]['site'] for sid in subject_ids])

    print(f"\n✓ 数据对齐完成")
    print(f"  被试数: {len(subject_ids)}")
    print(f"  标签分布: {np.bincount(labels)}")

    # 3. 构建FC矩阵
    fc_dict = construct_functional_graphs(
        timeseries_list,
        methods=['pearson', 'ledoit_wolf']
    )

    # 4. 提取节点特征
    features_dict = {}

    if args.use_statistical or not os.path.exists(args.encoder_path):
        # 使用统计特征（回退方案）
        print(f"\n{'='*80}")
        print("⚠️ Using Statistical Features (Fallback Mode)")
        print(f"{'='*80}")

        statistical_features = []
        for i, ts in enumerate(timeseries_list):
            feat = extract_statistical_features(ts)
            statistical_features.append(feat)

            if (i + 1) % 100 == 0:
                print(f"  Processed: {i+1}/{len(timeseries_list)}")

        statistical_features = np.array(statistical_features)
        features_dict['statistical'] = statistical_features

        print(f"\n✓ Statistical features extracted")
        print(f"  Feature dim: {statistical_features.shape[2]}")
        print(f"  Shape: {statistical_features.shape}")

        if not os.path.exists(args.encoder_path):
            print(f"\n⚠️  预训练模型不存在: {args.encoder_path}")
            print("  只使用统计特征")

    else:
        # 使用预训练特征
        print(f"\n{'='*80}")
        print("Using Pretrained Encoder Features")
        print(f"{'='*80}")

        pretrained_features = extract_node_features_pretrained(
            timeseries_list=timeseries_list,
            encoder_path=args.encoder_path,
            embedding_dim=args.embedding_dim,
            device=args.device
        )
        features_dict['temporal'] = np.array(pretrained_features)

        print(f"\n✓ Pretrained features extracted")
        print(f"  Feature dim: {args.embedding_dim}")
        print(f"  Shape: {features_dict['temporal'].shape}")

    # 5. 保存数据集
    saved_files = save_gnn_dataset(
        save_dir=args.save_dir,
        dataset_name=args.dataset,
        fc_dict=fc_dict,
        features_dict=features_dict,
        labels=labels,
        subject_ids=subject_ids,
        site_ids=site_ids,
        top_k=args.top_k
    )

    print(f"\n{'=' * 80}")
    print(f"✅ 数据准备完成！")
    print(f"{'=' * 80}")
    print(f"\n生成了 {len(saved_files)} 个数据集文件:")
    for f in saved_files:
        print(f"  - {f}")
    print(f"\n稀疏化策略: Top-K={args.top_k}")

    # 6. 数据质量提示
    print(f"\n{'=' * 80}")
    print("建议的验证步骤:")
    print(f"{'=' * 80}")
    print("1. 如果使用统计特征，预期准确率: 60-65%")
    print("2. 如果使用预训练特征但效果差，尝试:")
    print("   - 重新训练预训练模型（mask_ratio=0.6）")
    print("   - 或回退到统计特征: --use_statistical")
    print("3. 检查数据质量:")
    print("   - 确认特征和标签对齐")
    print("   - 验证FC矩阵合理性")


if __name__ == '__main__':
    main()