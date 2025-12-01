"""
准备GNN实验数据（修复版）
使用预训练的Node Encoder提取特征
包含 load_gnn_dataset 函数
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
from datetime import datetime
import argparse

from fc_construction import FCConstructor
from node_features import StatisticalFeatureExtractor  # ✅ 只导入统计特征
from extract_node_features import extract_node_features_pretrained  # ✅ 导入预训练版本


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
    """构建功能连接图"""
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


def extract_statistical_features(timeseries_list):
    """提取统计特征"""
    print(f"\n{'='*80}")
    print("Extracting statistical node features...")
    print(f"{'='*80}")

    extractor = StatisticalFeatureExtractor()
    features_list = []

    for i, ts in enumerate(timeseries_list):
        features = extractor.extract_features(ts)
        features_list.append(features)

        if (i + 1) % 100 == 0:
            print(f"  Processed: {i+1}/{len(timeseries_list)}")

    features_array = np.array(features_list)
    feature_dim = extractor.get_feature_dim()

    print(f"\n✓ Statistical features extracted")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Shape: {features_array.shape}")

    return features_array, feature_dim


def extract_pretrained_features(timeseries_list, encoder_path, embedding_dim, device):
    """✅ 使用预训练encoder提取特征"""
    print(f"\n{'='*80}")
    print("Extracting pretrained node features...")
    print(f"{'='*80}")

    # ✅ 调用正确的函数
    features_list = extract_node_features_pretrained(
        timeseries_list=timeseries_list,
        encoder_path=encoder_path,
        embedding_dim=embedding_dim,
        device=device
    )

    features_array = np.array(features_list)

    print(f"\n✓ Pretrained features extracted")
    print(f"  Feature dim: {embedding_dim}")
    print(f"  Shape: {features_array.shape}")

    return features_array, embedding_dim


def create_pyg_graphs(fc_matrices, node_features, labels, top_k=20):
    """
    创建PyTorch Geometric图对象（Top-K稀疏化版本）

    Args:
        fc_matrices: FC矩阵数组 [N_subjects, N_ROI, N_ROI]
        node_features: 节点特征数组 [N_subjects, N_ROI, feature_dim]
        labels: 标签数组
        top_k: 每个节点保留最强的k个连接（默认20，约占116节点的17%）

    Returns:
        graph_list: PyG Data对象列表
    """
    print(f"\n创建PyG图对象（Top-K稀疏化，k={top_k}）...")

    graph_list = []
    n_subjects = len(fc_matrices)
    invalid_count = 0

    for i in range(n_subjects):
        fc = fc_matrices[i].copy()
        x = node_features[i]
        y = labels[i]

        # ===== 1. 清理无效值 =====
        if np.any(np.isnan(fc)) or np.any(np.isinf(fc)):
            fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
            invalid_count += 1

        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            invalid_count += 1

        # ===== 2. 处理FC矩阵 =====
        # 取绝对值（连接强度）
        fc_abs = np.abs(fc)

        # 对角线设为0（去除自环）
        np.fill_diagonal(fc_abs, 0)

        # ===== 3. 🔥 Top-K 稀疏化 =====
        num_nodes = fc_abs.shape[0]
        k = min(top_k, num_nodes - 1)  # 防止k超过节点数

        # 对每一行，找出最强的k个连接
        # argsort返回从小到大的索引，取最后k个
        topk_indices = np.argsort(fc_abs, axis=1)[:, -k:]

        # 构建稀疏边列表
        edge_index_list = []
        edge_attr_list = []

        for row in range(num_nodes):
            for col in topk_indices[row]:
                if fc_abs[row, col] > 0:  # 额外保险
                    edge_index_list.append([row, col])
                    edge_attr_list.append(fc_abs[row, col])

        # 转换为Tensor
        if len(edge_index_list) == 0:
            # 如果没有边，创建一个最小图（每个节点连接到自己）
            edge_index = torch.arange(num_nodes).repeat(2, 1)
            edge_attr = torch.ones(num_nodes, 1) * 0.01
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).unsqueeze(1)

        # ===== 4. 创建PyG Data对象 =====
        data = Data(
            x=torch.FloatTensor(x),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.LongTensor([y])
        )

        # 最终验证
        assert not torch.isnan(data.x).any(), f"被试 {i} 的x仍包含NaN"
        assert not torch.isnan(data.edge_attr).any(), f"被试 {i} 的edge_attr仍包含NaN"

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
    """保存GNN数据集（添加top_k参数）"""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"保存GNN数据集（Top-K={top_k}）...")
    print(f"{'='*80}")

    saved_files = []

    for fc_method, fc_matrices in fc_dict.items():
        for feature_type, node_features in features_dict.items():
            print(f"\n处理: {fc_method} + {feature_type}")

            # 🔥 使用 Top-K 稀疏化
            graph_list = create_pyg_graphs(
                fc_matrices=fc_matrices,
                node_features=node_features,
                labels=labels,
                top_k=top_k  # 传入top_k参数
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
                    'top_k': top_k,  # 记录稀疏化参数
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }

            with open(filepath, 'wb') as f:
                pickle.dump(data_dict, f)

            print(f"  ✓ 保存至: {filepath}")
            saved_files.append(filepath)

    return saved_files


def load_gnn_dataset(filepath):
    """
    加载已保存的GNN数据集

    Args:
        filepath: .pkl文件路径

    Returns:
        graph_list: List[Data]
        labels: np.array
        metadata: dict
    """
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
    parser = argparse.ArgumentParser(description='Prepare GNN data with Top-K sparsification')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ABIDE', 'MDD'])
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./data/gnn_datasets')
    parser.add_argument('--encoder_path', type=str,
                        default='./pretrained_models/node_encoder_best.pth',
                        help='Path to pretrained encoder')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--top_k', type=int, default=20,  # 🔥 新增参数
                        help='Keep top-k strongest connections per node')

    args = parser.parse_args()

    print(f"\n{'=' * 80}")
    print(f"GNN数据准备（Top-K稀疏化）")
    print(f"{'=' * 80}")
    print(f"数据集: {args.dataset}")
    print(f"Top-K: {args.top_k} (保留每个节点最强的{args.top_k}个连接)")
    print(f"Encoder: {args.encoder_path}")
    print(f"{'=' * 80}\n")

    # 1-3. 数据加载和特征提取（保持不变）
    timeseries_list, labels, subject_ids, site_ids = load_timeseries_data(
        args.dataset, args.data_folder
    )

    fc_dict = construct_functional_graphs(
        timeseries_list,
        methods=['pearson', 'ledoit_wolf']
    )

    features_dict = {}

    # 只使用预训练特征（根据你的注释，statistical已被证明无效）
    if os.path.exists(args.encoder_path):
        pretrained_features, pretrained_dim = extract_pretrained_features(
            timeseries_list=timeseries_list,
            encoder_path=args.encoder_path,
            embedding_dim=args.embedding_dim,
            device=args.device
        )
        features_dict['temporal'] = pretrained_features
    else:
        print(f"\n⚠️  预训练模型不存在: {args.encoder_path}")
        print("  跳过预训练特征提取")

    # 4. 保存数据（传入top_k）
    saved_files = save_gnn_dataset(
        save_dir=args.save_dir,
        dataset_name=args.dataset,
        fc_dict=fc_dict,
        features_dict=features_dict,
        labels=labels,
        subject_ids=subject_ids,
        site_ids=site_ids,
        top_k=args.top_k  # 🔥 传入top_k参数
    )

    print(f"\n{'=' * 80}")
    print(f"✅ 数据准备完成！")
    print(f"{'=' * 80}")
    print(f"\n生成了 {len(saved_files)} 个数据集文件")
    print(f"稀疏化策略: Top-K={args.top_k}")


if __name__ == '__main__':
    main()