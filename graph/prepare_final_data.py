"""
prepare_final_data.py - 修复版 (解决NaN问题)
集成数据清洗逻辑，确保数据质量
生成双分支GNN所需的完整数据集
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
import argparse
from sklearn.covariance import LedoitWolf

from extract_node_features import extract_node_features_pretrained
from abide_data_baseline import ABIDEBaselineProcessor
from mdd_data_baseline import MDDBaselineProcessor

# ============================================================================
# 数据清洗工具函数
# ============================================================================
def clean_data(data_array, name="unknown"):
    """
    强制清洗NaN/Inf，替换为0

    Args:
        data_array: 需要清洗的数组
        name: 数据名称（用于日志）

    Returns:
        清洗后的数组
    """
    if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
        # print(f"  [Warning] Cleaning NaN/Inf in {name}")  # 调试时开启
        return np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)
    return data_array


# ============================================================================
# AAL116 结构定义
# ============================================================================
PAIRED_ROIS = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (16, 17), (18, 19), (20, 21), (22, 23), (24, 25), (26, 27), (28, 29),
    (30, 31), (32, 33), (34, 35), (36, 37), (38, 39), (40, 41), (42, 43),
    (44, 45), (46, 47), (48, 49), (50, 51), (52, 53), (54, 55), (56, 57),
    (58, 59), (60, 61), (62, 63), (64, 65), (66, 67), (68, 69), (70, 71),
    (72, 73), (74, 75), (76, 77), (78, 79), (80, 81), (82, 83), (84, 85),
    (86, 87), (88, 89), (90, 91), (92, 93), (94, 95), (96, 97), (98, 99),
    (100, 101), (102, 103), (104, 105), (106, 107)
]

VERMIS_INDICES = list(range(108, 116))

CEREBELLUM_TO_VERMIS = {
    90: 108, 91: 108, 92: 108, 93: 108, 94: 109, 95: 109, 96: 110, 97: 110,
    98: 111, 99: 111, 100: 112, 101: 112, 102: 113, 103: 113, 104: 114, 105: 114,
    106: 115, 107: 115
}


def build_structural_graph_aal116(num_rois=116, verbose=True):
    """
    构建结构图（稀疏生物学结构）

    规则：
    1. 跨半球对称连接
    2. 所有ROI连接到Global Mean节点
    3. Vermis内部顺序连接
    4. 小脑-Vermis连接

    Returns:
        edge_index: [2, num_edges]
    """
    edge_list = []
    global_idx = num_rois  # 116

    # 1. 跨半球对称连接
    for l, r in PAIRED_ROIS:
        edge_list.append([l, r])
        edge_list.append([r, l])

    # 2. ROI到Global Mean
    for i in range(num_rois):
        edge_list.append([i, global_idx])
        edge_list.append([global_idx, i])

    # 3. Vermis内部连接
    for i in range(len(VERMIS_INDICES) - 1):
        u, v = VERMIS_INDICES[i], VERMIS_INDICES[i+1]
        edge_list.append([u, v])
        edge_list.append([v, u])

    # 4. 小脑-Vermis连接
    for c, v in CEREBELLUM_TO_VERMIS.items():
        edge_list.append([c, v])
        edge_list.append([v, c])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    if verbose:
        print(f"  结构图边数: {edge_index.shape[1]}")

    return edge_index


def build_functional_graph(timeseries, k=20, method='ledoit_wolf'):
    """
    构建功能图（Robust版本，自动处理NaN）

    Args:
        timeseries: (T, N_ROI) 时间序列
        k: Top-K参数
        method: 'ledoit_wolf' 或 'pearson'

    Returns:
        edge_index: [2, num_edges]
        edge_attr: [num_edges, 1]
    """
    # 1. 输入清洗
    timeseries = clean_data(timeseries, "timeseries_input")
    num_rois = timeseries.shape[1]

    # 2. 计算相关矩阵
    try:
        if method == 'ledoit_wolf':
            lw = LedoitWolf(assume_centered=False)
            lw.fit(timeseries)
            cov = lw.covariance_
            d = np.sqrt(np.diag(cov))
            d[d == 0] = 1e-8  # 防止除以0
            corr = cov / np.outer(d, d)
        else:
            corr = np.corrcoef(timeseries.T)
    except Exception as e:
        print(f"  [Error] FC计算失败: {e}，使用零矩阵")
        corr = np.zeros((num_rois, num_rois))

    # 3. 输出清洗
    corr = clean_data(corr, "correlation_matrix")
    np.fill_diagonal(corr, 0)
    corr_abs = np.abs(corr)

    # 4. Top-K构图
    edge_list = []
    edge_weights = []

    for i in range(num_rois):
        idx = np.argsort(corr_abs[i])[-k:]
        for j in idx:
            val = corr_abs[i, j]
            if val > 1e-6:  # 忽略过小的权重
                edge_list.append([i, j])
                edge_weights.append(val)

    # 5. 空图防御
    if not edge_list:
        print("  [Warning] 功能图为空，使用自环")
        for i in range(num_rois):
            edge_list.append([i, i])
            edge_weights.append(1.0)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    return edge_index, edge_attr


def process_dataset(dataset_name, data_folder, encoder_path, save_dir,
                   fc_k=20, fc_method='ledoit_wolf'):
    """
    处理完整数据集（带自动清洗）

    Args:
        dataset_name: 'ABIDE' 或 'MDD'
        data_folder: 数据根目录
        encoder_path: 预训练encoder路径
        save_dir: 保存目录
        fc_k: 功能图Top-K参数
        fc_method: 功能图构建方法
    """
    print(f"\n{'='*80}")
    print(f"处理数据集: {dataset_name} (Auto-Clean NaN)")
    print(f"{'='*80}")

    # ===== 1. 加载原始数据 =====
    print("\n[步骤 1/5] 加载原始数据...")
    if dataset_name == 'ABIDE':
        processor = ABIDEBaselineProcessor(data_folder=data_folder)
        ts_list, labels, ids, _ = processor.download_and_extract(
            n_subjects=None, apply_zscore=True
        )
    elif dataset_name == 'MDD':
        processor = MDDBaselineProcessor(data_folder=data_folder)
        ts_list, labels, ids, _ = processor.load_roi_signals(apply_zscore=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 清洗原始时间序列
    ts_list = [clean_data(ts, f"ts_{i}") for i, ts in enumerate(ts_list)]

    print(f"  ✓ 加载了 {len(ts_list)} 个被试")
    print(f"  标签分布: HC={np.sum(labels==0)}, 患者={np.sum(labels==1)}")

    # ===== 2. 提取节点特征 =====
    print("\n[步骤 2/5] 提取节点特征...")
    features_116 = extract_node_features_pretrained(
        ts_list,
        encoder_path=encoder_path,
        embedding_dim=64
    )
    # 清洗特征
    features_116 = clean_data(features_116, "node_features")
    print(f"  ✓ 特征形状: {features_116.shape}")

    # ===== 3. 构建结构图 =====
    print("\n[步骤 3/5] 构建结构图...")
    struct_edge_index = build_structural_graph_aal116(num_rois=116, verbose=True)

    # ===== 4. 构建PyG数据对象 =====
    print("\n[步骤 4/5] 构建PyG数据对象...")
    data_list = []

    for i, (ts, feat, y) in enumerate(zip(ts_list, features_116, labels)):
        # 添加Global Mean特征
        global_feat = np.mean(feat, axis=0, keepdims=True)
        global_feat = clean_data(global_feat, f"global_feat_{i}")

        feat_117 = np.concatenate([feat, global_feat], axis=0)  # (117, D)

        # 构建功能图（仅基于116个ROI）
        func_edge_index, func_edge_attr = build_functional_graph(
            ts, k=fc_k, method=fc_method
        )

        # 封装为PyG Data对象
        data = Data(
            x=torch.FloatTensor(feat_117),
            y=torch.LongTensor([y]),
            edge_index_struct=struct_edge_index,
            edge_index_func=func_edge_index,
            edge_attr_func=func_edge_attr,
            subject_id=ids[i]
        )

        # 添加ROI掩码
        data.roi_mask = torch.zeros(117, dtype=torch.bool)
        data.roi_mask[:116] = True  # 前116个是ROI

        data_list.append(data)

        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(ts_list)}")

    print(f"  ✓ 创建了 {len(data_list)} 个图对象")

    # ===== 5. 保存数据 =====
    print("\n[步骤 5/5] 保存数据...")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{dataset_name}_DualBranch.pkl')

    with open(save_path, 'wb') as f:
        pickle.dump({
            'graph_list': data_list,
            'labels': labels,
            'ids': ids,
            'metadata': {
                'dataset': dataset_name,
                'num_subjects': len(data_list),
                'num_nodes': 117,
                'num_rois': 116,
                'fc_method': fc_method,
                'fc_k': fc_k,
                'struct_edges': struct_edge_index.shape[1],
                'cleaned': True  # 标记已清洗
            }
        }, f)

    print(f"  ✓ 数据已保存: {save_path}")
    print("  ✓ 所有潜在的NaN已被替换为0.0")

    # 打印统计信息
    print(f"\n{'='*80}")
    print("数据集统计")
    print(f"{'='*80}")
    print(f"数据集: {dataset_name}")
    print(f"被试数: {len(data_list)}")
    print(f"节点数: 117 (116 ROI + 1 Global)")
    print(f"节点特征维度: {feat_117.shape[1]}")
    print(f"结构图边数: {struct_edge_index.shape[1]}")
    print(f"平均功能图边数: {np.mean([d.edge_index_func.shape[1] for d in data_list]):.1f}")
    print(f"标签分布: {np.bincount(labels)}")
    print(f"{'='*80}")

    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='准备双分支GNN数据集（修复版）')
    parser.add_argument('--dataset', type=str, default='ABIDE',
                       choices=['ABIDE', 'MDD'])
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--encoder_path', type=str,
                       default='./pretrained_models/node_encoder_best.pth')
    parser.add_argument('--save_dir', type=str, default='./data/gnn_datasets')
    parser.add_argument('--fc_k', type=int, default=20)
    parser.add_argument('--fc_method', type=str, default='ledoit_wolf')

    args = parser.parse_args()

    save_path = process_dataset(
        dataset_name=args.dataset,
        data_folder=args.data_folder,
        encoder_path=args.encoder_path,
        save_dir=args.save_dir,
        fc_k=args.fc_k,
        fc_method=args.fc_method
    )

    print(f"\n{'='*80}")
    print("✅ 数据准备完成！")
    print(f"{'='*80}")
    print("\n下一步：运行基线实验")
    print("  python run_baseline.py --mode struct")
    print("  python run_baseline.py --mode struct --use_prompt")
    print("  python run_baseline.py --mode func")
    print("  python run_baseline.py --mode func --use_prompt")