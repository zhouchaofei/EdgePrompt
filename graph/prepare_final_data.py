"""
prepare_final_data.py - 完整版
目标：生成双分支 GNN 所需的完整数据集
按照《结构分支构建方案.md》（方案A × 论文）构建结构图

包含：
1. 节点特征 X (117节点: 116 ROI + 1 Global Mean)
2. 结构图边 edge_index_struct (跨半球对称 + Global连接 + Vermis + 小脑-Vermis)
3. 功能图边 edge_index_func (Ledoit-Wolf Top-K)
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
import argparse
from sklearn.covariance import LedoitWolf

# 导入你现有的模块
from extract_node_features import extract_node_features_pretrained
from abide_data_baseline import ABIDEBaselineProcessor
from mdd_data_baseline import MDDBaselineProcessor


# ============================================================================
# AAL116 配对表和映射表
# ============================================================================

# AAL116 左右配对脑区（索引从0开始）
# 格式：(left_idx, right_idx)
PAIRED_ROIS = [
    (0, 1),     # Precentral_L, Precentral_R
    (2, 3),     # Frontal_Sup_L, Frontal_Sup_R
    (4, 5),     # Frontal_Sup_Orb_L, Frontal_Sup_Orb_R
    (6, 7),     # Frontal_Mid_L, Frontal_Mid_R
    (8, 9),     # Frontal_Mid_Orb_L, Frontal_Mid_Orb_R
    (10, 11),   # Frontal_Inf_Oper_L, Frontal_Inf_Oper_R
    (12, 13),   # Frontal_Inf_Tri_L, Frontal_Inf_Tri_R
    (14, 15),   # Frontal_Inf_Orb_L, Frontal_Inf_Orb_R
    (16, 17),   # Rolandic_Oper_L, Rolandic_Oper_R
    (18, 19),   # Supp_Motor_Area_L, Supp_Motor_Area_R
    (20, 21),   # Olfactory_L, Olfactory_R
    (22, 23),   # Frontal_Sup_Medial_L, Frontal_Sup_Medial_R
    (24, 25),   # Frontal_Med_Orb_L, Frontal_Med_Orb_R
    (26, 27),   # Rectus_L, Rectus_R
    (28, 29),   # Insula_L, Insula_R
    (30, 31),   # Cingulum_Ant_L, Cingulum_Ant_R
    (32, 33),   # Cingulum_Mid_L, Cingulum_Mid_R
    (34, 35),   # Cingulum_Post_L, Cingulum_Post_R
    (36, 37),   # Hippocampus_L, Hippocampus_R
    (38, 39),   # ParaHippocampal_L, ParaHippocampal_R
    (40, 41),   # Amygdala_L, Amygdala_R
    (42, 43),   # Calcarine_L, Calcarine_R
    (44, 45),   # Cuneus_L, Cuneus_R
    (46, 47),   # Lingual_L, Lingual_R
    (48, 49),   # Occipital_Sup_L, Occipital_Sup_R
    (50, 51),   # Occipital_Mid_L, Occipital_Mid_R
    (52, 53),   # Occipital_Inf_L, Occipital_Inf_R
    (54, 55),   # Fusiform_L, Fusiform_R
    (56, 57),   # Postcentral_L, Postcentral_R
    (58, 59),   # Parietal_Sup_L, Parietal_Sup_R
    (60, 61),   # Parietal_Inf_L, Parietal_Inf_R
    (62, 63),   # SupraMarginal_L, SupraMarginal_R
    (64, 65),   # Angular_L, Angular_R
    (66, 67),   # Precuneus_L, Precuneus_R
    (68, 69),   # Paracentral_Lobule_L, Paracentral_Lobule_R
    (70, 71),   # Caudate_L, Caudate_R
    (72, 73),   # Putamen_L, Putamen_R
    (74, 75),   # Pallidum_L, Pallidum_R
    (76, 77),   # Thalamus_L, Thalamus_R
    (78, 79),   # Heschl_L, Heschl_R
    (80, 81),   # Temporal_Sup_L, Temporal_Sup_R
    (82, 83),   # Temporal_Pole_Sup_L, Temporal_Pole_Sup_R
    (84, 85),   # Temporal_Mid_L, Temporal_Mid_R
    (86, 87),   # Temporal_Pole_Mid_L, Temporal_Pole_Mid_R
    (88, 89),   # Temporal_Inf_L, Temporal_Inf_R
    (90, 91),   # Cerebelum_Crus1_L, Cerebelum_Crus1_R
    (92, 93),   # Cerebelum_Crus2_L, Cerebelum_Crus2_R
    (94, 95),   # Cerebelum_3_L, Cerebelum_3_R
    (96, 97),   # Cerebelum_4_5_L, Cerebelum_4_5_R
    (98, 99),   # Cerebelum_6_L, Cerebelum_6_R
    (100, 101),   # Cerebelum_7b_L, Cerebelum_7b_R
    (102, 103), # Cerebelum_8_L, Cerebelum_8_R
    (104, 105), # Cerebelum_9_L, Cerebelum_9_R
    (106, 107), # Cerebelum_10_L, Cerebelum_10_R
]

# Vermis 索引（108-115，对应原始的109-116但这里从0开始）
VERMIS_INDICES = list(range(108, 116))  # [108, 109, 110, 111, 112, 113, 114, 115]

# 小脑ROI到Vermis的映射（小脑索引 -> Vermis索引）
CEREBELLUM_TO_VERMIS = {
    # Crus1/2 -> Vermis_1_2
    90: 108,  # Cerebelum_Crus1_L -> Vermis_1_2
    91: 108,  # Cerebelum_Crus1_R -> Vermis_1_2
    92: 108,  # Cerebelum_Crus2_L -> Vermis_1_2
    93: 108,  # Cerebelum_Crus2_R -> Vermis_1_2
    # Lobule 3 -> Vermis_3
    94: 109,  # Cerebelum_3_L -> Vermis_3
    95: 109,  # Cerebelum_3_R -> Vermis_3
    # Lobule 4_5 -> Vermis_4_5
    96: 110,  # Cerebelum_4_5_L -> Vermis_4_5
    97: 110,  # Cerebelum_4_5_R -> Vermis_4_5
    # Lobule 6 -> Vermis_6
    98: 111,  # Cerebelum_6_L -> Vermis_6
    99: 111,  # Cerebelum_6_R -> Vermis_6
    # Lobule 7b -> Vermis_7
    100: 112,  # Cerebelum_7b_L -> Vermis_7
    101: 112,  # Cerebelum_7b_R -> Vermis_7
    # Lobule 8 -> Vermis_8
    102: 113,  # Cerebelum_8_L -> Vermis_8
    103: 113,  # Cerebelum_8_R -> Vermis_8
    # Lobule 9 -> Vermis_9
    104: 114,  # Cerebelum_9_L -> Vermis_9
    105: 114,  # Cerebelum_9_R -> Vermis_9
    # Lobule 10 -> Vermis_10
    106: 115,  # Cerebelum_10_L -> Vermis_10
    107: 115,  # Cerebelum_10_R -> Vermis_10
}


def build_structural_graph_aal116(num_rois=116, verbose=True):
    """
    按照《结构分支构建方案.md》构建结构图

    规则：
    1. 跨半球对称连接（所有成对的L-R ROI）
    2. 所有ROI连接到Global Mean节点（索引116）
    3. Vermis内部顺序连接
    4. 小脑ROI连接到对应的Vermis区域

    返回：
        edge_index: [2, num_edges] 的边索引张量
    """
    edge_list = []
    global_idx = num_rois  # 116

    # ===== 1. 跨半球对称连接 =====
    if verbose:
        print("  构建跨半球对称连接...")
    for left_idx, right_idx in PAIRED_ROIS:
        edge_list.append([left_idx, right_idx])
        edge_list.append([right_idx, left_idx])  # 无向图

    if verbose:
        print(f"    添加了 {len(PAIRED_ROIS) * 2} 条跨半球边")

    # ===== 2. 所有ROI到Global Mean =====
    if verbose:
        print("  构建ROI-Global连接...")
    for roi_idx in range(num_rois):
        edge_list.append([roi_idx, global_idx])
        edge_list.append([global_idx, roi_idx])

    if verbose:
        print(f"    添加了 {num_rois * 2} 条ROI-Global边")

    # ===== 3. Vermis内部顺序连接 =====
    if verbose:
        print("  构建Vermis内部连接...")
    vermis_edges = 0
    for i in range(len(VERMIS_INDICES) - 1):
        v1 = VERMIS_INDICES[i]
        v2 = VERMIS_INDICES[i + 1]
        edge_list.append([v1, v2])
        edge_list.append([v2, v1])
        vermis_edges += 2

    if verbose:
        print(f"    添加了 {vermis_edges} 条Vermis内部边")

    # ===== 4. 小脑-Vermis连接 =====
    if verbose:
        print("  构建小脑-Vermis连接...")
    cereb_vermis_edges = 0
    for cereb_idx, vermis_idx in CEREBELLUM_TO_VERMIS.items():
        edge_list.append([cereb_idx, vermis_idx])
        edge_list.append([vermis_idx, cereb_idx])
        cereb_vermis_edges += 2

    if verbose:
        print(f"    添加了 {cereb_vermis_edges} 条小脑-Vermis边")

    # 转换为tensor
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    if verbose:
        print(f"\n  ✓ 结构图构建完成")
        print(f"    节点数: {num_rois + 1} (116 ROI + 1 Global)")
        print(f"    边数: {edge_index.shape[1]} (无向图)")
        print(f"    图类型: 稀疏生物学结构图")

    return edge_index


def build_functional_graph(timeseries, k=20, method='ledoit_wolf', verbose=False):
    """
    构建功能图（仅116个ROI，不包含Global Mean）

    策略：Ledoit-Wolf正则化相关 + Top-K稀疏化

    Args:
        timeseries: (T, 116) 时间序列
        k: 每个节点保留的Top-K连接
        method: 'ledoit_wolf' 或 'pearson'

    Returns:
        edge_index: [2, num_edges]
        edge_attr: [num_edges, 1] 边权重
    """
    num_rois = timeseries.shape[1]

    # 计算相关矩阵
    if method == 'ledoit_wolf':
        lw = LedoitWolf()
        lw.fit(timeseries)
        cov = lw.covariance_
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
    else:
        corr = np.corrcoef(timeseries.T)

    # 去除对角线
    np.fill_diagonal(corr, 0)
    corr_abs = np.abs(corr)

    edge_list = []
    edge_weights = []

    # Top-K策略
    for i in range(num_rois):
        # 找到与节点i相关性最高的k个节点
        top_k_indices = np.argsort(corr_abs[i])[-k:]
        for j in top_k_indices:
            if corr_abs[i, j] > 0:
                edge_list.append([i, j])
                edge_weights.append(corr_abs[i, j])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    if verbose:
        print(f"  功能图: {edge_index.shape[1]} 条边 (Top-{k}, {method})")

    return edge_index, edge_attr


def process_dataset(dataset_name, data_folder, encoder_path, save_dir,
                   fc_k=20, fc_method='ledoit_wolf'):
    """
    处理完整数据集

    Args:
        dataset_name: 'ABIDE' 或 'MDD'
        data_folder: 数据根目录
        encoder_path: 预训练encoder路径
        save_dir: 保存目录
        fc_k: 功能图Top-K参数
        fc_method: 功能图构建方法
    """
    print("\n" + "="*80)
    print(f"处理数据集: {dataset_name}")
    print("="*80)

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

    print(f"  ✓ 加载了 {len(ts_list)} 个被试")
    print(f"  标签分布: HC={np.sum(labels==0)}, 患者={np.sum(labels==1)}")

    # ===== 2. 提取节点特征（116 ROI） =====
    print("\n[步骤 2/5] 提取节点特征...")
    features_116 = extract_node_features_pretrained(
        ts_list,
        encoder_path=encoder_path,
        embedding_dim=64
    )
    print(f"  ✓ 特征形状: {features_116.shape}")
    # 应该是 (N_subjects, 116, feature_dim)

    # ===== 3. 构建结构图（共享的图结构） =====
    print("\n[步骤 3/5] 构建结构图...")
    struct_edge_index = build_structural_graph_aal116(num_rois=116, verbose=True)

    # ===== 4. 为每个被试构建数据对象 =====
    print("\n[步骤 4/5] 构建PyG数据对象...")
    data_list = []

    for i, (ts, feat, y) in enumerate(zip(ts_list, features_116, labels)):
        # feat shape: (116, feature_dim)

        # --- A. 添加Global Mean特征（117个节点） ---
        # 策略1：Global特征 = 所有ROI特征的均值
        global_feat = np.mean(feat, axis=0, keepdims=True)  # (1, feature_dim)
        feat_117 = np.concatenate([feat, global_feat], axis=0)  # (117, feature_dim)

        # --- B. 构建功能图（仅基于116个ROI） ---
        func_edge_index, func_edge_attr = build_functional_graph(
            ts, k=fc_k, method=fc_method, verbose=False
        )

        # --- C. 封装为PyG Data对象 ---
        data = Data(
            x=torch.FloatTensor(feat_117),  # (117, feature_dim)
            y=torch.LongTensor([y]),

            # 结构分支：117个节点（包含Global）
            edge_index_struct=struct_edge_index,

            # 功能分支：116个节点（不包含Global）
            edge_index_func=func_edge_index,
            edge_attr_func=func_edge_attr,

            # 元信息
            subject_id=ids[i],
            num_nodes=117  # 明确指定节点数
        )

        # 添加节点掩码（用于区分ROI和Global）
        data.roi_mask = torch.zeros(117, dtype=torch.bool)
        data.roi_mask[:116] = True  # 前116个是ROI
        data.roi_mask[116] = False  # 第117个是Global

        data_list.append(data)

        if (i + 1) % 100 == 0:
            print(f"  进度: {i + 1}/{len(ts_list)}")

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
            }
        }, f)

    print(f"  ✓ 数据已保存: {save_path}")

    # 打印统计信息
    print("\n" + "="*80)
    print("数据集统计")
    print("="*80)
    print(f"数据集: {dataset_name}")
    print(f"被试数: {len(data_list)}")
    print(f"节点数: 117 (116 ROI + 1 Global Mean)")
    print(f"节点特征维度: {feat_117.shape[1]}")
    print(f"结构图边数: {struct_edge_index.shape[1]}")
    print(f"平均功能图边数: {np.mean([d.edge_index_func.shape[1] for d in data_list]):.1f}")
    print(f"标签分布: {np.bincount(labels)}")
    print("="*80)

    return save_path


def verify_graph_structure(save_path):
    """验证保存的图数据结构"""
    print("\n验证图结构...")

    with open(save_path, 'rb') as f:
        data = pickle.load(f)

    graph_list = data['graph_list']
    sample = graph_list[0]

    print(f"\n样本图对象属性:")
    print(f"  节点特征 x: {sample.x.shape}")
    print(f"  标签 y: {sample.y.shape}")
    print(f"  结构图边 edge_index_struct: {sample.edge_index_struct.shape}")
    print(f"  功能图边 edge_index_func: {sample.edge_index_func.shape}")
    print(f"  功能图边权重 edge_attr_func: {sample.edge_attr_func.shape}")
    print(f"  ROI掩码 roi_mask: {sample.roi_mask.shape}, True数量: {sample.roi_mask.sum()}")

    # 验证结构图的连接性
    struct_edges = sample.edge_index_struct
    global_node = 116
    global_connections = (struct_edges[0] == global_node).sum() + (struct_edges[1] == global_node).sum()
    print(f"\n结构图验证:")
    print(f"  Global节点(116)的连接数: {global_connections}")
    print(f"  预期应该是: {116 * 2} (每个ROI双向连接)")

    # 验证功能图只包含ROI
    func_edges = sample.edge_index_func
    max_node_in_func = func_edges.max().item()
    print(f"\n功能图验证:")
    print(f"  最大节点索引: {max_node_in_func} (应该 < 116)")

    print("\n✓ 图结构验证完成")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='准备双分支GNN数据集')
    parser.add_argument('--dataset', type=str, default='ABIDE',
                       choices=['ABIDE', 'MDD'],
                       help='数据集名称')
    parser.add_argument('--data_folder', type=str, default='./data',
                       help='数据根目录')
    parser.add_argument('--encoder_path', type=str,
                       default='./pretrained_models/node_encoder_best.pth',
                       help='预训练encoder路径')
    parser.add_argument('--save_dir', type=str, default='./data/gnn_datasets',
                       help='保存目录')
    parser.add_argument('--fc_k', type=int, default=20,
                       help='功能图Top-K参数')
    parser.add_argument('--fc_method', type=str, default='ledoit_wolf',
                       choices=['ledoit_wolf', 'pearson'],
                       help='功能图构建方法')
    parser.add_argument('--verify', action='store_true',
                       help='处理后验证图结构')

    args = parser.parse_args()

    # 处理数据集
    save_path = process_dataset(
        dataset_name=args.dataset,
        data_folder=args.data_folder,
        encoder_path=args.encoder_path,
        save_dir=args.save_dir,
        fc_k=args.fc_k,
        fc_method=args.fc_method
    )

    # 可选：验证
    if args.verify:
        verify_graph_structure(save_path)

    print("\n" + "="*80)
    print("✅ 数据准备完成！")
    print("="*80)
    print(f"\n使用方法：")
    print(f"```python")
    print(f"import pickle")
    print(f"with open('{save_path}', 'rb') as f:")
    print(f"    data = pickle.load(f)")
    print(f"graph_list = data['graph_list']")
    print(f"labels = data['labels']")
    print(f"```")
    print("="*80)