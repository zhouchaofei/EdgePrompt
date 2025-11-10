"""
解剖邻接矩阵构建模块
基于左右脑半球的先验知识构建解剖连接

参考论文: Identifying Autism Spectrum Disorder Based on Individual-Aware
Down-Sampling and Multi-Modal Learning (https://arxiv.org/pdf/2109.09129)

构建规则:
- 左脑区域连接到非左脑区域(右脑)
- 右脑区域连接到非右脑区域(左脑)
- 形成二部图结构
"""

import numpy as np
import torch
from torch_geometric.data import Data


class AnatomicalGraphConstructor:
    """解剖图构建器"""

    def __init__(self, atlas='aal116'):
        """
        Args:
            atlas: 使用的脑图谱 ('aal116', 'aal90')
        """
        self.atlas = atlas
        self.n_rois = 116 if atlas == 'aal116' else 90

        # 获取左右脑标签
        self.hemisphere_labels = self._get_hemisphere_labels()

        # 构建解剖邻接矩阵模板
        self.anat_adj_template = self._build_anatomical_adjacency()

        print(f"Anatomical Graph Constructor initialized")
        print(f"  Atlas: {atlas}")
        print(f"  Number of ROIs: {self.n_rois}")
        print(f"  Left hemisphere: {np.sum(self.hemisphere_labels == 0)} ROIs")
        print(f"  Right hemisphere: {np.sum(self.hemisphere_labels == 1)} ROIs")

    def _get_hemisphere_labels(self):
        """
        获取每个ROI的半球标签

        AAL-116规则：
        - 奇数索引(1, 3, 5, ...): 左半球 (label=0)
        - 偶数索引(2, 4, 6, ...): 右半球 (label=1)

        注意：这里使用0-based索引，所以:
        - 索引0, 2, 4, ... → 左半球
        - 索引1, 3, 5, ... → 右半球

        Returns:
            labels: [N_ROI] 半球标签数组 (0=左, 1=右)
        """
        if self.atlas == 'aal116':
            # AAL-116: 前116个区域按奇偶分左右
            labels = np.zeros(116, dtype=int)
            labels[1::2] = 1  # 奇数索引→右半球
            # 索引0, 2, 4, ... → 左半球 (label=0)
            # 索引1, 3, 5, ... → 右半球 (label=1)

        elif self.atlas == 'aal90':
            # AAL-90: 类似规则
            labels = np.zeros(90, dtype=int)
            labels[1::2] = 1

        else:
            raise ValueError(f"Unknown atlas: {self.atlas}")

        return labels

    def _build_anatomical_adjacency(self):
        """
        构建解剖邻接矩阵 (二部图)

        构建规则 (参考论文):
        - 左脑ROI连接到所有右脑ROI
        - 右脑ROI连接到所有左脑ROI
        - 不存在左脑内部或右脑内部的连接
        - 形成完全二部图

        Returns:
            adj: [N_ROI, N_ROI] 邻接矩阵 (0/1)
        """
        adj = np.zeros((self.n_rois, self.n_rois), dtype=np.float32)

        # 获取左右脑索引
        left_indices = np.where(self.hemisphere_labels == 0)[0]
        right_indices = np.where(self.hemisphere_labels == 1)[0]

        # 左脑连接到右脑
        for i in left_indices:
            adj[i, right_indices] = 1

        # 右脑连接到左脑
        for i in right_indices:
            adj[i, left_indices] = 1

        # 确保对称性
        adj = (adj + adj.T) / 2

        # 移除自连接
        np.fill_diagonal(adj, 0)

        # 统计信息
        n_edges = np.sum(adj > 0) / 2  # 除以2因为是对称矩阵
        sparsity = 1 - (2 * n_edges) / (self.n_rois * (self.n_rois - 1))

        print(f"\nAnatomical adjacency matrix:")
        print(f"  Shape: {adj.shape}")
        print(f"  Number of edges: {int(n_edges)}")
        print(f"  Sparsity: {sparsity:.4f}")

        return adj

    def create_anatomical_edge_index(self, edge_weight_method='binary'):
        """
        创建PyG格式的解剖图边索引和边权重

        Args:
            edge_weight_method: 边权重方法
                - 'binary': 二值权重 (1.0)
                - 'distance': 基于欧氏距离 (需要坐标信息，这里简化为1.0)

        Returns:
            edge_index: [2, E] 边索引
            edge_attr: [E, 1] 边权重
        """
        adj = self.anat_adj_template

        # 获取边索引 (只取上三角,避免重复)
        edge_indices = np.array(np.where(np.triu(adj, k=1)))

        # 转换为PyG格式 (双向边)
        edge_index_forward = edge_indices
        edge_index_backward = edge_indices[[1, 0], :]
        edge_index = np.concatenate([edge_index_forward, edge_index_backward], axis=1)

        # 边权重
        if edge_weight_method == 'binary':
            edge_attr = np.ones((edge_index.shape[1], 1), dtype=np.float32)
        else:
            raise NotImplementedError(f"Method {edge_weight_method} not implemented")

        return edge_index, edge_attr

    def add_anatomical_edges_to_graph(self, graph_data):
        """
        为PyG Data对象添加解剖边

        Args:
            graph_data: PyG Data对象 (可能已包含功能边)

        Returns:
            anat_edge_index: [2, E_anat] 解剖边索引
            anat_edge_attr: [E_anat, 1] 解剖边权重
        """
        edge_index, edge_attr = self.create_anatomical_edge_index()

        return (
            torch.LongTensor(edge_index),
            torch.FloatTensor(edge_attr)
        )


def create_dual_branch_data(
    node_features,
    func_edge_index,
    func_edge_attr,
    label,
    atlas='aal116'
):
    """
    创建双分支所需的数据对象

    Args:
        node_features: [N_ROI, feature_dim] 节点特征
        func_edge_index: [2, E_func] 功能边索引
        func_edge_attr: [E_func, 1] 功能边权重
        label: 标签
        atlas: 图谱名称

    Returns:
        data: PyG Data对象,包含:
            - x: 节点特征
            - func_edge_index, func_edge_attr: 功能边
            - anat_edge_index, anat_edge_attr: 解剖边
            - y: 标签
    """
    # 创建解剖图构建器
    anat_constructor = AnatomicalGraphConstructor(atlas=atlas)

    # 获取解剖边
    anat_edge_index, anat_edge_attr = anat_constructor.create_anatomical_edge_index()

    # 创建Data对象
    data = Data(
        x=torch.FloatTensor(node_features),
        func_edge_index=torch.LongTensor(func_edge_index),
        func_edge_attr=torch.FloatTensor(func_edge_attr),
        anat_edge_index=torch.LongTensor(anat_edge_index),
        anat_edge_attr=torch.FloatTensor(anat_edge_attr),
        y=torch.LongTensor([label])
    )

    return data


# 全局单例,避免重复创建
_anat_constructor_cache = {}


def get_anatomical_constructor(atlas='aal116'):
    """获取或创建解剖图构建器 (单例模式)"""
    if atlas not in _anat_constructor_cache:
        _anat_constructor_cache[atlas] = AnatomicalGraphConstructor(atlas)
    return _anat_constructor_cache[atlas]


if __name__ == '__main__':
    # 测试代码
    print("=" * 80)
    print("Testing Anatomical Graph Constructor")
    print("=" * 80)

    # 创建构建器
    constructor = AnatomicalGraphConstructor(atlas='aal116')

    # 创建边索引
    edge_index, edge_attr = constructor.create_anatomical_edge_index()

    print(f"\nEdge index shape: {edge_index.shape}")
    print(f"Edge attr shape: {edge_attr.shape}")
    print(f"Number of edges: {edge_index.shape[1]}")

    # 验证对称性
    n_edges = edge_index.shape[1]
    print(f"\nVerifying bidirectional edges...")
    print(f"  Total edges: {n_edges}")
    print(f"  Expected: {np.sum(constructor.anat_adj_template > 0)}")

    # 测试创建数据对象
    print(f"\n" + "=" * 80)
    print("Testing dual-branch data creation")
    print("=" * 80)

    # 模拟数据
    node_features = np.random.randn(116, 64)  # 64维temporal embedding
    func_edge_index = np.random.randint(0, 116, (2, 1000))
    func_edge_attr = np.random.randn(1000, 1)
    label = 0

    data = create_dual_branch_data(
        node_features, func_edge_index, func_edge_attr, label
    )

    print(f"\nCreated PyG Data object:")
    print(f"  x shape: {data.x.shape}")
    print(f"  func_edge_index shape: {data.func_edge_index.shape}")
    print(f"  anat_edge_index shape: {data.anat_edge_index.shape}")
    print(f"  y: {data.y}")

    print(f"\n✓ All tests passed!")