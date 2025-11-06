"""
GNN数据集加载器
支持灵活切换FC方法和节点特征类型
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch.utils.data import random_split


class BrainGraphDataset(Dataset):
    """
    脑图数据集（PyTorch Geometric格式）
    """

    def __init__(
        self,
        data_file,
        fc_method='pearson',
        node_feature_type='statistical',
        threshold=None,
        transform=None,
        pre_transform=None
    ):
        """
        Args:
            data_file: .npz数据文件路径
            fc_method: 'pearson' 或 'ledoit_wolf'
            node_feature_type: 'statistical' 或 'temporal'
            threshold: 边阈值（None表示全连接，或提供数值/百分比）
            transform: 数据变换
            pre_transform: 预变换
        """
        self.data_file = data_file
        self.fc_method = fc_method
        self.node_feature_type = node_feature_type
        self.threshold = threshold

        # 加载数据
        self._load_data()

        super().__init__(None, transform, pre_transform)

    def _load_data(self):
        """加载npz文件"""
        print(f"Loading GNN dataset from: {self.data_file}")

        data = np.load(self.data_file, allow_pickle=True)

        # 选择FC矩阵
        if self.fc_method == 'pearson':
            self.fc_matrices = data['fc_pearson']
        elif self.fc_method == 'ledoit_wolf':
            self.fc_matrices = data['fc_ledoit_wolf']
        else:
            raise ValueError(f"Unknown FC method: {self.fc_method}")

        # 选择节点特征
        if self.node_feature_type == 'statistical':
            self.node_features = data['node_features_statistical']
        elif self.node_feature_type == 'temporal':
            self.node_features = data['node_features_temporal']
        else:
            raise ValueError(f"Unknown node feature type: {self.node_feature_type}")

        self.labels = data['labels']
        self.subject_ids = data['subject_ids']

        # 元信息
        self.n_subjects = len(self.labels)
        self.n_rois = self.fc_matrices.shape[1]
        self.feature_dim = self.node_features.shape[2]

        # 读取temporal_method（如果存在）
        self.temporal_method = str(data['temporal_method']) if 'temporal_method' in data else 'unknown'

        print(f"  FC method: {self.fc_method}")
        print(f"  Node features: {self.node_feature_type}")
        if self.node_feature_type == 'temporal':
            print(f"  Temporal method: {self.temporal_method}")
        print(f"  Subjects: {self.n_subjects}")
        print(f"  ROIs: {self.n_rois}")
        print(f"  Feature dim: {self.feature_dim}")

    def len(self):
        return self.n_subjects

    def get(self, idx):
        """
        获取第idx个图

        Returns:
            Data对象，包含：
            - x: 节点特征 [N_ROI, feature_dim]
            - edge_index: 边索引 [2, num_edges]
            - edge_attr: 边权重 [num_edges, 1]
            - y: 标签 [1]
        """
        # 节点特征
        x = torch.FloatTensor(self.node_features[idx])  # [N_ROI, feature_dim]

        # FC矩阵
        fc = self.fc_matrices[idx]  # [N_ROI, N_ROI]

        # 构建边（根据阈值）
        if self.threshold is None:
            # 全连接图（去除自环）
            edge_index, edge_attr = self._build_full_graph(fc)
        else:
            # 阈值图
            edge_index, edge_attr = self._build_thresholded_graph(fc, self.threshold)

        # 标签
        y = torch.LongTensor([self.labels[idx]])

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            subject_id=self.subject_ids[idx]
        )

        return data

    def _build_full_graph(self, fc_matrix):
        """构建全连接图"""
        n_rois = fc_matrix.shape[0]

        # 创建所有可能的边（排除自环）
        src_nodes = []
        dst_nodes = []
        edge_weights = []

        for i in range(n_rois):
            for j in range(n_rois):
                if i != j:  # 排除自环
                    src_nodes.append(i)
                    dst_nodes.append(j)
                    edge_weights.append(fc_matrix[i, j])

        edge_index = torch.LongTensor([src_nodes, dst_nodes])
        edge_attr = torch.FloatTensor(edge_weights).unsqueeze(1)

        return edge_index, edge_attr

    def _build_thresholded_graph(self, fc_matrix, threshold):
        """构建阈值图"""
        # 根据绝对值阈值
        if isinstance(threshold, float) and 0 < threshold < 1:
            # 百分比阈值
            abs_fc = np.abs(fc_matrix)
            np.fill_diagonal(abs_fc, 0)
            threshold_value = np.percentile(abs_fc[abs_fc > 0], threshold * 100)
        else:
            # 绝对值阈值
            threshold_value = threshold

        # 找到超过阈值的边
        abs_fc = np.abs(fc_matrix)
        mask = (abs_fc >= threshold_value) & (np.eye(len(fc_matrix)) == 0)

        src_nodes, dst_nodes = np.where(mask)
        edge_weights = fc_matrix[src_nodes, dst_nodes]

        edge_index = torch.LongTensor(np.stack([src_nodes, dst_nodes]))
        edge_attr = torch.FloatTensor(edge_weights).unsqueeze(1)

        return edge_index, edge_attr


def load_brain_graph_dataset(
    dataset_name,
    data_folder='./data',
    fc_method='pearson',
    node_feature_type='statistical',
    threshold=None,
    temporal_method='pca'
):
    """
    加载脑图数据集的便捷函数

    Args:
        dataset_name: 'ABIDE' 或 'MDD'
        data_folder: 数据根目录
        fc_method: FC方法 ('pearson' / 'ledoit_wolf')
        node_feature_type: 节点特征类型 ('statistical' / 'temporal')
        threshold: 边阈值
        temporal_method: 时序编码方法 ('pca' / 'cnn' / 'transformer')

    Returns:
        dataset: BrainGraphDataset对象
    """
    # 确定数据路径
    if dataset_name == 'ABIDE':
        gnn_path = os.path.join(data_folder, 'ABIDE', 'gnn_data')
    elif dataset_name == 'MDD':
        gnn_path = os.path.join(data_folder, 'REST-meta-MDD', 'gnn_data')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 查找指定temporal_method的最新文件
    import glob
    pattern = f'{dataset_name.lower()}_gnn_dataset_{temporal_method}_*.npz'
    npz_files = glob.glob(os.path.join(gnn_path, pattern))

    if not npz_files:
        # 如果找不到指定temporal_method的文件，尝试查找任意文件
        pattern_any = f'{dataset_name.lower()}_gnn_dataset_*.npz'
        npz_files = glob.glob(os.path.join(gnn_path, pattern_any))

        if not npz_files:
            raise FileNotFoundError(
                f"No GNN dataset found in {gnn_path}. "
                f"Please run: python prepare_gnn_data.py --dataset {dataset_name}"
            )
        else:
            print(f"⚠️  Warning: No dataset found for temporal_method='{temporal_method}'")
            print(f"    Using available dataset: {os.path.basename(npz_files[-1])}")

    # 使用最新的文件
    data_file = sorted(npz_files)[-1]
    print(f"\nLoading dataset: {os.path.basename(data_file)}")

    dataset = BrainGraphDataset(
        data_file=data_file,
        fc_method=fc_method,
        node_feature_type=node_feature_type,
        threshold=threshold
    )

    return dataset


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    划分数据集为训练/验证/测试集

    Args:
        dataset: BrainGraphDataset对象
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset
