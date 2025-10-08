"""
REST-meta-MDD数据集处理模块 - 整合版
支持：
1. 传统单流图（基线实验）
2. 双流图（SF-DPL实验）
3. 时序特征 + 统计特征
"""
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from scipy import stats, signal
from sklearn.covariance import LedoitWolf
import networkx as nx
import glob
import re
import warnings
warnings.filterwarnings('ignore')


class MDDDataProcessor:
    """
    统一的MDD数据处理器

    核心功能：
    1. 加载REST-meta-MDD数据
    2. 生成多种格式的图数据
    3. 支持不同的节点特征类型
    """

    def __init__(self,
                 data_folder='./data',
                 time_strategy='adaptive',
                 min_time_length=100):
        """
        Args:
            time_strategy:
                - 'adaptive': 自适应确定时间长度
                - 'fixed': 固定到min_time_length
        """
        self.data_folder = data_folder
        self.time_strategy = time_strategy
        self.min_time_length = min_time_length

        self.mdd_path = os.path.join(data_folder, 'REST-meta-MDD')
        self.processed_path = os.path.join(self.mdd_path, 'processed')

        os.makedirs(self.processed_path, exist_ok=True)

    def load_roi_signals(self):
        """加载ROI信号数据"""
        print("加载REST-meta-MDD ROI数据...")

        roi_folder = os.path.join(
            self.mdd_path,
            'Results',
            'ROISignals_FunImgARCWF'
        )

        if not os.path.exists(roi_folder):
            print(f"ROI文件夹不存在: {roi_folder}")
            print("请确保数据已下载至正确位置")
            return None

        # 递归查找所有.mat文件
        mat_files = glob.glob(
            os.path.join(roi_folder, '**', '*.mat'),
            recursive=True
        )

        print(f"找到 {len(mat_files)} 个.mat文件")

        data_dict = {
            'time_series': [],
            'labels': [],
            'subject_ids': [],
            'file_paths': []
        }

        label_counts = {'MDD': 0, 'HC': 0, 'unknown': 0, 'error': 0}

        for file_idx, file_path in enumerate(mat_files):
            try:
                filename = os.path.basename(file_path)

                # 从文件名提取标签
                # 格式: ROISignals_S1-1-0001.mat
                # -1- = MDD患者, -2- = 健康对照
                match = re.search(r'-(\d)-', filename)
                if not match:
                    label_counts['unknown'] += 1
                    continue

                group_code = int(match.group(1))
                if group_code == 1:
                    label = 1  # MDD
                    label_counts['MDD'] += 1
                elif group_code == 2:
                    label = 0  # HC
                    label_counts['HC'] += 1
                else:
                    label_counts['unknown'] += 1
                    continue

                # 加载.mat文件
                mat_data = sio.loadmat(file_path)

                # 查找时间序列数据
                if 'ROISignals' not in mat_data:
                    label_counts['error'] += 1
                    continue

                time_series = mat_data['ROISignals']

                # ==========================================
                # 关键修改：只提取AAL图谱的116个ROI（列1-116）
                # ==========================================
                # 原始数据：[T, 1833]
                # 我们只要前116列
                if time_series.shape[1] < 116:
                    label_counts['error'] += 1
                    continue

                time_series = time_series[:, :116]  # [T, 116]

                # 检查有效性
                if time_series.shape[0] < 50:
                    label_counts['error'] += 1
                    continue

                # 检查是否有NaN或Inf
                if np.isnan(time_series).any() or np.isinf(time_series).any():
                    time_series = np.nan_to_num(time_series, nan=0.0, posinf=0.0, neginf=0.0)

                # 提取被试ID
                subject_id = filename.split('.')[0]

                # 保存数据
                data_dict['time_series'].append(time_series)
                data_dict['labels'].append(label)
                data_dict['subject_ids'].append(subject_id)
                data_dict['file_paths'].append(file_path)

                if (file_idx + 1) % 50 == 0:
                    print(f"已处理: {file_idx + 1}/{len(mat_files)}")

            except Exception as e:
                label_counts['error'] += 1
                if file_idx < 5:
                    print(f"处理文件 {os.path.basename(file_path)} 失败: {e}")
                continue

        # 打印统计信息
        print(f"\n{'='*60}")
        print(f"数据加载完成统计:")
        print(f"{'='*60}")
        print(f"成功加载: {len(data_dict['time_series'])} 个被试")
        print(f"  - MDD患者 (标签1): {label_counts['MDD']}")
        print(f"  - 健康对照 (标签0): {label_counts['HC']}")
        print(f"  - 无法识别标签: {label_counts['unknown']}")
        print(f"  - 处理错误: {label_counts['error']}")

        # 验证标签平衡性
        if label_counts['MDD'] > 0 and label_counts['HC'] > 0:
            ratio = max(label_counts['MDD'], label_counts['HC']) / min(label_counts['MDD'], label_counts['HC'])
            print(f"\n类别比例: {ratio:.2f}:1", end=" ")
            if ratio > 3:
                print("(类别不平衡)")
            else:
                print("(分布合理)")

        return data_dict

    def determine_time_length(self, time_series_list):
        """确定统一的时间长度"""
        if not time_series_list:
            return self.min_time_length

        lengths = [ts.shape[0] for ts in time_series_list[:50]]

        if not lengths:
            return self.min_time_length

        lengths = np.array(lengths)

        if self.time_strategy == 'fixed':
            return self.min_time_length
        elif self.time_strategy == 'adaptive':
            return int(np.percentile(lengths, 10))
        else:
            return self.min_time_length

    def process_time_series(self, time_series, target_length):
        """处理时间序列到目标长度"""
        T, N = time_series.shape

        if T >= target_length:
            return time_series[:target_length, :]
        else:
            # 插值上采样
            from scipy.interpolate import interp1d
            t_old = np.linspace(0, 1, T)
            t_new = np.linspace(0, 1, target_length)

            resampled = np.zeros((target_length, N))
            for i in range(N):
                try:
                    f = interp1d(t_old, time_series[:, i], kind='cubic')
                    resampled[:, i] = f(t_new)
                except:
                    resampled[:, i] = np.interp(t_new, t_old, time_series[:, i])

            return resampled

    # ==========================================
    # 节点特征提取（与ABIDE相同）
    # ==========================================

    def extract_temporal_features(self, time_series):
        """提取时序特征"""
        """
            提取时序特征（添加标准化）

            Args:
                time_series: [T, N]

            Returns:
                features: [N, T] 标准化后的时序特征
            """
        # 转置
        node_features = time_series.T  # [N, T] = [116, 78]

        # ==========================================
        # 关键修改：标准化每个节点的时间序列
        # ==========================================
        from sklearn.preprocessing import StandardScaler

        # 转为numpy进行标准化
        node_features_np = node_features  # 如果是torch tensor，转numpy
        if isinstance(node_features_np, torch.Tensor):
            node_features_np = node_features_np.numpy()

        # 对每个节点（脑区）的时间序列单独标准化
        scaler = StandardScaler()
        for i in range(node_features_np.shape[0]):
            # 将[T]变成[T, 1]以便标准化
            ts = node_features_np[i].reshape(-1, 1)
            node_features_np[i] = scaler.fit_transform(ts).flatten()

        # 转回torch tensor
        node_features = torch.tensor(node_features_np, dtype=torch.float)

        # 验证标准化效果
        if torch.isnan(node_features).any() or torch.isinf(node_features).any():
            print("警告：标准化后出现NaN或Inf，使用原始数据")
            return torch.tensor(time_series.T, dtype=torch.float)

        return node_features
        # return torch.tensor(time_series.T, dtype=torch.float)

    def extract_statistical_features(self, time_series):
        """提取统计特征"""
        features = []

        for i in range(time_series.shape[1]):
            ts = time_series[:, i]

            feat = [
                np.mean(ts),
                np.std(ts),
                np.median(ts),
                np.ptp(ts),
                stats.skew(ts) if not np.isnan(stats.skew(ts)) else 0,
                stats.kurtosis(ts) if not np.isnan(stats.kurtosis(ts)) else 0,
                np.percentile(ts, 25),
                np.percentile(ts, 75)
            ]

            # 频域特征
            try:
                freqs, psd = signal.welch(ts, fs=0.5, nperseg=min(len(ts), 64))
                # MDD相关的低频振荡
                freq_bands = [(0.01, 0.04), (0.04, 0.08), (0.08, 0.13)]
                for low, high in freq_bands:
                    idx = (freqs >= low) & (freqs <= high)
                    feat.append(np.mean(psd[idx]) if np.any(idx) else 0)
            except:
                feat.extend([0, 0, 0])

            features.append(feat)

        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, 0)

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        return torch.tensor(features, dtype=torch.float)

    def extract_enhanced_features(self, time_series, adj_matrix):
        """提取增强特征"""
        stat_feat = self.extract_statistical_features(time_series)

        N = time_series.shape[1]
        G = nx.from_numpy_array(np.abs(adj_matrix))

        degrees = dict(G.degree())
        clustering = nx.clustering(G)

        try:
            betweenness = nx.betweenness_centrality(G)
        except:
            betweenness = {i: 0 for i in range(N)}

        topo_feat = []
        for i in range(N):
            topo_feat.append([
                degrees[i] / N,
                clustering[i],
                betweenness[i]
            ])

        topo_feat = torch.tensor(topo_feat, dtype=torch.float)

        return torch.cat([stat_feat, topo_feat], dim=1)

    # ==========================================
    # 邻接矩阵构建
    # ==========================================

    def construct_anatomical_adj(self, n_regions=116):
        """构建基于解剖的邻接矩阵（与ABIDE相同）"""
        adj = np.zeros((n_regions, n_regions))

        # AAL图谱划分
        left_indices = list(range(0, 45))
        right_indices = list(range(45, 90))
        midline_indices = list(range(90, 116))

        for i in range(n_regions):
            if i in left_indices:
                adj[i, right_indices] = 1
                adj[i, midline_indices] = 1
            elif i in right_indices:
                adj[i, left_indices] = 1
                adj[i, midline_indices] = 1
            else:
                adj[i, left_indices] = 1
                adj[i, right_indices] = 1

        adj = (adj + adj.T) / 2
        adj[adj > 0] = 1
        np.fill_diagonal(adj, 0)

        return adj

    def construct_functional_adj(self, time_series, use_dynamic=True):
        """
        构建基于功能连接的邻接矩阵
        MDD强调动态连接（情绪波动）
        """
        # 相关矩阵
        corr = np.corrcoef(time_series.T)
        corr = np.nan_to_num(corr, 0)
        np.fill_diagonal(corr, 0)

        if use_dynamic:
            # 动态连接（MDD关键特征）
            dynamic_var = self._compute_dynamic_variance(time_series)
            func_conn = (corr + dynamic_var * 2) / 3  # 强调动态
        else:
            func_conn = corr

        # 阈值化
        threshold = np.percentile(np.abs(func_conn), 70)
        adj = np.zeros_like(func_conn)
        adj[np.abs(func_conn) > threshold] = func_conn[np.abs(func_conn) > threshold]

        return adj

    def _compute_dynamic_variance(self, time_series):
        """计算动态连接变异性"""
        window_size = min(50, time_series.shape[0] // 3)
        stride = 10

        n_regions = time_series.shape[1]
        dynamic_corrs = []

        for start in range(0, time_series.shape[0] - window_size, stride):
            end = start + window_size
            window_ts = time_series[start:end, :]

            try:
                window_corr = np.corrcoef(window_ts.T)
                window_corr = np.nan_to_num(window_corr, 0)
                dynamic_corrs.append(window_corr)
            except:
                continue

        if len(dynamic_corrs) > 1:
            variance = np.std(dynamic_corrs, axis=0)
        else:
            variance = np.zeros((n_regions, n_regions))

        return variance

    # ==========================================
    # 主处理函数
    # ==========================================

    def process_and_save(self,
                        feature_type='temporal',
                        graph_type='dual_stream',
                        use_dynamic=True):
        """
        统一的处理和保存函数

        Args:
            feature_type: 'temporal' / 'statistical' / 'enhanced'
            graph_type: 'dual_stream' / 'single_functional'
            use_dynamic: 是否使用动态连接
        """
        print(f"\n{'='*60}")
        print(f"MDD数据处理")
        print(f"特征类型: {feature_type}")
        print(f"图类型: {graph_type}")
        print(f"{'='*60}")

        # 加载数据
        data_dict = self.load_roi_signals()

        if data_dict is None or len(data_dict['time_series']) == 0:
            print("错误：未找到有效数据")
            return []

        # 确定时间长度
        target_length = self.determine_time_length(data_dict['time_series'])
        print(f"目标时间长度: {target_length}")

        # 处理每个被试
        graph_list = []

        for idx, time_series in enumerate(data_dict['time_series']):
            try:
                # 处理到目标长度
                time_series = self.process_time_series(time_series, target_length)

                if time_series.shape[0] < 50:
                    continue

                label = data_dict['labels'][idx]

                # 根据配置生成图
                if graph_type == 'dual_stream':
                    graphs = self._create_dual_stream_graphs(
                        time_series, label, feature_type, use_dynamic
                    )
                elif graph_type == 'single_functional':
                    graphs = self._create_single_graph(
                        time_series, label, feature_type, use_dynamic
                    )
                else:
                    continue

                graph_list.append(graphs)

                if (idx + 1) % 20 == 0:
                    print(f"已处理: {idx + 1}/{len(data_dict['time_series'])}")

            except Exception as e:
                print(f"处理样本{idx}失败: {e}")
                continue

        print(f"\n成功构建 {len(graph_list)} 个样本")

        # 保存
        save_name = f'mdd_{graph_type}_{feature_type}_{target_length}.pt'
        save_path = os.path.join(self.processed_path, save_name)
        torch.save(graph_list, save_path)
        print(f"数据已保存至: {save_path}")

        # 保存元信息
        if graph_list:
            sample = graph_list[0]
            if isinstance(sample, tuple):
                sample = sample[0]

            meta_info = {
                'n_subjects': len(graph_list),
                'time_length': target_length,
                'feature_type': feature_type,
                'graph_type': graph_type,
                'node_feature_dim': sample.x.shape[1],
                'n_regions': sample.x.shape[0]
            }

            meta_path = os.path.join(self.processed_path, 'meta_info.pt')
            torch.save(meta_info, meta_path)
            print(f"\n样本信息:")
            print(f"  节点特征: {sample.x.shape}")
            print(f"  边数: {sample.edge_index.shape[1]}")

        return graph_list

    def _create_dual_stream_graphs(self, time_series, label, feature_type, use_dynamic):
        """创建双流图"""
        N = time_series.shape[1]

        # 构建邻接矩阵
        struct_adj = self.construct_anatomical_adj(N)
        func_adj = self.construct_functional_adj(time_series, use_dynamic)

        # 提取节点特征
        if feature_type == 'temporal':
            node_features = self.extract_temporal_features(time_series)
        elif feature_type == 'statistical':
            node_features = self.extract_statistical_features(time_series)
        elif feature_type == 'enhanced':
            node_features = self.extract_enhanced_features(time_series, func_adj)

        # 转换为PyG格式
        struct_data = self._adj_to_pyg(struct_adj, node_features, label)
        func_data = self._adj_to_pyg(func_adj, node_features, label, weighted=True)

        return (func_data, struct_data)

    def _create_single_graph(self, time_series, label, feature_type, use_dynamic):
        """创建单流图"""
        N = time_series.shape[1]

        # 构建功能邻接
        adj = self.construct_functional_adj(time_series, use_dynamic)

        # 提取节点特征
        if feature_type == 'temporal':
            node_features = self.extract_temporal_features(time_series)
        elif feature_type == 'statistical':
            node_features = self.extract_statistical_features(time_series)
        elif feature_type == 'enhanced':
            node_features = self.extract_enhanced_features(time_series, adj)

        # 转换为PyG格式
        return self._adj_to_pyg(adj, node_features, label, weighted=True)

    def _adj_to_pyg(self, adj_matrix, node_features, label, weighted=False):
        """转换为PyG Data格式"""
        edge_index = []
        edge_attr = [] if weighted else None

        N = adj_matrix.shape[0]
        for i in range(N):
            for j in range(N):
                if adj_matrix[i, j] != 0:
                    edge_index.append([i, j])
                    if weighted:
                        edge_attr.append(adj_matrix[i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

        if weighted:
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.long)
        )


# ==========================================
# 便捷函数
# ==========================================

def prepare_all_data(data_folder='./data'):
    """一键准备所有格式的数据"""

    processor = MDDDataProcessor(
        data_folder=data_folder,
        time_strategy='adaptive'
    )

    configs = [
        # SF-DPL用：双流 + 时序特征
        {'feature_type': 'temporal', 'graph_type': 'dual_stream', 'use_dynamic': True},

        # 基线用：单流 + 统计特征
        {'feature_type': 'statistical', 'graph_type': 'single_functional', 'use_dynamic': True},

        # 基线用：单流 + 增强特征
        {'feature_type': 'enhanced', 'graph_type': 'single_functional', 'use_dynamic': True},
    ]

    results = {}

    for config in configs:
        name = f"{config['graph_type']}_{config['feature_type']}"
        print(f"\n处理配置: {name}")

        graph_list = processor.process_and_save(**config)

        results[name] = len(graph_list)

    print(f"\n{'='*60}")
    print("所有数据处理完成")
    print(f"{'='*60}")
    for name, count in results.items():
        print(f"{name}: {count} 个样本")

    return results


if __name__ == "__main__":
    # 一键准备所有数据
    prepare_all_data(data_folder='./data')