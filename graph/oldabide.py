"""
ABIDE数据集处理模块
支持下载、处理和构建脑功能图
"""
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from nilearn.datasets import fetch_abide_pcp
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import StandardScaler
from scipy import signal, stats
from sklearn.covariance import LedoitWolf
import warnings

warnings.filterwarnings('ignore')


class ABIDEDataProcessor:
    """ABIDE数据集处理器"""

    def __init__(self, data_folder='./data',
                 pipeline='cpac',
                 atlas='ho',
                 connectivity_kind='correlation',
                 threshold=0.3):
        """
        初始化ABIDE数据处理器

        Args:
            data_folder: 数据保存路径
            pipeline: 预处理管道 ('cpac', 'ccs', 'dparsf', 'niak')
            atlas: 脑区分割图谱 ('ho', 'cc200', 'cc400', 'aal', 'ez', 'dosenbach160')
            connectivity_kind: 连接性度量方式 ('correlation', 'partial correlation', 'covariance')
            threshold: 边权重阈值，用于构建稀疏图
        """
        self.data_folder = data_folder
        self.pipeline = pipeline
        self.atlas = atlas
        self.connectivity_kind = connectivity_kind
        self.threshold = threshold
        self.abide_path = os.path.join(data_folder, 'ABIDE')
        self.processed_path = os.path.join(self.abide_path, 'processed')

        # 创建必要的目录
        os.makedirs(self.processed_path, exist_ok=True)

    def download_data(self, n_subjects=None):
        """
        下载ABIDE数据集

        Args:
            n_subjects: 下载的被试数量，None表示全部

        Returns:
            data: 下载的数据字典
        """
        print(f"正在下载ABIDE数据集...")
        print(f"参数: pipeline={self.pipeline}, atlas={self.atlas}")

        # 根据atlas选择不同的数据类型
        if self.atlas == 'ho':
            derivatives = 'rois_ho'
        elif self.atlas in ['cc200', 'cc400']:
            derivatives = f'rois_{self.atlas}'
        elif self.atlas == 'aal':
            derivatives = 'rois_aal'
        elif self.atlas == 'ez':
            derivatives = 'rois_ez'
        elif self.atlas == 'dosenbach160':
            derivatives = 'rois_dosenbach160'
        else:
            derivatives = 'rois_ho'  # 默认使用HO

        data = fetch_abide_pcp(
            data_dir=self.abide_path,
            pipeline=self.pipeline,
            band_pass_filtering=True,
            global_signal_regression=True,
            derivatives=derivatives,
            n_subjects=n_subjects,
            verbose=1
        )

        # 调试信息
        print(f"下载完成，数据结构:")
        print(f"  - 可用属性: {dir(data)}")

        # 检查不同的可能属性名
        rois_attr = None
        for attr in ['rois_ho', 'rois_cc200', 'rois_cc400', 'rois_aal', 'rois_ez', 'rois_dosenbach160']:
            if hasattr(data, attr):
                rois_attr = attr
                rois_data = getattr(data, attr)
                print(f"  - 找到ROI数据: {attr}")
                print(f"  - ROI数据类型: {type(rois_data)}")
                if rois_data is not None:
                    print(f"  - ROI数据长度: {len(rois_data) if hasattr(rois_data, '__len__') else 'N/A'}")
                    if len(rois_data) > 0:
                        print(f"  - 第一个元素类型: {type(rois_data[0])}")
                        if isinstance(rois_data[0], np.ndarray):
                            print(f"  - 第一个元素形状: {rois_data[0].shape}")
                break

        if rois_attr is None:
            print("  - 警告：未找到ROI数据属性")

        return data

    def construct_brain_graph(self, time_series, method='correlation_matrix'):
        """
        从时间序列构建脑功能连接图

        Args:
            time_series: 脑区时间序列 (time_points, n_regions)
            method: 构建方法
                - 'correlation_matrix': 相关矩阵（改进版）
                - 'dynamic_connectivity': 动态连接性
                - 'phase_synchronization': 相位同步

        Returns:
            edge_index: 边索引
            edge_attr: 边权重
            node_features: 节点特征
        """
        n_regions = time_series.shape[1]

        if method == 'correlation_matrix':
            # 计算皮尔逊相关系数矩阵
            corr_matrix = np.corrcoef(time_series.T)

            # 处理NaN值
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            # Fisher z变换，稳定化相关系数
            with np.errstate(divide='ignore', invalid='ignore'):
                corr_matrix = np.arctanh(corr_matrix)
                corr_matrix[np.isinf(corr_matrix)] = 0
                corr_matrix[np.isnan(corr_matrix)] = 0

            # 设置对角线为0
            np.fill_diagonal(corr_matrix, 0)

            # 应用阈值创建稀疏图
            adj_matrix = np.abs(corr_matrix)

            # 使用自适应阈值：保留最强的连接
            if self.threshold < 1.0:
                threshold_value = np.percentile(adj_matrix[adj_matrix > 0],
                                                (1 - self.threshold) * 100)
                adj_matrix[adj_matrix < threshold_value] = 0

        elif method == 'dynamic_connectivity':
            # 滑动窗口动态连接性
            window_size = min(30, time_series.shape[0] // 3)
            stride = window_size // 2
            n_windows = (time_series.shape[0] - window_size) // stride + 1

            dynamic_corr = []
            for i in range(n_windows):
                start = i * stride
                end = start + window_size
                window_data = time_series[start:end]
                corr = np.corrcoef(window_data.T)
                corr = np.nan_to_num(corr, nan=0.0)
                dynamic_corr.append(corr)

            # 平均动态连接性
            adj_matrix = np.mean(dynamic_corr, axis=0)
            np.fill_diagonal(adj_matrix, 0)
            adj_matrix = np.abs(adj_matrix)

            # 应用阈值
            threshold_value = np.percentile(adj_matrix[adj_matrix > 0],
                                            (1 - self.threshold) * 100)
            adj_matrix[adj_matrix < threshold_value] = 0

        elif method == 'phase_synchronization':
            # 相位同步（使用希尔伯特变换）
            from scipy.signal import hilbert

            # 计算瞬时相位
            analytic_signal = hilbert(time_series, axis=0)
            phase = np.angle(analytic_signal)

            # 计算相位同步
            adj_matrix = np.zeros((n_regions, n_regions))
            for i in range(n_regions):
                for j in range(i + 1, n_regions):
                    # 相位差
                    phase_diff = phase[:, i] - phase[:, j]
                    # 相位锁定值 (PLV)
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    adj_matrix[i, j] = plv
                    adj_matrix[j, i] = plv

            # 应用阈值
            threshold_value = np.percentile(adj_matrix[adj_matrix > 0],
                                            (1 - self.threshold) * 100)
            adj_matrix[adj_matrix < threshold_value] = 0

        # 转换为PyG格式的边
        edge_index = []
        edge_attr = []
        for i in range(n_regions):
            for j in range(n_regions):
                if adj_matrix[i, j] > 0 and i != j:
                    edge_index.append([i, j])
                    edge_attr.append(adj_matrix[i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # 构建节点特征
        node_features = self.extract_node_features(time_series)

        return edge_index, edge_attr, node_features

    def extract_node_features(self, time_series):
        """
        从时间序列中提取节点特征

        Args:
            time_series: 脑区时间序列 (time_points, n_regions)

        Returns:
            node_features: 节点特征矩阵
        """
        features = []

        for i in range(time_series.shape[1]):
            region_series = time_series[:, i]

            # 统计特征
            feat = [
                np.mean(region_series),  # 均值
                np.std(region_series),  # 标准差
                np.median(region_series),  # 中位数
                stats.skew(region_series),  # 偏度
                stats.kurtosis(region_series),  # 峰度
                np.percentile(region_series, 25),  # 25分位数
                np.percentile(region_series, 75),  # 75分位数
                np.max(region_series) - np.min(region_series),  # 范围
            ]

            # 频域特征（功率谱密度）
            from scipy import signal
            freqs, psd = signal.welch(region_series, fs=1.0, nperseg=min(len(region_series), 256))

            # 提取不同频段的功率
            # Delta (0.01-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz)
            freq_bands = [(0.01, 0.04), (0.04, 0.08), (0.08, 0.13), (0.13, 0.30)]
            for low, high in freq_bands:
                idx = np.logical_and(freqs >= low, freqs <= high)
                if np.any(idx):
                    feat.append(np.mean(psd[idx]))
                else:
                    feat.append(0.0)

            features.append(feat)

        features = np.array(features, dtype=np.float32)

        # 处理NaN和Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 标准化
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        return torch.tensor(features, dtype=torch.float)

    def process_and_save(self, n_subjects=None, graph_method='correlation_matrix'):
        """
        处理并保存ABIDE数据集为PyG格式

        Args:
            n_subjects: 处理的被试数量
            graph_method: 图构建方法

        Returns:
            graph_list: PyG Data对象列表
        """
        # 下载数据
        data = self.download_data(n_subjects)

        # 获取标签
        phenotypic = data.phenotypic
        labels = phenotypic['DX_GROUP'].values - 1  # 0: 控制组, 1: ASD

        # 获取时间序列数据
        rois_data = None

        # 尝试不同的属性名
        for attr in ['rois_ho', 'rois_cc200', 'rois_cc400', 'rois_aal', 'rois_ez', 'rois_dosenbach160']:
            if hasattr(data, attr):
                rois_data = getattr(data, attr)
                if rois_data is not None:
                    print(f"使用ROI数据: {attr}")
                    break

        if rois_data is None:
            print("错误：未找到ROI数据，请检查数据下载是否成功")
            return []

        graph_list = []
        valid_indices = []

        print(f"正在构建脑功能图 (方法: {graph_method})...")
        print(f"ROI数据数量: {len(rois_data)}")

        for idx, roi_data in enumerate(rois_data):
            try:
                # 处理不同的数据格式
                if roi_data is None:
                    print(f"跳过被试 {idx}: ROI数据为None")
                    continue

                # 检查是否是numpy数组（直接的时间序列数据）
                if isinstance(roi_data, np.ndarray):
                    time_series = roi_data
                    print(f"处理被试 {idx}: 直接使用numpy数组")
                # 检查是否是文件路径
                elif isinstance(roi_data, str):
                    if not os.path.exists(roi_data):
                        print(f"跳过被试 {idx}: ROI文件不存在 ({roi_data})")
                        continue
                    print(f"处理被试 {idx}: 从文件加载 {roi_data}")
                    # 加载时间序列
                    time_series = pd.read_csv(roi_data, sep='\t', header=0).values
                else:
                    print(f"跳过被试 {idx}: 未知的数据类型 ({type(roi_data)})")
                    continue

                print(f"  时间序列形状: {time_series.shape}")

                # 检查数据有效性
                if time_series.shape[0] < 50:  # 时间点太少
                    print(f"跳过被试 {idx}: 时间序列太短 ({time_series.shape[0]} < 50)")
                    continue

                # 构建图
                edge_index, edge_attr, node_features = self.construct_brain_graph(
                    time_series, method=graph_method
                )

                # 创建PyG Data对象
                data_obj = Data(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.tensor([labels[idx]], dtype=torch.long)
                )

                graph_list.append(data_obj)
                valid_indices.append(idx)

                if (idx + 1) % 50 == 0:
                    print(f"已处理 {idx + 1}/{len(rois_data)} 个被试")

            except Exception as e:
                print(f"处理被试 {idx} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"成功构建 {len(graph_list)} 个脑功能图")

        # 保存处理后的数据
        if graph_list:
            save_path = os.path.join(self.processed_path, f'abide_graphs_{graph_method}.pt')
            torch.save(graph_list, save_path)
            print(f"数据已保存到: {save_path}")

            # 保存元信息
            meta_info = {
                'n_subjects': len(graph_list),
                'n_features': graph_list[0].x.shape[1] if graph_list else 0,
                'graph_method': graph_method,
                'atlas': self.atlas,
                'pipeline': self.pipeline,
                'valid_indices': valid_indices
            }

            meta_path = os.path.join(self.processed_path, f'meta_info_{graph_method}.pt')
            torch.save(meta_info, meta_path)
        else:
            print("警告：没有成功构建任何图")

        return graph_list

    def load_processed_data(self, graph_method='correlation_matrix'):
        """
        加载已处理的数据

        Args:
            graph_method: 图构建方法

        Returns:
            graph_list: PyG Data对象列表
        """
        save_path = os.path.join(self.processed_path, f'abide_graphs_{graph_method}.pt')

        if os.path.exists(save_path):
            print(f"加载已处理的数据: {save_path}")
            return torch.load(save_path)
        else:
            print("未找到已处理的数据，开始处理...")
            return self.process_and_save(graph_method=graph_method)


def load_abide_data(data_folder='./data', n_subjects=None, graph_method='correlation_matrix'):
    """
    便捷函数：加载ABIDE数据集

    Args:
        data_folder: 数据保存路径
        n_subjects: 被试数量
        graph_method: 图构建方法

    Returns:
        graph_list: 图数据列表
        input_dim: 输入特征维度
        output_dim: 输出类别数
    """
    processor = ABIDEDataProcessor(data_folder=data_folder)

    # 尝试加载已处理的数据
    graph_list = processor.load_processed_data(graph_method=graph_method)

    if not graph_list:
        # 如果没有已处理的数据，则重新处理
        graph_list = processor.process_and_save(n_subjects=n_subjects, graph_method=graph_method)

    if graph_list:
        input_dim = graph_list[0].x.shape[1]
        output_dim = 2  # 二分类：控制组 vs ASD
    else:
        input_dim = 0
        output_dim = 2

    return graph_list, input_dim, output_dim

# 双流数据处理扩展
class ABIDEDualStream:
    """ABIDE双流数据处理扩展"""

    def __init__(self, original_processor):
        """基于原有ABIDE处理器扩展"""
        self.processor = original_processor

    def construct_dual_stream(self, time_series, label):
        """为单个被试构建双流数据"""
        # 功能流：多种连接矩阵融合
        func_matrix = self._construct_functional_network(time_series)

        # 结构流：稳定连接
        struct_matrix = self._construct_structural_network(time_series)

        # 节点特征
        func_features = self._extract_functional_features(time_series)
        struct_features = self._extract_structural_features(time_series)

        # 转换为PyG格式
        func_data = self._matrix_to_pyg(func_matrix, func_features, label)
        struct_data = self._matrix_to_pyg(struct_matrix, struct_features, label)

        return func_data, struct_data

    def _construct_functional_network(self, time_series):
        """构建功能网络（添加 nan 处理）"""
        # 预处理时间序列
        time_series = np.nan_to_num(time_series, nan=0.0, posinf=0.0, neginf=0.0)

        # 检查是否有全0列
        for col in range(time_series.shape[1]):
            if np.std(time_series[:, col]) < 1e-6:
                time_series[:, col] = np.random.randn(time_series.shape[0]) * 0.01

        # 相关矩阵
        try:
            corr = np.corrcoef(time_series.T)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(corr, 0)
        except:
            corr = np.zeros((time_series.shape[1], time_series.shape[1]))

        # 偏相关（针对ASD）
        try:
            cov = LedoitWolf().fit(time_series)
            precision = cov.precision_
            diag = np.sqrt(np.diag(precision))
            diag = np.where(diag < 1e-6, 1.0, diag)  # 避免除零
            partial = -precision / np.outer(diag, diag)
            partial = np.nan_to_num(partial, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(partial, 0)
        except:
            partial = corr.copy()

        # 相位同步（针对ASD长程连接）
        phase_sync = self._phase_synchronization(time_series)

        # 融合：ASD特别关注相位同步
        func_matrix = (corr + partial + phase_sync * 1.5) / 3.5
        func_matrix = np.nan_to_num(func_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # 阈值化
        threshold = np.percentile(np.abs(func_matrix), 70)
        func_matrix[np.abs(func_matrix) < threshold] = 0

        return func_matrix

    def _construct_structural_network(self, time_series):
        """构建结构网络（伪）（添加 nan 处理）"""
        # 预处理
        time_series = np.nan_to_num(time_series, nan=0.0, posinf=0.0, neginf=0.0)

        # 使用稳定的强连接
        try:
            corr = np.corrcoef(time_series.T)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        except:
            corr = np.zeros((time_series.shape[1], time_series.shape[1]))

        # 只保留最强连接
        threshold = np.percentile(np.abs(corr), 95)
        struct = np.zeros_like(corr)
        struct[np.abs(corr) > threshold] = 1
        np.fill_diagonal(struct, 0)

        return struct

    def _phase_synchronization(self, time_series):
        """计算相位同步（ASD关键特征）（添加 nan 处理）"""
        from scipy.signal import hilbert
        n_regions = time_series.shape[1]
        phase_sync = np.zeros((n_regions, n_regions))

        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                try:
                    # 检查输入
                    ts_i = time_series[:, i]
                    ts_j = time_series[:, j]

                    if np.isnan(ts_i).any() or np.isnan(ts_j).any():
                        ts_i = np.nan_to_num(ts_i, nan=0.0)
                        ts_j = np.nan_to_num(ts_j, nan=0.0)

                    # 去除常数序列
                    if np.std(ts_i) < 1e-6 or np.std(ts_j) < 1e-6:
                        phase_sync[i, j] = phase_sync[j, i] = 0.0
                        continue

                    # Hilbert变换
                    phase_i = np.angle(hilbert(ts_i))
                    phase_j = np.angle(hilbert(ts_j))

                    # 相位锁定值
                    plv = np.abs(np.mean(np.exp(1j * (phase_i - phase_j))))

                    if np.isnan(plv) or np.isinf(plv):
                        plv = 0.0

                    phase_sync[i, j] = phase_sync[j, i] = plv

                except Exception as e:
                    phase_sync[i, j] = phase_sync[j, i] = 0.0

        # 最终检查
        phase_sync = np.nan_to_num(phase_sync, nan=0.0, posinf=0.0, neginf=0.0)
        return phase_sync

    def _extract_functional_features(self, time_series):
        """提取功能特征（添加 nan 处理）"""
        features = []

        for i in range(time_series.shape[1]):
            ts = time_series[:, i]

            # 检查并处理nan值
            if np.isnan(ts).any():
                ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)

            # 如果整个时间序列都是0或常数
            if np.std(ts) < 1e-6:
                ts = ts + np.random.randn(len(ts)) * 0.01

            try:
                feat = [
                    np.mean(ts),
                    np.std(ts) if np.std(ts) > 0 else 1e-6,
                    stats.skew(ts) if not np.isnan(stats.skew(ts)) else 0.0,
                    stats.kurtosis(ts) if not np.isnan(stats.kurtosis(ts)) else 0.0,
                    np.percentile(ts, 25),
                    np.percentile(ts, 75)
                ]

                # 频域特征
                try:
                    freqs, psd = signal.welch(ts, fs=0.5, nperseg=min(len(ts), 64))
                    freq_bands = [(0.01, 0.04), (0.04, 0.08), (0.08, 0.13), (0.13, 0.30)]
                    for low, high in freq_bands:
                        idx = (freqs >= low) & (freqs <= high)
                        if np.any(idx):
                            band_power = np.mean(psd[idx])
                            feat.append(band_power if not np.isnan(band_power) else 0.0)
                        else:
                            feat.append(0.0)
                except:
                    feat.extend([0.0] * 4)
            except:
                feat = [0.0] * 10

            # 再次检查是否有nan
            feat = [0.0 if np.isnan(f) or np.isinf(f) else f for f in feat]
            features.append(feat)

        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 标准化（处理全0列）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        # 检查是否有全0或常数列
        for col in range(features.shape[1]):
            if np.std(features[:, col]) < 1e-6:
                features[:, col] = np.random.randn(features.shape[0]) * 0.01

        features = scaler.fit_transform(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(features, dtype=torch.float)

    def _extract_structural_features(self, time_series):
        """提取结构特征（简化版）（添加 nan 处理）"""
        features = []

        for i in range(time_series.shape[1]):
            ts = time_series[:, i]

            # 检查并处理nan值
            if np.isnan(ts).any():
                ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)

            # 如果整个时间序列都是0或常数
            if np.std(ts) < 1e-6:
                ts = ts + np.random.randn(len(ts)) * 0.01

            try:
                feat = [
                    np.mean(ts),
                    np.std(ts) if np.std(ts) > 0 else 1e-6,
                    np.median(ts),
                    np.ptp(ts) if np.ptp(ts) > 0 else 1e-6
                ]
            except:
                feat = [0.0, 1.0, 0.0, 1.0]

            # 检查nan
            feat = [0.0 if np.isnan(f) or np.isinf(f) else f for f in feat]
            features.append(feat)

        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        # 检查全0列
        for col in range(features.shape[1]):
            if np.std(features[:, col]) < 1e-6:
                features[:, col] = np.random.randn(features.shape[0]) * 0.01

        features = scaler.fit_transform(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(features, dtype=torch.float)

    def _matrix_to_pyg(self, matrix, features, label):
        """转换为PyG Data对象（添加 nan 处理）"""
        # 确保矩阵无nan
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        edge_index = []
        edge_attr = []

        n = matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                weight = matrix[i, j]
                # 检查权重
                if np.isnan(weight) or np.isinf(weight):
                    weight = 0.0

                if weight != 0:
                    edge_index.extend([[i, j], [j, i]])
                    edge_attr.extend([weight, weight])

        # 如果没有边，添加自环
        if len(edge_index) == 0:
            for i in range(min(10, n)):
                edge_index.extend([[i, i]])
                edge_attr.extend([1.0])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # 再次检查
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if torch.isnan(edge_attr).any():
            edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

        return Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.long)
        )

# 在ABIDEDataProcessor类中添加双流处理方法
def process_dual_stream(self):
    """添加到ABIDEDataProcessor类的方法"""
    dual_processor = ABIDEDualStream(self)
    dual_stream_data = []

    # 先加载数据
    data = self.download_data(n_subjects=None)

    print("构建双流数据...")

    # 检查不同的可能属性名
    rois_data = None
    for attr in ['rois_ho', 'rois_cc200', 'rois_cc400', 'rois_aal', 'rois_ez', 'rois_dosenbach160']:
        if hasattr(data, attr):
            rois_data = getattr(data, attr)
            if rois_data is not None:
                break

    if rois_data is None:
        print("未找到ROI数据")
        return []

    # 获取标签
    labels = data.phenotypic['DX_GROUP'].values - 1

    for idx, roi_file in enumerate(rois_data):
        try:
            if roi_file is None:
                continue

            # 处理不同格式的数据
            if isinstance(roi_file, str) and os.path.exists(roi_file):
                time_series = pd.read_csv(roi_file, sep='\t', header=0).values
            elif isinstance(roi_file, np.ndarray):
                time_series = roi_file
            else:
                continue

            if time_series.shape[0] < 50:
                continue

            label = labels[idx]

            # 构建双流
            func_data, struct_data = dual_processor.construct_dual_stream(
                time_series, label
            )
            dual_stream_data.append((func_data, struct_data))

            if (idx + 1) % 10 == 0:
                print(f"处理进度: {idx + 1}/{len(rois_data)}")

        except Exception as e:
            print(f"处理被试{idx}失败: {e}")
            continue

    print(f"成功构建{len(dual_stream_data)}个双流样本")

    # 保存
    save_path = os.path.join(self.processed_path, 'abide_dual_stream.pt')
    torch.save(dual_stream_data, save_path)
    print(f"数据已保存至: {save_path}")

    return dual_stream_data


# 为了兼容，在ABIDEDataProcessor类后添加
ABIDEDataProcessor.process_dual_stream = process_dual_stream

# 添加主函数，使文件可以直接运行
if __name__ == "__main__":
    print("=" * 60)
    print("ABIDE数据集处理")
    print("=" * 60)

    # 创建处理器
    processor = ABIDEDataProcessor(
        data_folder='./data',
        pipeline='cpac',
        atlas='ho',
        connectivity_kind='correlation',
        threshold=0.3

    )

    # 1. 下载和处理基础数据
    print("\n1. 处理基础数据...")
    graph_list = processor.process_and_save(
        n_subjects=None,  # 处理所有被试
        graph_method='correlation_matrix'
    )
    print(f"基础数据处理完成: {len(graph_list)} 个样本")

    # 2. 构建双流数据
    print("\n2. 构建双流数据...")
    dual_stream_data = processor.process_dual_stream()
    print(f"双流数据构建完成: {len(dual_stream_data)} 个样本")

    print("\n处理完成！")