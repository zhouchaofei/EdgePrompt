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
from scipy import stats
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