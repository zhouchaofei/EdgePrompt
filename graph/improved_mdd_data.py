"""
改进的MDD数据处理
"""
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import scipy.io as sio
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
import glob
import re


class ImprovedMDDProcessor:
    """改进的MDD处理器"""

    def __init__(self, data_folder='./data',
                 time_strategy='adaptive',
                 min_time_length=100):
        self.data_folder = data_folder
        self.time_strategy = time_strategy
        self.min_time_length = min_time_length
        self.mdd_path = os.path.join(data_folder, 'REST-meta-MDD')
        self.processed_path = os.path.join(self.mdd_path, 'processed_v2')

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
            'file_paths': []
        }

        label_counts = {'MDD': 0, 'HC': 0, 'error': 0}

        for file_path in mat_files:
            try:
                filename = os.path.basename(file_path)

                # 从文件名提取标签
                # 格式: ROISignals_S1-1-0001.mat
                # -1- = MDD, -2- = HC
                match = re.search(r'-(\d)-', filename)
                if not match:
                    continue

                group_code = int(match.group(1))
                if group_code == 1:
                    label = 1  # MDD
                    label_counts['MDD'] += 1
                elif group_code == 2:
                    label = 0  # HC
                    label_counts['HC'] += 1
                else:
                    continue

                # 加载.mat文件
                mat_data = sio.loadmat(file_path)

                # 查找时间序列数据
                time_series = None
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        data = mat_data[key]
                        if isinstance(data, np.ndarray) and data.ndim == 2:
                            time_series = data
                            break

                if time_series is None:
                    continue

                # 确保维度正确
                if time_series.shape[1] > time_series.shape[0]:
                    time_series = time_series.T

                # 标准化到116个ROI
                if time_series.shape[1] != 116:
                    if time_series.shape[1] > 116:
                        time_series = time_series[:, :116]
                    else:
                        padding = np.zeros((
                            time_series.shape[0],
                            116 - time_series.shape[1]
                        ))
                        time_series = np.concatenate([time_series, padding], axis=1)

                data_dict['time_series'].append(time_series)
                data_dict['labels'].append(label)
                data_dict['file_paths'].append(file_path)

            except Exception as e:
                label_counts['error'] += 1
                continue

        print(f"\n加载完成:")
        print(f"  MDD: {label_counts['MDD']}")
        print(f"  HC: {label_counts['HC']}")
        print(f"  错误: {label_counts['error']}")

        return data_dict

    def determine_time_length(self, time_series_list):
        """确定时间长度"""
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
        """处理时间序列（同ABIDE）"""
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

    def compute_dynamic_connectivity_variance(self, time_series):
        """
        计算动态连接性的变异性（MDD关键特征）

        Args:
            time_series: [T, N]

        Returns:
            variance_matrix: [N, N] 连接强度的时间变异性
        """
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
            # 标准差作为变异性度量
            variance = np.std(dynamic_corrs, axis=0)
        else:
            variance = np.zeros((n_regions, n_regions))

        return variance

    def construct_dual_stream_graph(self, time_series, label):
        """
        构建MDD双流图

        MDD特点：
        - 功能流强调动态变化（情绪波动）
        - 结构流强调稳定连接
        """
        T, N = time_series.shape

        # 节点特征：时间序列
        node_features = torch.tensor(time_series.T, dtype=torch.float)

        # ==========================================
        # 结构流：稳定的强连接
        # ==========================================
        corr = np.corrcoef(time_series.T)
        corr = np.nan_to_num(corr, 0)

        # 只保留最强的连接
        threshold = np.percentile(np.abs(corr), 95)
        struct_adj = np.zeros_like(corr)
        struct_adj[np.abs(corr) > threshold] = 1
        np.fill_diagonal(struct_adj, 0)

        struct_edge_index = []
        for i in range(N):
            for j in range(N):
                if struct_adj[i, j] > 0:
                    struct_edge_index.append([i, j])

        struct_edge_index = torch.tensor(struct_edge_index, dtype=torch.long).t()

        struct_data = Data(
            x=node_features,
            edge_index=struct_edge_index,
            y=torch.tensor([label], dtype=torch.long)
        )

        # ==========================================
        # 功能流：动态变异性 + 相关
        # ==========================================
        # 计算动态变异性
        dynamic_var = self.compute_dynamic_connectivity_variance(time_series)

        # 融合静态相关和动态变异
        func_matrix = (corr + dynamic_var * 2) / 3  # 强调动态

        # 阈值化
        threshold = np.percentile(np.abs(func_matrix), 70)

        func_edge_index = []
        func_edge_attr = []

        for i in range(N):
            for j in range(N):
                if i != j and np.abs(func_matrix[i, j]) > threshold:
                    func_edge_index.append([i, j])
                    func_edge_attr.append(func_matrix[i, j])

        func_edge_index = torch.tensor(func_edge_index, dtype=torch.long).t()
        func_edge_attr = torch.tensor(func_edge_attr, dtype=torch.float)

        func_data = Data(
            x=node_features,
            edge_index=func_edge_index,
            edge_attr=func_edge_attr,
            y=torch.tensor([label], dtype=torch.long)
        )

        return func_data, struct_data

    def process_and_save(self):
        """处理并保存"""
        # 加载数据
        data_dict = self.load_roi_signals()

        if data_dict is None or len(data_dict['time_series']) == 0:
            print("数据加载失败")
            return []

        # 确定时间长度
        target_length = self.determine_time_length(data_dict['time_series'])
        print(f"目标时间长度: {target_length}")

        dual_stream_data = []

        for idx, time_series in enumerate(data_dict['time_series']):
            try:
                # 处理时间序列
                time_series = self.process_time_series(time_series, target_length)

                if time_series.shape[0] < 50:
                    continue

                label = data_dict['labels'][idx]

                # 构建双流图
                func_data, struct_data = self.construct_dual_stream_graph(
                    time_series, label
                )

                dual_stream_data.append((func_data, struct_data))

                if (idx + 1) % 20 == 0:
                    print(f"已处理: {idx + 1}/{len(data_dict['time_series'])}")

            except Exception as e:
                print(f"处理样本{idx}失败: {e}")
                continue

        print(f"\n成功构建 {len(dual_stream_data)} 个双流样本")

        # 保存
        save_path = os.path.join(
            self.processed_path,
            f'mdd_dual_stream_{self.time_strategy}_{target_length}.pt'
        )
        torch.save(dual_stream_data, save_path)
        print(f"数据已保存至: {save_path}")

        return dual_stream_data


if __name__ == "__main__":
    print("=" * 60)
    print("改进的MDD数据处理")
    print("=" * 60)

    processor = ImprovedMDDProcessor(
        data_folder='./data',
        time_strategy='adaptive',
        min_time_length=100
    )

    dual_stream = processor.process_and_save()

    if dual_stream:
        print(f"\n样本信息:")
        print(f"  节点特征维度: {dual_stream[0][0].x.shape}")
        print(f"  功能流边数: {dual_stream[0][0].edge_index.shape[1]}")
        print(f"  结构流边数: {dual_stream[0][1].edge_index.shape[1]}")