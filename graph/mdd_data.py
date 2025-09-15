"""
改进的REST-meta-MDD数据集处理模块
支持处理和构建脑功能图
"""
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy import signal
import warnings
import glob

warnings.filterwarnings('ignore')


class MDDDataProcessor:
    """REST-meta-MDD数据集处理器"""

    def __init__(self, data_folder='./data',
                 connectivity_kind='correlation',
                 threshold=0.3):
        """
        初始化MDD数据处理器
        """
        self.data_folder = data_folder
        self.connectivity_kind = connectivity_kind
        self.threshold = threshold
        self.mdd_path = os.path.join(data_folder, 'REST-meta-MDD')  # 修改为正确的文件夹名
        self.processed_path = os.path.join(self.mdd_path, 'processed')

        # 创建必要的目录
        os.makedirs(self.processed_path, exist_ok=True)

    def load_roi_signals(self):
        """
        加载ROISignals数据（优先使用）
        """
        print("正在加载REST-meta-MDD ROISignals数据...")

        data_dict = {
            'time_series': [],
            'labels': [],
            'subject_ids': [],
            'file_paths': []
        }

        # ROISignals文件夹路径
        results_path = os.path.join(self.mdd_path, 'Results')
        roi_folders = [
            'ROISignals_FunImgARCWF',
            'ROISignals_FunImgARglobalCWF'
        ]

        # 为不同文件夹分配标签（0: 健康对照, 1: MDD患者）
        # 注意：这里需要根据实际的数据组织调整
        folder_labels = {
            'ROISignals_FunImgARCWF': 0,  # 假设为健康对照
            'ROISignals_FunImgARglobalCWF': 1  # 假设为MDD患者
        }

        for folder_name in roi_folders:
            folder_path = os.path.join(results_path, folder_name)

            if os.path.exists(folder_path):
                print(f"处理文件夹: {folder_path}")

                # 查找所有的ROI信号文件（可能是.txt, .csv, .mat等格式）
                roi_files = glob.glob(os.path.join(folder_path, '*'))
                print(f"找到 {len(roi_files)} 个文件")

                for file_path in roi_files:
                    try:
                        subject_id = os.path.basename(file_path).split('.')[0]

                        # 根据文件扩展名选择加载方法
                        if file_path.endswith('.txt') or file_path.endswith('.csv'):
                            # 文本文件格式
                            time_series = np.loadtxt(file_path)
                        elif file_path.endswith('.mat'):
                            # MATLAB文件格式
                            import scipy.io as sio
                            mat_data = sio.loadmat(file_path)
                            # 需要找到包含时间序列的变量名
                            for key in mat_data.keys():
                                if not key.startswith('__'):
                                    time_series = mat_data[key]
                                    break
                        elif file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
                            # NIfTI格式
                            img = nib.load(file_path)
                            time_series = img.get_fdata()
                        else:
                            continue

                        # 确保数据是2D的 (time_points, n_regions)
                        if len(time_series.shape) == 1:
                            time_series = time_series.reshape(-1, 1)
                        elif len(time_series.shape) > 2:
                            # 如果是高维数据，尝试reshape
                            time_series = time_series.reshape(time_series.shape[0], -1)

                        # 检查时间序列的有效性
                        if time_series.shape[0] < 50:  # 时间点太少
                            print(f"  跳过 {subject_id}: 时间点太少 ({time_series.shape[0]})")
                            continue

                        if time_series.shape[1] < 10:  # ROI太少
                            print(f"  跳过 {subject_id}: ROI太少 ({time_series.shape[1]})")
                            continue

                        # 标准化到合理的ROI数量（如116个AAL ROI）
                        if time_series.shape[1] > 116:
                            time_series = time_series[:, :116]

                        data_dict['time_series'].append(time_series)
                        data_dict['labels'].append(folder_labels.get(folder_name, 0))
                        data_dict['subject_ids'].append(subject_id)
                        data_dict['file_paths'].append(file_path)

                        print(f"  成功加载 {subject_id}: {time_series.shape}")

                    except Exception as e:
                        print(f"  处理文件 {os.path.basename(file_path)} 时出错: {e}")
                        continue

        print(f"\n成功加载 {len(data_dict['time_series'])} 个被试数据")
        return data_dict

    def load_phenotypic_data(self):
        """
        加载表型数据（如果存在）
        """
        phenotypic_path = os.path.join(self.mdd_path, 'phenotypic.csv')
        if os.path.exists(phenotypic_path):
            phenotypic = pd.read_csv(phenotypic_path)
            print(f"加载表型数据: {phenotypic.shape}")
            return phenotypic
        return None

    def construct_brain_graph(self, time_series, method='correlation_matrix'):
        """
        从时间序列构建脑功能连接图
        """
        n_regions = time_series.shape[1]

        if method == 'correlation_matrix':
            # 计算相关矩阵
            corr_matrix = np.corrcoef(time_series.T)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            # Fisher z变换
            with np.errstate(divide='ignore', invalid='ignore'):
                corr_matrix = np.arctanh(corr_matrix)
                corr_matrix[np.isinf(corr_matrix)] = 0
                corr_matrix[np.isnan(corr_matrix)] = 0

            # 设置对角线为0
            np.fill_diagonal(corr_matrix, 0)

            # 应用阈值
            adj_matrix = np.abs(corr_matrix)

            if self.threshold < 1.0:
                threshold_value = np.percentile(adj_matrix[adj_matrix > 0],
                                              (1 - self.threshold) * 100)
                adj_matrix[adj_matrix < threshold_value] = 0

        elif method == 'dynamic_connectivity':
            # 动态连接性
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

            adj_matrix = np.mean(dynamic_corr, axis=0)
            np.fill_diagonal(adj_matrix, 0)
            adj_matrix = np.abs(adj_matrix)

            threshold_value = np.percentile(adj_matrix[adj_matrix > 0],
                                          (1 - self.threshold) * 100)
            adj_matrix[adj_matrix < threshold_value] = 0

        elif method == 'phase_synchronization':
            # 相位同步
            from scipy.signal import hilbert

            phase_matrix = np.zeros((n_regions, n_regions))

            for i in range(n_regions):
                for j in range(i+1, n_regions):
                    signal1 = hilbert(time_series[:, i])
                    signal2 = hilbert(time_series[:, j])

                    phase1 = np.angle(signal1)
                    phase2 = np.angle(signal2)

                    phase_diff = phase1 - phase2
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

                    phase_matrix[i, j] = plv
                    phase_matrix[j, i] = plv

            adj_matrix = phase_matrix

            if self.threshold < 1.0:
                threshold_value = np.percentile(adj_matrix[adj_matrix > 0],
                                              (1 - self.threshold) * 100)
                adj_matrix[adj_matrix < threshold_value] = 0

        # 转换为PyG格式
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
        """
        features = []

        for i in range(time_series.shape[1]):
            region_series = time_series[:, i]

            # 统计特征
            feat = [
                np.mean(region_series),
                np.std(region_series),
                np.median(region_series),
                stats.skew(region_series),
                stats.kurtosis(region_series),
                np.percentile(region_series, 25),
                np.percentile(region_series, 75),
                np.max(region_series) - np.min(region_series),
            ]

            # 频域特征
            freqs, psd = signal.welch(region_series, fs=1.0,
                                    nperseg=min(len(region_series), 256))

            freq_bands = [(0.01, 0.04), (0.04, 0.08), (0.08, 0.13), (0.13, 0.30)]
            for low, high in freq_bands:
                idx = np.logical_and(freqs >= low, freqs <= high)
                if np.any(idx):
                    feat.append(np.mean(psd[idx]))
                else:
                    feat.append(0.0)

            features.append(feat)

        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 标准化
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        return torch.tensor(features, dtype=torch.float)

    def process_and_save(self, graph_method='correlation_matrix'):
        """
        处理并保存MDD数据集为PyG格式
        """
        # 尝试加载ROI信号数据
        data_dict = self.load_roi_signals()

        # 如果ROI信号数据不存在，尝试其他方法
        if len(data_dict['time_series']) == 0:
            print("未找到ROISignals数据，创建模拟数据...")
            data_dict = self.create_mock_data(n_subjects=60)

        graph_list = []

        print(f"\n正在构建脑功能图 (方法: {graph_method})...")

        for idx, time_series in enumerate(data_dict['time_series']):
            try:
                # 构建图
                edge_index, edge_attr, node_features = self.construct_brain_graph(
                    time_series, method=graph_method
                )

                # 创建PyG Data对象
                data_obj = Data(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.tensor([data_dict['labels'][idx]], dtype=torch.long)
                )

                graph_list.append(data_obj)

                if (idx + 1) % 10 == 0:
                    print(f"已处理 {idx + 1}/{len(data_dict['time_series'])} 个被试")

            except Exception as e:
                print(f"处理被试 {idx} 时出错: {e}")
                continue

        print(f"\n成功构建 {len(graph_list)} 个脑功能图")

        # 保存处理后的数据
        if graph_list:
            save_path = os.path.join(self.processed_path, f'mdd_graphs_{graph_method}.pt')
            torch.save(graph_list, save_path)
            print(f"数据已保存到: {save_path}")

            # 保存元信息
            meta_info = {
                'n_subjects': len(graph_list),
                'n_features': graph_list[0].x.shape[1] if graph_list else 0,
                'graph_method': graph_method,
                'subject_ids': data_dict['subject_ids'][:len(graph_list)]
            }

            meta_path = os.path.join(self.processed_path, f'meta_info_{graph_method}.pt')
            torch.save(meta_info, meta_path)

        return graph_list

    def create_mock_data(self, n_subjects=60):
        """
        创建模拟数据（用于测试）
        """
        print("创建模拟MDD数据用于测试...")

        data_dict = {
            'time_series': [],
            'labels': [],
            'subject_ids': [],
            'file_paths': []
        }

        for i in range(n_subjects):
            # 模拟fMRI时间序列 (time_points=150, n_regions=116)
            time_series = np.random.randn(150, 116)

            # 添加一些相关性结构
            for j in range(116):
                time_series[:, j] = np.convolve(time_series[:, j],
                                              np.ones(3)/3, mode='same')

            # 标签：0=健康对照，1=MDD患者
            label = 0 if i < n_subjects // 2 else 1

            data_dict['time_series'].append(time_series)
            data_dict['labels'].append(label)
            data_dict['subject_ids'].append(f'subj_{i:04d}')
            data_dict['file_paths'].append(f'mock_file_{i:04d}.nii.gz')

        return data_dict

    def load_processed_data(self, graph_method='correlation_matrix'):
        """
        加载已处理的数据
        """
        save_path = os.path.join(self.processed_path, f'mdd_graphs_{graph_method}.pt')

        if os.path.exists(save_path):
            print(f"加载已处理的数据: {save_path}")
            return torch.load(save_path)
        else:
            print("未找到已处理的数据，开始处理...")
            return self.process_and_save(graph_method=graph_method)


def load_mdd_data(data_folder='./data', graph_method='correlation_matrix'):
    """
    便捷函数：加载MDD数据集
    """
    processor = MDDDataProcessor(data_folder=data_folder)

    # 尝试加载已处理的数据
    graph_list = processor.load_processed_data(graph_method=graph_method)

    if graph_list:
        input_dim = graph_list[0].x.shape[1]
        output_dim = 2  # 二分类任务
    else:
        input_dim = 0
        output_dim = 2

    return graph_list, input_dim, output_dim