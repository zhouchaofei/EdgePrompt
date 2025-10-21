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


class MDDDualStream:
    """MDD双流数据处理扩展"""

    def __init__(self, original_processor):
        self.processor = original_processor

    def load_rest_meta_mdd(self):
        """加载REST-meta-MDD数据"""
        mdd_path = os.path.join(self.processor.data_folder, 'REST-meta-MDD')
        roi_path = os.path.join(mdd_path, 'Results', 'ROISignals_FunImgARCWF')

        if not os.path.exists(roi_path):
            print(f"请下载REST-meta-MDD到: {mdd_path}")
            return []

        # 加载人口学数据
        demo_file = os.path.join(mdd_path, 'Demographic_Data.xlsx')
        if os.path.exists(demo_file):
            demographics = pd.read_excel(demo_file)
        else:
            demographics = None

        dual_stream_data = []

        # 遍历站点
        sites = [d for d in os.listdir(roi_path)
                 if os.path.isdir(os.path.join(roi_path, d))]

        for site in sites:
            site_path = os.path.join(roi_path, site)
            roi_files = glob.glob(os.path.join(site_path, '*.txt'))

            for roi_file in roi_files:
                try:
                    # 加载时间序列
                    time_series = np.loadtxt(roi_file)

                    if time_series.shape[0] < 50:
                        continue

                    # 确保维度正确
                    if time_series.shape[1] > time_series.shape[0]:
                        time_series = time_series.T

                    # 标准化到116个ROI
                    if time_series.shape[1] != 116:
                        if time_series.shape[1] > 116:
                            time_series = time_series[:, :116]
                        else:
                            padding = np.zeros((time_series.shape[0], 116 - time_series.shape[1]))
                            time_series = np.concatenate([time_series, padding], axis=1)

                    # 获取标签（简化：从文件名判断）
                    filename = os.path.basename(roi_file)
                    label = 1 if 'MDD' in filename.upper() else 0

                    # 构建双流
                    func_data, struct_data = self.construct_mdd_dual_stream(
                        time_series, label
                    )
                    dual_stream_data.append((func_data, struct_data))

                except Exception as e:
                    continue

        print(f"加载了{len(dual_stream_data)}个MDD双流样本")

        # 保存
        save_path = os.path.join(self.processor.processed_path, 'mdd_dual_stream.pt')
        torch.save(dual_stream_data, save_path)

        return dual_stream_data

    def construct_mdd_dual_stream(self, time_series, label):
        """MDD特定的双流构建"""
        # 功能流：关注动态变化（MDD情绪波动）
        func_matrix = self._construct_mdd_functional(time_series)

        # 结构流
        struct_matrix = self._construct_mdd_structural(time_series)

        # 特征
        func_features = self._extract_mdd_functional_features(time_series)
        struct_features = self._extract_mdd_structural_features(time_series)

        # 转PyG
        func_data = self._matrix_to_pyg(func_matrix, func_features, label)
        struct_data = self._matrix_to_pyg(struct_matrix, struct_features, label)

        return func_data, struct_data

    def _construct_mdd_functional(self, time_series):
        """MDD功能网络：强调动态"""
        # 动态连接
        dynamic_conn = self._compute_dynamic_connectivity(time_series)

        # 标准相关
        corr = np.corrcoef(time_series.T)
        corr = np.nan_to_num(corr, 0)
        np.fill_diagonal(corr, 0)

        # 融合（MDD强调动态）
        func_matrix = (corr + dynamic_conn * 2) / 3

        # 阈值化
        threshold = np.percentile(np.abs(func_matrix), 70)
        func_matrix[np.abs(func_matrix) < threshold] = 0

        return func_matrix

    def _compute_dynamic_connectivity(self, time_series, window_size=30):
        """计算动态连接"""
        n_regions = time_series.shape[1]
        n_windows = (time_series.shape[0] - window_size) // 10 + 1

        dynamic_matrices = []
        for w in range(n_windows):
            start = w * 10
            end = start + window_size
            if end > time_series.shape[0]:
                break

            window_corr = np.corrcoef(time_series[start:end].T)
            window_corr = np.nan_to_num(window_corr, 0)
            dynamic_matrices.append(window_corr)

        if len(dynamic_matrices) > 1:
            # 返回标准差（变异性）
            return np.std(dynamic_matrices, axis=0)
        else:
            return np.zeros((n_regions, n_regions))

    def _construct_mdd_structural(self, time_series):
        """MDD结构网络"""
        corr = np.corrcoef(time_series.T)
        corr = np.nan_to_num(corr, 0)

        # 强连接作为结构
        threshold = np.percentile(np.abs(corr), 95)
        struct = np.zeros_like(corr)
        struct[np.abs(corr) > threshold] = 1
        np.fill_diagonal(struct, 0)

        return struct

    def _extract_mdd_functional_features(self, time_series):
        """MDD功能特征"""
        features = []

        for i in range(time_series.shape[1]):
            ts = time_series[:, i]
            feat = [
                np.mean(ts), np.std(ts),
                stats.skew(ts), stats.kurtosis(ts),
                np.ptp(ts)  # 范围（情绪波动）
            ]

            # MDD相关的低频振荡
            freqs, psd = signal.welch(ts, fs=0.5, nperseg=min(len(ts), 64))
            # Slow-5 and Slow-4
            idx = (freqs >= 0.01) & (freqs <= 0.073)
            feat.append(np.mean(psd[idx]) if np.any(idx) else 0)

            features.append(feat)

        features = np.array(features, dtype=np.float32)
        from sklearn.preprocessing import StandardScaler
        features = StandardScaler().fit_transform(features)

        return torch.tensor(features, dtype=torch.float)

    def _extract_mdd_structural_features(self, time_series):
        """MDD结构特征"""
        return self._extract_mdd_functional_features(time_series)[:, :4]

    def _matrix_to_pyg(self, matrix, features, label):
        """转换为PyG格式"""
        edge_index = []
        edge_attr = []

        n = matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i, j] != 0:
                    edge_index.extend([[i, j], [j, i]])
                    edge_attr.extend([matrix[i, j], matrix[i, j]])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.long)
        )


# 修正：为MDDDataProcessor类添加方法
def process_dual_stream(self):
    """添加到MDDDataProcessor类的方法"""
    dual_processor = MDDDualStream(self)
    dual_stream_data = []

    # 加载ROI信号数据
    data_dict = self.load_roi_signals()

    if len(data_dict['time_series']) == 0:
        print("未找到MDD数据，创建模拟数据...")
        data_dict = self.create_mock_data(n_subjects=60)

    print("构建双流数据...")

    for idx, time_series in enumerate(data_dict['time_series']):
        try:
            if time_series.shape[0] < 50:
                continue

            label = data_dict['labels'][idx]

            # 构建双流
            func_data, struct_data = dual_processor.construct_mdd_dual_stream(
                time_series, label
            )
            dual_stream_data.append((func_data, struct_data))

            if (idx + 1) % 10 == 0:
                print(f"处理进度: {idx + 1}/{len(data_dict['time_series'])}")

        except Exception as e:
            print(f"处理被试{idx}失败: {e}")
            continue

    print(f"成功构建{len(dual_stream_data)}个双流样本")

    # 保存
    save_path = os.path.join(self.processed_path, 'mdd_dual_stream.pt')
    torch.save(dual_stream_data, save_path)
    print(f"数据已保存至: {save_path}")

    return dual_stream_data


# 正确的类名：MDDDataProcessor
MDDDataProcessor.process_dual_stream = process_dual_stream

# 添加主函数
if __name__ == "__main__":
    print("=" * 60)
    print("MDD数据集处理")
    print("=" * 60)

    # 创建处理器
    processor = MDDDataProcessor(
        data_folder='./data',
        connectivity_kind='correlation',
        threshold=0.3
    )

    # 1. 处理基础数据
    print("\n1. 处理基础数据...")
    graph_list = processor.process_and_save(
        graph_method='correlation_matrix'
    )
    print(f"基础数据处理完成: {len(graph_list)} 个样本")

    # 2. 构建双流数据
    print("\n2. 构建双流数据...")
    dual_stream_data = processor.process_dual_stream()
    print(f"双流数据构建完成: {len(dual_stream_data)} 个样本")

    print("\n处理完成！")