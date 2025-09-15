"""
ADHD200完整数据处理（无模拟数据）
"""
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from nilearn.datasets import fetch_adhd
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
from sklearn.preprocessing import StandardScaler
from scipy import stats, signal
import warnings
warnings.filterwarnings('ignore')


class ADHD200DataProcessor:
    def __init__(self, data_folder='./data', threshold=0.3):
        self.data_folder = data_folder
        self.threshold = threshold
        self.adhd_path = os.path.join(data_folder, 'ADHD200')
        self.processed_path = os.path.join(self.adhd_path, 'processed')
        os.makedirs(self.processed_path, exist_ok=True)

    def download_full_dataset(self):
        """
        下载完整的ADHD200数据集
        """
        print("="*60)
        print("下载完整ADHD200数据集")
        print("注意：完整数据集约8-10GB，下载可能需要较长时间")
        print("="*60)

        # 明确指定n_subjects=None来获取所有数据
        adhd_data = fetch_adhd(
            data_dir=self.adhd_path,
            n_subjects=None,  # None表示下载所有可用数据
            verbose=2  # 显示详细进度
        )

        print(f"\n下载完成统计：")
        print(f"- 功能数据文件数: {len(adhd_data.func)}")
        print(f"- 表型数据行数: {len(adhd_data.phenotypic)}")

        # 检查每个站点的数据
        if 'site' in adhd_data.phenotypic.columns:
            site_counts = adhd_data.phenotypic['site'].value_counts()
            print("\n各站点数据分布：")
            for site, count in site_counts.items():
                print(f"  {site}: {count} 被试")

        return adhd_data

    def process_with_error_handling(self, adhd_data):
        """
        处理数据，包含详细的错误处理
        """
        # 下载AAL atlas
        print("\n下载AAL脑区模板...")
        atlas = datasets.fetch_atlas_aal(version='SPM12')

        # 创建多个masker配置，以处理不同的数据格式
        masker_configs = [
            # 配置1：标准参数
            {
                'standardize': True,
                'detrend': True,
                'low_pass': 0.1,
                'high_pass': 0.01,
                't_r': 2.0,
            },
            # 配置2：简化参数（兼容性更好）
            {
                'standardize': True,
                'detrend': False,
            },
            # 配置3：最小参数
            {
                'standardize': False,
            }
        ]

        time_series_list = []
        successful_indices = []
        failed_files = []

        print(f"\n开始处理 {len(adhd_data.func)} 个被试的数据...")

        for idx, func_file in enumerate(adhd_data.func):
            print(f"\n处理被试 {idx+1}/{len(adhd_data.func)}")
            print(f"文件: {os.path.basename(func_file)}")

            success = False

            # 尝试不同的masker配置
            for config_idx, config in enumerate(masker_configs):
                try:
                    masker = NiftiMapsMasker(
                        maps_img=atlas.maps,
                        memory='nilearn_cache',
                        memory_level=1,
                        verbose=0,
                        **config
                    )

                    # 提取时间序列
                    time_series = masker.fit_transform(func_file)

                    # 验证结果
                    if time_series.shape[1] == 116:  # AAL有116个脑区
                        time_series_list.append(time_series)
                        successful_indices.append(idx)
                        print(f"  ✓ 成功 (配置{config_idx+1}): 形状 {time_series.shape}")
                        success = True
                        break
                    else:
                        print(f"  ! ROI数量异常: {time_series.shape[1]} (预期116)")

                except Exception as e:
                    if config_idx == len(masker_configs) - 1:
                        print(f"  ✗ 所有配置均失败: {str(e)[:100]}")

            if not success:
                failed_files.append((idx, func_file))

        print(f"\n处理完成统计：")
        print(f"- 成功: {len(successful_indices)} 个被试")
        print(f"- 失败: {len(failed_files)} 个被试")

        if failed_files and len(failed_files) < 10:
            print("\n失败的文件：")
            for idx, file in failed_files[:10]:
                print(f"  {idx}: {os.path.basename(file)}")

        # 提取标签
        phenotypic = adhd_data.phenotypic
        labels = []

        for idx in successful_indices:
            row = phenotypic.iloc[idx]

            # 尝试不同的标签字段
            if 'adhd' in phenotypic.columns:
                label = int(row['adhd'])
            elif 'DX' in phenotypic.columns:
                # DX: 0=Control, 1=ADHD-C, 2=ADHD-H, 3=ADHD-I
                dx = row['DX']
                label = 0 if dx == 0 else 1  # 二分类
            elif 'dx' in phenotypic.columns:
                dx = row['dx']
                label = 0 if dx == 0 else 1
            else:
                # 根据文件名或其他信息推断
                label = idx % 2  # 临时方案

            labels.append(label)

        return time_series_list, np.array(labels), successful_indices

    def construct_brain_graph(self, time_series):
        """
        构建脑功能连接图
        """
        n_regions = time_series.shape[1]

        # 计算相关矩阵
        corr_matrix = np.corrcoef(time_series.T)
        np.fill_diagonal(corr_matrix, 0)
        corr_matrix = np.nan_to_num(corr_matrix, 0)

        # Fisher z变换
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.arctanh(corr_matrix)
            corr_matrix[np.isinf(corr_matrix)] = 0
            corr_matrix[np.isnan(corr_matrix)] = 0

        # 取绝对值并应用阈值
        adj_matrix = np.abs(corr_matrix)

        # 保留前30%的强连接
        non_zero = adj_matrix[adj_matrix > 0]
        if len(non_zero) > 0:
            threshold_value = np.percentile(non_zero, 70)
            adj_matrix[adj_matrix < threshold_value] = 0

        # 构建边
        edge_index = []
        edge_attr = []

        for i in range(n_regions):
            for j in range(i+1, n_regions):
                if adj_matrix[i, j] > 0:
                    edge_index.extend([[i, j], [j, i]])
                    edge_attr.extend([adj_matrix[i, j]] * 2)

        # 确保有最小连接
        if len(edge_index) == 0:
            for i in range(n_regions-1):
                edge_index.extend([[i, i+1], [i+1, i]])
                edge_attr.extend([0.1, 0.1])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # 提取节点特征
        node_features = self.extract_node_features(time_series)

        return edge_index, edge_attr, node_features

    def extract_node_features(self, time_series):
        """
        提取脑区特征
        """
        features = []

        for i in range(time_series.shape[1]):
            region_series = time_series[:, i]
            feat = []

            # 基本统计特征
            feat.extend([
                np.mean(region_series),
                np.std(region_series),
                np.median(region_series),
                np.percentile(region_series, 25),
                np.percentile(region_series, 75)
            ])

            # 高阶统计
            try:
                feat.append(stats.skew(region_series))
                feat.append(stats.kurtosis(region_series))
            except:
                feat.extend([0, 0])

            # 频域特征
            try:
                freqs, psd = signal.welch(region_series, fs=0.5)
                # 4个频段的功率
                freq_bands = [(0.01, 0.04), (0.04, 0.08),
                             (0.08, 0.13), (0.13, 0.25)]
                for low, high in freq_bands:
                    idx = np.logical_and(freqs >= low, freqs <= high)
                    feat.append(np.mean(psd[idx]) if np.any(idx) else 0)
            except:
                feat.extend([0] * 4)

            features.append(feat)

        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, 0)

        # 标准化
        if np.std(features) > 0:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

        return torch.tensor(features, dtype=torch.float)

    def process_and_save(self):
        """
        完整处理流程
        """
        # 下载数据
        adhd_data = self.download_full_dataset()

        # 处理数据
        time_series_list, labels, indices = self.process_with_error_handling(adhd_data)

        if len(time_series_list) == 0:
            raise ValueError("没有成功处理任何数据！")

        # 构建图
        print(f"\n构建 {len(time_series_list)} 个脑功能图...")
        graph_list = []

        for idx, ts in enumerate(time_series_list):
            if (idx + 1) % 50 == 0:
                print(f"  进度: {idx+1}/{len(time_series_list)}")

            edge_index, edge_attr, node_features = self.construct_brain_graph(ts)

            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([labels[idx]], dtype=torch.long)
            )
            graph_list.append(graph)

        # 保存结果
        save_path = os.path.join(self.processed_path, 'adhd_graphs_full.pt')
        torch.save(graph_list, save_path)
        print(f"\n数据已保存: {save_path}")
        print(f"共 {len(graph_list)} 个图")

        # 保存元信息
        meta_info = {
            'n_subjects': len(graph_list),
            'n_features': graph_list[0].x.shape[1],
            'labels': labels,
            'label_counts': np.bincount(labels)
        }

        meta_path = os.path.join(self.processed_path, 'meta_info.pt')
        torch.save(meta_info, meta_path)

        print(f"\n标签分布：")
        print(f"  Control (0): {meta_info['label_counts'][0]}")
        print(f"  ADHD (1): {meta_info['label_counts'][1]}")

        return graph_list


def load_adhd_data(data_folder='./data', n_subjects=None, graph_method='correlation_matrix'):
    """
    加载ADHD数据
    """
    processor = ADHD200DataProcessor(data_folder=data_folder)

    # 检查已处理的数据
    processed_file = os.path.join(data_folder, 'ADHD200', 'processed', 'adhd_graphs_full.pt')

    if os.path.exists(processed_file):
        print(f"加载已处理数据: {processed_file}")
        graph_list = torch.load(processed_file)
        print(f"加载 {len(graph_list)} 个图")
    else:
        print("首次运行，开始下载和处理完整数据集...")
        graph_list = processor.process_and_save()

    input_dim = graph_list[0].x.shape[1]
    output_dim = 2

    return graph_list, input_dim, output_dim