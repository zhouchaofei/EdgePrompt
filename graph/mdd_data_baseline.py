"""
MDD数据集处理 - 增强版本（带标准化）
添加了ROI z-score标准化

主要改进：
1. 在计算FC前对时间序列进行z-score标准化
2. 提高不同被试之间的可比性
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
import glob
import re
import warnings

warnings.filterwarnings('ignore')


class MDDBaselineProcessor:
    """MDD数据处理器 - 增强版本"""

    def __init__(self, data_folder='./data'):
        """
        Args:
            data_folder: 数据保存根目录
        """
        self.data_folder = data_folder
        self.mdd_path = os.path.join(data_folder, 'REST-meta-MDD')
        self.baseline_path = os.path.join(self.mdd_path, 'baseline')

        os.makedirs(self.baseline_path, exist_ok=True)

        print(f"=" * 60)
        print(f"MDD Enhanced Baseline Processor")
        print(f"=" * 60)
        print(f"Data path: {self.mdd_path}")
        print(f"Save to: {self.baseline_path}")
        print(f"Features: ROI z-score normalization")
        print(f"=" * 60)

    def z_score_normalize_timeseries(self, timeseries):
        """
        对时间序列进行z-score标准化（按ROI）

        Args:
            timeseries: [T, N_ROI] 时间序列数据

        Returns:
            normalized_ts: 标准化后的时间序列
        """
        # 对每个ROI（列）进行z-score标准化
        # 避免除零错误
        epsilon = 1e-8

        mean = np.mean(timeseries, axis=0, keepdims=True)  # [1, N_ROI]
        std = np.std(timeseries, axis=0, keepdims=True) + epsilon  # [1, N_ROI]

        normalized_ts = (timeseries - mean) / std

        # 处理可能的NaN（如果某个ROI完全恒定）
        normalized_ts = np.nan_to_num(normalized_ts, nan=0.0, posinf=0.0, neginf=0.0)

        return normalized_ts

    def load_roi_signals(self, apply_zscore=True):
        """
        加载ROI信号数据（只提取AAL-116，带z-score标准化）

        Args:
            apply_zscore: 是否应用z-score标准化

        Returns:
            timeseries_list: 时间序列列表 [N_subjects, (T, 116)]
            labels: 标签列表 (0=HC, 1=MDD)
            subject_ids: 被试ID列表
            site_ids: 站点ID列表（MDD数据集可能是单站点）
        """
        print(f"\n1. Loading ROI signals from REST-meta-MDD...")

        if apply_zscore:
            print(f"   ✓ Z-score normalization will be applied")

        # ROI数据路径
        roi_folder = os.path.join(
            self.mdd_path,
            'Results',
            'ROISignals_FunImgARCWF'
        )

        if not os.path.exists(roi_folder):
            raise FileNotFoundError(
                f"ROI folder not found: {roi_folder}\n"
                f"Please ensure REST-meta-MDD data is downloaded and extracted correctly."
            )

        # 递归查找所有.mat文件
        mat_files = glob.glob(
            os.path.join(roi_folder, '**', '*.mat'),
            recursive=True
        )

        print(f"   Found {len(mat_files)} .mat files")

        if len(mat_files) == 0:
            raise ValueError(f"No .mat files found in {roi_folder}")

        timeseries_list = []
        labels = []
        subject_ids = []
        site_ids = []  # 可能需要从文件路径推断

        # 统计信息
        label_counts = {'MDD': 0, 'HC': 0, 'unknown': 0, 'error': 0}

        # 尝试从路径推断站点信息
        site_pattern = re.compile(r'S(\d+)-')  # S1, S2等可能表示不同站点

        for file_idx, file_path in enumerate(mat_files):
            try:
                filename = os.path.basename(file_path)

                # 提取可能的站点信息
                site_match = site_pattern.search(filename)
                if site_match:
                    site_id = f"Site_{site_match.group(1)}"
                else:
                    site_id = 'Site_1'  # 默认站点

                # ============================================
                # 从文件名提取标签
                # 格式: ROISignals_S1-1-0001.mat
                # -1- = MDD患者, -2- = 健康对照
                # ============================================
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

                time_series = mat_data['ROISignals']  # [T, 1833]

                # ============================================
                # 关键：只提取AAL图谱的116个ROI（列1-116）
                # ============================================
                if time_series.shape[1] < 116:
                    label_counts['error'] += 1
                    continue

                time_series = time_series[:, :116]  # [T, 116]

                # 基本质量检查
                if time_series.shape[0] < 50:  # 时间点太少
                    label_counts['error'] += 1
                    continue

                # ======================
                # 关键步骤：Z-score标准化
                # ======================
                if apply_zscore:
                    time_series = self.z_score_normalize_timeseries(time_series)
                else:
                    # 即使不做z-score，也要清理NaN/Inf
                    time_series = np.nan_to_num(
                        time_series,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0
                    )

                # 提取被试ID
                subject_id = filename.split('.')[0]

                # 保存数据
                timeseries_list.append(time_series)
                labels.append(label)
                subject_ids.append(subject_id)
                site_ids.append(site_id)

                if (file_idx + 1) % 50 == 0:
                    print(f"   Processed: {file_idx + 1}/{len(mat_files)}")

            except Exception as e:
                label_counts['error'] += 1
                if label_counts['error'] <= 5:
                    print(f"   Warning: Failed to process {filename}: {e}")
                continue

        labels = np.array(labels)
        site_ids = np.array(site_ids)

        # 打印统计信息
        print(f"\n   Loading completed:")
        print(f"   ----------------------------------------")
        print(f"   Successfully loaded: {len(timeseries_list)} subjects")
        print(f"     - MDD patients (label=1): {label_counts['MDD']}")
        print(f"     - Healthy controls (label=0): {label_counts['HC']}")
        print(f"   Failed:")
        print(f"     - Unknown label format: {label_counts['unknown']}")
        print(f"     - Processing errors: {label_counts['error']}")

        # 验证标签平衡性
        if label_counts['MDD'] > 0 and label_counts['HC'] > 0:
            ratio = max(label_counts['MDD'], label_counts['HC']) / \
                    min(label_counts['MDD'], label_counts['HC'])
            print(f"\n   Class balance ratio: {ratio:.2f}:1", end=" ")
            if ratio > 3:
                print("(⚠️  Imbalanced)")
            else:
                print("(✅ Balanced)")

        # 打印时间序列长度信息
        ts_lengths = [ts.shape[0] for ts in timeseries_list]
        print(f"\n   Time series lengths:")
        print(f"     Min: {min(ts_lengths)}")
        print(f"     Max: {max(ts_lengths)}")
        print(f"     Mean: {np.mean(ts_lengths):.1f}")
        print(f"     Median: {np.median(ts_lengths):.1f}")

        # 打印站点信息
        unique_sites = np.unique(site_ids)
        print(f"\n   Sites detected: {len(unique_sites)}")
        for site in unique_sites:
            site_mask = site_ids == site
            site_labels = labels[site_mask]
            n_hc = np.sum(site_labels == 0)
            n_mdd = np.sum(site_labels == 1)
            print(f"     {site}: Total={len(site_labels)} (HC={n_hc}, MDD={n_mdd})")

        return timeseries_list, labels, subject_ids, site_ids

    def compute_fc_matrices(self, timeseries_list, method='pearson'):
        """
        计算功能连接矩阵（输入已经是标准化后的时间序列）

        Args:
            timeseries_list: 时间序列列表（已标准化）
            method: 'pearson' or 'spearman'

        Returns:
            fc_matrices: FC矩阵列表 [N_subjects, (116, 116)]
        """
        print(f"\n2. Computing FC matrices ({method} correlation)...")
        print(f"   Note: Using z-score normalized time series")

        fc_matrices = []

        for i, ts in enumerate(timeseries_list):
            # 计算相关矩阵
            if method == 'pearson':
                fc = np.corrcoef(ts.T)  # [T, 116] -> [116, 116]
            elif method == 'spearman':
                fc = stats.spearmanr(ts)[0]
            else:
                raise ValueError(f"Unknown method: {method}")

            # 处理NaN和Inf
            fc = np.nan_to_num(fc, nan=0.0, posinf=1.0, neginf=-1.0)

            # 对角线设为0（去掉自相关）
            np.fill_diagonal(fc, 0)

            fc_matrices.append(fc)

            if (i + 1) % 100 == 0:
                print(f"   Computed: {i + 1}/{len(timeseries_list)}")

        fc_matrices = np.array(fc_matrices)

        print(f"\n   FC matrices statistics:")
        print(f"     Shape: {fc_matrices.shape}")
        print(f"     Mean: {fc_matrices.mean():.4f}")
        print(f"     Std: {fc_matrices.std():.4f}")
        print(f"     Min: {fc_matrices.min():.4f}")
        print(f"     Max: {fc_matrices.max():.4f}")

        # 检查异常值
        extreme_values = np.sum(np.abs(fc_matrices) > 0.99)
        total_values = fc_matrices.size
        print(f"     Extreme values (>0.99): {extreme_values} "
              f"({extreme_values / total_values * 100:.2f}%)")

        return fc_matrices

    def save_data(self, fc_matrices, labels, subject_ids, site_ids, method='pearson'):
        """
        保存数据为npz格式（包含站点信息）

        Args:
            fc_matrices: [N_subjects, 116, 116]
            labels: [N_subjects]
            subject_ids: [N_subjects]
            site_ids: [N_subjects] 站点ID
            method: FC计算方法
        """
        print(f"\n3. Saving data...")

        # 构建文件名
        filename = f'mdd_aal116_{method}_fc_normalized.npz'
        save_path = os.path.join(self.baseline_path, filename)

        # 保存
        np.savez_compressed(
            save_path,
            fc_matrices=fc_matrices,
            labels=labels,
            subject_ids=subject_ids,
            site_ids=site_ids,  # 保存站点信息
            atlas='aal',
            n_rois=116,
            method=method,
            n_subjects=len(labels),
            normalized=True  # 标记已做标准化
        )

        print(f"   ✅ Saved to: {save_path}")

        # 保存元信息为文本
        meta_file = os.path.join(self.baseline_path, 'mdd_aal116_meta_normalized.txt')
        with open(meta_file, 'w') as f:
            f.write(f"REST-meta-MDD Dataset - Enhanced Baseline Format\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Atlas: AAL-116\n")
            f.write(f"FC Method: {method}\n")
            f.write(f"Z-score Normalized: Yes\n")
            f.write(f"Number of subjects: {len(labels)}\n")
            f.write(f"Number of ROIs: 116\n")
            f.write(f"FC matrix shape: {fc_matrices.shape}\n")
            f.write(f"\nLabel distribution:\n")
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                label_name = 'HC' if u == 0 else 'MDD'
                f.write(f"  {label_name} (label={u}): {c}\n")
            f.write(f"\nFC statistics:\n")
            f.write(f"  Mean: {fc_matrices.mean():.4f}\n")
            f.write(f"  Std: {fc_matrices.std():.4f}\n")
            f.write(f"  Min: {fc_matrices.min():.4f}\n")
            f.write(f"  Max: {fc_matrices.max():.4f}\n")

            # 站点信息
            unique_sites = np.unique(site_ids)
            f.write(f"\nNumber of sites: {len(unique_sites)}\n")

        print(f"   ✅ Meta info saved to: {meta_file}")

        return save_path

    def process_and_save(self, fc_method='pearson', apply_zscore=True):
        """
        完整的处理流程（带标准化）

        Args:
            fc_method: FC计算方法
            apply_zscore: 是否应用z-score标准化

        Returns:
            save_path: 保存路径
        """
        # 加载时间序列（带标准化）
        timeseries_list, labels, subject_ids, site_ids = self.load_roi_signals(apply_zscore)

        if len(timeseries_list) == 0:
            print("\n❌ No valid subjects found!")
            return None

        # 计算FC矩阵
        fc_matrices = self.compute_fc_matrices(timeseries_list, fc_method)

        # 保存（包含站点信息）
        save_path = self.save_data(fc_matrices, labels, subject_ids, site_ids, fc_method)

        print(f"\n{'=' * 60}")
        print(f"✅ Processing completed successfully!")
        print(f"{'=' * 60}")

        return save_path


def load_mdd_baseline(data_folder='./data', method='pearson', normalized=True):
    """
    加载处理好的MDD baseline数据（支持标准化版本）

    Args:
        data_folder: 数据根目录
        method: FC方法
        normalized: 是否加载标准化版本

    Returns:
        fc_matrices: [N, 116, 116]
        labels: [N]
        subject_ids: [N]
        site_ids: [N] 站点ID（可能都是同一站点）
        meta: 元信息字典
    """
    baseline_path = os.path.join(data_folder, 'REST-meta-MDD', 'baseline')

    if normalized:
        filename = f'mdd_aal116_{method}_fc_normalized.npz'
    else:
        filename = f'mdd_aal116_{method}_fc.npz'

    file_path = os.path.join(baseline_path, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            f"Please run the data preparation script"
        )

    data = np.load(file_path, allow_pickle=True)

    fc_matrices = data['fc_matrices']
    labels = data['labels']
    subject_ids = data['subject_ids']

    # 尝试加载站点信息
    site_ids = data['site_ids'] if 'site_ids' in data else None

    meta = {
        'atlas': str(data['atlas']),
        'n_rois': int(data['n_rois']),
        'method': str(data['method']),
        'n_subjects': int(data['n_subjects']),
        'normalized': data.get('normalized', False)
    }

    print(f"Loaded MDD baseline data:")
    print(f"  Shape: {fc_matrices.shape}")
    print(f"  Atlas: {meta['atlas']}")
    print(f"  ROIs: {meta['n_rois']}")
    print(f"  Method: {meta['method']}")
    print(f"  Normalized: {meta['normalized']}")

    if site_ids is not None:
        print(f"  Sites: {len(np.unique(site_ids))}")

    return fc_matrices, labels, subject_ids, site_ids, meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Prepare REST-meta-MDD data for enhanced baseline experiment'
    )
    parser.add_argument(
        '--data_folder',
        type=str,
        default='./data',
        help='Root folder for data'
    )
    parser.add_argument(
        '--fc_method',
        type=str,
        default='pearson',
        choices=['pearson', 'spearman'],
        help='FC computation method'
    )
    parser.add_argument(
        '--no_zscore',
        action='store_true',
        help='Disable z-score normalization'
    )

    args = parser.parse_args()

    # 创建处理器
    processor = MDDBaselineProcessor(data_folder=args.data_folder)

    # 处理并保存
    save_path = processor.process_and_save(
        fc_method=args.fc_method,
        apply_zscore=not args.no_zscore  # 默认应用z-score
    )

    if save_path:
        print(f"\n📊 To use this data:")
        print(f"   from mdd_data_baseline import load_mdd_baseline")
        print(f"   fc, labels, ids, site_ids, meta = load_mdd_baseline(normalized=True)")