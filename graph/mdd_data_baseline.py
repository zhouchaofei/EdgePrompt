"""
MDD数据集处理 - Baseline实验版本
从REST-meta-MDD数据集提取AAL-116 ROI的时间序列并计算FC矩阵

功能：
1. 加载已下载的REST-meta-MDD数据
2. 从文件名提取标签
3. 只提取AAL图谱的116个ROI（列1-116）
4. 计算Pearson FC矩阵
5. 保存为npz格式
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
    """MDD数据处理器 - Baseline版本"""

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
        print(f"MDD Baseline Processor")
        print(f"=" * 60)
        print(f"Data path: {self.mdd_path}")
        print(f"Save to: {self.baseline_path}")
        print(f"=" * 60)

    def load_roi_signals(self):
        """
        加载ROI信号数据（只提取AAL-116）

        Returns:
            timeseries_list: 时间序列列表 [N_subjects, (T, 116)]
            labels: 标签列表 (0=HC, 1=MDD)
            subject_ids: 被试ID列表
        """
        print(f"\n1. Loading ROI signals from REST-meta-MDD...")

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

        # 统计信息
        label_counts = {'MDD': 0, 'HC': 0, 'unknown': 0, 'error': 0}

        for file_idx, file_path in enumerate(mat_files):
            try:
                filename = os.path.basename(file_path)

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

                # 检查NaN和Inf
                if np.isnan(time_series).any() or np.isinf(time_series).any():
                    # 清理异常值
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

                if (file_idx + 1) % 50 == 0:
                    print(f"   Processed: {file_idx + 1}/{len(mat_files)}")

            except Exception as e:
                label_counts['error'] += 1
                if label_counts['error'] <= 5:
                    print(f"   Warning: Failed to process {filename}: {e}")
                continue

        labels = np.array(labels)

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

        return timeseries_list, labels, subject_ids

    def compute_fc_matrices(self, timeseries_list, method='pearson'):
        """
        计算功能连接矩阵

        Args:
            timeseries_list: 时间序列列表
            method: 'pearson' or 'spearman'

        Returns:
            fc_matrices: FC矩阵列表 [N_subjects, (116, 116)]
        """
        print(f"\n2. Computing FC matrices ({method} correlation)...")

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

    def save_data(self, fc_matrices, labels, subject_ids, method='pearson'):
        """
        保存数据为npz格式

        Args:
            fc_matrices: [N_subjects, 116, 116]
            labels: [N_subjects]
            subject_ids: [N_subjects]
            method: FC计算方法
        """
        print(f"\n3. Saving data...")

        # 构建文件名
        filename = f'mdd_aal116_{method}_fc.npz'
        save_path = os.path.join(self.baseline_path, filename)

        # 保存
        np.savez_compressed(
            save_path,
            fc_matrices=fc_matrices,
            labels=labels,
            subject_ids=subject_ids,
            atlas='aal',
            n_rois=116,
            method=method,
            n_subjects=len(labels)
        )

        print(f"   ✅ Saved to: {save_path}")

        # 保存元信息为文本
        meta_file = os.path.join(self.baseline_path, 'mdd_aal116_meta.txt')
        with open(meta_file, 'w') as f:
            f.write(f"REST-meta-MDD Dataset - Baseline Format\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Atlas: AAL-116\n")
            f.write(f"FC Method: {method}\n")
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

        print(f"   ✅ Meta info saved to: {meta_file}")

        return save_path

    def process_and_save(self, fc_method='pearson'):
        """
        完整的处理流程

        Args:
            fc_method: FC计算方法

        Returns:
            save_path: 保存路径
        """
        # 加载时间序列
        timeseries_list, labels, subject_ids = self.load_roi_signals()

        if len(timeseries_list) == 0:
            print("\n❌ No valid subjects found!")
            return None

        # 计算FC矩阵
        fc_matrices = self.compute_fc_matrices(timeseries_list, fc_method)

        # 保存
        save_path = self.save_data(fc_matrices, labels, subject_ids, fc_method)

        print(f"\n{'=' * 60}")
        print(f"✅ Processing completed successfully!")
        print(f"{'=' * 60}")

        return save_path


def load_mdd_baseline(data_folder='./data', method='pearson'):
    """
    加载处理好的MDD baseline数据

    Args:
        data_folder: 数据根目录
        method: FC方法

    Returns:
        fc_matrices: [N, 116, 116]
        labels: [N]
        subject_ids: [N]
        meta: 元信息字典
    """
    baseline_path = os.path.join(data_folder, 'REST-meta-MDD', 'baseline')
    filename = f'mdd_aal116_{method}_fc.npz'
    file_path = os.path.join(baseline_path, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            f"Please run: python prepare_mdd_baseline.py"
        )

    data = np.load(file_path, allow_pickle=True)

    fc_matrices = data['fc_matrices']
    labels = data['labels']
    subject_ids = data['subject_ids']

    meta = {
        'atlas': str(data['atlas']),
        'n_rois': int(data['n_rois']),
        'method': str(data['method']),
        'n_subjects': int(data['n_subjects'])
    }

    print(f"Loaded MDD baseline data:")
    print(f"  Shape: {fc_matrices.shape}")
    print(f"  Atlas: {meta['atlas']}")
    print(f"  ROIs: {meta['n_rois']}")
    print(f"  Method: {meta['method']}")

    return fc_matrices, labels, subject_ids, meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Prepare REST-meta-MDD data for baseline experiment'
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

    args = parser.parse_args()

    # 创建处理器
    processor = MDDBaselineProcessor(data_folder=args.data_folder)

    # 处理并保存
    save_path = processor.process_and_save(fc_method=args.fc_method)

    if save_path:
        print(f"\n📊 To use this data:")
        print(f"   from prepare_mdd_baseline import load_mdd_baseline")
        print(f"   fc, labels, ids, meta = load_mdd_baseline()")