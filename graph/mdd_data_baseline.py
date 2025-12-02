"""
MDD数据集处理 - 修复版（强制排序防止错位）
主要修改：在glob后添加sorted()确保文件顺序固定
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
    """MDD数据处理器 - 修复版"""

    def __init__(self, data_folder='./data'):
        self.data_folder = data_folder
        self.mdd_path = os.path.join(data_folder, 'REST-meta-MDD')
        self.baseline_path = os.path.join(self.mdd_path, 'baseline')
        os.makedirs(self.baseline_path, exist_ok=True)

        print(f"=" * 60)
        print(f"MDD Enhanced Baseline Processor (Fixed)")
        print(f"=" * 60)
        print(f"Data path: {self.mdd_path}")
        print(f"Save to: {self.baseline_path}")
        print(f"Features: ROI z-score normalization + Sorted file loading")
        print(f"=" * 60)

    def z_score_normalize_timeseries(self, timeseries):
        """Z-score标准化"""
        epsilon = 1e-8
        mean = np.mean(timeseries, axis=0, keepdims=True)
        std = np.std(timeseries, axis=0, keepdims=True) + epsilon
        normalized_ts = (timeseries - mean) / std
        normalized_ts = np.nan_to_num(normalized_ts, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized_ts

    def load_roi_signals(self, apply_zscore=True):
        """加载ROI信号数据（修复版：强制排序）"""
        print(f"\n1. Loading ROI signals from REST-meta-MDD...")

        if apply_zscore:
            print(f"   ✓ Z-score normalization will be applied")

        roi_folder = os.path.join(
            self.mdd_path,
            'Results',
            'ROISignals_FunImgARCWF'
        )

        if not os.path.exists(roi_folder):
            raise FileNotFoundError(f"ROI folder not found: {roi_folder}")

        # 🔥 关键修复：添加 sorted() 确保文件顺序固定
        mat_files = sorted(glob.glob(
            os.path.join(roi_folder, '**', '*.mat'),
            recursive=True
        ))

        print(f"   Found {len(mat_files)} .mat files")
        print(f"   ✓ Files sorted to ensure consistent ordering")

        if len(mat_files) == 0:
            raise ValueError(f"No .mat files found in {roi_folder}")

        timeseries_list = []
        labels = []
        subject_ids = []
        site_ids = []

        label_counts = {'MDD': 0, 'HC': 0, 'unknown': 0, 'error': 0}
        site_pattern = re.compile(r'S(\d+)-')

        for file_idx, file_path in enumerate(mat_files):
            try:
                filename = os.path.basename(file_path)

                # 提取站点
                site_match = site_pattern.search(filename)
                site_id = f"Site_{site_match.group(1)}" if site_match else 'Site_1'

                # 提取标签
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

                # 加载数据
                mat_data = sio.loadmat(file_path)
                if 'ROISignals' not in mat_data:
                    label_counts['error'] += 1
                    continue

                time_series = mat_data['ROISignals']

                # 只提取前116个ROI
                if time_series.shape[1] < 116:
                    label_counts['error'] += 1
                    continue

                time_series = time_series[:, :116]

                # 质量检查
                if time_series.shape[0] < 50:
                    label_counts['error'] += 1
                    continue

                # Z-score标准化
                if apply_zscore:
                    time_series = self.z_score_normalize_timeseries(time_series)
                else:
                    time_series = np.nan_to_num(time_series, nan=0.0, posinf=0.0, neginf=0.0)

                subject_id = filename.split('.')[0]

                timeseries_list.append(time_series)
                labels.append(label)
                subject_ids.append(subject_id)
                site_ids.append(site_id)

                if (file_idx + 1) % 100 == 0:
                    print(f"   Processed: {file_idx + 1}/{len(mat_files)}")

            except Exception as e:
                label_counts['error'] += 1
                if label_counts['error'] <= 5:
                    print(f"   Warning: Failed to process {filename}: {e}")
                continue

        labels = np.array(labels)
        site_ids = np.array(site_ids)

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

        # 时间序列长度信息
        ts_lengths = [ts.shape[0] for ts in timeseries_list]
        print(f"\n   Time series lengths:")
        print(f"     Min: {min(ts_lengths)}")
        print(f"     Max: {max(ts_lengths)}")
        print(f"     Mean: {np.mean(ts_lengths):.1f}")

        # 站点信息
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
        """计算功能连接矩阵"""
        print(f"\n2. Computing FC matrices ({method} correlation)...")
        print(f"   Note: Using z-score normalized time series")

        fc_matrices = []

        for i, ts in enumerate(timeseries_list):
            if method == 'pearson':
                fc = np.corrcoef(ts.T)
            elif method == 'spearman':
                fc = stats.spearmanr(ts)[0]
            else:
                raise ValueError(f"Unknown method: {method}")

            fc = np.nan_to_num(fc, nan=0.0, posinf=1.0, neginf=-1.0)
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

        return fc_matrices

    def save_data(self, fc_matrices, labels, subject_ids, site_ids, method='pearson'):
        """保存数据"""
        print(f"\n3. Saving data...")

        filename = f'mdd_aal116_{method}_fc_normalized.npz'
        save_path = os.path.join(self.baseline_path, filename)

        np.savez_compressed(
            save_path,
            fc_matrices=fc_matrices,
            labels=labels,
            subject_ids=subject_ids,
            site_ids=site_ids,
            atlas='aal',
            n_rois=116,
            method=method,
            n_subjects=len(labels),
            normalized=True
        )

        print(f"   ✅ Saved to: {save_path}")

        meta_file = os.path.join(self.baseline_path, 'mdd_aal116_meta_normalized.txt')
        with open(meta_file, 'w') as f:
            f.write(f"REST-meta-MDD Dataset - Fixed Version\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Atlas: AAL-116\n")
            f.write(f"FC Method: {method}\n")
            f.write(f"Z-score Normalized: Yes\n")
            f.write(f"File Loading: Sorted (Fixed)\n")
            f.write(f"Number of subjects: {len(labels)}\n")
            f.write(f"FC matrix shape: {fc_matrices.shape}\n")

        print(f"   ✅ Meta info saved to: {meta_file}")

        return save_path

    def process_and_save(self, fc_method='pearson', apply_zscore=True):
        """完整处理流程"""
        timeseries_list, labels, subject_ids, site_ids = self.load_roi_signals(apply_zscore)

        if len(timeseries_list) == 0:
            print("\n❌ No valid subjects found!")
            return None

        fc_matrices = self.compute_fc_matrices(timeseries_list, fc_method)
        save_path = self.save_data(fc_matrices, labels, subject_ids, site_ids, fc_method)

        print(f"\n{'=' * 60}")
        print(f"✅ Processing completed successfully!")
        print(f"{'=' * 60}")

        return save_path


def load_mdd_baseline(data_folder='./data', method='pearson', normalized=True):
    """加载处理好的MDD baseline数据"""
    baseline_path = os.path.join(data_folder, 'REST-meta-MDD', 'baseline')

    if normalized:
        filename = f'mdd_aal116_{method}_fc_normalized.npz'
    else:
        filename = f'mdd_aal116_{method}_fc.npz'

    file_path = os.path.join(baseline_path, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = np.load(file_path, allow_pickle=True)

    fc_matrices = data['fc_matrices']
    labels = data['labels']
    subject_ids = data['subject_ids']
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
    print(f"  Method: {meta['method']}")
    print(f"  Normalized: {meta['normalized']}")

    return fc_matrices, labels, subject_ids, site_ids, meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prepare MDD data (Fixed Version)')
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--fc_method', type=str, default='pearson',
                        choices=['pearson', 'spearman'])
    parser.add_argument('--no_zscore', action='store_true')

    args = parser.parse_args()

    processor = MDDBaselineProcessor(data_folder=args.data_folder)
    save_path = processor.process_and_save(
        fc_method=args.fc_method,
        apply_zscore=not args.no_zscore
    )

    if save_path:
        print(f"\n📊 To use this data:")
        print(f"   from mdd_data_baseline import load_mdd_baseline")
        print(f"   fc, labels, ids, site_ids, meta = load_mdd_baseline(normalized=True)")