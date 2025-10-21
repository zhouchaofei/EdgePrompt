"""
ABIDE数据集处理 - Baseline实验版本
直接生成FC矩阵，用于传统机器学习分类

功能：
1. 下载ABIDE数据（使用nilearn）
2. 提取时间序列（不统一长度）
3. 计算Pearson FC矩阵
4. 保存为npz格式
"""

import os
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_abide_pcp
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class ABIDEBaselineProcessor:
    """ABIDE数据处理器 - Baseline版本"""

    def __init__(self, data_folder='./data', pipeline='cpac', atlas='aal'):
        """
        Args:
            data_folder: 数据保存根目录
            pipeline: 预处理管道 (cpac/ccs/niak/dparsf)
            atlas: 脑图谱 (aal/ho/cc200/cc400)
        """
        self.data_folder = data_folder
        self.pipeline = pipeline
        self.atlas = atlas

        self.abide_path = os.path.join(data_folder, 'ABIDE')
        self.baseline_path = os.path.join(self.abide_path, 'baseline')

        os.makedirs(self.baseline_path, exist_ok=True)

        print(f"=" * 60)
        print(f"ABIDE Baseline Processor")
        print(f"=" * 60)
        print(f"Pipeline: {pipeline}")
        print(f"Atlas: {atlas}")
        print(f"Save to: {self.baseline_path}")
        print(f"=" * 60)

    def download_and_extract(self, n_subjects=None):
        """
        下载ABIDE数据并提取时间序列

        Args:
            n_subjects: 下载的被试数量（None表示全部）

        Returns:
            timeseries_list: 时间序列列表
            labels: 标签列表 (0=HC, 1=ASD)
            subject_ids: 被试ID列表
        """
        print(f"\n1. Downloading ABIDE data...")
        print(f"   This may take a while on first run...")

        # 确定derivatives参数
        if self.atlas == 'aal':
            derivatives = 'rois_aal'
            print(f"   Using AAL-116 atlas")
        elif self.atlas == 'ho':
            derivatives = 'rois_ho'
            print(f"   Using Harvard-Oxford atlas")
        elif self.atlas == 'cc200':
            derivatives = 'rois_cc200'
            print(f"   Using CC200 atlas")
        elif self.atlas == 'cc400':
            derivatives = 'rois_cc400'
            print(f"   Using CC400 atlas")
        else:
            derivatives = 'rois_aal'
            print(f"   Unknown atlas, using AAL-116")

        # 下载数据
        data = fetch_abide_pcp(
            data_dir=self.abide_path,
            pipeline=self.pipeline,
            band_pass_filtering=True,
            global_signal_regression=False,
            derivatives=[derivatives],
            n_subjects=n_subjects,
            quality_checked=True,  # 只下载质检通过的数据
            verbose=1
        )

        print(f"\n   Downloaded {len(data.phenotypic)} subjects")

        # 提取ROI时间序列
        print(f"\n2. Extracting time series...")

        timeseries_list = []
        labels = []
        subject_ids = []

        # 获取ROI数据
        rois_data = getattr(data, derivatives, None)

        if rois_data is None:
            raise ValueError(f"No ROI data found for atlas: {self.atlas}")

        # 获取标签 (DX_GROUP: 1=ASD, 2=Control)
        phenotypic = data.phenotypic
        dx_labels = phenotypic['DX_GROUP'].values

        valid_count = 0
        invalid_count = 0

        for idx, roi_file in enumerate(rois_data):
            try:
                # 加载时间序列
                if isinstance(roi_file, str) and os.path.exists(roi_file):
                    ts = pd.read_csv(roi_file, sep='\t', header=0).values
                elif isinstance(roi_file, np.ndarray):
                    ts = roi_file
                else:
                    invalid_count += 1
                    continue

                # 基本质量检查
                if ts.shape[0] < 50:  # 时间点太少
                    invalid_count += 1
                    continue

                if np.isnan(ts).any() or np.isinf(ts).any():
                    # 尝试清理
                    ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)

                # 提取标签和ID
                label = dx_labels[idx] - 1  # 转为0/1 (0=HC, 1=ASD)
                subject_id = phenotypic.iloc[idx]['SUB_ID']

                # 保存
                timeseries_list.append(ts)
                labels.append(label)
                subject_ids.append(subject_id)

                valid_count += 1

                if valid_count % 50 == 0:
                    print(f"   Processed: {valid_count}/{len(rois_data)}")

            except Exception as e:
                invalid_count += 1
                if invalid_count <= 5:
                    print(f"   Warning: Failed to process subject {idx}: {e}")
                continue

        labels = np.array(labels)

        print(f"\n   Successfully loaded: {valid_count} subjects")
        print(f"   Failed: {invalid_count} subjects")

        # 打印标签分布
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n   Label distribution:")
        for u, c in zip(unique, counts):
            label_name = 'HC' if u == 0 else 'ASD'
            print(f"     {label_name} (label={u}): {c} subjects")

        return timeseries_list, labels, subject_ids

    def compute_fc_matrices(self, timeseries_list, method='pearson'):
        """
        计算功能连接矩阵

        Args:
            timeseries_list: 时间序列列表
            method: 'pearson' or 'spearman'

        Returns:
            fc_matrices: FC矩阵列表 [N_subjects, (N_ROI, N_ROI)]
        """
        print(f"\n3. Computing FC matrices ({method} correlation)...")

        fc_matrices = []

        for i, ts in enumerate(timeseries_list):
            # 计算相关矩阵
            if method == 'pearson':
                fc = np.corrcoef(ts.T)  # [T, N_ROI] -> [N_ROI, N_ROI]
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

        print(f"   FC matrices shape: {fc_matrices.shape}")
        print(f"   FC statistics:")
        print(f"     Mean: {fc_matrices.mean():.4f}")
        print(f"     Std: {fc_matrices.std():.4f}")
        print(f"     Min: {fc_matrices.min():.4f}")
        print(f"     Max: {fc_matrices.max():.4f}")

        return fc_matrices

    def save_data(self, fc_matrices, labels, subject_ids, method='pearson'):
        """
        保存数据为npz格式

        Args:
            fc_matrices: [N_subjects, N_ROI, N_ROI]
            labels: [N_subjects]
            subject_ids: [N_subjects]
            method: FC计算方法
        """
        print(f"\n4. Saving data...")

        # 构建文件名
        filename = f'abide_{self.atlas}_{method}_fc.npz'
        save_path = os.path.join(self.baseline_path, filename)

        # 保存
        np.savez_compressed(
            save_path,
            fc_matrices=fc_matrices,
            labels=labels,
            subject_ids=subject_ids,
            atlas=self.atlas,
            method=method,
            n_subjects=len(labels),
            n_rois=fc_matrices.shape[1]
        )

        print(f"   Saved to: {save_path}")

        # 保存元信息为文本
        meta_file = os.path.join(self.baseline_path, f'abide_{self.atlas}_meta.txt')
        with open(meta_file, 'w') as f:
            f.write(f"ABIDE Dataset - Baseline Format\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Atlas: {self.atlas}\n")
            f.write(f"Pipeline: {self.pipeline}\n")
            f.write(f"FC Method: {method}\n")
            f.write(f"Number of subjects: {len(labels)}\n")
            f.write(f"Number of ROIs: {fc_matrices.shape[1]}\n")
            f.write(f"FC matrix shape: {fc_matrices.shape}\n")
            f.write(f"\nLabel distribution:\n")
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                label_name = 'HC' if u == 0 else 'ASD'
                f.write(f"  {label_name} (label={u}): {c}\n")

        print(f"   Meta info saved to: {meta_file}")

        return save_path

    def process_and_save(self, n_subjects=None, fc_method='pearson'):
        """
        完整的处理流程

        Args:
            n_subjects: 被试数量
            fc_method: FC计算方法

        Returns:
            save_path: 保存路径
        """
        # 下载和提取时间序列
        timeseries_list, labels, subject_ids = self.download_and_extract(n_subjects)

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


def load_abide_baseline(data_folder='./data', atlas='aal', method='pearson'):
    """
    加载处理好的ABIDE baseline数据

    Args:
        data_folder: 数据根目录
        atlas: 脑图谱
        method: FC方法

    Returns:
        fc_matrices: [N, N_ROI, N_ROI]
        labels: [N]
        subject_ids: [N]
        meta: 元信息字典
    """
    baseline_path = os.path.join(data_folder, 'ABIDE', 'baseline')
    filename = f'abide_{atlas}_{method}_fc.npz'
    file_path = os.path.join(baseline_path, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}\n"
                                f"Please run: python prepare_abide_baseline.py")

    data = np.load(file_path, allow_pickle=True)

    fc_matrices = data['fc_matrices']
    labels = data['labels']
    subject_ids = data['subject_ids']

    meta = {
        'atlas': str(data['atlas']),
        'method': str(data['method']),
        'n_subjects': int(data['n_subjects']),
        'n_rois': int(data['n_rois'])
    }

    print(f"Loaded ABIDE baseline data:")
    print(f"  Shape: {fc_matrices.shape}")
    print(f"  Atlas: {meta['atlas']}")
    print(f"  Method: {meta['method']}")

    return fc_matrices, labels, subject_ids, meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prepare ABIDE data for baseline experiment')
    parser.add_argument('--data_folder', type=str, default='./data',
                        help='Root folder for data')
    parser.add_argument('--pipeline', type=str, default='cpac',
                        choices=['cpac', 'ccs', 'niak', 'dparsf'],
                        help='Preprocessing pipeline')
    parser.add_argument('--atlas', type=str, default='aal',
                        choices=['aal', 'ho', 'cc200', 'cc400'],
                        help='Brain atlas')
    parser.add_argument('--n_subjects', type=int, default=None,
                        help='Number of subjects to download (None=all)')
    parser.add_argument('--fc_method', type=str, default='pearson',
                        choices=['pearson', 'spearman'],
                        help='FC computation method')

    args = parser.parse_args()

    # 创建处理器
    processor = ABIDEBaselineProcessor(
        data_folder=args.data_folder,
        pipeline=args.pipeline,
        atlas=args.atlas
    )

    # 处理并保存
    save_path = processor.process_and_save(
        n_subjects=args.n_subjects,
        fc_method=args.fc_method
    )

    if save_path:
        print(f"\n📊 To use this data:")
        print(f"   from prepare_abide_baseline import load_abide_baseline")
        print(f"   fc, labels, ids, meta = load_abide_baseline()")