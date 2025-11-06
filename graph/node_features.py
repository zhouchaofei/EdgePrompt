"""
节点特征提取模块（完整版）
支持：
1. 统计特征：mean/std/skew/kurtosis/bandpower
2. 时序编码：PCA / 1D-CNN / Transformer
"""

import numpy as np
from scipy import stats, signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


# ==================== 统计特征提取器 ====================
class StatisticalFeatureExtractor:
    """
    统计特征提取器
    提取每个ROI的统计特征，形成固定维度的向量（12维）
    """

    def __init__(self, fs=None):
        """
        Args:
            fs: 采样频率（Hz），用于计算频带功率，默认0.5（TR=2s）
        """
        self.fs = fs if fs is not None else 0.5

    def extract_features(self, timeseries):
        """
        从单个被试的时间序列提取统计特征

        Args:
            timeseries: [T, N_ROI] 时间序列

        Returns:
            features: [N_ROI, 12] 统计特征矩阵
        """
        n_timepoints, n_rois = timeseries.shape

        features_list = []

        for roi_idx in range(n_rois):
            roi_signal = timeseries[:, roi_idx]

            # 1. 基础统计量 (4维)
            mean_val = np.mean(roi_signal)
            std_val = np.std(roi_signal)
            skew_val = stats.skew(roi_signal)
            kurt_val = stats.kurtosis(roi_signal)

            # 2. 频带功率 (3维)
            # Slow-5: 0.01-0.027 Hz, Slow-4: 0.027-0.073 Hz, Slow-3: 0.073-0.198 Hz
            freqs, psd = signal.welch(roi_signal, fs=self.fs, nperseg=min(256, n_timepoints))

            slow5_idx = np.where((freqs >= 0.01) & (freqs < 0.027))[0]
            slow5_power = np.trapz(psd[slow5_idx], freqs[slow5_idx]) if len(slow5_idx) > 0 else 0

            slow4_idx = np.where((freqs >= 0.027) & (freqs < 0.073))[0]
            slow4_power = np.trapz(psd[slow4_idx], freqs[slow4_idx]) if len(slow4_idx) > 0 else 0

            slow3_idx = np.where((freqs >= 0.073) & (freqs < 0.198))[0]
            slow3_power = np.trapz(psd[slow3_idx], freqs[slow3_idx]) if len(slow3_idx) > 0 else 0

            # 3. 振幅包络 (2维)
            analytic_signal = signal.hilbert(roi_signal)
            amplitude_envelope = np.abs(analytic_signal)
            envelope_mean = np.mean(amplitude_envelope)
            envelope_std = np.std(amplitude_envelope)

            # 4. 局部变异性 (2维)
            local_diff = np.diff(roi_signal)
            local_var = np.var(local_diff)
            local_range = np.max(roi_signal) - np.min(roi_signal)

            # 5. Hurst指数（长程相关性）(1维)
            hurst = self._compute_hurst(roi_signal)

            # 合并所有特征 (4+3+2+2+1 = 12维)
            roi_features = [
                mean_val, std_val, skew_val, kurt_val,
                slow5_power, slow4_power, slow3_power,
                envelope_mean, envelope_std,
                local_var, local_range,
                hurst
            ]

            features_list.append(roi_features)

        features = np.array(features_list)  # [N_ROI, 12]

        # 处理NaN和Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        return features

    def _compute_hurst(self, signal, max_lag=20):
        """计算Hurst指数（简化版R/S分析）"""
        try:
            lags = range(2, max_lag)
            tau = []

            for lag in lags:
                n_segments = len(signal) // lag
                if n_segments == 0:
                    continue
                segments = signal[:n_segments * lag].reshape(n_segments, lag)

                rs_values = []
                for seg in segments:
                    mean_seg = np.mean(seg)
                    deviations = seg - mean_seg
                    cumsum_dev = np.cumsum(deviations)

                    R = np.max(cumsum_dev) - np.min(cumsum_dev)
                    S = np.std(seg) + 1e-10

                    rs_values.append(R / S)

                tau.append(np.mean(rs_values))

            tau = np.array(tau)
            lags = np.array(list(lags[:len(tau)]))

            valid_idx = (tau > 0) & (lags > 0)
            if np.sum(valid_idx) < 3:
                return 0.5

            log_lags = np.log(lags[valid_idx])
            log_tau = np.log(tau[valid_idx])

            hurst = np.polyfit(log_lags, log_tau, 1)[0]
            hurst = np.clip(hurst, 0, 1)

            return hurst

        except:
            return 0.5


# ==================== 时序编码提取器 ====================
class TemporalEncodingExtractor:
    """
    时序编码特征提取器
    支持三种方法：PCA、1D-CNN、Transformer
    """

    def __init__(self, method='pca', output_dim=64):
        """
        Args:
            method: 'pca' / 'cnn' / 'transformer'
            output_dim: 输出特征维度
        """
        self.method = method
        self.output_dim = output_dim
        self.T_fixed = None  # 固定时间长度（仅PCA需要）

        if method == 'pca':
            self.encoder = None
            self.scaler = None
        elif method == 'cnn':
            self.encoder = self._build_cnn_encoder()
        elif method == 'transformer':
            self.encoder = self._build_transformer_encoder()
        else:
            raise ValueError(f"Unknown method: {method}. Choose from ['pca', 'cnn', 'transformer']")

    def _build_cnn_encoder(self):
        """
        构建1D-CNN编码器
        使用自适应池化，可以处理任意长度的时间序列
        """
        class CNN1DEncoder(nn.Module):
            def __init__(self, output_dim):
                super().__init__()

                self.conv_layers = nn.Sequential(
                    # 第一层：捕捉局部时间模式
                    nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(),
                    nn.BatchNorm1d(32),
                    nn.MaxPool1d(2),

                    # 第二层：更高层次的时间特征
                    nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.MaxPool1d(2),

                    # 第三层：全局特征
                    nn.Conv1d(64, output_dim, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1)  # 自适应池化到长度1
                )

            def forward(self, x):
                # x: [batch, 1, time] - 任意time长度
                x = self.conv_layers(x)  # [batch, output_dim, 1]
                x = x.squeeze(-1)         # [batch, output_dim]
                return x

        return CNN1DEncoder(self.output_dim)

    def _build_transformer_encoder(self):
        """
        构建简单的Transformer编码器
        使用位置编码和注意力机制捕捉长程依赖
        """
        class SimpleTransformerEncoder(nn.Module):
            def __init__(self, output_dim, d_model=64, nhead=4, num_layers=2):
                super().__init__()

                # 输入投影
                self.input_proj = nn.Linear(1, d_model)

                # Transformer编码器
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                # 输出投影
                self.output_proj = nn.Sequential(
                    nn.Linear(d_model, output_dim),
                    nn.ReLU()
                )

            def forward(self, x):
                # x: [batch, time, 1] - 任意time长度
                x = self.input_proj(x)  # [batch, time, d_model]
                x = self.transformer(x)  # [batch, time, d_model]
                x = x.mean(dim=1)        # [batch, d_model] - 平均池化
                x = self.output_proj(x)  # [batch, output_dim]
                return x

        return SimpleTransformerEncoder(self.output_dim)

    def fit(self, timeseries_list):
        """
        拟合编码器（仅PCA需要）

        Args:
            timeseries_list: 时间序列列表，每个形状为 [T_i, N_ROI]
        """
        if self.method != 'pca':
            print(f"  {self.method.upper()} encoder does not require fitting.")
            return

        print(f"  Fitting PCA encoder (target_dim={self.output_dim})...")

        # 1. 确定固定时间长度（取所有被试的最短长度）
        print(f"    Checking time series lengths for all {len(timeseries_list)} subjects...")
        lengths = [ts.shape[0] for ts in timeseries_list]
        self.T_fixed = min(lengths)

        print(f"    Detected min length across all subjects: T={self.T_fixed}")
        print(f"    Max length: {max(lengths)}, Mean: {np.mean(lengths):.1f}")

        # 2. 收集所有ROI的裁剪后信号
        all_signals = []

        for ts in timeseries_list:
            # 裁剪到固定长度
            ts_crop = ts[:self.T_fixed, :]

            # 验证裁剪后的形状
            if ts_crop.shape[0] != self.T_fixed:
                print(
                    f"    Warning: Skipping subject with insufficient time points ({ts_crop.shape[0]} < {self.T_fixed})")
                continue

            ts_crop = np.nan_to_num(ts_crop, nan=0.0, posinf=0.0, neginf=0.0)

            # 每个ROI作为一个样本
            for roi_idx in range(ts_crop.shape[1]):
                all_signals.append(ts_crop[:, roi_idx])

        # 3. 验证所有信号长度一致
        signal_lengths = [sig.shape[0] for sig in all_signals]
        if len(set(signal_lengths)) > 1:
            print(f"    Error: Signal lengths are not consistent: {set(signal_lengths)}")
            raise ValueError("Cannot stack signals with different lengths")

        all_signals = np.stack(all_signals, axis=0)  # [N_samples, T_fixed]

        print(f"    Total training samples: {all_signals.shape[0]}")

        # 4. 子采样（避免内存压力）
        max_samples = 50000
        if all_signals.shape[0] > max_samples:
            print(f"    Subsampling to {max_samples} samples for PCA fitting...")
            idx = np.random.choice(all_signals.shape[0], size=max_samples, replace=False)
            all_signals = all_signals[idx]

        # 5. 标准化
        self.scaler = StandardScaler()
        all_signals_scaled = self.scaler.fit_transform(all_signals)

        # 6. PCA降维（维度不超过T_fixed）
        n_components = min(self.output_dim, self.T_fixed)
        self.encoder = PCA(n_components=n_components)
        self.encoder.fit(all_signals_scaled)

        explained_var = np.sum(self.encoder.explained_variance_ratio_)
        self.output_dim = n_components  # 更新为实际维度

        print(f"    PCA fitted: n_components={n_components}, explained_variance={explained_var:.2%}")

    def extract_features(self, timeseries):
        """
        从单个被试的时间序列提取编码特征

        Args:
            timeseries: [T, N_ROI] 时间序列

        Returns:
            features: [N_ROI, output_dim] 编码特征矩阵
        """
        if self.method == 'pca':
            return self._extract_pca_features(timeseries)
        elif self.method == 'cnn':
            return self._extract_cnn_features(timeseries)
        elif self.method == 'transformer':
            return self._extract_transformer_features(timeseries)

    def _extract_pca_features(self, timeseries):
        """使用PCA提取特征"""
        if self.encoder is None or self.scaler is None or self.T_fixed is None:
            raise RuntimeError("PCA encoder not fitted. Call fit() first.")

        n_timepoints, n_rois = timeseries.shape

        # 检查时间序列长度是否足够
        if n_timepoints < self.T_fixed:
            # 如果不够，进行零填充
            print(f"    Warning: Time series too short ({n_timepoints} < {self.T_fixed}), padding with zeros")
            padding = np.zeros((self.T_fixed - n_timepoints, n_rois))
            ts_crop = np.vstack([timeseries, padding])
        else:
            # 裁剪到固定长度
            ts_crop = timeseries[:self.T_fixed, :]

        ts_crop = np.nan_to_num(ts_crop, nan=0.0, posinf=0.0, neginf=0.0)

        features_list = []

        for roi_idx in range(n_rois):
            roi_signal = ts_crop[:, roi_idx].reshape(1, -1)  # [1, T_fixed]

            # 标准化 + PCA变换
            roi_signal_scaled = self.scaler.transform(roi_signal)
            encoded = self.encoder.transform(roi_signal_scaled)  # [1, output_dim]

            features_list.append(encoded[0])

        features = np.array(features_list)  # [N_ROI, output_dim]

        return features

    def _extract_cnn_features(self, timeseries):
        """使用1D-CNN提取特征（支持任意长度）"""
        n_timepoints, n_rois = timeseries.shape

        self.encoder.eval()

        features_list = []

        with torch.no_grad():
            for roi_idx in range(n_rois):
                roi_signal = timeseries[:, roi_idx]

                # 转为tensor [1, 1, T]
                roi_tensor = torch.FloatTensor(roi_signal).unsqueeze(0).unsqueeze(0)

                # 编码
                encoded = self.encoder(roi_tensor)  # [1, output_dim]

                features_list.append(encoded[0].numpy())

        features = np.array(features_list)  # [N_ROI, output_dim]

        return features

    def _extract_transformer_features(self, timeseries):
        """使用Transformer提取特征（支持任意长度）"""
        n_timepoints, n_rois = timeseries.shape

        self.encoder.eval()

        features_list = []

        with torch.no_grad():
            for roi_idx in range(n_rois):
                roi_signal = timeseries[:, roi_idx]

                # 转为tensor [1, T, 1]
                roi_tensor = torch.FloatTensor(roi_signal).unsqueeze(0).unsqueeze(-1)

                # 编码
                encoded = self.encoder(roi_tensor)  # [1, output_dim]

                features_list.append(encoded[0].numpy())

        features = np.array(features_list)  # [N_ROI, output_dim]

        return features


# ==================== 批量提取函数 ====================
def extract_node_features_batch(
    timeseries_list,
    method='statistical',
    temporal_method='pca',
    fs=None,
    output_dim=64
):
    """
    批量提取节点特征

    Args:
        timeseries_list: 时间序列列表
        method: 'statistical' 或 'temporal'
        temporal_method: 'pca' / 'cnn' / 'transformer' (仅当method='temporal'时有效)
        fs: 采样频率
        output_dim: 输出维度（仅对temporal有效）

    Returns:
        node_features_list: 节点特征列表 [N_subjects, (N_ROI, feature_dim)]
        feature_dim: 特征维度
    """
    print(f"\n{'='*60}")
    print(f"Extracting node features...")
    print(f"  Method: {method}")
    if method == 'temporal':
        print(f"  Temporal encoding: {temporal_method}")
        print(f"  Target dimension: {output_dim}")
    print(f"{'='*60}")

    if method == 'statistical':
        # 统计特征提取
        extractor = StatisticalFeatureExtractor(fs=fs)

        node_features_list = []
        for i, ts in enumerate(timeseries_list):
            features = extractor.extract_features(ts)
            node_features_list.append(features)

            if (i + 1) % 100 == 0:
                print(f"  Processed: {i + 1}/{len(timeseries_list)}")

        feature_dim = node_features_list[0].shape[1]

    elif method == 'temporal':
        # 时序编码特征提取
        extractor = TemporalEncodingExtractor(method=temporal_method, output_dim=output_dim)

        # 先拟合编码器（仅PCA需要）
        extractor.fit(timeseries_list)

        # 提取特征
        node_features_list = []
        for i, ts in enumerate(timeseries_list):
            features = extractor.extract_features(ts)
            node_features_list.append(features)

            if (i + 1) % 100 == 0:
                print(f"  Processed: {i + 1}/{len(timeseries_list)}")

        feature_dim = extractor.output_dim  # 使用实际维度

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"\n{'='*60}")
    print(f"Feature extraction completed!")
    print(f"  Feature dimension: {feature_dim}")

    # 统计信息
    all_features = np.concatenate(node_features_list, axis=0)
    print(f"  Feature statistics:")
    print(f"    Mean: {all_features.mean():.4f}")
    print(f"    Std: {all_features.std():.4f}")
    print(f"    Min: {all_features.min():.4f}")
    print(f"    Max: {all_features.max():.4f}")
    print(f"{'='*60}\n")

    return node_features_list, feature_dim
