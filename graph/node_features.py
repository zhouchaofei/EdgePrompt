"""
节点特征提取模块
支持两种特征提取方式：
A. 统计特征（mean/std/skew/kurtosis/bandpower）
B. 时序编码（1D-CNN embedding）
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')


class StatisticalFeatureExtractor:
    """
    统计特征提取器
    为每个ROI提取统计特征：mean, std, skew, kurtosis, bandpower
    """

    def __init__(self, fs=0.5, bands=None):
        """
        Args:
            fs: 采样频率 (Hz)，fMRI通常为0.5Hz (TR=2s)
            bands: 频带定义 [(low, high), ...]
        """
        self.fs = fs

        # 默认频带：slow-5 (0.01-0.027), slow-4 (0.027-0.073), slow-3 (0.073-0.198)
        if bands is None:
            self.bands = [
                (0.01, 0.027),   # slow-5
                (0.027, 0.073),  # slow-4
                (0.073, 0.198)   # slow-3 (部分)
            ]
        else:
            self.bands = bands

    def extract_bandpower(self, signal):
        """
        提取不同频带的功率

        Args:
            signal: [T] 时间序列

        Returns:
            bandpowers: [n_bands] 各频带功率
        """
        # 计算功率谱密度
        freqs, psd = welch(signal, fs=self.fs, nperseg=min(len(signal), 128))

        bandpowers = []
        for low, high in self.bands:
            # 找到频带范围内的索引
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            # 计算该频带的平均功率
            bp = np.trapz(psd[idx_band], freqs[idx_band])
            bandpowers.append(bp)

        return np.array(bandpowers)

    def extract_features(self, timeseries):
        """
        从时间序列提取统计特征

        Args:
            timeseries: [T, N_ROI] 时间序列

        Returns:
            features: [N_ROI, n_features] 节点特征矩阵
        """
        T, N_ROI = timeseries.shape

        # 特征维度：mean(1) + std(1) + skew(1) + kurtosis(1) + bandpower(3) = 7
        # 可以根据需要扩展
        n_stat_features = 4  # mean, std, skew, kurtosis
        n_band_features = len(self.bands)
        n_features = n_stat_features + n_band_features

        features = np.zeros((N_ROI, n_features))

        for i in range(N_ROI):
            signal = timeseries[:, i]

            # 统计特征
            features[i, 0] = np.mean(signal)
            features[i, 1] = np.std(signal)
            features[i, 2] = stats.skew(signal)
            features[i, 3] = stats.kurtosis(signal)

            # 频带功率特征
            try:
                bandpowers = self.extract_bandpower(signal)
                features[i, 4:4+n_band_features] = bandpowers
            except:
                # 如果频谱分析失败，填充0
                features[i, 4:4+n_band_features] = 0

        # 处理NaN和Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features


class TemporalCNNEncoder(nn.Module):
    """
    1D-CNN时序编码器
    将时间序列编码为固定维度的embedding
    """

    def __init__(self, seq_len=140, out_dim=64):
        """
        Args:
            seq_len: 输入序列长度（时间点数）
            out_dim: 输出embedding维度
        """
        super().__init__()

        self.seq_len = seq_len
        self.out_dim = out_dim

        # 1D卷积层
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)

        # 池化层
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Linear(64, out_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len] 时间序列

        Returns:
            embedding: [batch, out_dim]
        """
        # Reshape: [batch, 1, seq_len]
        x = x.unsqueeze(1)

        # 卷积提取特征
        x = self.relu(self.conv1(x))
        x = self.dropout(x)

        x = self.relu(self.conv2(x))
        x = self.dropout(x)

        x = self.relu(self.conv3(x))

        # 全局池化
        x = self.pool(x)  # [batch, 64, 1]
        x = x.squeeze(-1)  # [batch, 64]

        # 全连接层
        x = self.fc(x)  # [batch, out_dim]

        return x


class TemporalFeatureExtractor:
    """
    时序特征提取器（基于1D-CNN）
    """

    def __init__(self, out_dim=64, device='cuda'):
        """
        Args:
            out_dim: 输出embedding维度
            device: 计算设备
        """
        self.out_dim = out_dim
        self.device = device
        self.encoder = None

    def _init_encoder(self, seq_len):
        """初始化编码器"""
        self.encoder = TemporalCNNEncoder(seq_len=seq_len, out_dim=self.out_dim)
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()  # 使用预训练或随机初始化的权重

    def extract_features(self, timeseries, batch_size=32):
        """
        从时间序列提取temporal embedding

        Args:
            timeseries: [T, N_ROI] 时间序列
            batch_size: 批处理大小

        Returns:
            features: [N_ROI, out_dim] 节点特征矩阵
        """
        T, N_ROI = timeseries.shape

        # 初始化编码器
        if self.encoder is None:
            self._init_encoder(T)

        features = []

        with torch.no_grad():
            # 分批处理ROI
            for i in range(0, N_ROI, batch_size):
                end_i = min(i + batch_size, N_ROI)

                # 提取batch
                batch_signals = timeseries[:, i:end_i].T  # [batch, T]
                batch_signals = torch.FloatTensor(batch_signals).to(self.device)

                # 编码
                embeddings = self.encoder(batch_signals)  # [batch, out_dim]
                features.append(embeddings.cpu().numpy())

        features = np.concatenate(features, axis=0)  # [N_ROI, out_dim]

        return features


def extract_node_features_batch(
    timeseries_list,
    method='statistical',
    **kwargs
):
    """
    批量提取节点特征

    Args:
        timeseries_list: 时间序列列表
        method: 'statistical' 或 'temporal'
        **kwargs: 额外参数

    Returns:
        features_list: 节点特征列表
    """
    print(f"\nExtracting node features using method: {method}")

    if method == 'statistical':
        extractor = StatisticalFeatureExtractor()
        features_list = []

        for i, ts in enumerate(timeseries_list):
            features = extractor.extract_features(ts)
            features_list.append(features)

            if (i + 1) % 100 == 0:
                print(f"  Processed: {i + 1}/{len(timeseries_list)}")

        features_list = np.array(features_list)
        print(f"  Feature shape: {features_list.shape}")
        print(f"  Feature dim per node: {features_list.shape[-1]}")

    elif method == 'temporal':
        out_dim = kwargs.get('out_dim', 64)
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        extractor = TemporalFeatureExtractor(out_dim=out_dim, device=device)
        features_list = []

        for i, ts in enumerate(timeseries_list):
            features = extractor.extract_features(ts)
            features_list.append(features)

            if (i + 1) % 100 == 0:
                print(f"  Processed: {i + 1}/{len(timeseries_list)}")

        features_list = np.array(features_list)
        print(f"  Feature shape: {features_list.shape}")
        print(f"  Feature dim per node: {features_list.shape[-1]}")

    else:
        raise ValueError(f"Unknown method: {method}")

    return features_list


if __name__ == "__main__":
    # 测试代码
    print("Testing feature extractors...")

    # 模拟时间序列：140个时间点，116个ROI
    ts = np.random.randn(140, 116)

    # 测试统计特征
    print("\n1. Statistical Features:")
    stat_extractor = StatisticalFeatureExtractor()
    stat_features = stat_extractor.extract_features(ts)
    print(f"   Shape: {stat_features.shape}")
    print(f"   Sample features (ROI 0): {stat_features[0]}")

    # 测试时序编码
    print("\n2. Temporal Features:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    temp_extractor = TemporalFeatureExtractor(out_dim=64, device=device)
    temp_features = temp_extractor.extract_features(ts)
    print(f"   Shape: {temp_features.shape}")
    print(f"   Sample features (ROI 0): {temp_features[0, :5]}")

    print("\n✅ Feature extractors test passed!")
