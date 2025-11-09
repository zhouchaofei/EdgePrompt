"""
节点特征提取模块
支持两种特征：
1. 统计特征：mean/std/skew/kurtosis/bandpower
2. 时序编码：1D-CNN提取固定长度embedding
"""

import numpy as np
from scipy import stats, signal
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


class StatisticalFeatureExtractor:
    """统计特征提取器"""

    def __init__(self, fs=0.5, bands=None):
        """
        Args:
            fs: 采样率 (Hz)，fMRI通常是0.5Hz (TR=2s)
            bands: 频带定义 [(low, high), ...]
        """
        self.fs = fs
        if bands is None:
            # 默认频带：slow-5 (0.01-0.027), slow-4 (0.027-0.073), slow-3 (0.073-0.198)
            self.bands = [
                (0.01, 0.027),   # slow-5
                (0.027, 0.073),  # slow-4
                (0.073, 0.198),  # slow-3
            ]
        else:
            self.bands = bands

    def extract_features(self, timeseries):
        """
        从时间序列提取统计特征

        Args:
            timeseries: [T, N_ROI] 时间序列

        Returns:
            features: [N_ROI, n_features] 特征矩阵
        """
        n_roi = timeseries.shape[1]
        features_list = []

        for roi_idx in range(n_roi):
            roi_signal = timeseries[:, roi_idx]
            roi_features = self._compute_roi_features(roi_signal)
            features_list.append(roi_features)

        features = np.array(features_list)  # [N_ROI, n_features]

        return features

    def _compute_roi_features(self, signal_1d):
        """计算单个ROI的特征（增强NaN处理）"""
        features = []

        # 1. 基础统计特征 - 添加安全检查
        try:
            mean_val = np.mean(signal_1d)
            std_val = np.std(signal_1d)
            skew_val = skew(signal_1d)
            kurt_val = kurtosis(signal_1d)

            # 检查并替换无效值
            features.append(0.0 if np.isnan(mean_val) or np.isinf(mean_val) else mean_val)
            features.append(0.0 if np.isnan(std_val) or np.isinf(std_val) else std_val)
            features.append(0.0 if np.isnan(skew_val) or np.isinf(skew_val) else skew_val)
            features.append(0.0 if np.isnan(kurt_val) or np.isinf(kurt_val) else kurt_val)
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # 2. 频域特征：各频带的能量
        try:
            freqs, psd = signal.periodogram(signal_1d, fs=self.fs)

            for low, high in self.bands:
                band_mask = (freqs >= low) & (freqs <= high)
                if band_mask.sum() > 0:
                    band_power = np.trapz(psd[band_mask], freqs[band_mask])
                    band_power = 0.0 if np.isnan(band_power) or np.isinf(band_power) else band_power
                else:
                    band_power = 0.0
                features.append(band_power)
        except:
            features.extend([0.0] * len(self.bands))

        # 3. 其他特征
        try:
            median_val = np.median(signal_1d)
            q1_val = np.percentile(signal_1d, 25)
            q3_val = np.percentile(signal_1d, 75)

            features.append(0.0 if np.isnan(median_val) or np.isinf(median_val) else median_val)
            features.append(0.0 if np.isnan(q1_val) or np.isinf(q1_val) else q1_val)
            features.append(0.0 if np.isnan(q3_val) or np.isinf(q3_val) else q3_val)
        except:
            features.extend([0.0, 0.0, 0.0])

        # 4. 时间域动态特征
        try:
            diff_std = np.std(np.diff(signal_1d))
            features.append(0.0 if np.isnan(diff_std) or np.isinf(diff_std) else diff_std)
        except:
            features.append(0.0)

        features = np.array(features)

        # 最后再做一次全局检查
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def get_feature_dim(self):
        """返回特征维度"""
        # 4 (basic stats) + len(bands) (bandpower) + 4 (other) = 11 for 3 bands
        return 4 + len(self.bands) + 4

    def get_feature_names(self):
        """返回特征名称"""
        names = ['mean', 'std', 'skewness', 'kurtosis']
        for low, high in self.bands:
            names.append(f'bandpower_{low}_{high}')
        names.extend(['median', 'Q1', 'Q3', 'diff_std'])
        return names


class TemporalEncoder(nn.Module):
    """时序编码器：使用1D-CNN提取固定长度embedding"""

    def __init__(self, input_length, embedding_dim=64, dropout=0.1):
        """
        Args:
            input_length: 输入时间序列长度
            embedding_dim: 输出embedding维度
            dropout: dropout比率
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # 1D卷积层
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_length] or [batch_size, 1, input_length]

        Returns:
            embedding: [batch_size, embedding_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, T]

        # Conv blocks
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv3(x)))

        # Global pooling
        x = self.global_pool(x)  # [B, 128, 1]
        x = x.squeeze(-1)        # [B, 128]

        # FC
        embedding = self.fc(x)   # [B, embedding_dim]

        return embedding


class TemporalFeatureExtractor:
    """时序特征提取器（使用预训练的encoder）"""

    def __init__(self, embedding_dim=64, device='cuda'):
        """
        Args:
            embedding_dim: embedding维度
            device: 计算设备
        """
        self.embedding_dim = embedding_dim
        self.device = device
        self.encoder = None

    def fit(self, timeseries_list, labels, epochs=20, batch_size=32, lr=0.001):
        """
        预训练encoder（使用自监督或监督学习）
        这里使用简单的监督学习

        Args:
            timeseries_list: 时间序列列表
            labels: 标签
            epochs: 训练轮数
            batch_size: batch大小
            lr: 学习率
        """
        print(f"\nPre-training temporal encoder...")
        print(f"  Embedding dim: {self.embedding_dim}")
        print(f"  Device: {self.device}")

        # 获取时间序列长度（假设所有序列长度相同或取最小值）
        min_length = min(ts.shape[0] for ts in timeseries_list)

        # 创建encoder
        self.encoder = TemporalEncoder(
            input_length=min_length,
            embedding_dim=self.embedding_dim
        ).to(self.device)

        # 准备训练数据
        X_train = []
        y_train = []

        for ts, label in zip(timeseries_list, labels):
            # 对每个ROI，截取到min_length
            ts_truncated = ts[:min_length, :]  # [min_length, N_ROI]

            # 将每个ROI的时间序列作为一个样本
            for roi_idx in range(ts_truncated.shape[1]):
                X_train.append(ts_truncated[:, roi_idx])
                y_train.append(label)

        X_train = torch.FloatTensor(np.array(X_train)).to(self.device)  # [N_samples, T]
        y_train = torch.LongTensor(np.array(y_train)).to(self.device)

        # 创建分类头（用于预训练）
        classifier = nn.Linear(self.embedding_dim, 2).to(self.device)

        # 优化器
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(classifier.parameters()),
            lr=lr
        )
        criterion = nn.CrossEntropyLoss()

        # 训练
        self.encoder.train()
        classifier.train()

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward
                embeddings = self.encoder(batch_x)
                outputs = classifier(embeddings)
                loss = criterion(outputs, batch_y)

                # Backward
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_y.size(0)

            if (epoch + 1) % 5 == 0:
                acc = 100. * correct / total
                print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(dataloader):.4f}, "
                      f"Acc={acc:.2f}%")

        print(f"  ✓ Encoder pre-training completed")

    def extract_features(self, timeseries):
        """
        从时间序列提取embedding特征

        Args:
            timeseries: [T, N_ROI] 时间序列

        Returns:
            features: [N_ROI, embedding_dim] 特征矩阵
        """
        if self.encoder is None:
            raise ValueError("Encoder not fitted. Call fit() first.")

        self.encoder.eval()

        n_roi = timeseries.shape[1]
        features_list = []

        with torch.no_grad():
            for roi_idx in range(n_roi):
                roi_signal = timeseries[:, roi_idx]

                # 转换为tensor
                roi_tensor = torch.FloatTensor(roi_signal).unsqueeze(0).to(self.device)

                # 提取embedding
                embedding = self.encoder(roi_tensor)
                features_list.append(embedding.cpu().numpy())

        features = np.vstack(features_list)  # [N_ROI, embedding_dim]

        return features

    def get_feature_dim(self):
        """返回特征维度"""
        return self.embedding_dim


def extract_all_features(timeseries_list, labels, feature_type='statistical',
                         embedding_dim=64, device='cuda'):
    """
    批量提取所有被试的节点特征

    Args:
        timeseries_list: 时间序列列表
        labels: 标签
        feature_type: 'statistical' or 'temporal'
        embedding_dim: temporal编码的维度
        device: 计算设备

    Returns:
        features_list: 特征列表 [N_subjects, (N_ROI, feature_dim)]
        feature_dim: 特征维度
    """
    print(f"\n{'='*60}")
    print(f"Extracting {feature_type} features...")
    print(f"{'='*60}")

    if feature_type == 'statistical':
        extractor = StatisticalFeatureExtractor()
        feature_dim = extractor.get_feature_dim()

        print(f"Feature dimension: {feature_dim}")
        print(f"Features: {', '.join(extractor.get_feature_names())}")

        features_list = []
        for i, ts in enumerate(timeseries_list):
            features = extractor.extract_features(ts)
            features_list.append(features)

            if (i + 1) % 100 == 0:
                print(f"  Processed: {i+1}/{len(timeseries_list)}")

    elif feature_type == 'temporal':
        extractor = TemporalFeatureExtractor(
            embedding_dim=embedding_dim,
            device=device
        )

        # 预训练encoder
        extractor.fit(timeseries_list, labels)

        feature_dim = extractor.get_feature_dim()
        print(f"Feature dimension: {feature_dim}")

        features_list = []
        for i, ts in enumerate(timeseries_list):
            features = extractor.extract_features(ts)
            features_list.append(features)

            if (i + 1) % 100 == 0:
                print(f"  Processed: {i+1}/{len(timeseries_list)}")

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    print(f"\n✓ Feature extraction completed")
    print(f"  Number of subjects: {len(features_list)}")
    print(f"  Feature shape per subject: {features_list[0].shape}")

    return features_list, feature_dim