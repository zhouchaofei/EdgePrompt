"""
节点特征提取模块 - 自监督预训练版本
支持三种自监督方法：
1. Autoencoder重构（推荐，最稳定）
2. 对比学习（Contrastive Learning）
3. 时间掩码预测（Temporal Masking）
"""

import numpy as np
from scipy import stats, signal
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 统计特征提取器（保持不变）
# ============================================================================
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
            self.bands = [
                (0.01, 0.027),   # slow-5
                (0.027, 0.073),  # slow-4
                (0.073, 0.198),  # slow-3
            ]
        else:
            self.bands = bands

    def extract_features(self, timeseries):
        """从时间序列提取统计特征"""
        n_roi = timeseries.shape[1]
        features_list = []

        for roi_idx in range(n_roi):
            roi_signal = timeseries[:, roi_idx]
            roi_features = self._compute_roi_features(roi_signal)
            features_list.append(roi_features)

        features = np.array(features_list)
        return features

    def _compute_roi_features(self, signal_1d):
        """计算单个ROI的特征"""
        features = []

        # 1. 基础统计特征
        try:
            mean_val = np.mean(signal_1d)
            std_val = np.std(signal_1d)
            skew_val = skew(signal_1d)
            kurt_val = kurtosis(signal_1d)

            features.append(0.0 if np.isnan(mean_val) or np.isinf(mean_val) else mean_val)
            features.append(0.0 if np.isnan(std_val) or np.isinf(std_val) else std_val)
            features.append(0.0 if np.isnan(skew_val) or np.isinf(skew_val) else skew_val)
            features.append(0.0 if np.isnan(kurt_val) or np.isinf(kurt_val) else kurt_val)
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # 2. 频域特征
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
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def get_feature_dim(self):
        return 4 + len(self.bands) + 4

    def get_feature_names(self):
        names = ['mean', 'std', 'skewness', 'kurtosis']
        for low, high in self.bands:
            names.append(f'bandpower_{low}_{high}')
        names.extend(['median', 'Q1', 'Q3', 'diff_std'])
        return names


# ============================================================================
# 自监督时序编码器（核心修改）
# ============================================================================

class TemporalEncoder(nn.Module):
    """时序编码器：1D-CNN提取embedding"""

    def __init__(self, input_length, embedding_dim=64, dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Encoder
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

        # Embedding层
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

        # Encoder
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv3(x)))

        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)

        # Embedding
        embedding = self.fc(x)

        return embedding


class TemporalDecoder(nn.Module):
    """时序解码器：用于重构（Autoencoder）"""

    def __init__(self, embedding_dim, output_length, dropout=0.1):
        super().__init__()

        self.output_length = output_length

        # 上采样
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128 * (output_length // 4))
        )

        # 反卷积
        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)

        self.deconv3 = nn.Conv1d(32, 1, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, embedding):
        """
        Args:
            embedding: [batch_size, embedding_dim]
        Returns:
            reconstructed: [batch_size, 1, output_length]
        """
        x = self.fc(embedding)
        x = x.view(x.size(0), 128, -1)

        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)

        # 确保输出长度正确
        if x.size(2) != self.output_length:
            x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=False)

        return x


# ============================================================================
# 自监督预训练策略
# ============================================================================

class SelfSupervisedTrainer:
    """自监督预训练的基类"""

    def __init__(self, encoder, device='cuda'):
        self.encoder = encoder
        self.device = device

    def augment_timeseries(self, ts):
        """
        时间序列数据增强

        Args:
            ts: [batch_size, length] numpy array
        Returns:
            ts_aug: 增强后的时间序列
        """
        batch_size, length = ts.shape
        ts_aug = ts.copy()

        for i in range(batch_size):
            # 随机选择一种增强方式
            aug_type = np.random.choice(['jitter', 'scale', 'shift', 'none'], p=[0.3, 0.3, 0.2, 0.2])

            if aug_type == 'jitter':
                # 添加高斯噪声
                noise = np.random.normal(0, 0.05, length)
                ts_aug[i] += noise

            elif aug_type == 'scale':
                # 幅度缩放
                scale = np.random.uniform(0.8, 1.2)
                ts_aug[i] *= scale

            elif aug_type == 'shift':
                # 时间平移
                shift = np.random.randint(-length // 10, length // 10)
                ts_aug[i] = np.roll(ts_aug[i], shift)

        return ts_aug


class AutoencoderTrainer(SelfSupervisedTrainer):
    """方案1: 自编码器重构（推荐）"""

    def __init__(self, encoder, input_length, device='cuda'):
        super().__init__(encoder, device)

        # 创建解码器
        self.decoder = TemporalDecoder(
            embedding_dim=encoder.embedding_dim,
            output_length=input_length
        ).to(device)

    def fit(self, timeseries_list, epochs=20, batch_size=128, lr=0.001):
        """
        自监督预训练：重构原始时间序列

        Args:
            timeseries_list: 时间序列列表（不需要标签！）
            epochs: 训练轮数
            batch_size: batch大小
            lr: 学习率
        """
        print(f"\n{'='*60}")
        print(f"自监督预训练: Autoencoder Reconstruction")
        print(f"{'='*60}")
        print(f"  Method: 时间序列重构")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {epochs}")

        # 准备数据（所有ROI的时间序列）
        X_train = []
        min_length = min(ts.shape[0] for ts in timeseries_list)

        for ts in timeseries_list:
            ts_truncated = ts[:min_length, :]
            for roi_idx in range(ts_truncated.shape[1]):
                X_train.append(ts_truncated[:, roi_idx])

        X_train = np.array(X_train)
        print(f"  Training samples: {len(X_train)}")
        print(f"  Time series length: {min_length}")

        # 转换为tensor
        X_train = torch.FloatTensor(X_train).to(self.device)

        # 优化器
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
            weight_decay=1e-5
        )

        # 训练
        self.encoder.train()
        self.decoder.train()

        dataset = torch.utils.data.TensorDataset(X_train)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        best_loss = float('inf')

        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0

            for (batch_x,) in dataloader:
                optimizer.zero_grad()

                # Forward
                embedding = self.encoder(batch_x)
                reconstructed = self.decoder(embedding)

                # 重构损失
                loss = F.mse_loss(reconstructed.squeeze(1), batch_x)

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    max_norm=1.0
                )
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches

            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.6f} (Best={best_loss:.6f})")

        print(f"  ✓ Autoencoder预训练完成")
        print(f"{'='*60}\n")


class ContrastiveTrainer(SelfSupervisedTrainer):
    """方案2: 对比学习"""

    def __init__(self, encoder, device='cuda', temperature=0.07):
        super().__init__(encoder, device)
        self.temperature = temperature

    def contrastive_loss(self, embeddings1, embeddings2):
        """
        InfoNCE对比损失

        Args:
            embeddings1, embeddings2: [batch_size, embedding_dim]
        """
        batch_size = embeddings1.shape[0]

        # L2归一化
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)

        # 计算相似度矩阵
        similarity = torch.matmul(embeddings1, embeddings2.T) / self.temperature

        # 对角线元素是正样本对
        labels = torch.arange(batch_size).to(self.device)

        # 双向对比损失
        loss1 = F.cross_entropy(similarity, labels)
        loss2 = F.cross_entropy(similarity.T, labels)

        return (loss1 + loss2) / 2

    def fit(self, timeseries_list, epochs=20, batch_size=128, lr=0.001):
        """
        自监督预训练：对比学习

        同一ROI的两种增强视图应该相似
        """
        print(f"\n{'='*60}")
        print(f"自监督预训练: Contrastive Learning")
        print(f"{'='*60}")
        print(f"  Method: SimCLR风格对比学习")
        print(f"  Temperature: {self.temperature}")
        print(f"  Device: {self.device}")

        # 准备数据
        X_train = []
        min_length = min(ts.shape[0] for ts in timeseries_list)

        for ts in timeseries_list:
            ts_truncated = ts[:min_length, :]
            for roi_idx in range(ts_truncated.shape[1]):
                X_train.append(ts_truncated[:, roi_idx])

        X_train = np.array(X_train)
        print(f"  Training samples: {len(X_train)}")

        # 优化器
        optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=lr,
            weight_decay=1e-5
        )

        # 训练
        self.encoder.train()

        best_loss = float('inf')

        for epoch in range(epochs):
            # 每个epoch随机打乱
            indices = np.random.permutation(len(X_train))
            total_loss = 0
            n_batches = 0

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_x = X_train[batch_indices]

                # 生成两个增强视图
                batch_x1 = self.augment_timeseries(batch_x)
                batch_x2 = self.augment_timeseries(batch_x)

                batch_x1 = torch.FloatTensor(batch_x1).to(self.device)
                batch_x2 = torch.FloatTensor(batch_x2).to(self.device)

                optimizer.zero_grad()

                # 编码两个视图
                embeddings1 = self.encoder(batch_x1)
                embeddings2 = self.encoder(batch_x2)

                # 对比损失
                loss = self.contrastive_loss(embeddings1, embeddings2)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches

            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f} (Best={best_loss:.4f})")

        print(f"  ✓ 对比学习预训练完成")
        print(f"{'='*60}\n")


class MaskedPredictionTrainer(SelfSupervisedTrainer):
    """方案3: 时间掩码预测（类似BERT）"""

    def __init__(self, encoder, input_length, device='cuda', mask_ratio=0.15):
        super().__init__(encoder, device)
        self.mask_ratio = mask_ratio

        # 预测头（预测被掩码的时间点）
        self.prediction_head = nn.Sequential(
            nn.Linear(encoder.embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_length)
        ).to(device)

    def create_masked_input(self, x):
        """
        随机掩码时间点

        Args:
            x: [batch_size, length]
        Returns:
            x_masked: 掩码后的输入
            mask: 掩码位置
        """
        batch_size, length = x.shape
        mask = torch.rand(batch_size, length) < self.mask_ratio

        x_masked = x.clone()
        x_masked[mask] = 0  # 将被掩码的位置设为0

        return x_masked, mask

    def fit(self, timeseries_list, epochs=20, batch_size=128, lr=0.001):
        """
        自监督预训练：掩码预测
        """
        print(f"\n{'='*60}")
        print(f"自监督预训练: Masked Prediction")
        print(f"{'='*60}")
        print(f"  Method: 时间掩码预测（类似BERT）")
        print(f"  Mask ratio: {self.mask_ratio}")
        print(f"  Device: {self.device}")

        # 准备数据
        X_train = []
        min_length = min(ts.shape[0] for ts in timeseries_list)

        for ts in timeseries_list:
            ts_truncated = ts[:min_length, :]
            for roi_idx in range(ts_truncated.shape[1]):
                X_train.append(ts_truncated[:, roi_idx])

        X_train = np.array(X_train)
        X_train = torch.FloatTensor(X_train).to(self.device)

        print(f"  Training samples: {len(X_train)}")

        # 优化器
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.prediction_head.parameters()),
            lr=lr,
            weight_decay=1e-5
        )

        # 训练
        self.encoder.train()
        self.prediction_head.train()

        dataset = torch.utils.data.TensorDataset(X_train)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        best_loss = float('inf')

        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0

            for (batch_x,) in dataloader:
                # 创建掩码输入
                batch_x_masked, mask = self.create_masked_input(batch_x)

                optimizer.zero_grad()

                # 编码
                embedding = self.encoder(batch_x_masked)

                # 预测原始值
                prediction = self.prediction_head(embedding)

                # 只计算被掩码位置的损失
                loss = F.mse_loss(prediction[mask], batch_x[mask])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.prediction_head.parameters()),
                    max_norm=1.0
                )
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches

            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.6f} (Best={best_loss:.6f})")

        print(f"  ✓ 掩码预测预训练完成")
        print(f"{'='*60}\n")


# ============================================================================
# 时序特征提取器（主接口）
# ============================================================================

class TemporalFeatureExtractor:
    """
    时序特征提取器（自监督预训练版本）

    使用方法:
        extractor = TemporalFeatureExtractor(
            embedding_dim=64,
            pretrain_method='autoencoder'  # 或 'contrastive', 'masked'
        )
        extractor.fit(timeseries_list)  # 不需要标签！
        features = extractor.extract_features(timeseries)
    """

    def __init__(self, embedding_dim=64, device='cuda', pretrain_method='autoencoder'):
        """
        Args:
            embedding_dim: embedding维度
            device: 计算设备
            pretrain_method: 预训练方法
                - 'autoencoder': 自编码器重构（推荐，最稳定）
                - 'contrastive': 对比学习
                - 'masked': 掩码预测
        """
        self.embedding_dim = embedding_dim
        self.device = device
        self.pretrain_method = pretrain_method
        self.encoder = None
        self.trainer = None

    def fit(self, timeseries_list, epochs=20, batch_size=128, lr=0.001):
        """
        ✅ 自监督预训练（不使用标签）

        Args:
            timeseries_list: 时间序列列表（不需要labels参数！）
            epochs: 训练轮数
            batch_size: batch大小
            lr: 学习率
        """
        print(f"\n{'='*80}")
        print(f"时序特征提取器 - 自监督预训练")
        print(f"{'='*80}")
        print(f"  Embedding dim: {self.embedding_dim}")
        print(f"  Method: {self.pretrain_method}")
        print(f"  Device: {self.device}")
        print(f"  ✅ 无需标签信息")
        print(f"{'='*80}")

        # 获取时间序列长度
        min_length = min(ts.shape[0] for ts in timeseries_list)

        # 创建encoder
        self.encoder = TemporalEncoder(
            input_length=min_length,
            embedding_dim=self.embedding_dim
        ).to(self.device)

        # 选择预训练方法
        if self.pretrain_method == 'autoencoder':
            self.trainer = AutoencoderTrainer(
                encoder=self.encoder,
                input_length=min_length,
                device=self.device
            )
        elif self.pretrain_method == 'contrastive':
            self.trainer = ContrastiveTrainer(
                encoder=self.encoder,
                device=self.device
            )
        elif self.pretrain_method == 'masked':
            self.trainer = MaskedPredictionTrainer(
                encoder=self.encoder,
                input_length=min_length,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown pretrain_method: {self.pretrain_method}")

        # 开始预训练
        self.trainer.fit(
            timeseries_list=timeseries_list,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )

        print(f"✅ 预训练完成，可以开始提取特征")

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


# ============================================================================
# 批量特征提取（接口保持兼容）
# ============================================================================

def extract_all_features(timeseries_list, labels=None, feature_type='statistical',
                         embedding_dim=64, device='cuda', pretrain_method='autoencoder'):
    """
    批量提取所有被试的节点特征

    ✅ 修改：labels参数变为可选，自监督方法不需要

    Args:
        timeseries_list: 时间序列列表
        labels: 标签（仅用于统计，不用于预训练）
        feature_type: 'statistical' or 'temporal'
        embedding_dim: temporal编码的维度
        device: 计算设备
        pretrain_method: 自监督方法 ('autoencoder', 'contrastive', 'masked')

    Returns:
        features_list: 特征列表 [N_subjects, (N_ROI, feature_dim)]
        feature_dim: 特征维度
    """
    print(f"\n{'='*80}")
    print(f"提取 {feature_type} 特征...")
    print(f"{'='*80}")

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
            device=device,
            pretrain_method=pretrain_method
        )

        # ✅ 自监督预训练（不需要标签）
        extractor.fit(timeseries_list)

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


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("测试自监督时序特征提取器")
    print("="*80)

    # 创建模拟数据
    np.random.seed(42)
    n_subjects = 10
    n_timepoints = 200
    n_rois = 116

    timeseries_list = []
    for i in range(n_subjects):
        ts = np.random.randn(n_timepoints, n_rois)
        # 添加一些时间相关性
        for roi in range(n_rois):
            ts[:, roi] = np.convolve(ts[:, roi], np.ones(5)/5, mode='same')
        timeseries_list.append(ts)

    print(f"\n生成模拟数据:")
    print(f"  被试数: {n_subjects}")
    print(f"  时间点: {n_timepoints}")
    print(f"  ROI数: {n_rois}")

    # 测试三种自监督方法
    for method in ['autoencoder', 'contrastive', 'masked']:
        print(f"\n{'='*80}")
        print(f"测试方法: {method}")
        print(f"{'='*80}")

        extractor = TemporalFeatureExtractor(
            embedding_dim=64,
            device='cpu',  # 测试用CPU
            pretrain_method=method
        )

        # 预训练（不需要标签！）
        extractor.fit(timeseries_list, epochs=5, batch_size=32)

        # 提取特征
        features = extractor.extract_features(timeseries_list[0])
        print(f"\n提取的特征形状: {features.shape}")
        print(f"特征统计:")
        print(f"  Mean: {features.mean():.4f}")
        print(f"  Std: {features.std():.4f}")

    print(f"\n{'='*80}")
    print("✅ 所有测试通过")
    print(f"{'='*80}")