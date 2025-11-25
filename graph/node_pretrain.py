"""
Node-level预训练模型
支持Masked Reconstruction + Temporal Contrastive双任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NodeEncoder(nn.Module):
    """ROI时间序列编码器（1D-CNN）"""

    def __init__(self, input_length, embedding_dim=64, dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim

        # 1D-CNN Encoder
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Projection to embedding
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
        x = x.squeeze(-1)  # [B, 128]

        # Embedding
        embedding = self.fc(x)

        return embedding


class MaskedDecoder(nn.Module):
    """Masked重建解码器"""

    def __init__(self, embedding_dim, output_length, dropout=0.1):
        super().__init__()

        self.output_length = output_length

        # Upsampling
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128 * (output_length // 4))
        )

        # Deconvolution
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


class DualTaskPretrainer(nn.Module):
    """
    双任务预训练模型
    Task 1: Masked Reconstruction
    Task 2: Temporal Contrastive
    """

    def __init__(self, input_length, embedding_dim=64, temperature=0.07):
        super().__init__()

        self.encoder = NodeEncoder(input_length, embedding_dim)
        self.decoder = MaskedDecoder(embedding_dim, input_length)

        # Contrastive learning投影头
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.temperature = temperature

    def forward(self, x, task='reconstruction'):
        """
        Args:
            x: [batch_size, input_length]
            task: 'reconstruction' or 'contrastive'
        """
        embedding = self.encoder(x)

        if task == 'reconstruction':
            reconstructed = self.decoder(embedding)
            return reconstructed
        elif task == 'contrastive':
            projection = self.projection(embedding)
            return F.normalize(projection, dim=1)
        else:
            return embedding

    def contrastive_loss(self, z1, z2):
        """
        InfoNCE对比损失
        Args:
            z1, z2: [batch_size, embedding_dim] 两个增强视图的投影
        """
        batch_size = z1.shape[0]

        # 计算相似度矩阵
        similarity = torch.matmul(z1, z2.T) / self.temperature

        # 对角线是正样本对
        labels = torch.arange(batch_size).to(z1.device)

        # 双向对比损失
        loss1 = F.cross_entropy(similarity, labels)
        loss2 = F.cross_entropy(similarity.T, labels)

        return (loss1 + loss2) / 2


def create_masked_input(x, mask_ratio=0.15, mask_strategy='random'):
    """
    创建masked输入

    Args:
        x: [batch_size, length] 时间序列
        mask_ratio: mask比例
        mask_strategy: 'random' or 'block'

    Returns:
        x_masked: masked输入
        mask: mask位置 [batch_size, length]
    """
    batch_size, length = x.shape
    mask = torch.zeros_like(x, dtype=torch.bool)

    if mask_strategy == 'random':
        # 随机mask单个timepoint
        for i in range(batch_size):
            mask_indices = torch.randperm(length)[:int(length * mask_ratio)]
            mask[i, mask_indices] = True

    elif mask_strategy == 'block':
        # Mask连续的block
        for i in range(batch_size):
            block_length = int(length * mask_ratio)
            start_idx = torch.randint(0, length - block_length + 1, (1,)).item()
            mask[i, start_idx:start_idx + block_length] = True

    x_masked = x.clone()
    x_masked[mask] = 0  # 被mask的位置设为0

    return x_masked, mask


def augment_timeseries(x, aug_type=None):
    """
    时间序列数据增强

    Args:
        x: [batch_size, length]
        aug_type: 增强类型，None则随机选择

    Returns:
        x_aug: 增强后的时间序列
    """
    batch_size, length = x.shape
    x_aug = x.clone()

    for i in range(batch_size):
        if aug_type is None:
            aug = np.random.choice(['jitter', 'scale', 'shift', 'none'],
                                   p=[0.3, 0.3, 0.2, 0.2])
        else:
            aug = aug_type

        if aug == 'jitter':
            # 添加高斯噪声
            noise = torch.randn_like(x_aug[i]) * 0.05
            x_aug[i] += noise

        elif aug == 'scale':
            # 幅度缩放
            scale = torch.empty(1).uniform_(0.8, 1.2).item()
            x_aug[i] *= scale

        elif aug == 'shift':
            # 时间平移
            shift = torch.randint(-length // 10, length // 10 + 1, (1,)).item()
            x_aug[i] = torch.roll(x_aug[i], shift)

    return x_aug