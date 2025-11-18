"""
Node-level 预训练：时序 → node embedding
结合 Masked Reconstruction + Temporal Contrastive Learning

数据：ABIDE + MDD 合并
模型：1D-CNN Encoder → Global Pooling → Node Embedding (d=64)
任务：
  1. Masked Reconstruction: 随机mask timepoints/ROIs，重建原始信号
  2. Temporal Contrastive: 时间增强对比学习（InfoNCE）
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import random

# 导入数据加载函数
from abide_data_baseline import ABIDEBaselineProcessor
from mdd_data_baseline import MDDBaselineProcessor


# ============================================================================
# 数据增强策略
# ============================================================================
class TimeSeriesAugmentation:
    """时间序列数据增强"""

    @staticmethod
    def add_noise(ts, noise_std=0.1):
        """添加高斯噪声"""
        noise = torch.randn_like(ts) * noise_std
        return ts + noise

    @staticmethod
    def time_shift(ts, max_shift=5):
        """时间平移"""
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return ts
        elif shift > 0:
            return torch.cat([ts[shift:], ts[-shift:]], dim=0)
        else:
            return torch.cat([ts[:shift], ts[:-shift]], dim=0)

    @staticmethod
    def time_warp(ts, sigma=0.2):
        """时间扭曲（速度变化）"""
        T = ts.shape[0]
        # 生成平滑的扭曲因子
        warp = torch.cumsum(torch.randn(T) * sigma + 1, dim=0)
        warp = warp / warp[-1] * T  # 归一化到[0, T]

        # 线性插值
        indices = torch.clamp(warp, 0, T - 1).long()
        return ts[indices]

    @staticmethod
    def frequency_mask(ts, mask_ratio=0.3):
        """频域mask（模拟带通滤波）"""
        # FFT
        fft = torch.fft.rfft(ts, dim=0)
        n_freqs = fft.shape[0]

        # 随机mask一些频段
        n_mask = int(n_freqs * mask_ratio)
        mask_indices = torch.randperm(n_freqs)[:n_mask]
        fft[mask_indices] = 0

        # IFFT
        return torch.fft.irfft(fft, n=ts.shape[0], dim=0)

    @staticmethod
    def random_crop_resize(ts, crop_ratio=0.9):
        """随机裁剪并resize回原长度"""
        T = ts.shape[0]
        crop_len = int(T * crop_ratio)
        start = random.randint(0, T - crop_len)
        cropped = ts[start:start + crop_len]

        # 简单线性插值resize
        indices = torch.linspace(0, crop_len - 1, T).long()
        return cropped[indices]


# ============================================================================
# 数据集
# ============================================================================
class NodePretrainDataset(Dataset):
    """Node预训练数据集（支持双任务）"""

    def __init__(self, timeseries_list, mask_ratio=0.15,
                 roi_mask_ratio=0.2, augment_prob=0.5):
        """
        Args:
            timeseries_list: List of [T, N_ROI] arrays
            mask_ratio: timepoint mask比例
            roi_mask_ratio: ROI维度mask比例
            augment_prob: 数据增强概率
        """
        self.timeseries_list = timeseries_list
        self.mask_ratio = mask_ratio
        self.roi_mask_ratio = roi_mask_ratio
        self.augment_prob = augment_prob
        self.aug = TimeSeriesAugmentation()

    def __len__(self):
        return len(self.timeseries_list)

    def __getitem__(self, idx):
        # 获取原始时间序列 [T, N_ROI]
        ts = torch.FloatTensor(self.timeseries_list[idx])
        T, N = ts.shape

        # ========== Task 1: Masked Reconstruction ==========
        # 随机mask timepoints
        mask_t = torch.rand(T) < self.mask_ratio
        ts_masked = ts.clone()
        ts_masked[mask_t] = 0  # 简单置零

        # 随机mask ROIs（某些ROI全部置零）
        mask_roi = torch.rand(N) < self.roi_mask_ratio
        ts_masked[:, mask_roi] = 0

        # 合并mask信息（用于loss计算）
        # 创建完整的mask矩阵
        mask_full = torch.zeros(T, N, dtype=torch.bool)
        mask_full[mask_t, :] = True
        mask_full[:, mask_roi] = True

        # ========== Task 2: Temporal Contrastive ==========
        # 生成两个增强版本
        if random.random() < self.augment_prob:
            ts_aug1 = self._augment(ts)
            ts_aug2 = self._augment(ts)
        else:
            ts_aug1 = ts
            ts_aug2 = ts

        return {
            'original': ts,  # [T, N_ROI] 原始信号
            'masked': ts_masked,  # [T, N_ROI] masked信号
            'mask': mask_full,  # [T, N_ROI] mask位置
            'aug1': ts_aug1,  # [T, N_ROI] 增强版本1
            'aug2': ts_aug2,  # [T, N_ROI] 增强版本2
        }

    def _augment(self, ts):
        """随机应用数据增强"""
        aug_funcs = [
            lambda x: self.aug.add_noise(x, noise_std=0.05),
            lambda x: self.aug.time_shift(x, max_shift=3),
            lambda x: self.aug.time_warp(x, sigma=0.1),
            lambda x: self.aug.frequency_mask(x, mask_ratio=0.2),
            lambda x: self.aug.random_crop_resize(x, crop_ratio=0.9),
        ]

        # 随机选择1-2个增强
        n_aug = random.randint(1, 2)
        selected = random.sample(aug_funcs, n_aug)

        ts_aug = ts
        for aug_func in selected:
            ts_aug = aug_func(ts_aug)

        return ts_aug


# ============================================================================
# 1D-CNN Encoder
# ============================================================================
class NodeEncoder(nn.Module):
    """1D-CNN编码器：time-series → node embedding"""

    def __init__(self, n_timepoints, d_node=64, dropout=0.1):
        """
        Args:
            n_timepoints: 时间点数量（会根据实际数据padding/truncate）
            d_node: node embedding维度
            dropout: dropout率
        """
        super().__init__()

        self.d_node = d_node

        # 1D-CNN layers（处理时间维度）
        self.conv_layers = nn.ModuleList([
            # Conv1: [T, 1] -> [T, 64]
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2)  # T -> T/2
            ),
            # Conv2: [T/2, 64] -> [T/2, 128]
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2)  # T/2 -> T/4
            ),
            # Conv3: [T/4, 128] -> [T/4, 256]
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout),
            ),
        ])

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Projection to node embedding
        self.projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_node)
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, N_ROI] 或 [B, T] （单个ROI）
        Returns:
            embeddings: [B, N_ROI, d_node] 或 [B, d_node]
        """
        if x.dim() == 3:
            # [B, T, N_ROI] -> process each ROI separately
            B, T, N = x.shape

            # Reshape to [B*N, T]
            x = x.permute(0, 2, 1).reshape(B * N, T)

            # Encode
            emb = self._encode_single(x)  # [B*N, d_node]

            # Reshape back
            emb = emb.reshape(B, N, self.d_node)

            return emb
        else:
            # [B, T] -> single ROI
            return self._encode_single(x)

    def _encode_single(self, x):
        """编码单个时间序列 [B, T] -> [B, d_node]"""
        # [B, T] -> [B, 1, T]
        x = x.unsqueeze(1)

        # CNN layers
        for conv in self.conv_layers:
            x = conv(x)

        # Global pooling: [B, 256, T/4] -> [B, 256, 1] -> [B, 256]
        x = self.pool(x).squeeze(-1)

        # Projection: [B, 256] -> [B, d_node]
        x = self.projection(x)

        return x


# ============================================================================
# Decoder (for reconstruction)
# ============================================================================
class NodeDecoder(nn.Module):
    """解码器：node embedding → time-series"""

    def __init__(self, d_node=64, n_timepoints=200):
        super().__init__()

        self.n_timepoints = n_timepoints

        # Expand
        self.expand = nn.Sequential(
            nn.Linear(d_node, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        # Deconv layers
        # 需要根据目标长度调整
        init_len = n_timepoints // 4

        self.initial_shape = nn.Linear(256, 256 * init_len)

        self.deconv_layers = nn.ModuleList([
            # [256, L] -> [128, L*2]
            nn.Sequential(
                nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            ),
            # [128, L*2] -> [64, L*4]
            nn.Sequential(
                nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            ),
            # [64, L*4] -> [1, L*4]
            nn.Conv1d(64, 1, kernel_size=3, padding=1)
        ])

    def forward(self, emb):
        """
        Args:
            emb: [B, N_ROI, d_node]
        Returns:
            recon: [B, T, N_ROI]
        """
        B, N, _ = emb.shape

        # Reshape: [B, N, d_node] -> [B*N, d_node]
        emb = emb.reshape(B * N, -1)

        # Expand
        x = self.expand(emb)  # [B*N, 256]

        # Reshape to [B*N, 256, init_len]
        init_len = self.n_timepoints // 4
        x = self.initial_shape(x).reshape(B * N, 256, init_len)

        # Deconv
        for deconv in self.deconv_layers:
            x = deconv(x)

        # [B*N, 1, T] -> [B*N, T]
        x = x.squeeze(1)

        # Adjust to exact length
        if x.shape[1] != self.n_timepoints:
            x = F.interpolate(
                x.unsqueeze(1),
                size=self.n_timepoints,
                mode='linear',
                align_corners=False
            ).squeeze(1)

        # Reshape: [B*N, T] -> [B, T, N]
        x = x.reshape(B, N, self.n_timepoints).permute(0, 2, 1)

        return x


# ============================================================================
# Contrastive Loss (InfoNCE)
# ============================================================================
class InfoNCELoss(nn.Module):
    """InfoNCE对比损失"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Args:
            z1: [B, N, d_node]
            z2: [B, N, d_node]
        Returns:
            loss: scalar
        """
        B, N, d = z1.shape

        # Flatten: [B, N, d] -> [B*N, d]
        z1 = z1.reshape(B * N, d)
        z2 = z2.reshape(B * N, d)

        # L2 normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Compute similarity matrix
        logits = torch.mm(z1, z2.t()) / self.temperature  # [B*N, B*N]

        # Labels: diagonal is positive
        labels = torch.arange(B * N, device=z1.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


# ============================================================================
# 训练器
# ============================================================================
class NodePretrainer:
    """Node预训练器（双任务）"""

    def __init__(self, encoder, decoder, device,
                 lambda_recon=1.0, lambda_contrast=0.5):
        """
        Args:
            encoder: NodeEncoder
            decoder: NodeDecoder
            device: torch device
            lambda_recon: reconstruction loss权重
            lambda_contrast: contrastive loss权重
        """
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device

        self.lambda_recon = lambda_recon
        self.lambda_contrast = lambda_contrast

        # Losses
        self.recon_loss_fn = nn.MSELoss()
        self.contrast_loss_fn = InfoNCELoss(temperature=0.07)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=1e-3,
            weight_decay=1e-4
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )

    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.encoder.train()
        self.decoder.train()

        total_loss = 0
        total_recon_loss = 0
        total_contrast_loss = 0

        pbar = tqdm(dataloader, desc='Training')
        for batch in pbar:
            # Move to device
            original = batch['original'].to(self.device)
            masked = batch['masked'].to(self.device)
            mask = batch['mask'].to(self.device)
            aug1 = batch['aug1'].to(self.device)
            aug2 = batch['aug2'].to(self.device)

            self.optimizer.zero_grad()

            # ========== Task 1: Masked Reconstruction ==========
            # Encode masked input
            emb_masked = self.encoder(masked)  # [B, N, d]

            # Decode
            recon = self.decoder(emb_masked)  # [B, T, N]

            # Compute reconstruction loss (only on masked positions)
            recon_loss = self.recon_loss_fn(
                recon[mask],
                original[mask]
            )

            # ========== Task 2: Temporal Contrastive ==========
            # Encode both augmented versions
            emb_aug1 = self.encoder(aug1)  # [B, N, d]
            emb_aug2 = self.encoder(aug2)  # [B, N, d]

            # Contrastive loss
            contrast_loss = self.contrast_loss_fn(emb_aug1, emb_aug2)

            # ========== Total Loss ==========
            loss = (self.lambda_recon * recon_loss +
                    self.lambda_contrast * contrast_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()

            # Stats
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_contrast_loss += contrast_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'contrast': f'{contrast_loss.item():.4f}'
            })

        self.scheduler.step()

        n = len(dataloader)
        return {
            'loss': total_loss / n,
            'recon_loss': total_recon_loss / n,
            'contrast_loss': total_contrast_loss / n
        }

    def save_checkpoint(self, save_path, epoch, metrics):
        """保存checkpoint"""
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'd_node': self.encoder.d_node,
        }, save_path)
        print(f"✅ Checkpoint saved to {save_path}")


# ============================================================================
# 主函数
# ============================================================================
def load_combined_data(data_folder, n_abide=None, n_mdd=None):
    """加载并合并ABIDE+MDD数据"""
    print("=" * 60)
    print("Loading ABIDE + MDD combined data...")
    print("=" * 60)

    timeseries_all = []

    # ========== ABIDE ==========
    try:
        print("\n[1/2] Loading ABIDE data...")
        abide_processor = ABIDEBaselineProcessor(
            data_folder=data_folder,
            pipeline='cpac',
            atlas='aal'
        )

        abide_ts, _, _, _ = abide_processor.download_and_extract(
            n_subjects=n_abide,
            apply_zscore=True  # 已在源码中做了z-score
        )

        print(f"  ✅ ABIDE: {len(abide_ts)} subjects loaded")
        timeseries_all.extend(abide_ts)

    except Exception as e:
        print(f"  ⚠️  ABIDE loading failed: {e}")

    # ========== MDD ==========
    try:
        print("\n[2/2] Loading MDD data...")
        mdd_processor = MDDBaselineProcessor(data_folder=data_folder)

        mdd_ts, _, _, _ = mdd_processor.load_roi_signals(apply_zscore=True)

        print(f"  ✅ MDD: {len(mdd_ts)} subjects loaded")
        timeseries_all.extend(mdd_ts)

    except Exception as e:
        print(f"  ⚠️  MDD loading failed: {e}")

    print(f"\n{'=' * 60}")
    print(f"✅ Total: {len(timeseries_all)} subjects combined")
    print(f"{'=' * 60}\n")

    return timeseries_all


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    timeseries_all = load_combined_data(
        args.data_folder,
        n_abide=args.n_abide,
        n_mdd=args.n_mdd
    )

    if len(timeseries_all) == 0:
        print("❌ No data loaded! Exiting...")
        return

    # ========== 关键：统计时间序列长度 ==========
    ts_lengths = [ts.shape[0] for ts in timeseries_all]
    min_T = min(ts_lengths)
    max_T = max(ts_lengths)
    mean_T = np.mean(ts_lengths)
    median_T = np.median(ts_lengths)

    print(f"\n{'=' * 60}")
    print(f"Time Series Length Statistics:")
    print(f"{'=' * 60}")
    print(f"  Min:    {min_T}")
    print(f"  Max:    {max_T}")
    print(f"  Mean:   {mean_T:.1f}")
    print(f"  Median: {median_T:.1f}")
    print(f"  25th percentile: {np.percentile(ts_lengths, 25):.1f}")
    print(f"  75th percentile: {np.percentile(ts_lengths, 75):.1f}")


    # ========== 过滤太短的序列 ==========
    min_acceptable_length = 100  # 建议：至少2分钟数据（TR=2s → 60个点/分钟）

    timeseries_filtered = []
    filtered_count = 0

    for ts in timeseries_all:
        if ts.shape[0] >= min_acceptable_length:
            timeseries_filtered.append(ts)
        else:
            filtered_count += 1

    if filtered_count > 0:
        print(f"\n⚠️  Filtered out {filtered_count} subjects with < {min_acceptable_length} timepoints")
        print(f"   Kept: {len(timeseries_filtered)}/{len(timeseries_all)} subjects")

    timeseries_all = timeseries_filtered

    # 重新统计
    ts_lengths = [ts.shape[0] for ts in timeseries_all]
    min_T = min(ts_lengths)
    max_T = max(ts_lengths)
    mean_T = np.mean(ts_lengths)
    median_T = np.median(ts_lengths)
    print(f"\nAfter filtering:")
    print(f"  New Min: {min_T}")
    print(f"  New Max: {max_T}")
    print(f"  New Mean: {mean_T:.1f}")
    print(f"  New Median: {median_T:.1f}\n")


    # ========== 智能选择目标长度 ==========
    if args.n_timepoints > 0:
        target_T = args.n_timepoints
        print(f"\n  Using specified target length: {target_T}")
    else:
        # 策略1：使用中位数（推荐）
        target_T = int(median_T)
        # 策略2：使用75分位数（保留更多信息）
        # target_T = int(np.percentile(ts_lengths, 75))
        # 策略3：使用固定经验值
        # target_T = 150  # 或 180, 200
        print(f"\n  Auto-selected target length (median): {target_T}")

    # 统计padding和truncate的比例
    n_pad = sum(1 for l in ts_lengths if l < target_T)
    n_truncate = sum(1 for l in ts_lengths if l > target_T)
    n_exact = sum(1 for l in ts_lengths if l == target_T)

    print(f"\n  Processing strategy:")
    print(f"    Will pad:     {n_pad}/{len(ts_lengths)} subjects ({n_pad / len(ts_lengths) * 100:.1f}%)")
    print(f"    Will truncate: {n_truncate}/{len(ts_lengths)} subjects ({n_truncate / len(ts_lengths) * 100:.1f}%)")
    print(f"    Exact match:  {n_exact}/{len(ts_lengths)} subjects ({n_exact / len(ts_lengths) * 100:.1f}%)")
    print(f"{'=' * 60}\n")

    # # Pad/truncate to fixed length
    # max_T = max(ts.shape[0] for ts in timeseries_all)
    # print(f"Max time points: {max_T}")
    #
    # # 使用统一长度（pad or truncate）
    # target_T = args.n_timepoints if args.n_timepoints > 0 else max_T
    # print(f"Target time points: {target_T}")

    timeseries_processed = []
    for ts in timeseries_all:
        T_curr = ts.shape[0]
        if T_curr < target_T:
            # Pad
            pad = np.zeros((target_T - T_curr, ts.shape[1]))
            ts_proc = np.vstack([ts, pad])
        else:
            # Truncate
            ts_proc = ts[:target_T]
        timeseries_processed.append(ts_proc)

    # Dataset
    dataset = NodePretrainDataset(
        timeseries_processed,
        mask_ratio=args.mask_ratio,
        roi_mask_ratio=args.roi_mask_ratio,
        augment_prob=0.5
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Models
    encoder = NodeEncoder(
        n_timepoints=target_T,
        d_node=args.d_node,
        dropout=args.dropout
    )

    decoder = NodeDecoder(
        d_node=args.d_node,
        n_timepoints=target_T
    )

    print(f"\nEncoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # Trainer
    trainer = NodePretrainer(
        encoder, decoder, device,
        lambda_recon=args.lambda_recon,
        lambda_contrast=args.lambda_contrast
    )

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"Starting training for {args.epochs} epochs...")
    print(f"{'=' * 60}\n")

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        metrics = trainer.train_epoch(dataloader)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Total Loss: {metrics['loss']:.4f}")
        print(f"  Recon Loss: {metrics['recon_loss']:.4f}")
        print(f"  Contrast Loss: {metrics['contrast_loss']:.4f}")

        # Save checkpoint
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            save_path = os.path.join(args.save_dir, 'node_encoder_best.pth')
            trainer.save_checkpoint(save_path, epoch, metrics)

        # Regular checkpoint
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f'node_encoder_epoch{epoch}.pth')
            trainer.save_checkpoint(save_path, epoch, metrics)

    # Final save
    final_path = os.path.join(args.save_dir, 'node_encoder.pth')
    trainer.save_checkpoint(final_path, args.epochs, metrics)

    print(f"\n{'=' * 60}")
    print(f"✅ Training completed!")
    print(f"Best model saved to: {os.path.join(args.save_dir, 'node_encoder_best.pth')}")
    print(f"Final model saved to: {final_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Node-level Pretraining')

    # Data
    parser.add_argument('--data_folder', type=str, default='./data',
                        help='Root data folder')
    parser.add_argument('--n_abide', type=int, default=None,
                        help='Number of ABIDE subjects (None=all)')
    parser.add_argument('--n_mdd', type=int, default=None,
                        help='Number of MDD subjects (None=all)')
    parser.add_argument('--n_timepoints', type=int, default=200,
                        help='Target time points (0=use max)')

    # Model
    parser.add_argument('--d_node', type=int, default=64,
                        help='Node embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (subject-level)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--mask_ratio', type=float, default=0.15,
                        help='Timepoint mask ratio')
    parser.add_argument('--roi_mask_ratio', type=float, default=0.2,
                        help='ROI mask ratio')
    parser.add_argument('--lambda_recon', type=float, default=1.0,
                        help='Reconstruction loss weight')
    parser.add_argument('--lambda_contrast', type=float, default=0.5,
                        help='Contrastive loss weight')

    # Misc
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Save directory')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Checkpoint save interval')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)