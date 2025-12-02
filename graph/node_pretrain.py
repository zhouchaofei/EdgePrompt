"""
离线节点预训练 - 增强版
修改：
1. 输入层添加BatchNorm
2. 增加mask_ratio到0.6
3. 优化训练策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime


class TimeSeriesWindowDataset(Dataset):
    """滑窗时间序列数据集"""

    def __init__(self, timeseries_list, window_size=64, stride=32):
        self.window_size = window_size
        self.stride = stride
        self.chunks = []

        print(f"\n准备预训练数据...")
        print(f"  窗口大小: {window_size}")
        print(f"  滑动步长: {stride}")

        for ts_idx, ts in enumerate(timeseries_list):
            T, N_rois = ts.shape

            if T < window_size:
                continue

            num_windows = (T - window_size) // stride + 1

            for roi_idx in range(N_rois):
                roi_signal = ts[:, roi_idx]

                for w in range(num_windows):
                    start = w * stride
                    end = start + window_size
                    chunk = roi_signal[start:end]

                    # Z-score within window
                    mean = np.mean(chunk)
                    std = np.std(chunk) + 1e-6
                    chunk_norm = (chunk - mean) / std

                    if np.any(np.isnan(chunk_norm)) or np.any(np.isinf(chunk_norm)):
                        continue

                    self.chunks.append(chunk_norm)

            if (ts_idx + 1) % 50 == 0:
                print(f"  处理进度: {ts_idx + 1}/{len(timeseries_list)}")

        self.chunks = np.array(self.chunks, dtype=np.float32)

        print(f"\n✓ 数据准备完成")
        print(f"  总样本数: {len(self.chunks)}")
        print(f"  样本形状: {self.chunks.shape}")
        print(f"  数据统计: mean={self.chunks.mean():.4f}, std={self.chunks.std():.4f}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.chunks[idx])


class MAE_Encoder(nn.Module):
    """1D-CNN Encoder with Input Batch Normalization"""

    def __init__(self, input_length, embedding_dim=64, dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim

        # 🔥 关键修改：输入层BatchNorm
        self.input_bn = nn.BatchNorm1d(1)

        # 3层1D-CNN
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, L]

        # 🔥 输入归一化
        x = self.input_bn(x)

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


class MAE_Decoder(nn.Module):
    """1D-CNN Transpose Decoder"""

    def __init__(self, embedding_dim, output_length, dropout=0.1):
        super().__init__()

        self.output_length = output_length

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128 * (output_length // 4))
        )

        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)

        self.deconv3 = nn.Conv1d(32, 1, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, embedding):
        x = self.fc(embedding)
        x = x.view(x.size(0), 128, -1)

        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)

        if x.size(2) != self.output_length:
            x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=False)

        return x.squeeze(1)


class MaskedAutoencoder(nn.Module):
    """Masked Autoencoder"""

    def __init__(self, input_length, embedding_dim=64, mask_ratio=0.6):  # 🔥 mask_ratio改为0.6
        super().__init__()

        self.input_length = input_length
        self.mask_ratio = mask_ratio

        self.encoder = MAE_Encoder(input_length, embedding_dim)
        self.decoder = MAE_Decoder(embedding_dim, input_length)

    def create_mask(self, x):
        B, L = x.shape
        mask = torch.rand(B, L, device=x.device) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0
        return x_masked, mask

    def forward(self, x):
        x_masked, mask = self.create_mask(x)
        embedding = self.encoder(x_masked)
        reconstructed = self.decoder(embedding)
        return reconstructed, mask


def train_mae_offline(timeseries_list,
                      window_size=64,
                      embedding_dim=64,
                      mask_ratio=0.6,  # 🔥 默认0.6
                      epochs=50,
                      batch_size=128,
                      lr=1e-3,
                      device='cuda',
                      save_dir='./pretrained_models'):
    """离线MAE预训练（增强版）"""

    print("\n" + "="*80)
    print("离线节点预训练 - Masked Autoencoder (Enhanced)")
    print("="*80)
    print(f"窗口大小: {window_size}")
    print(f"Embedding维度: {embedding_dim}")
    print(f"Mask比例: {mask_ratio} (增强版)")  # 🔥
    print(f"Input BatchNorm: ✓ Enabled")  # 🔥
    print(f"Batch大小: {batch_size}")
    print(f"学习率: {lr}")
    print(f"训练轮数: {epochs}")
    print(f"设备: {device}")
    print("="*80)

    # 准备数据
    dataset = TimeSeriesWindowDataset(
        timeseries_list,
        window_size=window_size,
        stride=window_size // 2
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 创建模型
    model = MaskedAutoencoder(
        input_length=window_size,
        embedding_dim=embedding_dim,
        mask_ratio=mask_ratio
    ).to(device)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
        betas=(0.9, 0.95)
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.01
    )

    # 训练
    best_loss = float('inf')
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_x in dataloader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()

            # Forward
            reconstructed, mask = model(batch_x)

            # 只计算被mask位置的重建损失
            loss = F.mse_loss(reconstructed[mask], batch_x[mask])

            # Backward
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        scheduler.step()

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss

            os.makedirs(save_dir, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'input_length': window_size,
                    'embedding_dim': embedding_dim,
                    'mask_ratio': mask_ratio
                }
            }

            save_path = os.path.join(save_dir, 'node_encoder_best.pth')
            torch.save(checkpoint, save_path)

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.6f} "
                  f"(Best={best_loss:.6f}) LR={optimizer.param_groups[0]['lr']:.6f}")

    print(f"\n✓ 预训练完成！")
    print(f"  最佳损失: {best_loss:.6f}")
    print(f"  模型保存至: {save_path}")

    visualize_reconstruction(model, dataset, device, save_dir)

    return model, best_loss


def visualize_reconstruction(model, dataset, device, save_dir):
    """可视化重建效果"""
    import matplotlib.pyplot as plt

    print("\n生成重建可视化...")

    model.eval()

    indices = np.random.choice(len(dataset), 5, replace=False)

    fig, axes = plt.subplots(5, 1, figsize=(12, 10))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            x = dataset[idx].unsqueeze(0).to(device)

            reconstructed, mask = model(x)

            x = x.cpu().numpy()[0]
            reconstructed = reconstructed.cpu().numpy()[0]
            mask = mask.cpu().numpy()[0]

            axes[i].plot(x, 'b-', label='Original', linewidth=1.5)
            axes[i].plot(reconstructed, 'r--', label='Reconstructed', linewidth=1.5, alpha=0.8)

            mask_indices = np.where(mask)[0]
            axes[i].scatter(mask_indices, x[mask_indices], c='orange', s=10,
                           label='Masked points', zorder=5)

            axes[i].set_title(f'Sample {i+1}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'reconstruction_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 可视化保存至: {save_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='离线节点预训练（增强版）')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ABIDE', 'MDD', 'BOTH'])
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--mask_ratio', type=float, default=0.6)  # 🔥 默认0.6
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./pretrained_models')

    args = parser.parse_args()

    # 加载数据
    timeseries_list = []

    if args.dataset in ['ABIDE', 'BOTH']:
        from abide_data_baseline import ABIDEBaselineProcessor
        processor = ABIDEBaselineProcessor(data_folder=args.data_folder)
        ts_abide, _, _, _ = processor.download_and_extract(n_subjects=None, apply_zscore=True)
        timeseries_list.extend(ts_abide)
        print(f"加载 ABIDE: {len(ts_abide)} 个被试")

    if args.dataset in ['MDD', 'BOTH']:
        from mdd_data_baseline import MDDBaselineProcessor
        processor = MDDBaselineProcessor(data_folder=args.data_folder)
        ts_mdd, _, _, _ = processor.load_roi_signals(apply_zscore=True)
        timeseries_list.extend(ts_mdd)
        print(f"加载 MDD: {len(ts_mdd)} 个被试")

    print(f"\n总共 {len(timeseries_list)} 个被试用于预训练")

    # 开始训练
    train_mae_offline(
        timeseries_list=timeseries_list,
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        mask_ratio=args.mask_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir
    )