"""
使用预训练的Node Encoder提取节点特征
修改：增加滑窗+平均池化策略
"""

import torch
import numpy as np
from node_pretrain import MAE_Encoder


def sliding_window_inference(timeseries, encoder, window_size, stride, device):
    """
    滑窗推理+平均池化

    Args:
        timeseries: (T,) 单个ROI的时间序列
        encoder: 预训练的encoder
        window_size: 窗口大小
        stride: 滑动步长
        device: 设备

    Returns:
        embedding: (embedding_dim,) ROI的特征向量
    """
    T = len(timeseries)

    if T < window_size:
        # 如果序列太短，padding到window_size
        padded = np.zeros(window_size)
        padded[:T] = timeseries
        timeseries = padded
        T = window_size

    # 滑窗切片
    num_windows = (T - window_size) // stride + 1
    embeddings = []

    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        chunk = timeseries[start:end]

        # Z-score标准化（与预训练保持一致）
        mean = np.mean(chunk)
        std = np.std(chunk) + 1e-6
        chunk_norm = (chunk - mean) / std

        # 转tensor
        chunk_tensor = torch.FloatTensor(chunk_norm).unsqueeze(0).to(device)  # [1, window_size]

        # 编码
        with torch.no_grad():
            embedding = encoder(chunk_tensor)  # [1, embedding_dim]

        embeddings.append(embedding.cpu().numpy())

    # 平均池化
    embeddings = np.vstack(embeddings)  # [num_windows, embedding_dim]
    avg_embedding = np.mean(embeddings, axis=0)  # [embedding_dim]

    return avg_embedding


def extract_node_features_pretrained(timeseries_list, encoder_path,
                                     embedding_dim=64, device='cuda'):
    """
    使用预训练encoder提取节点特征

    Args:
        timeseries_list: 时间序列列表 [N_subjects, (T, N_ROI)]
        encoder_path: 预训练encoder路径
        embedding_dim: embedding维度
        device: 计算设备

    Returns:
        features_list: 特征列表 [N_subjects, (N_ROI, embedding_dim)]
    """
    print(f"\n{'=' * 80}")
    print("使用预训练Encoder提取节点特征")
    print(f"{'=' * 80}")
    print(f"Encoder路径: {encoder_path}")

    # 加载预训练checkpoint
    checkpoint = torch.load(encoder_path, map_location=device)
    config = checkpoint['config']

    window_size = config['input_length']
    stride = window_size // 2  # 50%重叠

    print(f"  ✓ 加载checkpoint (epoch {checkpoint['epoch']})")
    print(f"  窗口大小: {window_size}")
    print(f"  Embedding维度: {embedding_dim}")
    print(f"  滑动步长: {stride}")

    # 创建encoder并加载权重
    encoder = MAE_Encoder(
        input_length=window_size,
        embedding_dim=embedding_dim
    ).to(device)

    # 只加载encoder部分
    encoder_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('encoder.'):
            encoder_state_dict[k.replace('encoder.', '')] = v

    encoder.load_state_dict(encoder_state_dict)
    # encoder.train()
    # print("⚠️ 注意: 已强制模型进入 .train() 模式以避开 Batch Norm 统计量偏移问题")

    print(f"  ✓ Encoder加载完成")

    # 提取特征
    features_list = []

    for i, ts in enumerate(timeseries_list):
        if ts.shape[0] == 116 and ts.shape[1] != 116:
            # 假设 Time > 116，或者单纯就是反了
            ts = ts.T

        if ts.shape[1] != 116:
            print(f"❌ 严重警告: 样本 {i} 的形状不对! Shape={ts.shape}. 期望第二个维度是116(ROIs)")
        # ts shape: (T, N_ROI)
        T, n_roi = ts.shape

        roi_features = []

        for roi_idx in range(n_roi):
            roi_signal = ts[:, roi_idx]  # (T,)

            # 滑窗推理+平均池化
            embedding = sliding_window_inference(
                timeseries=roi_signal,
                encoder=encoder,
                window_size=window_size,
                stride=stride,
                device=device
            )

            roi_features.append(embedding)

        features = np.vstack(roi_features)  # [N_ROI, embedding_dim]
        features_list.append(features)

        if (i + 1) % 50 == 0:
            print(f"  处理进度: {i + 1}/{len(timeseries_list)}")

    print(f"\n✓ 特征提取完成")
    print(f"  被试数: {len(features_list)}")
    print(f"  特征形状: {features_list[0].shape}")

    return features_list


if __name__ == '__main__':
    # 测试代码
    from abide_data_baseline import ABIDEBaselineProcessor

    processor = ABIDEBaselineProcessor()
    timeseries_list, labels, _, _ = processor.download_and_extract(n_subjects=10)

    features = extract_node_features_pretrained(
        timeseries_list,
        encoder_path='./pretrained_models/node_encoder_best.pth',
        embedding_dim=64,
        device='cuda'
    )

    print(f"\n提取的特征形状: {features[0].shape}")