"""
混合特征提取器 - 终极增强版
策略：
1. 预训练特征采用 Mean+Std+Max 池化，保留动态信息
2. 强制拼接统计特征 (FC Profile)，保证下限
"""

import torch
import numpy as np
from node_pretrain import MAE_Encoder
import warnings


def extract_statistical_features_single(timeseries):
    """
    提取单个被试的统计特征（FC Profile）

    Args:
        timeseries: (T, N_ROI) 时间序列

    Returns:
        fc: (N_ROI, N_ROI) 功能连接矩阵，每行是该ROI与其他ROI的相关性
    """
    # FC Profile (连接指纹) - 非常强的特征
    fc = np.corrcoef(timeseries.T)  # (N_ROI, N_ROI)
    np.fill_diagonal(fc, 0)

    return fc


def sliding_window_inference(timeseries, encoder, window_size, stride, device):
    """
    滑窗推理 + 多重池化 (Mean/Std/Max)

    Args:
        timeseries: (T,) 单个ROI的时间序列
        encoder: 预训练的encoder
        window_size: 窗口大小
        stride: 滑动步长
        device: 设备

    Returns:
        combined_embedding: (embedding_dim * 3,) 包含Mean/Std/Max的综合特征
    """
    T = len(timeseries)

    if T < window_size:
        padded = np.zeros(window_size)
        padded[:T] = timeseries
        timeseries = padded
        T = window_size

    num_windows = (T - window_size) // stride + 1
    embeddings = []

    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        chunk = timeseries[start:end]

        # Z-score within window
        mean = np.mean(chunk)
        std = np.std(chunk) + 1e-6
        chunk_norm = (chunk - mean) / std

        chunk_tensor = torch.FloatTensor(chunk_norm).unsqueeze(0).to(device)  # [1, L]

        with torch.no_grad():
            embedding = encoder(chunk_tensor)  # [1, D]

        embeddings.append(embedding.cpu().numpy())

    # [Windows, D]
    embeddings = np.vstack(embeddings)

    # 🔥 多重池化，保留动态信息
    emb_mean = np.mean(embeddings, axis=0)  # [D]
    emb_std = np.std(embeddings, axis=0)    # [D]
    emb_max = np.max(embeddings, axis=0)    # [D]

    # 拼接: [D*3]
    return np.concatenate([emb_mean, emb_std, emb_max])


def extract_node_features_pretrained(timeseries_list, encoder_path,
                                     embedding_dim=64, device='cuda'):
    """
    提取混合特征 (Hybrid Features = Deep Features + Statistical Features)

    Args:
        timeseries_list: 时间序列列表 [N_subjects, (T, N_ROI)]
        encoder_path: 预训练encoder路径
        embedding_dim: embedding维度
        device: 计算设备

    Returns:
        features_list: 混合特征列表 [N_subjects, (N_ROI, feature_dim)]
                      feature_dim = embedding_dim*3 + N_ROI (如果有预训练模型)
                      或 feature_dim = N_ROI (仅统计特征)
    """
    print(f"\n{'='*80}")
    print("提取混合节点特征 (Temporal + Statistical)")
    print(f"{'='*80}")

    # 1. 尝试加载预训练模型
    has_encoder = False
    encoder = None
    window_size = None
    stride = None

    try:
        checkpoint = torch.load(encoder_path, map_location=device)
        config = checkpoint['config']
        window_size = config['input_length']
        stride = window_size // 2

        encoder = MAE_Encoder(input_length=window_size, embedding_dim=embedding_dim).to(device)

        # 加载权重
        encoder_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('encoder.'):
                encoder_state_dict[k.replace('encoder.', '')] = v
        encoder.load_state_dict(encoder_state_dict)
        encoder.eval()  # 推理模式

        has_encoder = True
        print("  ✓ 预训练模型加载成功，将提取 Deep Features + Statistical Features")
        print(f"    窗口大小: {window_size}")
        print(f"    Embedding维度: {embedding_dim}")
        print(f"    滑动步长: {stride}")
    except Exception as e:
        print(f"  ⚠️ 模型加载失败: {e}")
        print("  ⚠️ 将只使用统计特征 (Statistical Features Only)")

    features_list = []

    for i, ts in enumerate(timeseries_list):
        # 确保形状正确: (T, N_ROI)
        if ts.shape[0] == 116 and ts.shape[1] != 116:
            ts = ts.T

        T, n_roi = ts.shape

        # A. 提取统计特征 (Base Feature)
        # shape: (N_ROI, N_ROI) - 每一行是该ROI与其他ROI的相关性
        stat_feat = extract_statistical_features_single(ts)

        # B. 提取时序特征 (Deep Feature)
        if has_encoder:
            deep_feats = []
            for roi_idx in range(n_roi):
                # 多重池化: [embedding_dim * 3]
                emb = sliding_window_inference(
                    ts[:, roi_idx], encoder, window_size, stride, device
                )
                deep_feats.append(emb)
            deep_feats = np.array(deep_feats)  # (N_ROI, embedding_dim*3)

            # 🔥 C. 特征融合
            # 最终特征 = [Deep(embedding_dim*3) + Stat(N_ROI)] = (embedding_dim*3 + N_ROI)维
            # 例如: embedding_dim=64 -> Deep=192, Stat=116 -> Total=308维
            combined = np.column_stack([deep_feats, stat_feat])
        else:
            # 仅统计特征
            combined = stat_feat

        features_list.append(combined)

        if (i + 1) % 50 == 0:
            print(f"  处理进度: {i + 1}/{len(timeseries_list)}")

    features_list = np.array(features_list)
    print(f"\n✓ 特征提取完成")
    print(f"  被试数: {len(features_list)}")
    print(f"  特征形状: {features_list[0].shape}")  # 应该是 (116, 308) 或 (116, 116)
    print(f"  特征类型: {'Hybrid (Deep+Stat)' if has_encoder else 'Statistical Only'}")

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