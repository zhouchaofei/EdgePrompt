"""
使用预训练的Node Encoder提取节点特征
"""

import torch
import numpy as np
from node_pretrain import NodeEncoder


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
    print("Extracting Node Features with Pretrained Encoder")
    print(f"{'=' * 80}")
    print(f"Encoder: {encoder_path}")

    # 加载预训练encoder
    # checkpoint = torch.load(encoder_path, map_location=device)
    # config = checkpoint['config']
    #
    # input_length = config['input_length']
    checkpoint = torch.load(encoder_path)
    config = checkpoint['config']

    input_length = config['input_length']  # ✅ 使用保存的长度

    # ✅ 检查长度兼容性
    actual_min_length = min(ts.shape[0] for ts in timeseries_list)
    if actual_min_length < input_length:
        print(f"⚠️  Warning: Data length {actual_min_length} < "
              f"pretrained length {input_length}")
        print(f"   Using shorter length: {actual_min_length}")
        input_length = actual_min_length

    encoder = NodeEncoder(
        input_length=input_length,
        embedding_dim=embedding_dim
    ).to(device)

    # 加载权重（只加载encoder部分）
    state_dict = checkpoint['model_state_dict']
    encoder_state_dict = {
        k.replace('encoder.', ''): v
        for k, v in state_dict.items()
        if k.startswith('encoder.')
    }
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()

    print(f"✓ Encoder loaded from epoch {checkpoint['epoch']}")
    print(f"  Input length: {input_length}")
    print(f"  Embedding dim: {embedding_dim}")

    # 提取特征
    features_list = []

    with torch.no_grad():
        for i, ts in enumerate(timeseries_list):
            # 截断到相同长度
            ts = ts[:input_length, :]
            n_roi = ts.shape[1]

            roi_features = []

            for roi_idx in range(n_roi):
                roi_signal = ts[:, roi_idx]
                roi_tensor = torch.FloatTensor(roi_signal).unsqueeze(0).to(device)

                # 提取embedding
                embedding = encoder(roi_tensor)
                roi_features.append(embedding.cpu().numpy())

            features = np.vstack(roi_features)  # [N_ROI, embedding_dim]
            features_list.append(features)

            if (i + 1) % 50 == 0:
                print(f"  Processed: {i + 1}/{len(timeseries_list)}")

    print(f"\n✓ Feature extraction completed")
    print(f"  Number of subjects: {len(features_list)}")
    print(f"  Feature shape per subject: {features_list[0].shape}")

    return features_list


if __name__ == '__main__':
    # 测试代码
    from abide_data_baseline import ABIDEBaselineProcessor

    processor = ABIDEBaselineProcessor()
    timeseries_list, labels, _, _ = processor.download_and_extract(n_subjects=10)

    features = extract_node_features_pretrained(
        timeseries_list,
        encoder_path='./pretrained_models/ABIDE_node_encoder.pth',
        embedding_dim=64,
        device='cuda'
    )

    print(f"\nExtracted features shape: {features[0].shape}")