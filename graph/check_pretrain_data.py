"""
诊断预训练数据的问题
"""
import torch
import numpy as np
from torch_geometric.data import DataLoader


def diagnose_dataset(data_path, dataset_name):
    """诊断数据集"""
    print(f"\n{'=' * 60}")
    print(f"诊断数据集: {dataset_name}")
    print(f"{'=' * 60}")

    # 加载数据
    data = torch.load(data_path)

    # 检查是否是双流
    if isinstance(data[0], tuple):
        print("数据类型: 双流")
        func_list = [func for func, struct in data]
    else:
        print("数据类型: 单流")
        func_list = data

    print(f"\n1. 样本数量: {len(func_list)}")

    # 检查维度
    sample = func_list[0]
    print(f"2. 节点特征维度: {sample.x.shape}")
    print(f"3. 边数量: {sample.edge_index.shape[1]}")

    # 统计数据值范围
    all_values = []
    for data in func_list[:10]:  # 检查前10个样本
        all_values.extend(data.x.flatten().numpy())

    all_values = np.array(all_values)
    print(f"\n4. 数据统计:")
    print(f"   均值: {all_values.mean():.6f}")
    print(f"   标准差: {all_values.std():.6f}")
    print(f"   最小值: {all_values.min():.6f}")
    print(f"   最大值: {all_values.max():.6f}")
    print(f"   范围: [{all_values.min():.6f}, {all_values.max():.6f}]")

    # 检查是否有NaN或Inf
    print(f"\n5. 数据质量:")
    print(f"   包含NaN: {np.isnan(all_values).any()}")
    print(f"   包含Inf: {np.isinf(all_values).any()}")

    # 计算理论重建损失下界
    print(f"\n6. 理论损失估计:")
    mse_baseline = np.mean((all_values - all_values.mean()) ** 2)
    print(f"   基线MSE（预测均值）: {mse_baseline:.4f}")

    # DataLoader测试
    print(f"\n7. DataLoader信息:")
    loader = DataLoader(func_list, batch_size=32, shuffle=False)
    print(f"   Batch数量: {len(loader)}")
    print(f"   每个batch平均样本数: {len(func_list) / len(loader):.1f}")

    return {
        'n_samples': len(func_list),
        'feature_dim': sample.x.shape,
        'mean': all_values.mean(),
        'std': all_values.std(),
        'min': all_values.min(),
        'max': all_values.max(),
        'n_batches': len(loader)
    }


if __name__ == "__main__":
    # 检查ABIDE
    abide_stats = diagnose_dataset(
        './data/ABIDE/processed/abide_dual_stream_temporal_78.pt',
        'ABIDE'
    )

    # 检查MDD
    mdd_stats = diagnose_dataset(
        './data/REST-meta-MDD/processed/mdd_dual_stream_temporal_150.pt',
        'MDD'
    )

    # 对比
    print(f"\n{'=' * 60}")
    print(f"对比分析")
    print(f"{'=' * 60}")

    print(f"\n样本数量比例: {abide_stats['n_samples'] / mdd_stats['n_samples']:.2f}")
    print(f"特征维度比例: {abide_stats['feature_dim'][1] / mdd_stats['feature_dim'][1]:.2f}")
    print(f"数据标准差比例: {abide_stats['std'] / mdd_stats['std']:.2f}")
    print(f"Batch数量比例: {abide_stats['n_batches'] / mdd_stats['n_batches']:.2f}")