import torch
import os

# 检查ABIDE
abide_path = './data/ABIDE/processed/abide_dual_stream_temporal_78.pt'
if os.path.exists(abide_path):
    data = torch.load(abide_path)
    print(f'ABIDE样本数: {len(data)}')
    func, struct = data[0]
    print(f'功能图节点特征: {func.x.shape}')
    print(f'结构图节点特征: {struct.x.shape}')
else:
    print('ABIDE数据未找到！')

# 检查MDD
mdd_path = './data/REST-meta-MDD/processed/mdd_dual_stream_temporal_150.pt'
if os.path.exists(mdd_path):
    data = torch.load(mdd_path)
    print(f'MDD样本数: {len(data)}')
    func, struct = data[0]
    print(f'功能图节点特征: {func.x.shape}')
    print(f'结构图节点特征: {struct.x.shape}')
else:
    print('MDD数据未找到！')