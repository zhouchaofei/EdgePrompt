# 让我们先深入分析这个.mat文件
import scipy.io as sio
import numpy as np

# 加载文件
mat_data = sio.loadmat('data/REST-meta-MDD/Results/ROISignals_FunImgARCWF/ROISignals_S1-1-0015.mat')

# 查看所有键值
print("所有键:", mat_data.keys())

# 查看ROISignals的详细信息
roi_signals = mat_data['ROISignals']
print(f"Shape: {roi_signals.shape}")
print(f"数据类型: {roi_signals.dtype}")

# 检查是否有其他有用信息
for key in mat_data.keys():
    if not key.startswith('__'):
        data = mat_data[key]
        print(f"\n{key}:")
        print(f"  类型: {type(data)}")
        if isinstance(data, np.ndarray):
            print(f"  Shape: {data.shape}")
            print(f"  前几个值: {data.flat[:5]}")

# 可能的情况分析：
# 1. 1833 = 116 ROI × 多个特征统计量
# 2. 包含多个被试数据
# 3. 使用不同的ROI模板（如AAL-1833是错误的）
# 4. 数据结构：[时间点, 特征]但特征不是简单的ROI

# 让我们检查是否能找到116的规律
print(f"\n1833除以可能的因子:")
for i in [90, 116, 160, 200, 246]:
    if 1833 % i == 0:
        print(f"  1833 / {i} = {1833 / i}")
    else:
        print(f"  1833 / {i} = {1833 / i:.2f} (不整除)")