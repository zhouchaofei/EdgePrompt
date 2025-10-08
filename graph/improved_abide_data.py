"""
改进的ABIDE数据处理：结合两种方法的优点
"""
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from nilearn.datasets import fetch_abide_pcp
from sklearn.preprocessing import StandardScaler
from scipy import stats, signal
from sklearn.covariance import LedoitWolf
import warnings

warnings.filterwarnings('ignore')


class ImprovedABIDEProcessor:
    """
    改进的ABIDE处理器
    核心改进：
    1. 节点特征使用原始时间序列
    2. 邻接矩阵结合解剖和功能信息
    3. 不丢弃时序信息
    """

    def __init__(self, data_folder='./data',
                 atlas='ho',
                 time_strategy='adaptive',
                 min_time_length=78):
        """
        Args:
            time_strategy:
                - 'fixed': 固定截断到min_time_length
                - 'adaptive': 自适应找最小长度
                - 'pad': 填充到最大长度
        """
        self.data_folder = data_folder
        self.atlas = atlas
        self.time_strategy = time_strategy
        self.min_time_length = min_time_length
        self.abide_path = os.path.join(data_folder, 'ABIDE')
        self.processed_path = os.path.join(self.abide_path, 'processed_v2')

        os.makedirs(self.processed_path, exist_ok=True)

    def download_data(self, n_subjects=None):
        """下载ABIDE数据"""
        print(f"下载ABIDE数据集...")

        data = fetch_abide_pcp(
            data_dir=self.abide_path,
            pipeline='cpac',
            band_pass_filtering=True,
            global_signal_regression=False,  # 不做全局信号回归
            derivatives='rois_ho',
            n_subjects=n_subjects,
            verbose=1
        )

        return data

    def determine_time_length(self, rois_data):
        """确定统一的时间长度"""
        lengths = []

        for idx, roi_file in enumerate(rois_data[:50]):  # 检查前50个
            try:
                if isinstance(roi_file, str) and os.path.exists(roi_file):
                    ts = pd.read_csv(roi_file, sep='\t', header=0).values
                    lengths.append(ts.shape[0])
            except:
                continue

        if not lengths:
            return self.min_time_length

        lengths = np.array(lengths)

        if self.time_strategy == 'fixed':
            return self.min_time_length
        elif self.time_strategy == 'adaptive':
            # 使用10th percentile，保留90%数据的完整性
            return int(np.percentile(lengths, 10))
        elif self.time_strategy == 'pad':
            return int(np.max(lengths))
        else:
            return self.min_time_length

    def process_time_series(self, time_series, target_length):
        """
        处理时间序列到目标长度

        Args:
            time_series: [T, N] 原始时间序列
            target_length: 目标长度

        Returns:
            processed: [target_length, N]
        """
        T, N = time_series.shape

        if self.time_strategy == 'pad':
            if T >= target_length:
                return time_series[:target_length, :]
            else:
                # 零填充
                padded = np.zeros((target_length, N))
                padded[:T, :] = time_series
                return padded
        else:
            # 截断或采样
            if T >= target_length:
                return time_series[:target_length, :]
            else:
                # 上采样
                from scipy.interpolate import interp1d
                t_old = np.linspace(0, 1, T)
                t_new = np.linspace(0, 1, target_length)

                resampled = np.zeros((target_length, N))
                for i in range(N):
                    f = interp1d(t_old, time_series[:, i], kind='cubic')
                    resampled[:, i] = f(t_new)

                return resampled

    def construct_anatomical_adj(self, n_regions=116):
        """
        构建基于解剖的邻接矩阵（借鉴construct_graph.py）

        Args:
            n_regions: 脑区数量（AAL=116, HO=111）

        Returns:
            adj: [n_regions, n_regions] 二值邻接矩阵
        """
        adj = np.zeros((n_regions, n_regions))

        # AAL图谱的左右脑划分
        # AAL: 1-45左脑，46-90右脑，91-116中线/小脑
        if n_regions == 116:
            left_indices = list(range(0, 45))
            right_indices = list(range(45, 90))
            midline_indices = list(range(90, 116))
        elif n_regions == 111:  # HO图谱
            # 简化：假设前55左脑，后55右脑，最后1个全局
            left_indices = list(range(0, 55))
            right_indices = list(range(55, 110))
            midline_indices = [110]
        else:
            raise ValueError(f"不支持的脑区数量: {n_regions}")

        # 构建跨半球连接（借鉴construct_graph.py的设计）
        for i in range(n_regions):
            if i in left_indices:
                # 左脑连接右脑和中线
                adj[i, right_indices] = 1
                adj[i, midline_indices] = 1
            elif i in right_indices:
                # 右脑连接左脑和中线
                adj[i, left_indices] = 1
                adj[i, midline_indices] = 1
            else:
                # 中线连接所有非中线
                adj[i, left_indices] = 1
                adj[i, right_indices] = 1

        # 对称化
        adj = (adj + adj.T) / 2
        adj[adj > 0] = 1
        np.fill_diagonal(adj, 0)

        return adj

    def construct_functional_weights(self, time_series, adj_anatomical):
        """
        为解剖边计算功能连接权重

        Args:
            time_series: [T, N]
            adj_anatomical: [N, N] 解剖邻接矩阵

        Returns:
            edge_weights: dict {(i,j): weight}
        """
        # 计算相关矩阵
        corr = np.corrcoef(time_series.T)
        corr = np.nan_to_num(corr, 0)

        # 相位同步（ASD关键）
        phase_sync = self._compute_phase_sync(time_series)

        # 融合
        func_conn = (corr + phase_sync) / 2

        # 为已有的解剖边赋予功能权重
        edge_weights = {}
        edges = np.where(adj_anatomical > 0)

        for i, j in zip(edges[0], edges[1]):
            edge_weights[(i, j)] = func_conn[i, j]

        return edge_weights

    def _compute_phase_sync(self, time_series):
        """计算相位同步矩阵"""
        from scipy.signal import hilbert

        n_regions = time_series.shape[1]
        phase_sync = np.zeros((n_regions, n_regions))

        # 计算瞬时相位
        phases = np.angle(hilbert(time_series, axis=0))

        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                try:
                    phase_diff = phases[:, i] - phases[:, j]
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    phase_sync[i, j] = phase_sync[j, i] = plv
                except:
                    phase_sync[i, j] = phase_sync[j, i] = 0

        return phase_sync

    def construct_dual_stream_graph(self, time_series, label):
        """
        构建双流图（结构+功能）

        核心改进：节点特征使用原始时间序列！

        Args:
            time_series: [T, N] 时间序列
            label: 标签

        Returns:
            func_data: 功能流图
            struct_data: 结构流图
        """
        T, N = time_series.shape

        # 节点特征：直接使用时间序列（关键改进！）
        # 转置：[N, T] 每个节点是T维时间序列
        node_features = torch.tensor(time_series.T, dtype=torch.float)

        # ==========================================
        # 结构流：解剖邻接 + 时序节点特征
        # ==========================================
        struct_adj = self.construct_anatomical_adj(N)

        struct_edge_index = []
        for i in range(N):
            for j in range(N):
                if struct_adj[i, j] > 0:
                    struct_edge_index.append([i, j])

        struct_edge_index = torch.tensor(struct_edge_index, dtype=torch.long).t()

        struct_data = Data(
            x=node_features,  # [N, T] - 时序特征！
            edge_index=struct_edge_index,
            y=torch.tensor([label], dtype=torch.long)
        )

        # ==========================================
        # 功能流：解剖邻接 + 功能权重 + 时序节点特征
        # ==========================================
        edge_weights = self.construct_functional_weights(time_series, struct_adj)

        func_edge_index = []
        func_edge_attr = []

        for (i, j), weight in edge_weights.items():
            if weight > 0:  # 只保留正权重
                func_edge_index.append([i, j])
                func_edge_attr.append(weight)

        func_edge_index = torch.tensor(func_edge_index, dtype=torch.long).t()
        func_edge_attr = torch.tensor(func_edge_attr, dtype=torch.float)

        func_data = Data(
            x=node_features,  # [N, T] - 同样的时序特征！
            edge_index=func_edge_index,
            edge_attr=func_edge_attr,
            y=torch.tensor([label], dtype=torch.long)
        )

        return func_data, struct_data

    def process_and_save(self, n_subjects=None):
        """处理并保存数据"""
        # 下载数据
        data = self.download_data(n_subjects)

        # 获取ROI数据
        rois_data = data.rois_ho
        labels = data.phenotypic['DX_GROUP'].values - 1

        # 确定时间长度
        target_length = self.determine_time_length(rois_data)
        print(f"目标时间长度: {target_length}")

        dual_stream_data = []

        for idx, roi_file in enumerate(rois_data):
            try:
                # 加载时间序列
                if isinstance(roi_file, str) and os.path.exists(roi_file):
                    time_series = pd.read_csv(roi_file, sep='\t', header=0).values
                elif isinstance(roi_file, np.ndarray):
                    time_series = roi_file
                else:
                    continue

                # 处理到目标长度
                time_series = self.process_time_series(time_series, target_length)

                # 检查有效性
                if time_series.shape[0] < 50:
                    continue

                label = labels[idx]

                # 构建双流图
                func_data, struct_data = self.construct_dual_stream_graph(
                    time_series, label
                )

                dual_stream_data.append((func_data, struct_data))

                if (idx + 1) % 20 == 0:
                    print(f"已处理: {idx + 1}/{len(rois_data)}")

            except Exception as e:
                print(f"处理被试{idx}失败: {e}")
                continue

        print(f"\n成功构建 {len(dual_stream_data)} 个双流样本")

        # 保存
        save_path = os.path.join(
            self.processed_path,
            f'abide_dual_stream_{self.time_strategy}_{target_length}.pt'
        )
        torch.save(dual_stream_data, save_path)
        print(f"数据已保存至: {save_path}")

        # 保存元信息
        meta_info = {
            'n_subjects': len(dual_stream_data),
            'time_length': target_length,
            'n_regions': dual_stream_data[0][0].x.shape[0],
            'time_strategy': self.time_strategy,
        }

        meta_path = os.path.join(self.processed_path, 'meta_info.pt')
        torch.save(meta_info, meta_path)

        return dual_stream_data


if __name__ == "__main__":
    print("=" * 60)
    print("改进的ABIDE数据处理")
    print("=" * 60)

    # 测试不同策略
    strategies = ['adaptive', 'fixed']

    for strategy in strategies:
        print(f"\n策略: {strategy}")
        processor = ImprovedABIDEProcessor(
            data_folder='./data',
            time_strategy=strategy,
            min_time_length=78
        )

        dual_stream = processor.process_and_save(n_subjects=None)

        if dual_stream:
            print(f"\n样本信息:")
            print(f"  节点特征维度: {dual_stream[0][0].x.shape}")
            print(f"  边数（功能流）: {dual_stream[0][0].edge_index.shape[1]}")
            print(f"  边数（结构流）: {dual_stream[0][1].edge_index.shape[1]}")