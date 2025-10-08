"""
ABIDE数据集处理模块 - 整合版
支持：
1. 传统单流图（基线实验）
2. 双流图（SF-DPL实验）
3. 时序特征 + 统计特征
"""
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from nilearn.datasets import fetch_abide_pcp
from sklearn.preprocessing import StandardScaler
from scipy import stats, signal
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


class ABIDEDataProcessor:
    """
    统一的ABIDE数据处理器

    核心功能：
    1. 下载和预处理数据
    2. 生成多种格式的图数据
    3. 支持不同的节点特征类型
    """

    def __init__(self,
                 data_folder='./data',
                 pipeline='cpac',
                 atlas='aal',  # 使用AAL图谱
                 time_strategy='adaptive',
                 min_time_length=78):
        """
        Args:
            time_strategy:
                - 'adaptive': 自适应确定时间长度
                - 'fixed': 固定到min_time_length
        """

        self.data_folder = data_folder
        self.pipeline = pipeline
        self.atlas = atlas
        self.time_strategy = time_strategy
        self.min_time_length = min_time_length

        self.abide_path = os.path.join(data_folder, 'ABIDE')
        self.processed_path = os.path.join(self.abide_path, 'processed')

        os.makedirs(self.processed_path, exist_ok=True)

        # 验证图谱类型
        if self.atlas not in ['aal']:
            print(f"警告：为了与MDD统一，建议使用AAL图谱")
            print(f"当前使用: {self.atlas}")

    def download_data(self, n_subjects=None):
        """下载ABIDE数据 - AAL图谱"""
        print(f"下载ABIDE数据集...")
        print(f"参数: pipeline={self.pipeline}, atlas={self.atlas}")

        # 确定derivatives
        if self.atlas == 'aal':
            derivatives = 'rois_aal'
            print("使用AAL-116图谱（与MDD数据集统一）")
        elif self.atlas == 'ho':
            derivatives = 'rois_ho'
            print("警告：HO图谱只有111个ROI，与MDD的AAL-116不统一")
        elif self.atlas in ['cc200', 'cc400']:
            derivatives = f'rois_{self.atlas}'
        else:
            derivatives = 'rois_aal'
            print("默认使用AAL-116图谱")

        data = fetch_abide_pcp(
            data_dir=self.abide_path,
            pipeline=self.pipeline,
            band_pass_filtering=True,
            global_signal_regression=False,
            derivatives=derivatives,
            n_subjects=n_subjects,
            verbose=1
        )

        print(f"下载完成，共 {len(data.phenotypic)} 个被试")

        return data

    def determine_time_length(self, rois_data):
        """确定统一的时间长度"""
        lengths = []

        for idx, roi_file in enumerate(rois_data[:50]):
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
            # 使用10th percentile
            return int(np.percentile(lengths, 10))
        else:
            return self.min_time_length

    def process_time_series(self, time_series, target_length):
        """处理时间序列到目标长度"""
        T, N = time_series.shape

        if T >= target_length:
            return time_series[:target_length, :]
        else:
            # 插值上采样
            from scipy.interpolate import interp1d
            t_old = np.linspace(0, 1, T)
            t_new = np.linspace(0, 1, target_length)

            resampled = np.zeros((target_length, N))
            for i in range(N):
                try:
                    f = interp1d(t_old, time_series[:, i], kind='cubic')
                    resampled[:, i] = f(t_new)
                except:
                    resampled[:, i] = np.interp(t_new, t_old, time_series[:, i])

            return resampled

    # ==========================================
    # 节点特征提取：多种方式
    # ==========================================

    def extract_temporal_features(self, time_series):
        """
        提取时序特征（用于SF-DPL）

        Args:
            time_series: [T, N]

        Returns:
            features: [N, T] 每个节点是T维时间序列
        """
        """
            提取时序特征（添加标准化）

            Args:
                time_series: [T, N]

            Returns:
                features: [N, T] 标准化后的时序特征
            """
        # 转置
        node_features = time_series.T  # [N, T] = [116, 78]

        # ==========================================
        # 关键修改：标准化每个节点的时间序列
        # ==========================================
        from sklearn.preprocessing import StandardScaler

        # 转为numpy进行标准化
        node_features_np = node_features  # 如果是torch tensor，转numpy
        if isinstance(node_features_np, torch.Tensor):
            node_features_np = node_features_np.numpy()

        # 对每个节点（脑区）的时间序列单独标准化
        scaler = StandardScaler()
        for i in range(node_features_np.shape[0]):
            # 将[T]变成[T, 1]以便标准化
            ts = node_features_np[i].reshape(-1, 1)
            node_features_np[i] = scaler.fit_transform(ts).flatten()

        # 转回torch tensor
        node_features = torch.tensor(node_features_np, dtype=torch.float)

        # 验证标准化效果
        if torch.isnan(node_features).any() or torch.isinf(node_features).any():
            print("警告：标准化后出现NaN或Inf，使用原始数据")
            return torch.tensor(time_series.T, dtype=torch.float)

        return node_features

    def extract_statistical_features(self, time_series):
        """
        提取统计特征（用于基线）

        Args:
            time_series: [T, N]

        Returns:
            features: [N, D] 每个节点是D维统计特征
        """
        features = []

        for i in range(time_series.shape[1]):
            ts = time_series[:, i]

            # 时域统计
            feat = [
                np.mean(ts),
                np.std(ts),
                np.median(ts),
                np.ptp(ts),
                stats.skew(ts) if not np.isnan(stats.skew(ts)) else 0,
                stats.kurtosis(ts) if not np.isnan(stats.kurtosis(ts)) else 0,
                np.percentile(ts, 25),
                np.percentile(ts, 75)
            ]

            # 频域特征
            try:
                freqs, psd = signal.welch(ts, fs=0.5, nperseg=min(len(ts), 64))
                freq_bands = [(0.01, 0.04), (0.04, 0.08), (0.08, 0.13)]
                for low, high in freq_bands:
                    idx = (freqs >= low) & (freqs <= high)
                    feat.append(np.mean(psd[idx]) if np.any(idx) else 0)
            except:
                feat.extend([0, 0, 0])

            features.append(feat)

        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, 0)

        # 标准化
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        return torch.tensor(features, dtype=torch.float)

    def extract_enhanced_features(self, time_series, adj_matrix):
        """
        提取增强特征（统计 + 图拓扑）

        Args:
            time_series: [T, N]
            adj_matrix: [N, N]

        Returns:
            features: [N, D+K]
        """
        # 统计特征
        stat_feat = self.extract_statistical_features(time_series)

        # 图拓扑特征
        N = time_series.shape[1]
        G = nx.from_numpy_array(np.abs(adj_matrix))

        degrees = dict(G.degree())
        clustering = nx.clustering(G)

        try:
            betweenness = nx.betweenness_centrality(G)
        except:
            betweenness = {i: 0 for i in range(N)}

        topo_feat = []
        for i in range(N):
            topo_feat.append([
                degrees[i] / N,
                clustering[i],
                betweenness[i]
            ])

        topo_feat = torch.tensor(topo_feat, dtype=torch.float)

        # 拼接
        return torch.cat([stat_feat, topo_feat], dim=1)

    # ==========================================
    # 邻接矩阵构建
    # ==========================================

    def construct_anatomical_adj(self, n_regions=116):
        """构建基于解剖的邻接矩阵"""
        adj = np.zeros((n_regions, n_regions))

        # AAL图谱的左右脑划分
        if n_regions == 116:
            left_indices = list(range(0, 45))
            right_indices = list(range(45, 90))
            midline_indices = list(range(90, 116))
        elif n_regions == 111:  # HO
            left_indices = list(range(0, 55))
            right_indices = list(range(55, 110))
            midline_indices = [110]
        else:
            # 默认划分
            left_indices = list(range(0, n_regions // 2))
            right_indices = list(range(n_regions // 2, n_regions))
            midline_indices = []

        # 双边图连接
        for i in range(n_regions):
            if i in left_indices:
                adj[i, right_indices] = 1
                adj[i, midline_indices] = 1
            elif i in right_indices:
                adj[i, left_indices] = 1
                adj[i, midline_indices] = 1
            else:
                adj[i, left_indices] = 1
                adj[i, right_indices] = 1

        # 对称化
        adj = (adj + adj.T) / 2
        adj[adj > 0] = 1
        np.fill_diagonal(adj, 0)

        return adj

    def construct_functional_adj(self, time_series, use_phase_sync=True):
        """构建基于功能连接的邻接矩阵"""
        # 相关矩阵
        corr = np.corrcoef(time_series.T)
        corr = np.nan_to_num(corr, 0)
        np.fill_diagonal(corr, 0)

        if use_phase_sync:
            # 相位同步（ASD关键特征）
            phase_sync = self._compute_phase_sync(time_series)
            func_conn = (corr + phase_sync) / 2
        else:
            func_conn = corr

        # 阈值化
        threshold = np.percentile(np.abs(func_conn), 70)
        adj = np.zeros_like(func_conn)
        adj[np.abs(func_conn) > threshold] = func_conn[np.abs(func_conn) > threshold]

        return adj

    def _compute_phase_sync(self, time_series):
        """计算相位同步矩阵"""
        from scipy.signal import hilbert

        n_regions = time_series.shape[1]
        phase_sync = np.zeros((n_regions, n_regions))

        try:
            phases = np.angle(hilbert(time_series, axis=0))

            for i in range(n_regions):
                for j in range(i+1, n_regions):
                    try:
                        phase_diff = phases[:, i] - phases[:, j]
                        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                        phase_sync[i, j] = phase_sync[j, i] = plv
                    except:
                        pass
        except:
            pass

        return phase_sync

    # ==========================================
    # 主处理函数：生成不同格式的数据
    # ==========================================

    def process_and_save(self,
                        n_subjects=None,
                        feature_type='temporal',
                        graph_type='dual_stream',
                        use_phase_sync=True):
        """
        统一的处理和保存函数

        Args:
            feature_type:
                - 'temporal': 时序特征（用于SF-DPL）
                - 'statistical': 统计特征（用于基线）
                - 'enhanced': 统计+拓扑特征（用于基线）

            graph_type:
                - 'dual_stream': 双流图（结构+功能）
                - 'single_functional': 单流功能图
                - 'single_anatomical': 单流解剖图

            use_phase_sync: 是否使用相位同步

        Returns:
            graph_list: 图数据列表
        """
        print(f"\n{'='*60}")
        print(f"ABIDE数据处理")
        print(f"特征类型: {feature_type}")
        print(f"图类型: {graph_type}")
        print(f"{'='*60}")

        # 下载数据
        data = self.download_data(n_subjects)

        # 获取ROI数据和标签
        rois_data = None
        for attr in ['rois_ho', 'rois_cc200', 'rois_aal']:
            if hasattr(data, attr):
                rois_data = getattr(data, attr)
                if rois_data is not None:
                    print(f"使用ROI数据: {attr}")
                    break

        if rois_data is None:
            print("错误：未找到ROI数据")
            return []

        labels = data.phenotypic['DX_GROUP'].values - 1

        # 确定时间长度
        target_length = self.determine_time_length(rois_data)
        print(f"目标时间长度: {target_length}")

        # 处理每个被试
        graph_list = []

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

                if time_series.shape[0] < 50:
                    continue

                label = labels[idx]

                # 根据配置生成图
                if graph_type == 'dual_stream':
                    graphs = self._create_dual_stream_graphs(
                        time_series, label, feature_type, use_phase_sync
                    )
                elif graph_type == 'single_functional':
                    graphs = self._create_single_graph(
                        time_series, label, feature_type, 'functional', use_phase_sync
                    )
                elif graph_type == 'single_anatomical':
                    graphs = self._create_single_graph(
                        time_series, label, feature_type, 'anatomical', use_phase_sync
                    )
                else:
                    continue

                graph_list.append(graphs)

                if (idx + 1) % 20 == 0:
                    print(f"已处理: {idx + 1}/{len(rois_data)}")

            except Exception as e:
                print(f"处理被试{idx}失败: {e}")
                continue

        print(f"\n成功构建 {len(graph_list)} 个样本")

        # 保存
        save_name = f'abide_{graph_type}_{feature_type}_{target_length}.pt'
        save_path = os.path.join(self.processed_path, save_name)
        torch.save(graph_list, save_path)
        print(f"数据已保存至: {save_path}")

        # 保存元信息
        if graph_list:
            sample = graph_list[0]
            if isinstance(sample, tuple):
                sample = sample[0]

            meta_info = {
                'n_subjects': len(graph_list),
                'time_length': target_length,
                'feature_type': feature_type,
                'graph_type': graph_type,
                'node_feature_dim': sample.x.shape[1],
                'n_regions': sample.x.shape[0]
            }

            meta_path = os.path.join(self.processed_path, 'meta_info.pt')
            torch.save(meta_info, meta_path)
            print(f"\n样本信息:")
            print(f"  节点特征: {sample.x.shape}")
            print(f"  边数: {sample.edge_index.shape[1]}")

        return graph_list

    def _create_dual_stream_graphs(self, time_series, label, feature_type, use_phase_sync):
        """创建双流图"""
        N = time_series.shape[1]

        # 构建邻接矩阵
        struct_adj = self.construct_anatomical_adj(N)
        func_adj = self.construct_functional_adj(time_series, use_phase_sync)

        # 提取节点特征
        if feature_type == 'temporal':
            node_features = self.extract_temporal_features(time_series)
        elif feature_type == 'statistical':
            node_features = self.extract_statistical_features(time_series)
        elif feature_type == 'enhanced':
            node_features = self.extract_enhanced_features(time_series, func_adj)

        # 转换为PyG格式
        struct_data = self._adj_to_pyg(struct_adj, node_features, label)
        func_data = self._adj_to_pyg(func_adj, node_features, label, weighted=True)

        return (func_data, struct_data)

    def _create_single_graph(self, time_series, label, feature_type, adj_type, use_phase_sync):
        """创建单流图"""
        N = time_series.shape[1]

        # 构建邻接矩阵
        if adj_type == 'functional':
            adj = self.construct_functional_adj(time_series, use_phase_sync)
        elif adj_type == 'anatomical':
            adj = self.construct_anatomical_adj(N)

        # 提取节点特征
        if feature_type == 'temporal':
            node_features = self.extract_temporal_features(time_series)
        elif feature_type == 'statistical':
            node_features = self.extract_statistical_features(time_series)
        elif feature_type == 'enhanced':
            node_features = self.extract_enhanced_features(time_series, adj)

        # 转换为PyG格式
        return self._adj_to_pyg(adj, node_features, label, weighted=(adj_type=='functional'))

    def _adj_to_pyg(self, adj_matrix, node_features, label, weighted=False):
        """转换为PyG Data格式"""
        edge_index = []
        edge_attr = [] if weighted else None

        N = adj_matrix.shape[0]
        for i in range(N):
            for j in range(N):
                if adj_matrix[i, j] != 0:
                    edge_index.append([i, j])
                    if weighted:
                        edge_attr.append(adj_matrix[i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

        if weighted:
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.long)
        )


# ==========================================
# 便捷函数
# ==========================================

def prepare_all_data(data_folder='./data', n_subjects=None):
    """一键准备所有格式的数据"""

    processor = ABIDEDataProcessor(
        data_folder=data_folder,
        time_strategy='adaptive'
    )

    configs = [
        # SF-DPL用：双流 + 时序特征
        {'feature_type': 'temporal', 'graph_type': 'dual_stream', 'use_phase_sync': True},

        # 基线用：单流 + 统计特征
        {'feature_type': 'statistical', 'graph_type': 'single_functional', 'use_phase_sync': True},

        # 基线用：单流 + 增强特征
        {'feature_type': 'enhanced', 'graph_type': 'single_functional', 'use_phase_sync': True},
    ]

    results = {}

    for config in configs:
        name = f"{config['graph_type']}_{config['feature_type']}"
        print(f"\n处理配置: {name}")

        graph_list = processor.process_and_save(
            n_subjects=n_subjects,
            **config
        )

        results[name] = len(graph_list)

    print(f"\n{'='*60}")
    print("所有数据处理完成")
    print(f"{'='*60}")
    for name, count in results.items():
        print(f"{name}: {count} 个样本")

    return results


# 在 abide_data.py 文件的最末尾添加以下代码

def load_abide_data(data_folder='./data', n_subjects=None, graph_method='correlation_matrix'):
    """
    便捷函数：加载ABIDE数据集（单流版本，用于传统方法）

    Args:
        data_folder: 数据保存路径
        n_subjects: 被试数量
        graph_method: 图构建方法（虽然传入但不使用，为了兼容性）

    Returns:
        graph_list: 图数据列表
        input_dim: 输入特征维度
        output_dim: 输出类别数
    """
    print("=" * 60)
    print("加载ABIDE数据集（单流版本）")
    print("=" * 60)

    # 检查是否存在双流数据
    dual_stream_path = os.path.join(data_folder, 'ABIDE/processed/abide_dual_stream_temporal_78.pt')

    if os.path.exists(dual_stream_path):
        print(f"检测到双流数据文件: {dual_stream_path}")
        print("从双流数据提取功能流作为单流数据...")

        # 加载双流数据
        dual_stream_data = torch.load(dual_stream_path)

        # 提取功能流（第一个元素）
        graph_list = [func_data for func_data, struct_data in dual_stream_data]

        if graph_list:
            input_dim = graph_list[0].x.shape[1]
            output_dim = 2  # 二分类

            print(f"成功加载 {len(graph_list)} 个样本")
            print(f"输入维度: {input_dim}")
            print(f"输出类别数: {output_dim}")
        else:
            print("警告：数据为空")
            input_dim = 0
            output_dim = 2

    else:
        print(f"未找到双流数据: {dual_stream_path}")
        print("请先运行数据处理脚本生成数据")
        print("\n运行命令：")
        print("python abide_data.py")

        # 返回空数据
        graph_list = []
        input_dim = 0
        output_dim = 2

    print("=" * 60)

    return graph_list, input_dim, output_dim


if __name__ == "__main__":
    # 主函数保持不变
    print("=" * 60)
    print("ABIDE数据处理")
    print("=" * 60)

    # 询问用户选择
    print("\n选择操作：")
    print("1. 生成双流数据（推荐，用于SF-DPL）")
    print("2. 测试加载功能")

    choice = input("\n请选择 (1/2): ").strip()

    if choice == '1':
        # 生成双流数据
        from abide_data import prepare_all_data

        prepare_all_data(
            data_folder='./data',
            n_subjects=None,
            force_reprocess=False
        )

    elif choice == '2':
        # 测试加载
        graph_list, input_dim, output_dim = load_abide_data(
            data_folder='./data'
        )

        if graph_list:
            print(f"\n加载成功！")
            print(f"样本数: {len(graph_list)}")
            print(f"第一个样本信息:")
            print(f"  节点特征: {graph_list[0].x.shape}")
            print(f"  边索引: {graph_list[0].edge_index.shape}")
            print(f"  标签: {graph_list[0].y.item()}")

    else:
        print("无效选择")

# if __name__ == "__main__":
#     # 一键准备所有数据
#     prepare_all_data(data_folder='./data', n_subjects=None)