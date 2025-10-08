"""
统一预训练框架 - 完整版
支持：
1. 双流数据（dual_stream）→ SF-DPL预训练
2. 单流数据（single_functional）→ 基线预训练
3. 不同特征类型（temporal, statistical, enhanced）
"""
"""
统一预训练框架 - 修正版
核心修正：
1. 时序特征统一到最大时间长度
2. 统计特征保持固定维度
3. 跨数据集预训练时维度兼容
"""
"""
统一预训练框架 - 修正版
"""
"""
统一预训练框架 - 完整版
支持：
1. 多种数据格式（dual_stream, single_functional_statistical, single_functional_enhanced）
2. 多种预训练任务（GraphMAE, EdgePrediction）
3. 同数据集和跨数据集预训练
4. 自动处理时间长度差异
5. 批处理支持
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import json
import glob
from torch_geometric.loader import DataLoader  # 修正：使用新的导入路径
from typing import Optional


# ==========================================
# 特征编码器
# ==========================================

class FeatureEncoder(nn.Module):
    """支持时间长度自适应的特征编码器"""

    def __init__(self,
                 feature_type: str,
                 input_dim: int,
                 hidden_dim: int,
                 target_dim: Optional[int] = None):
        """
        Args:
            feature_type: 'temporal', 'statistical', 'enhanced'
            input_dim: 实际输入维度
            hidden_dim: 隐藏层维度
            target_dim: 统一目标维度（用于跨数据集）
        """
        super().__init__()
        self.feature_type = feature_type
        self.input_dim = input_dim
        self.target_dim = target_dim if target_dim else input_dim

        if feature_type == 'temporal':
            # 时序编码器：1D CNN
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.fc = nn.Linear(64, hidden_dim)

            # 时间长度适配
            self.need_adapt = (input_dim != self.target_dim)
            if self.need_adapt:
                print(f"  时间长度适配: {input_dim} → {self.target_dim}")

        else:  # statistical 或 enhanced
            # 统计特征编码器：MLP
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.need_adapt = False

    def adapt_temporal(self, x):
        """适配时间序列长度（线性插值）"""
        """
                时间序列适配（支持上采样和降采样）

                Args:
                    x: [N, T_input]

                Returns:
                    adapted: [N, T_target]
                """
        if not self.need_adapt:
            return x

        N, T_input = x.shape
        T_target = self.target_dim

        if T_input == T_target:
            return x

        elif T_input < T_target:
            # ==========================================
            # 场景1: 上采样（插值）
            # 例如：ABIDE(78) → MDD(150)
            # ==========================================
            print(f"  上采样: {T_input} → {T_target}")

            x = F.interpolate(
                x.unsqueeze(1),  # [N, 1, T_input]
                size=T_target,  # T_target
                mode='linear',
                align_corners=False
            ).squeeze(1)  # [N, T_target]

            return x

        else:
            # ==========================================
            # 场景2: 降采样
            # 例如：MDD(150) → ABIDE(78)
            # ==========================================
            print(f"  降采样: {T_input} → {T_target}")

            # 方法1: 简单截取（快速但可能丢失信息）
            # return x[:, :T_target]

            # 方法2: 插值降采样（保留更多信息）
            x = F.interpolate(
                x.unsqueeze(1),  # [N, 1, T_input]
                size=T_target,  # T_target
                mode='linear',
                align_corners=False
            ).squeeze(1)  # [N, T_target]

            return x

    def forward(self, x):
        """
        Args:
            x: [N, D]
        Returns:
            encoded: [N, hidden_dim]
            adapted_x: [N, target_dim]
        """
        if self.feature_type == 'temporal':
            # 先适配时间长度
            adapted_x = self.adapt_temporal(x)

            # CNN编码
            feat = self.encoder(adapted_x.unsqueeze(1)).squeeze(-1)
            encoded = self.fc(feat)

            return encoded, adapted_x
        else:
            # 统计特征直接编码
            encoded = self.encoder(x)
            return encoded, x


# ==========================================
# 预训练任务
# ==========================================

class GraphMAETask(nn.Module):
    """GraphMAE任务 - 支持批处理"""

    def __init__(self, hidden_dim, target_dim, mask_rate=0.50):
        super().__init__()
        self.mask_rate = mask_rate
        self.target_dim = target_dim

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, target_dim)
        )

    def forward(self, graph_embedding, original_x, batch_info):
        """
        Args:
            graph_embedding: [batch_size, hidden_dim]
            original_x: [N_total, target_dim]
            batch_info: [N_total] 或 None
        """
        device = graph_embedding.device

        if batch_info is None:
            batch_size = 1
            batch_info = torch.zeros(original_x.shape[0], dtype=torch.long, device=device)
        else:
            batch_size = batch_info.max().item() + 1

        total_loss = 0
        valid_graphs = 0

        # 对每个图单独处理
        for b in range(batch_size):
            mask = (batch_info == b)
            graph_x = original_x[mask]
            N_b = graph_x.shape[0]

            if N_b == 0:
                continue

            # 掩码该图的节点
            num_mask = max(1, int(N_b * self.mask_rate))
            mask_idx = torch.randperm(N_b, device=device)[:num_mask]
            target_x = graph_x[mask_idx]

            # 重建
            reconstructed = self.decoder(graph_embedding[b:b + 1])
            reconstructed = reconstructed.expand(num_mask, -1)

            # 损失
            loss = F.mse_loss(reconstructed, target_x)
            total_loss += loss
            valid_graphs += 1

        return total_loss / max(valid_graphs, 1)


class EdgePredictionTask(nn.Module):
    """边预测任务 - 支持批处理"""

    def __init__(self, hidden_dim):
        super().__init__()

        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_embeddings, pos_edges, neg_edges, batch_info=None):
        """
        Args:
            node_embeddings: [N_total, hidden_dim]
            pos_edges: [E_pos, 2]
            neg_edges: [E_neg, 2]
            batch_info: [N_total] 或 None
        """
        if len(pos_edges) == 0 or len(neg_edges) == 0:
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)

        # 预测正边
        pos_src = node_embeddings[pos_edges[:, 0]]
        pos_dst = node_embeddings[pos_edges[:, 1]]
        pos_feat = torch.cat([pos_src, pos_dst], dim=1)
        pos_pred = self.edge_predictor(pos_feat).squeeze()

        # 预测负边
        neg_src = node_embeddings[neg_edges[:, 0]]
        neg_dst = node_embeddings[neg_edges[:, 1]]
        neg_feat = torch.cat([neg_src, neg_dst], dim=1)
        neg_pred = self.edge_predictor(neg_feat).squeeze()

        # BCE损失
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_pred, torch.ones_like(pos_pred)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_pred, torch.zeros_like(neg_pred)
        )

        return (pos_loss + neg_loss) / 2


# ==========================================
# 统一预训练模型
# ==========================================

class UnifiedPretrainModel(nn.Module):
    """统一预训练模型"""

    def __init__(self,
                 feature_type: str,
                 input_dim: int,
                 target_dim: int,
                 hidden_dim: int = 128,
                 num_layer: int = 5,
                 pretrain_task: str = 'graphmae',
                 mask_rate: float = 0.75):
        super().__init__()

        self.feature_type = feature_type
        self.pretrain_task = pretrain_task

        # 特征编码器
        self.feature_encoder = FeatureEncoder(
            feature_type, input_dim, hidden_dim, target_dim
        )

        # GNN编码器
        from model import GIN
        self.gnn = GIN(
            num_layer=num_layer,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim
        )

        # 任务模块
        if pretrain_task == 'graphmae':
            self.task_module = GraphMAETask(hidden_dim, target_dim, mask_rate)
        elif pretrain_task == 'edge_prediction':
            self.task_module = EdgePredictionTask(hidden_dim)

    def forward(self, data):
        from torch_geometric.data import Data

        # 1. 特征编码
        node_emb, adapted_x = self.feature_encoder(data.x)
        batch_info = data.batch if hasattr(data, 'batch') else None

        # 2. 根据任务准备GNN输入
        if self.pretrain_task == 'graphmae':
            # 掩码节点
            if batch_info is None:
                # 单图
                N = node_emb.shape[0]
                num_mask = int(N * self.task_module.mask_rate)
                mask_idx = random.sample(range(N), num_mask)
                node_emb_masked = node_emb.clone()
                node_emb_masked[mask_idx] = 0
            else:
                # 批处理：对每个图单独掩码
                node_emb_masked = node_emb.clone()
                batch_size = batch_info.max().item() + 1

                for b in range(batch_size):
                    mask = (batch_info == b)
                    graph_nodes = torch.where(mask)[0]
                    N_b = len(graph_nodes)

                    if N_b > 0:
                        num_mask = max(1, int(N_b * self.task_module.mask_rate))
                        mask_idx = torch.randperm(N_b, device=node_emb.device)[:num_mask]
                        actual_mask_idx = graph_nodes[mask_idx]
                        node_emb_masked[actual_mask_idx] = 0

            temp_data = Data(
                x=node_emb_masked,
                edge_index=data.edge_index,
                batch=batch_info
            )

            # 3. GNN编码
            graph_emb = self.gnn(temp_data, pooling='mean')

            # 4. 计算损失
            loss = self.task_module(graph_emb, adapted_x, batch_info)

        elif self.pretrain_task == 'edge_prediction':
            temp_data = Data(
                x=node_emb,
                edge_index=data.edge_index,
                batch=batch_info
            )

            # 3. GNN编码（节点级别）
            node_emb_gnn = self.gnn.get_node_embeddings(temp_data)

            # 4. 采样边
            pos_edges, neg_edges = self._sample_edges_batch(
                data.edge_index, batch_info, node_emb.shape[0]
            )

            # 5. 计算损失
            loss = self.task_module(node_emb_gnn, pos_edges, neg_edges, batch_info)

        return loss

    def _sample_edges_batch(self, edge_index, batch_info, num_nodes):
        """批处理边采样"""
        device = edge_index.device

        if batch_info is None:
            # 单图情况
            pos_edges = edge_index.t()
            num_sample = min(len(pos_edges), 100)

            if num_sample > 0:
                idx = torch.randperm(len(pos_edges))[:num_sample]
                pos_edges = pos_edges[idx]

            # 负采样
            neg_edges = []
            while len(neg_edges) < num_sample:
                i, j = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
                if i != j:
                    neg_edges.append([i, j])
            neg_edges = torch.tensor(neg_edges, dtype=torch.long, device=device)

            return pos_edges, neg_edges

        else:
            # 批处理情况
            all_pos_edges = []
            all_neg_edges = []
            batch_size = batch_info.max().item() + 1

            for b in range(batch_size):
                # 该图的节点
                mask = (batch_info == b)
                graph_nodes = torch.where(mask)[0]
                N_b = len(graph_nodes)

                if N_b == 0:
                    continue

                # 该图的边
                edge_mask = torch.isin(edge_index[0], graph_nodes)
                graph_edge_index = edge_index[:, edge_mask]
                pos_edges = graph_edge_index.t()

                # 采样正边
                num_sample = min(len(pos_edges), 50)
                if num_sample > 0:
                    idx = torch.randperm(len(pos_edges), device=device)[:num_sample]
                    all_pos_edges.append(pos_edges[idx])

                    # 负采样
                    node_list = graph_nodes.cpu().tolist()
                    neg_edges = []
                    attempts = 0

                    while len(neg_edges) < num_sample and attempts < num_sample * 10:
                        i = random.choice(node_list)
                        j = random.choice(node_list)
                        if i != j:
                            neg_edges.append([i, j])
                        attempts += 1

                    if neg_edges:
                        all_neg_edges.append(
                            torch.tensor(neg_edges, dtype=torch.long, device=device)
                        )

            if all_pos_edges:
                pos_edges = torch.cat(all_pos_edges, dim=0)
                neg_edges = torch.cat(all_neg_edges, dim=0)
            else:
                pos_edges = torch.empty((0, 2), dtype=torch.long, device=device)
                neg_edges = torch.empty((0, 2), dtype=torch.long, device=device)

            return pos_edges, neg_edges


# ==========================================
# 数据集注册表
# ==========================================

class DatasetRegistry:
    """数据集注册表 - 管理所有数据格式"""

    DATASETS = {
        'ABIDE': {
            'data_formats': {
                'dual_stream_temporal': {
                    'path_pattern': './data/ABIDE/processed/abide_dual_stream_temporal_*.pt',
                    'feature_type': 'temporal',
                    'time_length': 78
                },
                'single_functional_statistical': {
                    'path_pattern': './data/ABIDE/processed/abide_single_functional_statistical_*.pt',
                    'feature_type': 'statistical',
                    'time_length': None
                },
                'single_functional_enhanced': {
                    'path_pattern': './data/ABIDE/processed/abide_single_functional_enhanced_*.pt',
                    'feature_type': 'enhanced',
                    'time_length': None
                }
            },
            'n_regions': 116
        },
        'MDD': {
            'data_formats': {
                'dual_stream_temporal': {
                    'path_pattern': './data/REST-meta-MDD/processed/mdd_dual_stream_temporal_*.pt',
                    'feature_type': 'temporal',
                    'time_length': 150
                },
                'single_functional_statistical': {
                    'path_pattern': './data/REST-meta-MDD/processed/mdd_single_functional_statistical_*.pt',
                    'feature_type': 'statistical',
                    'time_length': None
                },
                'single_functional_enhanced': {
                    'path_pattern': './data/REST-meta-MDD/processed/mdd_single_functional_enhanced_*.pt',
                    'feature_type': 'enhanced',
                    'time_length': None
                }
            },
            'n_regions': 116
        }
    }

    @classmethod
    def get_target_dim(cls, source_dataset: str, target_dataset: str, data_format: str):
        """
        获取目标维度（修正版）

        核心逻辑：
        - 同数据集：target_dim = source数据集的维度
        - 跨数据集：target_dim = target数据集的维度

        Args:
            source_dataset: 源数据集名称
            target_dataset: 目标数据集名称
            data_format: 数据格式

        Returns:
            target_dim: 目标维度
        """
        # 获取目标数据集的配置
        if target_dataset not in cls.DATASETS:
            raise ValueError(f"未知目标数据集: {target_dataset}")

        target_config = cls.DATASETS[target_dataset]

        if data_format not in target_config['data_formats']:
            raise ValueError(f"目标数据集{target_dataset}不支持格式: {data_format}")

        format_config = target_config['data_formats'][data_format]

        # 返回目标数据集的维度
        target_dim = format_config.get('time_length')

        print(f"维度适配: {source_dataset}({cls.DATASETS[source_dataset]['data_formats'][data_format]['time_length']}) "
              f"→ {target_dataset}({target_dim})")

        return target_dim

    # 保留原来的get_unified_dim（用于其他用途）
    @classmethod
    def get_unified_dim(cls, data_format):
        """获取统一维度（全局最大值）- 仅用于特殊情况"""
        max_length = 0
        is_temporal = False

        for dataset_config in cls.DATASETS.values():
            if data_format in dataset_config['data_formats']:
                format_config = dataset_config['data_formats'][data_format]

                if format_config['feature_type'] == 'temporal':
                    is_temporal = True
                    time_len = format_config.get('time_length', 0)
                    if time_len:
                        max_length = max(max_length, time_len)

        return max_length if is_temporal else None


    @classmethod
    def load_dataset(cls, dataset_name: str, data_format: str):
        """加载数据集"""
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"未知数据集: {dataset_name}")

        dataset_config = cls.DATASETS[dataset_name]

        if data_format not in dataset_config['data_formats']:
            raise ValueError(f"未知数据格式: {data_format}")

        format_config = dataset_config['data_formats'][data_format]

        # 查找数据文件
        pattern = format_config['path_pattern']
        files = glob.glob(pattern)

        if not files:
            raise FileNotFoundError(
                f"未找到数据文件\n"
                f"模式: {pattern}\n"
                f"请先运行数据处理脚本"
            )

        data_path = files[0]
        print(f"加载数据: {data_path}")

        # 加载数据
        loaded_data = torch.load(data_path)

        # 根据格式处理
        if data_format == 'dual_stream_temporal':
            # ==========================================
            # 双流数据：只使用功能流！
            # ==========================================
            data_list = [func for func, struct in loaded_data]

            print(f"双流数据：提取功能流进行预训练")
            print(f"  总样本数: {len(loaded_data)}")
            print(f"  功能流样本数: {len(data_list)}")
        else:
            # 单流数据：直接使用
            data_list = loaded_data

        # 获取实际维度
        if len(data_list) > 0:
            input_dim = data_list[0].x.shape[1]
            print(f"实际特征维度: {input_dim}")
        else:
            input_dim = None

        config = {
            'n_regions': dataset_config['n_regions'],
            'feature_type': format_config['feature_type'],
            'input_dim': input_dim,
            'data_format': data_format,
            'time_length': format_config.get('time_length')
        }

        return data_list, config


# ==========================================
# 训练器
# ==========================================

class PretrainTrainer:
    """预训练训练器"""

    def __init__(self,
                 source_dataset: str,
                 target_dataset: str,
                 data_format: str = 'dual_stream_temporal',
                 task: str = 'graphmae',
                 hidden_dim: int = 128,
                 num_layer: int = 5,
                 mask_rate: float = 0.75,
                 batch_size: int = 32,
                 lr: float = 1e-3,
                 device: str = 'cuda'):

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.data_format = data_format
        self.task = task
        self.device = device

        # 加载数据
        self.train_data, source_config = DatasetRegistry.load_dataset(
            source_dataset, data_format
        )

        # 确定统一维度
        # ==========================================
        # 修正：根据目标数据集确定target_dim
        # ==========================================
        if source_config['feature_type'] == 'temporal':
            # 使用新的get_target_dim方法
            target_dim = DatasetRegistry.get_target_dim(
                source_dataset,
                target_dataset,
                data_format
            )
        else:
            # 统计特征不需要适配
            target_dim = source_config['input_dim']

        print(f"\n{'=' * 60}")
        print(f"预训练配置")
        print(f"{'=' * 60}")
        print(f"源数据集: {source_dataset} (维度: {source_config['input_dim']})")
        print(f"目标数据集: {target_dataset} (维度: {target_dim})")
        print(f"特征类型: {source_config['feature_type']}")
        print(f"训练样本数: {len(self.train_data)}")

        # 判断是否需要适配
        need_adapt = (source_config['input_dim'] != target_dim)
        if need_adapt:
            print(f"\n⚠️  需要维度适配: {source_config['input_dim']} → {target_dim}")
            if source_config['input_dim'] < target_dim:
                print(f"   方式: 线性插值（上采样）")
            else:
                print(f"   方式: 降采样")
        else:
            print(f"\n✓  维度匹配，无需适配")

        print(f"{'=' * 60}\n")

        # 数据加载器
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

        # 创建模型
        self.model = UnifiedPretrainModel(
            feature_type=source_config['feature_type'],
            input_dim=source_config['input_dim'],  # 源数据集维度
            target_dim=target_dim,  # 目标数据集维度（修正！）
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            pretrain_task=task,
            mask_rate=mask_rate
        ).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # 保存配置
        self.config = {
            'source_dataset': source_dataset,
            'target_dataset': target_dataset,
            'data_format': data_format,
            'feature_type': source_config['feature_type'],
            'task': task,
            'hidden_dim': hidden_dim,
            'num_layer': num_layer,
            'input_dim': source_config['input_dim'],
            'target_dim': target_dim,  # 正确的target_dim
            'mask_rate': mask_rate,
            'need_adapt': need_adapt
        }

        # 检查数据格式
        if self.data_format == 'dual_stream_temporal':
            print(f"\n{'=' * 60}")
            print(f"双流预训练配置")
            print(f"{'=' * 60}")
            print(f"预训练策略: 只使用功能流")
            print(f"原因: 结构流是固定的解剖连接，不需要预训练")
            print(f"功能流: 包含个体化的功能连接模式，需要学习")
            print(f"{'=' * 60}\n")

    def train(self, num_epochs: int = 100, save_dir: str = './pretrained_models'):
        """训练"""
        print("开始预训练...")

        best_loss = float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0

            for batch in self.train_loader:
                batch = batch.to(self.device)

                self.optimizer.zero_grad()
                loss = self.model(batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(save_dir, epoch, avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}")

        print(f"\n预训练完成！最佳损失: {best_loss:.4f}")

    def save_model(self, save_dir: str, epoch: int, loss: float):
        """保存模型"""
        format_dir = os.path.join(save_dir, self.data_format)
        os.makedirs(format_dir, exist_ok=True)

        save_name = f'{self.task}_{self.source_dataset}_for_{self.target_dataset}.pth'
        save_path = os.path.join(format_dir, save_name)

        torch.save({
            'feature_encoder': self.model.feature_encoder.state_dict(),
            'gnn': self.model.gnn.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'loss': loss
        }, save_path)

        config_path = save_path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)


# ==========================================
# 主函数
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='统一预训练框架')

    parser.add_argument('--source', type=str, required=True,
                        choices=['ABIDE', 'MDD', 'ADHD'])
    parser.add_argument('--target', type=str, required=True,
                        choices=['ABIDE', 'MDD', 'ADHD'])
    parser.add_argument('--data_format', type=str,
                        default='dual_stream_temporal',
                        choices=['dual_stream_temporal',
                                 'single_functional_statistical',
                                 'single_functional_enhanced'])
    parser.add_argument('--task', type=str, default='graphmae',
                        choices=['graphmae', 'edge_prediction'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layer', type=int, default=5)
    parser.add_argument('--mask_rate', type=float, default=0.75)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./pretrained_models')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 设置设备
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # 创建训练器
    trainer = PretrainTrainer(
        source_dataset=args.source,
        target_dataset=args.target,
        data_format=args.data_format,
        task=args.task,
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layer,
        mask_rate=args.mask_rate,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

    # 训练
    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)


if __name__ == "__main__":
    main()