"""
GraphMAE预训练模块
实现自监督掩码图自编码器的预训练策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from model import GIN
import os
from tqdm import tqdm


class GraphMAE(nn.Module):
    """GraphMAE预训练模型"""

    def __init__(self, num_layer, input_dim, hidden_dim,
                 mask_rate=0.75, decoder_hidden_dim=64,
                 decoder_layers=1, drop_ratio=0.0):
        super(GraphMAE, self).__init__()

        self.mask_rate = mask_rate
        self.hidden_dim = hidden_dim

        # 编码器使用GIN
        self.encoder = GIN(
            num_layer=num_layer,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            drop_ratio=drop_ratio
        )

        # 解码器：重建被掩码的节点特征
        decoder_modules = []
        decoder_in = hidden_dim
        for _ in range(decoder_layers - 1):
            decoder_modules.append(nn.Linear(decoder_in, decoder_hidden_dim))
            decoder_modules.append(nn.ReLU())
            decoder_modules.append(nn.Dropout(drop_ratio))
            decoder_in = decoder_hidden_dim
        decoder_modules.append(nn.Linear(decoder_in, input_dim))

        self.decoder = nn.Sequential(*decoder_modules)

        # 掩码token（可学习的向量，用于替换被掩码的节点特征）
        self.mask_token = nn.Parameter(torch.zeros(1, input_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def encoding_mask_noise(self, x, mask_rate=0.75):
        """
        对节点特征进行随机掩码
        Args:
            x: 节点特征 [N, D]
            mask_rate: 掩码比例
        Returns:
            masked_x: 掩码后的特征
            mask: 掩码标记
        """
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # 计算要掩码的节点数量
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        # 创建掩码
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        mask[mask_nodes] = True

        # 应用掩码token
        out_x = x.clone()
        out_x[mask_nodes] = self.mask_token

        return out_x, mask

    def forward(self, data, return_rep=False):
        """
        前向传播
        Args:
            data: 图数据
            return_rep: 是否返回表示（用于下游任务）
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if return_rep:
            # 推理模式：返回图表示
            node_rep = self.encoder(data, pooling='none')
            graph_rep = global_mean_pool(node_rep, batch)
            return graph_rep

        # 训练模式：掩码重建
        # 对每个图独立进行掩码
        masked_x_list = []
        mask_list = []

        # 获取每个图的节点
        unique_batch = torch.unique(batch)
        for b in unique_batch:
            node_mask = (batch == b)
            graph_x = x[node_mask]

            # 对该图的节点特征进行掩码
            masked_graph_x, graph_mask = self.encoding_mask_noise(
                graph_x, self.mask_rate
            )
            masked_x_list.append(masked_graph_x)
            mask_list.append(graph_mask)

        # 合并所有图的掩码特征
        masked_x = torch.cat(masked_x_list, dim=0)
        mask = torch.cat(mask_list, dim=0)

        # 使用掩码后的特征进行编码
        data_masked = data.clone()
        data_masked.x = masked_x
        node_rep = self.encoder(data_masked, pooling='none')

        # 解码：重建原始特征
        recon_x = self.decoder(node_rep)

        # 只计算被掩码节点的重建损失
        loss = F.mse_loss(recon_x[mask], x[mask], reduction='mean')

        return loss, recon_x, mask


class GraphMAETrainer:
    """GraphMAE预训练器"""

    def __init__(self, model, device, lr=0.001, weight_decay=5e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            loss, _, _ = self.model(batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        """评估"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                loss, _, _ = self.model(batch)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader=None, epochs=200,
              save_path=None, verbose=True):
        """
        训练GraphMAE
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            save_path: 模型保存路径
            verbose: 是否打印训练信息
        """
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)

            # 验证
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
            else:
                val_loss = train_loss

            # 保存最佳模型
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'loss': val_loss
                }, save_path)
                if verbose:
                    print(f"保存最佳模型 (epoch {epoch})")

            # 打印进度
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")

        return best_val_loss