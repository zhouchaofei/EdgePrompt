"""
Edge Prediction预训练模块
实现基于边预测的自监督学习策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling, add_self_loops
from model import GIN
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score


class EdgePredictor(nn.Module):
    """边预测器"""

    def __init__(self, hidden_dim):
        super(EdgePredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, edge_index):
        """
        预测边的存在概率
        Args:
            z: 节点表示 [N, D]
            edge_index: 边索引 [2, E]
        """
        row, col = edge_index
        edge_features = torch.cat([z[row], z[col]], dim=-1)
        return self.predictor(edge_features).squeeze()


class EdgePredictionModel(nn.Module):
    """Edge Prediction预训练模型"""

    def __init__(self, num_layer, input_dim, hidden_dim, drop_ratio=0.0):
        super(EdgePredictionModel, self).__init__()

        # GNN编码器
        self.encoder = GIN(
            num_layer=num_layer,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            drop_ratio=drop_ratio
        )

        # 边预测器
        self.edge_predictor = EdgePredictor(hidden_dim)

    def encode(self, data):
        """编码图数据"""
        return self.encoder(data, pooling='none')

    def decode(self, z, pos_edge_index, neg_edge_index):
        """
        解码：预测边的存在
        Args:
            z: 节点表示
            pos_edge_index: 正样本边
            neg_edge_index: 负样本边
        """
        pos_pred = self.edge_predictor(z, pos_edge_index)
        neg_pred = self.edge_predictor(z, neg_edge_index)

        return pos_pred, neg_pred

    def forward(self, data, pos_edge_index, neg_edge_index):
        """前向传播"""
        z = self.encode(data)
        return self.decode(z, pos_edge_index, neg_edge_index)

    def loss(self, pos_pred, neg_pred):
        """计算损失"""
        # 二元交叉熵损失
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_pred, torch.ones_like(pos_pred)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_pred, torch.zeros_like(neg_pred)
        )
        return pos_loss + neg_loss


class EdgePredictionTrainer:
    """Edge Prediction训练器"""

    def __init__(self, model, device, lr=0.001, weight_decay=5e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def split_edges(self, data, val_ratio=0.1, test_ratio=0.1):
        """
        分割边用于训练/验证/测试
        """
        num_edges = data.edge_index.size(1)
        perm = torch.randperm(num_edges)

        num_val = int(val_ratio * num_edges)
        num_test = int(test_ratio * num_edges)

        val_edge_index = data.edge_index[:, perm[:num_val]]
        test_edge_index = data.edge_index[:, perm[num_val:num_val + num_test]]
        train_edge_index = data.edge_index[:, perm[num_val + num_test:]]

        return train_edge_index, val_edge_index, test_edge_index

    def train_epoch(self, train_loader, mask_ratio=0.15):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(self.device)

            # 分割边：保留一部分边作为训练，掩码一部分作为正样本
            num_edges = batch.edge_index.size(1)
            num_mask = int(mask_ratio * num_edges)

            perm = torch.randperm(num_edges)
            mask_edge_index = batch.edge_index[:, perm[:num_mask]]
            train_edge_index = batch.edge_index[:, perm[num_mask:]]

            # 创建用于训练的数据（移除掩码边）
            batch_train = batch.clone()
            batch_train.edge_index = train_edge_index

            # 负采样
            neg_edge_index = negative_sampling(
                edge_index=train_edge_index,
                num_nodes=batch.num_nodes,
                num_neg_samples=mask_edge_index.size(1)
            )

            self.optimizer.zero_grad()

            # 前向传播
            pos_pred, neg_pred = self.model(
                batch_train,
                mask_edge_index,
                neg_edge_index
            )

            # 计算损失
            loss = self.model.loss(pos_pred, neg_pred)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader, mask_ratio=0.15):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_pos_pred = []
        all_neg_pred = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                # 与训练时相同的边分割策略
                num_edges = batch.edge_index.size(1)
                num_mask = int(mask_ratio * num_edges)

                perm = torch.randperm(num_edges)
                mask_edge_index = batch.edge_index[:, perm[:num_mask]]
                train_edge_index = batch.edge_index[:, perm[num_mask:]]

                batch_train = batch.clone()
                batch_train.edge_index = train_edge_index

                neg_edge_index = negative_sampling(
                    edge_index=train_edge_index,
                    num_nodes=batch.num_nodes,
                    num_neg_samples=mask_edge_index.size(1)
                )

                pos_pred, neg_pred = self.model(
                    batch_train,
                    mask_edge_index,
                    neg_edge_index
                )

                loss = self.model.loss(pos_pred, neg_pred)
                total_loss += loss.item()

                all_pos_pred.append(torch.sigmoid(pos_pred).cpu())
                all_neg_pred.append(torch.sigmoid(neg_pred).cpu())

        # 计算AUC
        all_pos_pred = torch.cat(all_pos_pred)
        all_neg_pred = torch.cat(all_neg_pred)

        labels = torch.cat([
            torch.ones(len(all_pos_pred)),
            torch.zeros(len(all_neg_pred))
        ])
        preds = torch.cat([all_pos_pred, all_neg_pred])

        auc = roc_auc_score(labels.numpy(), preds.numpy())
        ap = average_precision_score(labels.numpy(), preds.numpy())

        return total_loss / len(val_loader), auc, ap

    def train(self, train_loader, val_loader=None, epochs=200,
              save_path=None, verbose=True):
        """
        训练Edge Prediction模型
        """
        best_val_auc = 0

        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)

            # 验证
            if val_loader is not None:
                val_loss, val_auc, val_ap = self.evaluate(val_loader)
            else:
                val_loss, val_auc, val_ap = train_loss, 0, 0

            # 保存最佳模型
            if save_path and val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'auc': val_auc,
                    'ap': val_ap
                }, save_path)
                if verbose:
                    print(f"保存最佳模型 (epoch {epoch})")

            # 打印进度
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val AUC: {val_auc:.4f}, "
                      f"Val AP: {val_ap:.4f}")

        return best_val_auc