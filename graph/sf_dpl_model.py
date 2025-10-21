"""
SF-DPL模型 - 改进版
核心改进：
1. 大幅降低辅助损失权重
2. 差异化初始化两个编码器
3. 添加特征标准化
4. 渐进式训练策略
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from model import GIN


class StructurePrompt(nn.Module):
    """结构提示模块 - 简化版"""

    def __init__(self, hidden_dim, num_prompts=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts

        # 结构提示向量
        self.prompts = nn.Parameter(torch.randn(num_prompts, hidden_dim) * 0.01)

        # 简化的注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, num_prompts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # 计算注意力权重
        weights = self.attention(x)  # [B, num_prompts]

        # 加权提示
        prompted = torch.matmul(weights, self.prompts)  # [B, hidden_dim]

        return x + prompted * 0.1  # 降低提示强度


class FunctionPrompt(nn.Module):
    """功能提示模块 - 简化版"""

    def __init__(self, hidden_dim, num_prompts=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts

        self.prompts = nn.Parameter(torch.randn(num_prompts, hidden_dim) * 0.01)

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 静态提示
        static_prompt = self.prompts.mean(dim=0).unsqueeze(0).expand_as(x)

        # 门控权重
        gate_weight = self.gate(x)

        return x + static_prompt * gate_weight * 0.1


class SF_DPL(nn.Module):
    """SF-DPL主模型 - 大幅改进版"""

    def __init__(self,
                 num_layer=5,
                 struct_input_dim=None,
                 func_input_dim=None,
                 hidden_dim=128,
                 num_classes=2,
                 drop_ratio=0.3,
                 num_prompts=5,
                 ortho_weight=0.01,    # ⭐ 大幅降低（从1.0降到0.01）
                 decorr_weight=0.005):  # ⭐ 大幅降低（从0.5降到0.005）
        super().__init__()

        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.ortho_weight = ortho_weight
        self.decorr_weight = decorr_weight
        self.drop_ratio = drop_ratio

        # 动态确定输入维度
        self.struct_input_dim = struct_input_dim
        self.func_input_dim = func_input_dim

        # 编码器稍后创建
        self.struct_encoder = None
        self.func_encoder = None

        # 提示模块
        self.struct_prompt = StructurePrompt(hidden_dim, num_prompts)
        self.func_prompt = FunctionPrompt(hidden_dim, num_prompts)

        # ⭐ 添加BatchNorm稳定训练
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # ⭐ 添加BN
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def _create_encoder(self, input_dim):
        """动态创建编码器"""
        return GIN(
            num_layer=self.num_layer,
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            drop_ratio=self.drop_ratio
        )

    def forward(self, struct_data, func_data):
        # 首次调用时创建编码器
        if self.struct_encoder is None:
            self.struct_input_dim = struct_data.x.shape[1]
            self.struct_encoder = self._create_encoder(self.struct_input_dim).to(struct_data.x.device)

        if self.func_encoder is None:
            self.func_input_dim = func_data.x.shape[1]
            self.func_encoder = self._create_encoder(self.func_input_dim).to(func_data.x.device)

        # 结构流编码
        struct_feat = self.struct_encoder(struct_data, pooling='mean')
        struct_feat = self.bn1(struct_feat)  # ⭐ 标准化
        struct_feat = self.struct_prompt(struct_feat)

        # 功能流编码
        func_feat = self.func_encoder(func_data, pooling='mean')
        func_feat = self.bn2(func_feat)  # ⭐ 标准化
        func_feat = self.func_prompt(func_feat)

        # ⭐ 计算辅助损失（大幅降低权重）
        ortho_loss = self.compute_orthogonality_loss(struct_feat, func_feat)
        decorr_loss = self.compute_decorrelation_loss(struct_feat, func_feat)

        total_aux_loss = ortho_loss + decorr_loss

        # 特征融合
        combined = torch.cat([struct_feat, func_feat], dim=-1)
        fused = self.fusion(combined)

        # 分类
        logits = self.classifier(fused)

        return logits, total_aux_loss

    def compute_orthogonality_loss(self, feat1, feat2):
        """正交化损失 - 添加安全检查"""
        if torch.isnan(feat1).any() or torch.isnan(feat2).any():
            return torch.tensor(0.0, device=feat1.device)

        # ⭐ 使用更稳定的归一化
        eps = 1e-8
        feat1_norm = F.normalize(feat1, p=2, dim=1, eps=eps)
        feat2_norm = F.normalize(feat2, p=2, dim=1, eps=eps)

        # 余弦相似度
        similarity = torch.sum(feat1_norm * feat2_norm, dim=1)

        if torch.isnan(similarity).any():
            return torch.tensor(0.0, device=feat1.device)

        # ⭐ 使用平方而不是绝对值（更温和）
        ortho_loss = torch.mean(similarity ** 2)

        if torch.isnan(ortho_loss):
            return torch.tensor(0.0, device=feat1.device)

        return ortho_loss * self.ortho_weight

    def compute_decorrelation_loss(self, feat1, feat2):
        """解耦损失 - 添加安全检查"""
        if torch.isnan(feat1).any() or torch.isnan(feat2).any():
            return torch.tensor(0.0, device=feat1.device)

        # 中心化
        feat1_centered = feat1 - feat1.mean(dim=0, keepdim=True)
        feat2_centered = feat2 - feat2.mean(dim=0, keepdim=True)

        # 协方差矩阵
        batch_size = feat1.size(0)
        cov_matrix = torch.matmul(feat1_centered.t(), feat2_centered) / (batch_size - 1)

        # Frobenius范数
        decorr_loss = torch.norm(cov_matrix, p='fro') ** 2

        if torch.isnan(decorr_loss):
            return torch.tensor(0.0, device=feat1.device)

        return decorr_loss * self.decorr_weight

    def load_pretrained_weights(self, struct_path=None, func_path=None):
        """加载预训练权重 - 添加差异化扰动"""

        def load_with_perturbation(encoder, path, perturb_std=0.02):
            """加载权重并添加扰动"""
            print(f"加载预训练权重: {path}")
            checkpoint = torch.load(path, map_location='cpu')

            if 'gnn' in checkpoint:
                state_dict = checkpoint['gnn']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # ⭐ 为每个编码器添加不同的随机扰动
            perturbed_state = {}
            for k, v in state_dict.items():
                # 跳过第一层（保持预训练特征）
                if 'convs.0' in k:
                    perturbed_state[k] = v
                else:
                    # 添加高斯噪声
                    noise = torch.randn_like(v) * perturb_std * v.std()
                    perturbed_state[k] = v + noise

            encoder.load_state_dict(perturbed_state, strict=False)
            print("✓ 权重加载完成（已添加扰动）")

        if struct_path and self.struct_encoder:
            load_with_perturbation(self.struct_encoder, struct_path, perturb_std=0.02)

        if func_path and self.func_encoder:
            load_with_perturbation(self.func_encoder, func_path, perturb_std=0.03)  # 功能流扰动更大


# ⭐ 训练辅助函数
def train_sf_dpl_one_epoch(model, train_loader, optimizer, device):
    """训练一个epoch - 改进版"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_aux_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        func_data, struct_data = batch
        func_data = func_data.to(device)
        struct_data = struct_data.to(device)

        # 检查输入
        if torch.isnan(func_data.x).any() or torch.isnan(struct_data.x).any():
            print(f"警告: Batch {batch_idx} 输入含有NaN")
            continue

        optimizer.zero_grad()
        logits, aux_loss = model(struct_data, func_data)

        if torch.isnan(logits).any():
            print(f"警告: Batch {batch_idx} 输出含有NaN")
            continue

        # ⭐ 主任务损失占主导
        ce_loss = F.cross_entropy(logits, func_data.y)
        loss = ce_loss + aux_loss  # 由于aux_loss权重已大幅降低，这里直接相加

        if torch.isnan(loss):
            print(f"警告: Batch {batch_idx} 总损失为NaN")
            continue

        loss.backward()

        # ⭐ 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_aux_loss += aux_loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0, 0, 0

    return (total_loss / num_batches,
            total_ce_loss / num_batches,
            total_aux_loss / num_batches)