"""
Structure-Function Decoupled Prompt Learning (SF-DPL) Model
结构-功能解耦提示学习模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from model import GIN


class StructurePrompt(nn.Module):
    """结构提示模块"""

    def __init__(self, hidden_dim, num_prompts=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts

        # 结构提示向量
        self.prompts = nn.Parameter(torch.randn(num_prompts, hidden_dim))

        # 注意力权重
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_prompts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """应用结构提示"""
        # 计算注意力权重
        weights = self.attention(x)  # [N, num_prompts]

        # 加权求和提示向量
        prompted = torch.matmul(weights, self.prompts)  # [N, hidden_dim]

        return x + prompted


class FunctionPrompt(nn.Module):
    """功能提示模块"""

    def __init__(self, hidden_dim, num_prompts=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts

        # 功能提示向量
        self.prompts = nn.Parameter(torch.randn(num_prompts, hidden_dim))

        # 动态门控
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 提示生成网络
        self.prompt_gen = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_feat=None):
        """应用功能提示"""
        # 生成动态提示
        dynamic_prompt = self.prompt_gen(x)

        # 选择静态提示
        static_prompt = self.prompts.mean(dim=0).unsqueeze(0).expand_as(x)

        # 门控融合
        if edge_feat is not None:
            gate_input = torch.cat([x, edge_feat], dim=-1)
        else:
            gate_input = torch.cat([x, x], dim=-1)

        gate_weight = self.gate(gate_input)

        # 融合动态和静态提示
        final_prompt = gate_weight * dynamic_prompt + (1 - gate_weight) * static_prompt

        return x + final_prompt


class SF_DPL(nn.Module):
    """
    结构-功能解耦提示学习模型

    核心思想：
    1. 双流架构：分别处理结构连接和功能动态
    2. 解耦机制：通过正交化损失确保两种特征互补
    3. 提示学习：只微调少量参数，保护预训练知识
    """

    def __init__(self,
                 num_layer=5,
                 struct_input_dim=4,      # 修改：结构流输入维度
                 func_input_dim=10,       # 修改：功能流输入维度
                 hidden_dim=128,
                 num_classes=2,
                 drop_ratio=0.3,
                 num_prompts=5):
        super().__init__()

        # 双流GNN编码器（不同输入维度）
        self.struct_encoder = GIN(
            num_layer=num_layer,
            input_dim=struct_input_dim,  # 结构流维度
            hidden_dim=hidden_dim,
            drop_ratio=drop_ratio
        )

        self.func_encoder = GIN(
            num_layer=num_layer,
            input_dim=func_input_dim,    # 功能流维度
            hidden_dim=hidden_dim,
            drop_ratio=drop_ratio
        )

        # 提示模块
        self.struct_prompt = StructurePrompt(hidden_dim, num_prompts)
        self.func_prompt = FunctionPrompt(hidden_dim, num_prompts)

        # 正交化损失权重
        self.ortho_weight = 0.01

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, struct_data, func_data):
        """前向传播"""
        # 结构流编码
        struct_feat = self.struct_encoder(struct_data, pooling='mean')
        struct_feat = self.struct_prompt(struct_feat)

        # 功能流编码
        func_feat = self.func_encoder(func_data, pooling='mean')
        func_feat = self.func_prompt(func_feat)

        # 计算正交化损失
        ortho_loss = self.compute_orthogonality_loss(struct_feat, func_feat)

        # 特征融合
        combined = torch.cat([struct_feat, func_feat], dim=-1)
        fused = self.fusion(combined)

        # 分类
        logits = self.classifier(fused)

        return logits, ortho_loss

    def compute_orthogonality_loss(self, feat1, feat2):
        """
        计算正交化损失，确保两个特征解耦

        原理：
        - 通过最小化余弦相似度，迫使结构和功能特征学习正交的信息
        - 这确保了两个流捕获互补而非冗余的特征
        """
        # 检查输入是否有nan
        if torch.isnan(feat1).any() or torch.isnan(feat2).any():
            return torch.tensor(0.0, device=feat1.device)

        # 归一化（添加epsilon避免除零）
        feat1_norm = F.normalize(feat1, p=2, dim=1, eps=1e-8)
        feat2_norm = F.normalize(feat2, p=2, dim=1, eps=1e-8)

        # 计算余弦相似度
        similarity = torch.matmul(feat1_norm, feat2_norm.t())

        # 检查相似度是否有nan
        if torch.isnan(similarity).any():
            return torch.tensor(0.0, device=feat1.device)

        # 正交化损失（相似度应该接近0）
        ortho_loss = torch.abs(similarity).mean()

        # 再次检查
        if torch.isnan(ortho_loss):
            return torch.tensor(0.0, device=feat1.device)

        return ortho_loss * self.ortho_weight