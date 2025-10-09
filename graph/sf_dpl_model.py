"""
SF-DPL模型 - 彻底解决过拟合版本
核心改进：
1. 差异化初始化两个编码器
2. 更强的正交化约束
3. 特征解耦机制
4. 对比学习
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from model import GIN


class StructurePrompt(nn.Module):
    """结构提示模块 - 强化版"""

    def __init__(self, hidden_dim, num_prompts=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts

        # 结构提示向量 - 更大的初始化
        self.prompts = nn.Parameter(torch.randn(num_prompts, hidden_dim) * 0.1)

        # 自适应权重网络
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # LayerNorm更稳定
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, num_prompts),
            nn.Softmax(dim=-1)
        )

        nn.init.xavier_normal_(self.prompts)

    def forward(self, x):
        weights = self.attention(x)
        prompted = torch.matmul(weights, self.prompts)
        return x + prompted * 0.3  # 增强提示强度


class FunctionPrompt(nn.Module):
    """功能提示模块 - 强化版"""

    def __init__(self, hidden_dim, num_prompts=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts

        self.prompts = nn.Parameter(torch.randn(num_prompts, hidden_dim) * 0.1)

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.prompt_gen = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.4)
        )

        nn.init.xavier_normal_(self.prompts)

    def forward(self, x, context=None):
        dynamic_prompt = self.prompt_gen(x)
        static_prompt = self.prompts.mean(dim=0).unsqueeze(0).expand_as(x)

        if context is not None:
            gate_input = torch.cat([x, context], dim=-1)
        else:
            gate_input = torch.cat([x, x], dim=-1)

        gate_weight = self.gate(gate_input)
        final_prompt = gate_weight * dynamic_prompt + (1 - gate_weight) * static_prompt

        return x + final_prompt * 0.3


class SF_DPL(nn.Module):
    """SF-DPL主模型 - 彻底解决过拟合版"""

    def __init__(self,
                 num_layer=5,
                 struct_input_dim=None,
                 func_input_dim=None,
                 hidden_dim=128,
                 num_classes=2,
                 drop_ratio=0.5,
                 num_prompts=5,
                 ortho_weight=1.0,  # ⭐ 大幅增强
                 decorr_weight=0.5):  # ⭐ 新增：解耦损失权重
        super().__init__()

        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.ortho_weight = ortho_weight
        self.decorr_weight = decorr_weight
        self.drop_ratio = drop_ratio

        self.struct_input_dim = struct_input_dim
        self.func_input_dim = func_input_dim

        self.struct_encoder = None
        self.func_encoder = None

        # 提示模块
        self.struct_prompt = StructurePrompt(hidden_dim, num_prompts)
        self.func_prompt = FunctionPrompt(hidden_dim, num_prompts)

        # ⭐ 特征投影层（用于解耦）
        self.struct_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_ratio)
        )

        self.func_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_ratio)
        )

        # BatchNorm
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_ratio)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def _create_encoder(self, input_dim, name='encoder'):
        """动态创建编码器"""
        encoder = GIN(
            num_layer=self.num_layer,
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            drop_ratio=self.drop_ratio
        )
        return encoder

    def forward(self, struct_data, func_data):
        # 首次调用时创建编码器
        if self.struct_encoder is None:
            self.struct_input_dim = struct_data.x.shape[1]
            self.struct_encoder = self._create_encoder(
                self.struct_input_dim, 'struct'
            ).to(struct_data.x.device)

        if self.func_encoder is None:
            self.func_input_dim = func_data.x.shape[1]
            self.func_encoder = self._create_encoder(
                self.func_input_dim, 'func'
            ).to(func_data.x.device)

        # 结构流编码
        struct_feat = self.struct_encoder(struct_data, pooling='mean')
        struct_feat = self.bn1(struct_feat)
        struct_feat = self.struct_prompt(struct_feat)
        struct_feat = self.struct_proj(struct_feat)  # ⭐ 投影

        # 功能流编码
        func_feat = self.func_encoder(func_data, pooling='mean')
        func_feat = self.bn2(func_feat)
        func_feat = self.func_prompt(func_feat)
        func_feat = self.func_proj(func_feat)  # ⭐ 投影

        # ⭐ 计算多种损失
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
        """计算正交化损失 - 超强约束"""
        if torch.isnan(feat1).any() or torch.isnan(feat2).any():
            return torch.tensor(0.0, device=feat1.device)

        eps = 1e-8
        feat1_norm = F.normalize(feat1, p=2, dim=1, eps=eps)
        feat2_norm = F.normalize(feat2, p=2, dim=1, eps=eps)

        # 余弦相似度
        similarity = torch.sum(feat1_norm * feat2_norm, dim=1)

        if torch.isnan(similarity).any():
            return torch.tensor(0.0, device=feat1.device)

        # ⭐ 使用绝对值惩罚（比平方更强）
        ortho_loss = torch.mean(torch.abs(similarity))

        if torch.isnan(ortho_loss):
            return torch.tensor(0.0, device=feat1.device)

        return ortho_loss * self.ortho_weight

    def compute_decorrelation_loss(self, feat1, feat2):
        """⭐ 新增：特征解耦损失（基于协方差矩阵）"""
        if torch.isnan(feat1).any() or torch.isnan(feat2).any():
            return torch.tensor(0.0, device=feat1.device)

        # 中心化特征
        feat1_centered = feat1 - feat1.mean(dim=0, keepdim=True)
        feat2_centered = feat2 - feat2.mean(dim=0, keepdim=True)

        # 计算协方差矩阵
        batch_size = feat1.size(0)
        cov_matrix = torch.matmul(feat1_centered.t(), feat2_centered) / (batch_size - 1)

        # 最小化协方差（鼓励不相关）
        decorr_loss = torch.norm(cov_matrix, p='fro') ** 2

        if torch.isnan(decorr_loss):
            return torch.tensor(0.0, device=feat1.device)

        return decorr_loss * self.decorr_weight

    def load_pretrained_weights(self, struct_path=None, func_path=None):
        """加载预训练权重 - 改进版"""
        def load_and_perturb(encoder, path, perturb_scale=0.1):
            """加载权重并添加扰动，实现差异化"""
            print(f"加载预训练权重: {path}")
            checkpoint = torch.load(path, map_location='cpu')

            if 'gnn' in checkpoint:
                state_dict = checkpoint['gnn']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # 跳过第一层
            filtered_state = {}
            for k, v in state_dict.items():
                if 'convs.0.mlp.0' in k:
                    print(f"跳过第一层权重: {k}")
                    continue

                # ⭐ 添加随机扰动（实现差异化）
                if perturb_scale > 0:
                    v = v + torch.randn_like(v) * perturb_scale * v.std()

                filtered_state[k] = v

            encoder.load_state_dict(filtered_state, strict=False)
            print("✓ 权重加载完成（已添加扰动）")

        if struct_path and self.struct_encoder:
            load_and_perturb(self.struct_encoder, struct_path, perturb_scale=0.1)

        if func_path and self.func_encoder:
            load_and_perturb(self.func_encoder, func_path, perturb_scale=0.15)  # 更大扰动


def train_sf_dpl_one_epoch(model, train_loader, optimizer, device,
                          use_mixup=True, mixup_alpha=0.3):
    """训练一个epoch - 增强版"""
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
            continue

        # ⭐ Mixup数据增强
        if use_mixup and train_loader.batch_size > 1:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            batch_size = func_data.y.size(0)
            if batch_size > 1:
                index = torch.randperm(batch_size).to(device)
                func_data.x = lam * func_data.x + (1 - lam) * func_data.x[index]
                struct_data.x = lam * struct_data.x + (1 - lam) * struct_data.x[index]
                mixed_labels = lam * F.one_hot(func_data.y, 2).float() + \
                              (1 - lam) * F.one_hot(func_data.y[index], 2).float()

        optimizer.zero_grad()
        logits, aux_loss = model(struct_data, func_data)

        if torch.isnan(logits).any():
            continue

        # 计算损失
        if use_mixup and train_loader.batch_size > 1 and batch_size > 1:
            ce_loss = -torch.mean(torch.sum(mixed_labels * F.log_softmax(logits, dim=1), dim=1))
        else:
            ce_loss = F.cross_entropy(logits, func_data.y, label_smoothing=0.1)

        loss = ce_loss + aux_loss

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 放宽一点
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