"""
SF-DPL模型 - 抗过拟合增强版
主要改进：
1. 增强dropout (0.3 → 0.5)
2. 增强正交化约束 (0.01 → 0.1)
3. 添加BatchNorm
4. 梯度裁剪更严格
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from model import GIN


class StructurePrompt(nn.Module):
    """结构提示模块 - 添加正则化"""

    def __init__(self, hidden_dim, num_prompts=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts

        # 结构提示向量
        self.prompts = nn.Parameter(torch.randn(num_prompts, hidden_dim) * 0.01)  # ⭐ 小初始化

        # 自适应权重网络 + Dropout
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),  # ⭐ 添加dropout
            nn.Linear(hidden_dim // 2, num_prompts),
            nn.Softmax(dim=-1)
        )

        nn.init.xavier_normal_(self.prompts)

    def forward(self, x):
        weights = self.attention(x)
        prompted = torch.matmul(weights, self.prompts)
        return x + prompted * 0.1  # ⭐ 降低提示强度


class FunctionPrompt(nn.Module):
    """功能提示模块 - 添加正则化"""

    def __init__(self, hidden_dim, num_prompts=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts

        self.prompts = nn.Parameter(torch.randn(num_prompts, hidden_dim) * 0.01)

        # 门控机制 + Dropout
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # ⭐ 添加dropout
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.prompt_gen = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.3)  # ⭐ 添加dropout
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

        return x + final_prompt * 0.1  # ⭐ 降低提示强度


class SF_DPL(nn.Module):
    """SF-DPL主模型 - 抗过拟合增强版"""

    def __init__(self,
                 num_layer=5,
                 struct_input_dim=None,
                 func_input_dim=None,
                 hidden_dim=128,
                 num_classes=2,
                 drop_ratio=0.5,  # ⭐ 从0.3增加到0.5
                 num_prompts=5,
                 ortho_weight=0.1):  # ⭐ 从0.01增加到0.1
        super().__init__()

        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.ortho_weight = ortho_weight
        self.drop_ratio = drop_ratio

        self.struct_input_dim = struct_input_dim
        self.func_input_dim = func_input_dim

        self.struct_encoder = None
        self.func_encoder = None

        # 提示模块
        self.struct_prompt = StructurePrompt(hidden_dim, num_prompts)
        self.func_prompt = FunctionPrompt(hidden_dim, num_prompts)

        # ⭐ 添加BatchNorm
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 融合层 + 更强的正则化
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # ⭐ 添加BN
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # ⭐ 添加BN
            nn.ReLU(),
            nn.Dropout(drop_ratio)
        )

        # 分类器 + 更强的正则化
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),  # ⭐ 添加BN
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
            drop_ratio=self.drop_ratio  # ⭐ 使用更高的dropout
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
        struct_feat = self.bn1(struct_feat)  # ⭐ 添加BN
        struct_feat = self.struct_prompt(struct_feat)

        # 功能流编码
        func_feat = self.func_encoder(func_data, pooling='mean')
        func_feat = self.bn2(func_feat)  # ⭐ 添加BN
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
        """计算正交化损失 - 更严格的约束"""
        if torch.isnan(feat1).any() or torch.isnan(feat2).any():
            return torch.tensor(0.0, device=feat1.device)

        eps = 1e-8
        feat1_norm = F.normalize(feat1, p=2, dim=1, eps=eps)
        feat2_norm = F.normalize(feat2, p=2, dim=1, eps=eps)

        # 计算余弦相似度
        similarity = torch.sum(feat1_norm * feat2_norm, dim=1)

        if torch.isnan(similarity).any():
            return torch.tensor(0.0, device=feat1.device)

        # ⭐ 使用平方惩罚（更强的约束）
        ortho_loss = torch.mean(similarity ** 2)

        if torch.isnan(ortho_loss):
            return torch.tensor(0.0, device=feat1.device)

        return ortho_loss * self.ortho_weight

    def load_pretrained_weights(self, struct_path=None, func_path=None):
        """加载预训练权重"""
        if struct_path and self.struct_encoder:
            print(f"加载结构编码器预训练权重: {struct_path}")
            checkpoint = torch.load(struct_path, map_location='cpu')

            if 'gnn' in checkpoint:
                state_dict = checkpoint['gnn']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # 跳过第一层（维度可能不匹配）
            filtered_state = {}
            for k, v in state_dict.items():
                if 'convs.0.mlp.0' in k:
                    print(f"跳过第一层权重: {k}")
                    continue
                filtered_state[k] = v

            self.struct_encoder.load_state_dict(filtered_state, strict=False)
            print("✓ 结构编码器权重加载完成")

        if func_path and self.func_encoder:
            print(f"加载功能编码器预训练权重: {func_path}")
            checkpoint = torch.load(func_path, map_location='cpu')

            if 'gnn' in checkpoint:
                state_dict = checkpoint['gnn']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            filtered_state = {}
            for k, v in state_dict.items():
                if 'convs.0.mlp.0' in k:
                    print(f"跳过第一层权重: {k}")
                    continue
                filtered_state[k] = v

            self.func_encoder.load_state_dict(filtered_state, strict=False)
            print("✓ 功能编码器权重加载完成")


# ⭐ 训练辅助函数 - 增强版
def train_sf_dpl_one_epoch(model, train_loader, optimizer, device, use_mixup=False, mixup_alpha=0.2):
    """
    训练一个epoch - 抗过拟合增强版

    Args:
        model: SF-DPL模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        use_mixup: 是否使用Mixup数据增强
        mixup_alpha: Mixup参数
    """
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_ortho_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        func_data, struct_data = batch
        func_data = func_data.to(device)
        struct_data = struct_data.to(device)

        # 检查输入
        if torch.isnan(func_data.x).any() or torch.isnan(struct_data.x).any():
            print(f"警告: Batch {batch_idx} 包含NaN，跳过")
            continue

        # ⭐ 可选：Mixup数据增强
        if use_mixup and num_batches > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            batch_size = func_data.y.size(0)
            index = torch.randperm(batch_size).to(device)

            # Mixup特征
            func_data.x = lam * func_data.x + (1 - lam) * func_data.x[index]
            struct_data.x = lam * struct_data.x + (1 - lam) * struct_data.x[index]

        # 前向传播
        optimizer.zero_grad()
        logits, ortho_loss = model(struct_data, func_data)

        # 检查输出
        if torch.isnan(logits).any():
            print(f"警告: Batch {batch_idx} 输出包含NaN，跳过")
            continue

        # ⭐ 计算损失（使用Label Smoothing）
        ce_loss = F.cross_entropy(logits, func_data.y, label_smoothing=0.1)
        loss = ce_loss + ortho_loss

        # 检查损失
        if torch.isnan(loss):
            print(f"警告: Batch {batch_idx} 损失为NaN，跳过")
            continue

        # 反向传播
        loss.backward()

        # ⭐ 梯度裁剪（更严格）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_ortho_loss += ortho_loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0, 0, 0

    return (total_loss / num_batches,
            total_ce_loss / num_batches,
            total_ortho_loss / num_batches)