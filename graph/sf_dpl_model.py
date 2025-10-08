"""
Structure-Function Decoupled Prompt Learning (SF-DPL) Model - 优化版
核心改进：
1. 自动适配输入维度
2. 增强的正交化损失
3. 预训练权重加载
4. 梯度裁剪防止爆炸
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from model import GIN


class StructurePrompt(nn.Module):
    """结构提示模块 - 捕获解剖连接模式"""

    def __init__(self, hidden_dim, num_prompts=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts

        # 结构提示向量（针对不同的解剖模式）
        self.prompts = nn.Parameter(torch.randn(num_prompts, hidden_dim))

        # 自适应权重网络
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_prompts),
            nn.Softmax(dim=-1)
        )

        # 初始化
        nn.init.xavier_normal_(self.prompts)

    def forward(self, x):
        """
        应用结构提示
        Args:
            x: [N, hidden_dim] 节点特征
        Returns:
            prompted_x: [N, hidden_dim] 添加提示后的特征
        """
        # 计算注意力权重 [N, num_prompts]
        weights = self.attention(x)

        # 加权求和提示向量 [N, hidden_dim]
        prompted = torch.matmul(weights, self.prompts)

        return x + prompted


class FunctionPrompt(nn.Module):
    """功能提示模块 - 捕获动态功能连接"""

    def __init__(self, hidden_dim, num_prompts=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts

        # 功能提示向量（针对不同的功能状态）
        self.prompts = nn.Parameter(torch.randn(num_prompts, hidden_dim))

        # 动态门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 提示生成网络
        self.prompt_gen = nn.Linear(hidden_dim, hidden_dim)

        # 初始化
        nn.init.xavier_normal_(self.prompts)

    def forward(self, x, context=None):
        """
        应用功能提示
        Args:
            x: [N, hidden_dim] 节点特征
            context: [N, hidden_dim] 上下文信息（如边特征聚合）
        """
        # 生成动态提示
        dynamic_prompt = self.prompt_gen(x)

        # 静态提示
        static_prompt = self.prompts.mean(dim=0).unsqueeze(0).expand_as(x)

        # 门控融合
        if context is not None:
            gate_input = torch.cat([x, context], dim=-1)
        else:
            gate_input = torch.cat([x, x], dim=-1)

        gate_weight = self.gate(gate_input)

        # 融合
        final_prompt = gate_weight * dynamic_prompt + (1 - gate_weight) * static_prompt

        return x + final_prompt


class SF_DPL(nn.Module):
    """
    结构-功能解耦提示学习模型（优化版）

    核心改进：
    1. 自动检测输入维度
    2. 稳定的正交化损失
    3. 支持预训练权重加载
    """

    def __init__(self,
                 num_layer=5,
                 struct_input_dim=None,  # 自动检测
                 func_input_dim=None,    # 自动检测
                 hidden_dim=128,
                 num_classes=2,
                 drop_ratio=0.5,
                 num_prompts=5,
                 ortho_weight=0.1):
        super().__init__()

        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.ortho_weight = ortho_weight

        # ⭐ 关键：输入维度自动适配
        # 在forward时动态创建编码器
        self.struct_input_dim = struct_input_dim
        self.func_input_dim = func_input_dim

        self.struct_encoder = None
        self.func_encoder = None

        # 提示模块
        self.struct_prompt = StructurePrompt(hidden_dim, num_prompts)
        self.func_prompt = FunctionPrompt(hidden_dim, num_prompts)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _create_encoder(self, input_dim, name='encoder'):
        """动态创建编码器"""
        encoder = GIN(
            num_layer=self.num_layer,
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            drop_ratio=0.3
        )
        return encoder

    def forward(self, struct_data, func_data):
        """
        前向传播
        Args:
            struct_data: 结构图数据
            func_data: 功能图数据
        Returns:
            logits: [B, num_classes]
            ortho_loss: 正交化损失
        """
        # ⭐ 首次调用时创建编码器
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
        计算正交化损失（优化版）

        改进：
        1. 添加epsilon避免数值不稳定
        2. 检查nan值
        3. 使用余弦相似度而非点积
        """
        # 检查输入
        if torch.isnan(feat1).any() or torch.isnan(feat2).any():
            return torch.tensor(0.0, device=feat1.device)

        # 归一化（添加epsilon）
        eps = 1e-8
        feat1_norm = F.normalize(feat1, p=2, dim=1, eps=eps)
        feat2_norm = F.normalize(feat2, p=2, dim=1, eps=eps)

        # 计算余弦相似度矩阵
        similarity = torch.matmul(feat1_norm, feat2_norm.t())

        # 检查相似度
        if torch.isnan(similarity).any():
            return torch.tensor(0.0, device=feat1.device)

        # 正交化损失（期望相似度接近0）
        ortho_loss = torch.abs(similarity).mean()

        # 再次检查
        if torch.isnan(ortho_loss):
            return torch.tensor(0.0, device=feat1.device)

        return ortho_loss * self.ortho_weight

    def load_pretrained_weights(self, struct_path=None, func_path=None):
        """
        加载预训练权重

        Args:
            struct_path: 结构流预训练模型路径
            func_path: 功能流预训练模型路径
        """
        if struct_path and self.struct_encoder:
            print(f"加载结构编码器预训练权重: {struct_path}")
            checkpoint = torch.load(struct_path, map_location='cpu')

            # 处理不同的checkpoint格式
            if 'gnn' in checkpoint:
                state_dict = checkpoint['gnn']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # 加载（允许部分加载）
            self.struct_encoder.load_state_dict(state_dict, strict=False)
            print("结构编码器权重加载完成")

        if func_path and self.func_encoder:
            print(f"加载功能编码器预训练权重: {func_path}")
            checkpoint = torch.load(func_path, map_location='cpu')

            if 'gnn' in checkpoint:
                state_dict = checkpoint['gnn']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            self.func_encoder.load_state_dict(state_dict, strict=False)
            print("功能编码器权重加载完成")


# ==========================================
# 训练辅助函数
# ==========================================

def train_sf_dpl_one_epoch(model, train_loader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_ortho_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # 解包双流数据
        func_data, struct_data = batch
        func_data = func_data.to(device)
        struct_data = struct_data.to(device)

        # 检查输入
        if torch.isnan(func_data.x).any() or torch.isnan(struct_data.x).any():
            print(f"警告: Batch {batch_idx} 包含NaN，跳过")
            continue

        # 前向传播
        optimizer.zero_grad()
        logits, ortho_loss = model(struct_data, func_data)

        # 检查输出
        if torch.isnan(logits).any():
            print(f"警告: Batch {batch_idx} 输出包含NaN，跳过")
            continue

        # 计算损失
        ce_loss = F.cross_entropy(logits, func_data.y)
        loss = ce_loss + ortho_loss

        # 检查损失
        if torch.isnan(loss):
            print(f"警告: Batch {batch_idx} 损失为NaN，跳过")
            continue

        # 反向传播
        loss.backward()

        # ⭐ 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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


def evaluate_sf_dpl(model, loader, device):
    """评估模型"""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            func_data, struct_data = batch
            func_data = func_data.to(device)
            struct_data = struct_data.to(device)

            logits, ortho_loss = model(struct_data, func_data)

            # 检查nan
            if torch.isnan(logits).any():
                continue

            loss = F.cross_entropy(logits, func_data.y) + ortho_loss
            total_loss += loss.item()
            num_batches += 1

            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(func_data.y.cpu().numpy())

    # 计算指标
    import numpy as np
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    metrics = {
        'loss': total_loss / num_batches if num_batches > 0 else 0,
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='binary', zero_division=0),
    }

    # AUC（需要至少两个类别）
    if len(np.unique(all_labels)) > 1:
        try:
            metrics['auc'] = roc_auc_score(all_labels, all_probs)
        except:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0

    return metrics