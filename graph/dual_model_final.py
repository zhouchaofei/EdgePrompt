import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from model import GIN
from prompt import NodePromptplus, EdgePromptplus
from layers import HGPSLPool


class DualBranchGNN(nn.Module):
    """
    双分支GNN用于脑网络分类

    架构：
    - 结构分支：NodePromptplus + GIN + HGPSL Pooling
    - 功能分支：EdgePromptplus + GIN (无Pooling)
    - 融合层：Concat + MLP分类器

    参数：
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        num_classes: 分类类别数
        num_layers: GNN层数
        pooling_ratio: 结构分支的pooling比例
        dropout: Dropout概率
        num_anchors: Prompt锚点数量
        use_consistency: 是否使用一致性损失
    """

    def __init__(self, input_dim, hidden_dim=64, num_classes=2,
                 num_layers=3, pooling_ratio=0.5, dropout=0.5,
                 num_anchors=5, use_consistency=False):
        super(DualBranchGNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_consistency = use_consistency

        # ============================================================
        # 分支 A: 结构分支 (Anatomical)
        # 特点：稀疏图 + NodePrompt+ + Pooling
        # ============================================================

        # 1. Node Prompt (应用在输入层)
        self.struct_prompt = NodePromptplus(input_dim, num_anchors)

        # 2. GIN Encoder (分两段以插入Pooling)
        # 第一段：1层
        self.struct_gin_pre = GIN(
            num_layer=1,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            drop_ratio=dropout
        )

        # 3. HGPSL Pooling
        self.struct_pool = HGPSLPool(
            in_channels=hidden_dim,
            ratio=pooling_ratio,
            sample=True,  # 使用快速采样模式
            sparse=True,  # 使用Sparsemax
            sl=True,  # 启用结构学习
            lamb=1.0
        )

        # 4. GIN Encoder (第二段：剩余层)
        if num_layers > 1:
            self.struct_gin_post = GIN(
                num_layer=num_layers - 1,
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                drop_ratio=dropout
            )
        else:
            self.struct_gin_post = None

        # ============================================================
        # 分支 B: 功能分支 (Functional)
        # 特点：稠密图 + EdgePrompt+ + 无Pooling
        # ============================================================

        # 1. Edge Prompt (每层都需要)
        dim_list = [input_dim] + [hidden_dim] * (num_layers - 1)
        self.func_prompt = EdgePromptplus(dim_list, num_anchors)

        # 2. GIN Encoder (完整的num_layers层)
        self.func_gin = GIN(
            num_layer=num_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            drop_ratio=dropout
        )

        # ============================================================
        # 融合与分类
        # ============================================================

        # Readout: Mean + Max pooling -> 2*hidden_dim per branch
        fusion_dim = hidden_dim * 4  # (2 + 2) * hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # 可选：每个分支独立的分类器（用于一致性损失）
        if use_consistency:
            self.struct_classifier = nn.Linear(hidden_dim * 2, num_classes)
            self.func_classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward_struct_branch(self, x, edge_index, edge_attr, batch):
        """
        结构分支前向传播

        流程：NodePrompt+ -> GIN -> Pooling -> GIN -> Readout
        """
        # 1. 应用Node Prompt
        x = self.struct_prompt.add(x)

        # 2. 第一段GIN（1层）
        x = self.struct_gin_pre.convs[0](x, edge_index, edge_prompt=False)
        x = self.struct_gin_pre.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. HGPSL Pooling
        # 如果没有edge_attr，创建全1向量
        if edge_attr is None:
            edge_attr = x.new_ones((edge_index.size(1), 1))

        x, edge_index, edge_attr, batch = self.struct_pool(
            x, edge_index, edge_attr, batch
        )

        # 4. 第二段GIN（剩余层）
        if self.struct_gin_post is not None:
            for i in range(len(self.struct_gin_post.convs)):
                x = self.struct_gin_post.convs[i](x, edge_index, edge_prompt=False)
                x = self.struct_gin_post.batch_norms[i](x)

                if i == len(self.struct_gin_post.convs) - 1:
                    # 最后一层
                    x = F.dropout(x, p=self.dropout, training=self.training)
                else:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

        # 5. Readout (Mean + Max)
        z = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch)
        ], dim=1)

        return z

    def forward_func_branch(self, x, edge_index, batch):
        """
        功能分支前向传播

        流程：EdgePrompt+ + GIN (每层注入prompt) -> Readout
        """
        h = x

        # 逐层传播，每层注入EdgePrompt
        for layer in range(self.num_layers):
            # 1. 生成当前层的Edge Prompt
            edge_prompt = self.func_prompt.get_prompt(h, edge_index, layer)

            # 2. GIN卷积（注入prompt）
            h = self.func_gin.convs[layer](h, edge_index, edge_prompt=edge_prompt)
            h = self.func_gin.batch_norms[layer](h)

            # 3. 激活和Dropout
            if layer == self.num_layers - 1:
                # 最后一层
                h = F.dropout(h, p=self.dropout, training=self.training)
            else:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        # 4. Readout (Mean + Max)
        z = torch.cat([
            global_mean_pool(h, batch),
            global_max_pool(h, batch)
        ], dim=1)

        return z

    def forward(self, data):
        """
        完整前向传播

        Args:
            data: PyG Data对象，必须包含：
                - x: [N_total, D] 节点特征（117个节点）
                - batch: [N_total] batch索引
                - edge_index_struct: [2, E_s] 结构图边
                - edge_index_func: [2, E_f] 功能图边
                - roi_mask: [N_total] ROI掩码（前116个为True）
                - edge_attr_struct: (可选) [E_s, 1] 结构图边权重

        Returns:
            logits: [B, num_classes] 分类logits
            z_struct: [B, hidden_dim*2] 结构分支特征
            z_func: [B, hidden_dim*2] 功能分支特征
            (可选) logits_struct, logits_func: 独立分类器的输出
        """
        x, batch = data.x, data.batch

        # ============================================================
        # 分支 A: 结构分支 (全部117个节点)
        # ============================================================
        z_struct = self.forward_struct_branch(
            x=x,
            edge_index=data.edge_index_struct,
            edge_attr=data.edge_attr_struct if hasattr(data, 'edge_attr_struct') else None,
            batch=batch
        )

        # ============================================================
        # 分支 B: 功能分支 (仅116个ROI节点)
        # ============================================================
        # 过滤出ROI节点
        roi_mask = data.roi_mask.bool()
        x_func = x[roi_mask]
        batch_func = batch[roi_mask]

        z_func = self.forward_func_branch(
            x=x_func,
            edge_index=data.edge_index_func,
            batch=batch_func
        )

        # ============================================================
        # 融合与分类
        # ============================================================
        z_fusion = torch.cat([z_struct, z_func], dim=1)  # [B, 4*hidden_dim]
        logits = self.classifier(z_fusion)

        # 如果启用一致性损失，计算每个分支的独立预测
        if self.use_consistency:
            logits_struct = self.struct_classifier(z_struct)
            logits_func = self.func_classifier(z_func)
            return logits, z_struct, z_func, logits_struct, logits_func

        return logits, z_struct, z_func

    def get_embeddings(self, data):
        """
        仅提取特征向量（用于可视化或下游任务）

        Returns:
            z_struct: 结构分支特征
            z_func: 功能分支特征
            z_fusion: 融合特征
        """
        with torch.no_grad():
            if self.use_consistency:
                logits, z_struct, z_func, _, _ = self.forward(data)
            else:
                logits, z_struct, z_func = self.forward(data)
            z_fusion = torch.cat([z_struct, z_func], dim=1)

        return z_struct, z_func, z_fusion


# ============================================================
# 损失函数
# ============================================================

def orthogonality_loss(z1, z2):
    """
    正交损失：强制两个分支学习不相关的特征

    原理：最小化两个特征向量的余弦相似度

    Args:
        z1, z2: [B, D] 两个分支的特征向量

    Returns:
        loss: 标量，范围[0, 1]
    """
    # L2归一化
    z1_norm = F.normalize(z1, p=2, dim=1)
    z2_norm = F.normalize(z2, p=2, dim=1)

    # 计算批内平均余弦相似度的绝对值
    cos_sim = torch.sum(z1_norm * z2_norm, dim=1)  # [B]
    loss = torch.mean(torch.abs(cos_sim))

    return loss


def consistency_loss(logits1, logits2, temperature=2.0):
    """
    一致性损失：确保两个分支的预测分布一致

    使用KL散度衡量两个概率分布的差异

    Args:
        logits1, logits2: [B, num_classes] 两个分支的logits
        temperature: 温度参数（软化概率分布）

    Returns:
        loss: 标量
    """
    # 使用温度软化softmax（类似Knowledge Distillation）
    prob1 = F.softmax(logits1 / temperature, dim=1)
    prob2 = F.softmax(logits2 / temperature, dim=1)

    # 对称KL散度（双向）
    kl_1_2 = F.kl_div(prob1.log(), prob2, reduction='batchmean')
    kl_2_1 = F.kl_div(prob2.log(), prob1, reduction='batchmean')

    return (kl_1_2 + kl_2_1) / 2


def compute_total_loss(logits, labels, z_struct, z_func,
                       logits_struct=None, logits_func=None,
                       lambda_orth=0.1, lambda_cons=0.05):
    """
    计算总损失

    Args:
        logits: 融合分类器的logits
        labels: 真实标签
        z_struct, z_func: 两个分支的特征
        logits_struct, logits_func: (可选) 独立分类器的logits
        lambda_orth: 正交损失权重
        lambda_cons: 一致性损失权重

    Returns:
        total_loss: 总损失
        loss_dict: 各项损失的字典（用于记录）
    """
    # 1. 分类损失（必须）
    loss_cls = F.cross_entropy(logits, labels)

    # 2. 正交损失（推荐）
    loss_orth = orthogonality_loss(z_struct, z_func)

    # 3. 一致性损失（可选）
    if logits_struct is not None and logits_func is not None:
        loss_cons = consistency_loss(logits_struct, logits_func)
        total_loss = loss_cls + lambda_orth * loss_orth + lambda_cons * loss_cons

        return total_loss, {
            'cls': loss_cls.item(),
            'orth': loss_orth.item(),
            'cons': loss_cons.item(),
            'total': total_loss.item()
        }
    else:
        total_loss = loss_cls + lambda_orth * loss_orth

        return total_loss, {
            'cls': loss_cls.item(),
            'orth': loss_orth.item(),
            'total': total_loss.item()
        }


# ============================================================
# 使用示例
# ============================================================

if __name__ == '__main__':
    # 模拟数据
    from torch_geometric.data import Data, Batch

    # 创建模型
    model = DualBranchGNN(
        input_dim=308,  # 特征维度
        hidden_dim=64,
        num_classes=2,
        num_layers=3,
        pooling_ratio=0.5,
        dropout=0.5,
        num_anchors=5,
        use_consistency=True  # 启用一致性损失
    )

    print("模型参数量:", sum(p.numel() for p in model.parameters()))

    # 创建一个样本图
    data = Data(
        x=torch.randn(117, 308),  # 117个节点
        edge_index_struct=torch.randint(0, 117, (2, 500)),
        edge_index_func=torch.randint(0, 116, (2, 300)),
        y=torch.tensor([1]),
        roi_mask=torch.cat([torch.ones(116), torch.zeros(1)]).bool(),
        batch=torch.zeros(117, dtype=torch.long)
    )

    # 前向传播
    logits, z_struct, z_func, logits_struct, logits_func = model(data)

    print("\n输出形状:")
    print(f"logits: {logits.shape}")
    print(f"z_struct: {z_struct.shape}")
    print(f"z_func: {z_func.shape}")

    # 计算损失
    total_loss, loss_dict = compute_total_loss(
        logits, data.y, z_struct, z_func,
        logits_struct, logits_func,
        lambda_orth=0.1,
        lambda_cons=0.05
    )

    print("\n损失:")
    for k, v in loss_dict.items():
        print(f"{k}: {v:.4f}")