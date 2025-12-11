"""
dual_model_final.py - 双分支GNN模型（修复版）
结构分支 + 功能分支 + 跨分支正交损失 + 一致性损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

from model import GIN
from prompt import NodePromptplus, EdgePromptplus
from layers import HGPSLPool


class DualBranchGNN(nn.Module):
    """
    双分支GNN模型

    架构：
    1. 结构分支：固定图 + NodePromptplus + GIN + HGPSL Pooling
    2. 功能分支：加权图 + EdgePromptplus + GIN
    3. 融合层：拼接 + 分类器
    """

    def __init__(self, input_dim, hidden_dim=64, num_classes=2, num_layers=3,
                 pooling_ratio=0.5, dropout=0.5, num_anchors=5, use_consistency=False):
        super(DualBranchGNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.use_consistency = use_consistency

        # ===== 结构分支 =====
        self.struct_prompt = NodePromptplus(input_dim, num_anchors=num_anchors)
        self.struct_gin_pre = GIN(num_layer=1, input_dim=input_dim, hidden_dim=hidden_dim, drop_ratio=dropout)
        self.struct_pool = HGPSLPool(hidden_dim, ratio=pooling_ratio, sample=True, sparse=True, sl=True)
        self.struct_gin_post = GIN(num_layer=2, input_dim=hidden_dim, hidden_dim=hidden_dim, drop_ratio=dropout)

        # ===== 功能分支 =====
        self.func_prompt = EdgePromptplus(
            [input_dim, hidden_dim, hidden_dim],
            num_anchors=num_anchors
        )
        self.func_gin = GIN(num_layer=num_layers, input_dim=input_dim, hidden_dim=hidden_dim, drop_ratio=dropout)

        # ===== 融合分类器 =====
        # 双分支Readout后各产生hidden_dim*2的特征（mean+max pooling）
        fusion_dim = hidden_dim * 2 * 2  # 结构 + 功能
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # ===== 一致性损失用的分支分类器（可选）=====
        if use_consistency:
            self.struct_classifier = nn.Linear(hidden_dim * 2, num_classes)
            self.func_classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, data):
        """
        前向传播

        Returns:
            logits: 融合分类logits
            z_struct: 结构分支图级表示
            z_func: 功能分支图级表示
            logits_struct: 结构分支独立分类（如果use_consistency=True）
            logits_func: 功能分支独立分类（如果use_consistency=True）
        """
        x_orig = data.x
        batch = data.batch

        # 运行时NaN防御
        x_orig = torch.nan_to_num(x_orig, nan=0.0, posinf=0.0, neginf=0.0)

        # ========== 结构分支 ==========
        x_struct = x_orig.clone()
        edge_index_struct = data.edge_index_struct
        batch_struct = batch.clone()

        # 1. Node Prompt
        x_struct = self.struct_prompt.add(x_struct)

        # 2. GIN第1层
        x_struct = self.struct_gin_pre.convs[0](x_struct, edge_index_struct, edge_prompt=False)
        x_struct = F.relu(self.struct_gin_pre.batch_norms[0](x_struct))

        # 3. HGPSL Pooling（修复：创建1D伪edge_attr）
        pseudo_edge_attr = x_struct.new_ones(edge_index_struct.size(1))
        x_struct, edge_index_struct, edge_attr_struct, batch_struct = self.struct_pool(
            x_struct, edge_index_struct, pseudo_edge_attr, batch_struct
        )

        # 4. GIN第2层（2个子层）
        for i in range(len(self.struct_gin_post.convs)):
            x_struct = self.struct_gin_post.convs[i](x_struct, edge_index_struct, edge_prompt=False)
            x_struct = F.relu(self.struct_gin_post.batch_norms[i](x_struct))

        # 5. Readout
        z_struct = torch.cat([
            global_mean_pool(x_struct, batch_struct),
            global_max_pool(x_struct, batch_struct)
        ], dim=1)  # (B, hidden_dim * 2)

        # ========== 功能分支 ==========
        # 只使用ROI节点（去除Global节点）
        mask = data.roi_mask.bool()

        # 创建节点索引映射：旧索引 -> 新索引
        old_to_new = torch.full((x_orig.size(0),), -1, dtype=torch.long, device=x_orig.device)
        old_to_new[mask] = torch.arange(mask.sum(), device=x_orig.device)

        # 过滤节点和batch
        x_func = x_orig[mask]
        batch_func = batch[mask]

        # 重新映射edge_index
        edge_index_func = data.edge_index_func
        row, col = edge_index_func[0], edge_index_func[1]

        # 只保留两端节点都在ROI中的边
        valid_edges = mask[row] & mask[col]
        row = old_to_new[row[valid_edges]]
        col = old_to_new[col[valid_edges]]
        edge_index_func = torch.stack([row, col], dim=0)

        # 逐层GIN + Edge Prompt
        for i in range(self.func_gin.num_layer):
            # 获取Edge Prompt
            edge_prompt = self.func_prompt.get_prompt(x_func, edge_index_func, layer=i)

            # GIN卷积
            x_func = self.func_gin.convs[i](x_func, edge_index_func, edge_prompt=edge_prompt)
            x_func = self.func_gin.batch_norms[i](x_func)

            # 激活函数（最后一层不用ReLU）
            if i < self.func_gin.num_layer - 1:
                x_func = F.relu(x_func)

        # Readout
        z_func = torch.cat([
            global_mean_pool(x_func, batch_func),
            global_max_pool(x_func, batch_func)
        ], dim=1)  # (B, hidden_dim * 2)

        # ========== 融合与分类 ==========
        z_fused = torch.cat([z_struct, z_func], dim=1)  # (B, hidden_dim * 4)
        logits = self.classifier(z_fused)

        # 一致性损失用的分支分类（可选）
        if self.use_consistency:
            logits_struct = self.struct_classifier(z_struct)
            logits_func = self.func_classifier(z_func)
            return logits, z_struct, z_func, logits_struct, logits_func
        else:
            return logits, z_struct, z_func


def compute_total_loss(logits, labels, z_struct, z_func,
                       logits_struct=None, logits_func=None,
                       lambda_orth=0.1, lambda_cons=0.05):
    """
    计算总损失

    组成：
    1. 分类损失（主任务）
    2. 正交损失（促进两分支学习互补特征）
    3. 一致性损失（保证各分支独立预测的一致性）

    Args:
        logits: 融合分类logits
        labels: 真实标签
        z_struct: 结构分支表示
        z_func: 功能分支表示
        logits_struct: 结构分支独立分类（可选）
        logits_func: 功能分支独立分类（可选）
        lambda_orth: 正交损失权重
        lambda_cons: 一致性损失权重

    Returns:
        total_loss: 总损失
        loss_dict: 各损失分量
    """
    # 1. 分类损失
    loss_cls = F.cross_entropy(logits, labels)

    # 2. 正交损失（Frobenius范数）
    # 目标：z_struct 和 z_func 的表示尽可能正交
    # 计算相似度矩阵：S = z_struct @ z_func^T
    # 正交损失：||S||_F^2
    z_struct_norm = F.normalize(z_struct, p=2, dim=1)
    z_func_norm = F.normalize(z_func, p=2, dim=1)
    similarity = torch.mm(z_struct_norm, z_func_norm.t())  # (B, B)
    loss_orth = torch.norm(similarity, p='fro') ** 2 / z_struct.size(0)

    # 3. 一致性损失（可选）
    loss_cons = torch.tensor(0.0, device=logits.device)
    if logits_struct is not None and logits_func is not None:
        # 各分支的独立分类也应该正确
        loss_cons = (
                            F.cross_entropy(logits_struct, labels) +
                            F.cross_entropy(logits_func, labels)
                    ) / 2.0

    # 总损失
    total_loss = loss_cls + lambda_orth * loss_orth + lambda_cons * loss_cons

    # 返回各损失分量（用于监控）
    loss_dict = {
        'cls': loss_cls.item(),
        'orth': loss_orth.item(),
        'cons': loss_cons.item(),
        'total': total_loss.item()
    }

    return total_loss, loss_dict


if __name__ == '__main__':
    # 测试模型
    import torch_geometric
    from torch_geometric.data import Data, Batch

    # 创建测试数据
    test_data = Data(
        x=torch.randn(117, 308),  # 117个节点，308维特征
        edge_index_struct=torch.randint(0, 117, (2, 500)),
        edge_index_func=torch.randint(0, 116, (2, 300)),
        edge_attr_func=torch.rand(300, 1),
        y=torch.tensor([0]),
        batch=torch.zeros(117, dtype=torch.long),
        roi_mask=torch.cat([torch.ones(116, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)])
    )

    # 创建batch
    batch = Batch.from_data_list([test_data, test_data])

    # 测试模型
    model = DualBranchGNN(
        input_dim=308,
        hidden_dim=64,
        num_classes=2,
        use_consistency=True
    )

    print("测试双分支模型...")
    logits, z_struct, z_func, logits_struct, logits_func = model(batch)

    print(f"✓ 融合分类logits: {logits.shape}")
    print(f"✓ 结构分支表示: {z_struct.shape}")
    print(f"✓ 功能分支表示: {z_func.shape}")
    print(f"✓ 结构分支分类: {logits_struct.shape}")
    print(f"✓ 功能分支分类: {logits_func.shape}")

    # 测试损失
    loss, loss_dict = compute_total_loss(
        logits, batch.y, z_struct, z_func,
        logits_struct, logits_func
    )

    print(f"\n损失测试:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\n✅ 模型测试通过！")