import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot


class EdgePrompt(nn.Module):
    def __init__(self, dim_list):
        super(EdgePrompt, self).__init__()
        self.global_prompt = nn.ParameterList([nn.Parameter(torch.Tensor(1, dim)) for dim in dim_list])
        self.reset_parameters()

    def reset_parameters(self):
        for prompt in self.global_prompt:
            glorot(prompt)

    def get_prompt(self, x, edge_index, layer):
        return self.global_prompt[layer]


class EdgePromptplus(nn.Module):
    def __init__(self, dim_list, num_anchors):
        super(EdgePromptplus, self).__init__()
        self.anchor_prompt = nn.ParameterList([nn.Parameter(torch.Tensor(num_anchors, dim)) for dim in dim_list])
        self.w = nn.ModuleList([nn.Linear(2 * dim, num_anchors) for dim in dim_list])
        self.reset_parameters()

    def reset_parameters(self):
        for anchor in self.anchor_prompt:
            glorot(anchor)
        for w in self.w:
            w.reset_parameters()

    def get_prompt(self, x, edge_index, layer):
        combined_x = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        b = F.softmax(F.leaky_relu(self.w[layer](combined_x)), dim=1)
        prompt = b.mm(self.anchor_prompt[layer])
        return prompt


class NodePrompt(nn.Module):
    """简单节点提示：为节点特征添加可学习向量"""

    def __init__(self, input_dim):
        super(NodePrompt, self).__init__()
        self.global_prompt = nn.Parameter(torch.Tensor(1, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_prompt)

    def add(self, x):
        return x + self.global_prompt


class NodePromptplus(nn.Module):
    """基于注意力的节点提示：多个可学习锚点"""

    def __init__(self, input_dim, num_anchors):
        super(NodePromptplus, self).__init__()
        self.anchor_prompt = nn.Parameter(torch.Tensor(num_anchors, input_dim))
        self.attention = nn.Linear(input_dim, num_anchors)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.anchor_prompt)
        self.attention.reset_parameters()

    def add(self, x):
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)
        prompt = weights.mm(self.anchor_prompt)
        return x + prompt


class SerialNodeEdgePrompt(nn.Module):
    """方案一：串行融合 - 输入层节点提示 + 各层边提示"""

    def __init__(self, dim_list, edge_type='EdgePrompt', node_type='NodePrompt',
                 edge_num_anchors=5, node_num_anchors=5):
        super(SerialNodeEdgePrompt, self).__init__()

        # 节点提示（仅用于输入层）
        self.node_type = node_type
        if node_type == 'NodePrompt':
            self.node_prompt = NodePrompt(dim_list[0])
        elif node_type == 'NodePromptplus':
            self.node_prompt = NodePromptplus(dim_list[0], node_num_anchors)
        else:
            self.node_prompt = None

        # 边提示（用于所有层）
        self.edge_type = edge_type
        if edge_type == 'EdgePrompt':
            self.edge_prompt = EdgePrompt(dim_list)
        elif edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = None

    def get_node_prompt(self, x):
        """对输入特征应用节点提示"""
        if self.node_prompt is not None:
            return self.node_prompt.add(x)
        return x

    def get_edge_prompt(self, x, edge_index, layer):
        """获取特定层的边提示"""
        if self.edge_prompt is not None:
            return self.edge_prompt.get_prompt(x, edge_index, layer)
        return False


class ParallelNodeEdgePrompt(nn.Module):
    """方案二：并行融合 - 每层同时使用节点和边提示"""

    def __init__(self, dim_list, edge_type='EdgePrompt', node_type='NodePrompt',
                 edge_num_anchors=5, node_num_anchors=5, fusion_method='weighted'):
        super(ParallelNodeEdgePrompt, self).__init__()

        self.fusion_method = fusion_method

        # 节点提示（所有层）
        self.node_type = node_type
        if node_type == 'NodePrompt':
            self.node_prompts = nn.ModuleList([NodePrompt(dim) for dim in dim_list])
        elif node_type == 'NodePromptplus':
            self.node_prompts = nn.ModuleList([NodePromptplus(dim, node_num_anchors) for dim in dim_list])
        else:
            self.node_prompts = None

        # 边提示（所有层）
        self.edge_type = edge_type
        if edge_type == 'EdgePrompt':
            self.edge_prompt = EdgePrompt(dim_list)
        elif edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = None

        # 融合权重
        if fusion_method == 'weighted':
            self.fusion_weights = nn.ParameterList([nn.Parameter(torch.tensor([0.5, 0.5])) for _ in dim_list])
        elif fusion_method == 'gated':
            self.gates = nn.ModuleList([nn.Linear(dim, 1) for dim in dim_list])

    def get_prompts(self, x, edge_index, layer):
        """获取特定层的节点和边提示"""
        node_prompted_x = x
        edge_prompt_value = False

        # 应用节点提示
        if self.node_prompts is not None:
            node_prompted_x = self.node_prompts[layer].add(x)

        # 获取边提示
        if self.edge_prompt is not None:
            edge_prompt_value = self.edge_prompt.get_prompt(x, edge_index, layer)

        return node_prompted_x, edge_prompt_value


class InteractiveNodeEdgePrompt(nn.Module):
    """方案三：交互融合 - 节点-边提示通过注意力机制交互"""

    def __init__(self, dim_list, edge_type='EdgePrompt', node_type='NodePrompt',
                 edge_num_anchors=5, node_num_anchors=5):
        super(InteractiveNodeEdgePrompt, self).__init__()

        # 节点提示
        self.node_type = node_type
        if node_type == 'NodePrompt':
            self.node_prompts = nn.ModuleList([NodePrompt(dim) for dim in dim_list])
        elif node_type == 'NodePromptplus':
            self.node_prompts = nn.ModuleList([NodePromptplus(dim, node_num_anchors) for dim in dim_list])
        else:
            self.node_prompts = None

        # 边提示
        self.edge_type = edge_type
        if edge_type == 'EdgePrompt':
            self.edge_prompt = EdgePrompt(dim_list)
        elif edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = None

        # 交互注意力机制 - 修复头数问题
        self.node_edge_attention = nn.ModuleList()
        for dim in dim_list:
            # 确保 num_heads 能整除 embed_dim
            if dim % 4 == 0:
                num_heads = 4
            elif dim % 2 == 0:
                num_heads = 2
            else:
                num_heads = 1

            self.node_edge_attention.append(
                nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
            )

        # 特征融合层
        self.fusion_layers = nn.ModuleList([
            nn.Linear(dim * 2, dim) for dim in dim_list
        ])

    def get_interactive_prompts(self, x, edge_index, layer):
        """获取交互式提示"""
        # 获取基础节点提示
        node_prompted_x = x
        if self.node_prompts is not None:
            node_prompted_x = self.node_prompts[layer].add(x)

        # 获取边提示
        edge_prompt = None
        if self.edge_prompt is not None:
            edge_prompt = self.edge_prompt.get_prompt(x, edge_index, layer)

        # 交互注意力
        if edge_prompt is not None:
            # 将边提示转换为节点级别的信息
            edge_info = torch.zeros_like(x)

            # 安全地处理边提示信息
            if edge_prompt.size(0) == edge_index.size(1):  # 每条边一个提示
                # 累加来自所有边的信息到节点
                edge_info.index_add_(0, edge_index[0], edge_prompt)
                edge_info.index_add_(0, edge_index[1], edge_prompt)
            else:  # 全局边提示
                edge_info = edge_info + edge_prompt

            # 应用多头注意力
            node_feats = node_prompted_x.unsqueeze(0)  # [1, N, D]
            edge_feats = edge_info.unsqueeze(0)  # [1, N, D]

            try:
                attended_features, _ = self.node_edge_attention[layer](
                    node_feats, edge_feats, edge_feats
                )
                attended_features = attended_features.squeeze(0)  # [N, D]

                # 特征融合
                combined_features = torch.cat([node_prompted_x, attended_features], dim=1)
                final_x = self.fusion_layers[layer](combined_features)

                return final_x, edge_prompt
            except Exception as e:
                print(f"Attention mechanism failed at layer {layer}: {e}")
                # 如果注意力机制失败，回退到简单的加法融合
                final_x = node_prompted_x + edge_info
                return final_x, edge_prompt

        return node_prompted_x, edge_prompt

class ComplementaryNodeEdgePrompt(nn.Module):
    """方案四：互补学习融合 - 通过辅助任务引导节点和边提示学习不同信息"""

    def __init__(self, dim_list, edge_type='EdgePrompt', node_type='NodePrompt',
                 edge_num_anchors=5, node_num_anchors=5,
                 recon_weight=0.1, link_pred_weight=0.1):
        super(ComplementaryNodeEdgePrompt, self).__init__()

        # 节点提示
        self.node_type = node_type
        if node_type == 'NodePrompt':
            self.node_prompts = nn.ModuleList([NodePrompt(dim) for dim in dim_list])
        elif node_type == 'NodePromptplus':
            self.node_prompts = nn.ModuleList([NodePromptplus(dim, node_num_anchors) for dim in dim_list])
        else:
            self.node_prompts = None

        # 边提示
        self.edge_type = edge_type
        if edge_type == 'EdgePrompt':
            self.edge_prompt = EdgePrompt(dim_list)
        elif edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = None

        # 重构解码器（用于节点特征重构）
        self.recon_decoders = nn.ModuleList([
            nn.Linear(dim, dim) for dim in dim_list
        ])

        # 链接预测器（用于边存在性预测）
        self.link_predictors = nn.ModuleList([
            nn.Linear(dim * 2, 1) for dim in dim_list
        ])

        # 辅助任务权重
        self.recon_weight = recon_weight
        self.link_pred_weight = link_pred_weight

        # 融合门控
        self.fusion_gates = nn.ModuleList([
            nn.Linear(dim * 2, 2) for dim in dim_list
        ])

    def get_complementary_prompts(self, x, edge_index, layer):
        """获取互补的节点和边提示，同时返回辅助任务所需的中间结果"""
        # 节点提示
        node_prompted_x = x
        if self.node_prompts is not None:
            node_prompted_x = self.node_prompts[layer].add(x)

        # 边提示
        edge_prompt = None
        if self.edge_prompt is not None:
            edge_prompt = self.edge_prompt.get_prompt(x, edge_index, layer)

        # 边信息聚合到节点
        edge_aggregated = torch.zeros_like(x)
        if edge_prompt is not None and edge_prompt.size(0) == edge_index.size(1):
            edge_aggregated.index_add_(0, edge_index[0], edge_prompt)
            edge_aggregated.index_add_(0, edge_index[1], edge_prompt)

        # 门控融合
        combined = torch.cat([node_prompted_x, edge_aggregated], dim=1)
        gates = F.softmax(self.fusion_gates[layer](combined), dim=1)

        # 加权组合
        final_x = gates[:, 0:1] * node_prompted_x + gates[:, 1:2] * edge_aggregated

        # 返回最终结果和中间结果（用于计算辅助损失）
        return final_x, edge_prompt, node_prompted_x, edge_aggregated

    def compute_auxiliary_losses(self, x, edge_index, layer,
                                 node_prompted_x, edge_aggregated):
        """计算辅助损失"""
        losses = {}

        # 1. 节点特征重构损失
        if node_prompted_x is not None:
            reconstructed = self.recon_decoders[layer](node_prompted_x)
            recon_loss = F.mse_loss(reconstructed, x)
            losses['recon'] = recon_loss * self.recon_weight

        # 2. 链接预测损失
        if edge_aggregated is not None and edge_aggregated.sum() != 0:
            # 正样本：存在的边
            row, col = edge_index
            pos_features = torch.cat([
                edge_aggregated[row],
                edge_aggregated[col]
            ], dim=1)
            pos_pred = self.link_predictors[layer](pos_features)

            # 负采样
            num_neg = min(edge_index.size(1), 100)  # 限制负样本数量
            neg_row = torch.randint(0, x.size(0), (num_neg,), device=x.device)
            neg_col = torch.randint(0, x.size(0), (num_neg,), device=x.device)
            neg_features = torch.cat([
                edge_aggregated[neg_row],
                edge_aggregated[neg_col]
            ], dim=1)
            neg_pred = self.link_predictors[layer](neg_features)

            # 二元交叉熵损失
            link_loss = F.binary_cross_entropy_with_logits(
                torch.cat([pos_pred, neg_pred], dim=0),
                torch.cat([
                    torch.ones(pos_pred.size(0), 1, device=x.device),
                    torch.zeros(neg_pred.size(0), 1, device=x.device)
                ], dim=0)
            )
            losses['link'] = link_loss * self.link_pred_weight

        return losses


class ContrastiveNodeEdgePrompt(nn.Module):
    """方案五：对比学习融合 - 通过对比学习增强提示的鲁棒性"""

    def __init__(self, dim_list, edge_type='EdgePrompt', node_type='NodePrompt',
                 edge_num_anchors=5, node_num_anchors=5,
                 temperature=0.5, contrast_weight=0.1):
        super(ContrastiveNodeEdgePrompt, self).__init__()

        # 节点提示
        self.node_type = node_type
        if node_type == 'NodePrompt':
            self.node_prompts = nn.ModuleList([NodePrompt(dim) for dim in dim_list])
        elif node_type == 'NodePromptplus':
            self.node_prompts = nn.ModuleList([NodePromptplus(dim, node_num_anchors) for dim in dim_list])
        else:
            self.node_prompts = None

        # 边提示
        self.edge_type = edge_type
        if edge_type == 'EdgePrompt':
            self.edge_prompt = EdgePrompt(dim_list)
        elif edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = None

        # 投影头（用于对比学习）
        self.projection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim // 2)
            ) for dim in dim_list
        ])

        # 对比学习参数
        self.temperature = temperature
        self.contrast_weight = contrast_weight

    def augment_graph(self, x, edge_index, aug_type='dropout'):
        """图数据增强"""
        if aug_type == 'dropout':
            x_aug = F.dropout(x, p=0.2, training=self.training)
            edge_aug = edge_index
        elif aug_type == 'edge_drop':
            edge_mask = torch.rand(edge_index.size(1), device=edge_index.device) > 0.1
            edge_aug = edge_index[:, edge_mask]
            x_aug = x
        else:
            x_aug = x
            edge_aug = edge_index
        return x_aug, edge_aug

    def get_contrastive_prompts(self, x, edge_index, layer):
        """获取对比学习增强的提示"""
        # 生成两个增强视图
        if self.training:
            x_aug1, edge_aug1 = self.augment_graph(x, edge_index, 'dropout')
            x_aug2, edge_aug2 = self.augment_graph(x, edge_index, 'edge_drop')
        else:
            x_aug1, edge_aug1 = x, edge_index
            x_aug2, edge_aug2 = x, edge_index

        # 视图1的提示
        node_prompted_x1 = x_aug1
        if self.node_prompts is not None:
            node_prompted_x1 = self.node_prompts[layer].add(x_aug1)

        edge_prompt1 = None
        if self.edge_prompt is not None:
            edge_prompt1 = self.edge_prompt.get_prompt(x_aug1, edge_aug1, layer)

        # 视图2的提示
        node_prompted_x2 = x_aug2
        if self.node_prompts is not None:
            node_prompted_x2 = self.node_prompts[layer].add(x_aug2)

        edge_prompt2 = None
        if self.edge_prompt is not None:
            edge_prompt2 = self.edge_prompt.get_prompt(x_aug2, edge_aug2, layer)

        # 融合
        final_x1 = self._fuse_prompts(node_prompted_x1, edge_prompt1, edge_aug1)
        final_x2 = self._fuse_prompts(node_prompted_x2, edge_prompt2, edge_aug2)

        # 主视图（用于下游任务）
        node_prompted_x = x
        if self.node_prompts is not None:
            node_prompted_x = self.node_prompts[layer].add(x)

        edge_prompt = None
        if self.edge_prompt is not None:
            edge_prompt = self.edge_prompt.get_prompt(x, edge_index, layer)

        final_x = self._fuse_prompts(node_prompted_x, edge_prompt, edge_index)

        return final_x, edge_prompt, (final_x1, final_x2)

    def _fuse_prompts(self, node_prompted_x, edge_prompt, edge_index):
        """融合节点和边提示"""
        if edge_prompt is not None and edge_prompt.size(0) == edge_index.size(1):
            edge_aggregated = torch.zeros_like(node_prompted_x)
            edge_aggregated.index_add_(0, edge_index[0], edge_prompt)
            edge_aggregated.index_add_(0, edge_index[1], edge_prompt)
            return node_prompted_x + 0.5 * edge_aggregated
        return node_prompted_x

    def compute_contrastive_loss(self, views, layer, batch=None):
        """计算对比损失"""
        if views is None or not self.training:
            return torch.tensor(0.0, device=views[0].device if views else None)

        z1, z2 = views

        # 通过投影头
        z1_proj = self.projection_heads[layer](z1)
        z2_proj = self.projection_heads[layer](z2)

        # 如果有batch信息，进行图级别的对比
        if batch is not None:
            # 图级别池化
            from torch_geometric.nn import global_mean_pool
            z1_graph = global_mean_pool(z1_proj, batch)
            z2_graph = global_mean_pool(z2_proj, batch)

            # 归一化
            z1_graph = F.normalize(z1_graph, dim=1)
            z2_graph = F.normalize(z2_graph, dim=1)

            # 计算相似度矩阵
            batch_size = z1_graph.size(0)

            # 正样本对：同一个图的两个视图
            sim_matrix = torch.matmul(z1_graph, z2_graph.t()) / self.temperature

            # 创建标签（对角线元素为正样本）
            labels = torch.arange(batch_size, device=z1_graph.device)

            # NT-Xent损失
            loss_1 = F.cross_entropy(sim_matrix, labels)
            loss_2 = F.cross_entropy(sim_matrix.t(), labels)
            loss = (loss_1 + loss_2) / 2

        else:
            # 节点级别的对比
            z1_norm = F.normalize(z1_proj, dim=1)
            z2_norm = F.normalize(z2_proj, dim=1)

            # 正样本对的相似度
            pos_sim = (z1_norm * z2_norm).sum(dim=1) / self.temperature

            # 简化的对比损失
            loss = -pos_sim.mean()

        return loss * self.contrast_weight


class SpectralNodeEdgePrompt(nn.Module):
    """方案六：图频域融合 - 基于图傅立叶变换的节点-边提示融合"""

    def __init__(self, dim_list, edge_type='EdgePrompt', node_type='NodePrompt',
                 edge_num_anchors=5, node_num_anchors=5, num_filters=8):
        super(SpectralNodeEdgePrompt, self).__init__()

        # 节点提示
        self.node_type = node_type
        if node_type == 'NodePrompt':
            self.node_prompts = nn.ModuleList([NodePrompt(dim) for dim in dim_list])
        elif node_type == 'NodePromptplus':
            self.node_prompts = nn.ModuleList([NodePromptplus(dim, node_num_anchors) for dim in dim_list])
        else:
            self.node_prompts = None

        # 边提示
        self.edge_type = edge_type
        if edge_type == 'EdgePrompt':
            self.edge_prompt = EdgePrompt(dim_list)
        elif edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = None

        # 谱滤波器（在频域学习滤波器）
        self.spectral_filters = nn.ParameterList([
            nn.Parameter(torch.randn(num_filters, dim)) for dim in dim_list
        ])

        # 频率选择门控
        self.freq_gates = nn.ModuleList([
            nn.Linear(dim, num_filters) for dim in dim_list
        ])

        # 融合权重
        self.fusion_weights = nn.ParameterList([
            nn.Parameter(torch.tensor([0.5, 0.5])) for _ in dim_list
        ])

        self.num_filters = num_filters

    def compute_graph_fourier_transform(self, x, edge_index):
        """计算图傅立叶变换"""
        try:
            from torch_geometric.utils import get_laplacian, to_dense_adj

            # 计算归一化拉普拉斯矩阵
            num_nodes = x.size(0)

            # 添加自环
            edge_index_with_self = torch.cat([
                edge_index,
                torch.stack([torch.arange(num_nodes, device=edge_index.device),
                             torch.arange(num_nodes, device=edge_index.device)])
            ], dim=1)

            # 获取拉普拉斯矩阵
            edge_index_lap, edge_weight = get_laplacian(
                edge_index_with_self,
                normalization='sym',
                num_nodes=num_nodes
            )

            # 转换为稠密矩阵（注意：仅适用于小图）
            L = to_dense_adj(edge_index_lap, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]

            # 特征分解
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(L)

                # 图傅立叶变换：X_hat = U^T * X
                x_freq = torch.matmul(eigenvectors.t(), x)

                return x_freq, eigenvectors, eigenvalues, True
            except:
                # 如果特征分解失败，返回原始特征
                return x, None, None, False

        except Exception as e:
            # 如果导入或计算失败，返回原始特征
            return x, None, None, False

    def get_spectral_prompts(self, x, edge_index, layer):
        """在频域融合提示"""
        # 获取基础提示
        node_prompted_x = x
        if self.node_prompts is not None:
            node_prompted_x = self.node_prompts[layer].add(x)

        edge_prompt = None
        if self.edge_prompt is not None:
            edge_prompt = self.edge_prompt.get_prompt(x, edge_index, layer)

        # 尝试进行频域变换
        x_freq, eigenvectors, eigenvalues, success = self.compute_graph_fourier_transform(
            node_prompted_x, edge_index
        )

        if success and eigenvectors is not None:
            # 在频域应用滤波
            filtered_x = torch.zeros_like(x_freq)

            # 计算频率门控权重
            x_mean = x.mean(dim=0, keepdim=True)
            freq_gates = F.softmax(self.freq_gates[layer](x_mean), dim=1)

            # 应用谱滤波器
            for i in range(self.num_filters):
                filter_response = self.spectral_filters[layer][i]
                filtered_x += freq_gates[0, i] * x_freq * filter_response.unsqueeze(0)

            # 逆变换回空间域
            final_node = torch.matmul(eigenvectors, filtered_x)

        else:
            # 如果频域变换失败，使用简单融合
            final_node = node_prompted_x

        # 融合边提示
        if edge_prompt is not None and edge_prompt.size(0) == edge_index.size(1):
            edge_aggregated = torch.zeros_like(final_node)
            edge_aggregated.index_add_(0, edge_index[0], edge_prompt)
            edge_aggregated.index_add_(0, edge_index[1], edge_prompt)

            # 加权融合
            w = F.softmax(self.fusion_weights[layer], dim=0)
            final_x = w[0] * final_node + w[1] * edge_aggregated
        else:
            final_x = final_node

        return final_x, edge_prompt