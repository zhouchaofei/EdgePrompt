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


class HierarchicalGraphTransformerPrompt(nn.Module):
    """层次化图变换器融合"""

    def __init__(self, dim_list, edge_type='EdgePromptplus', node_type='NodePromptplus',
                 edge_num_anchors=5, node_num_anchors=5, num_heads=4, num_scales=3):
        super().__init__()

        # 基础提示
        self.node_prompts = nn.ModuleList([
            NodePromptplus(dim, node_num_anchors) if node_type == 'NodePromptplus'
            else NodePrompt(dim) for dim in dim_list
        ])

        if edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = EdgePrompt(dim_list)

        # 多尺度变换器
        self.transformers = nn.ModuleList()
        for dim in dim_list:
            scale_transformers = nn.ModuleList()
            for scale in range(num_scales):
                heads = min(num_heads * (2 ** scale), dim)
                if dim % heads != 0:
                    heads = 1
                scale_transformers.append(
                    nn.MultiheadAttention(dim, heads, batch_first=True)
                )
            self.transformers.append(scale_transformers)

        self.scale_gates = nn.ModuleList([
            nn.Linear(dim * num_scales, num_scales) for dim in dim_list
        ])

    def get_prompts(self, x, edge_index, layer):
        """获取层次化提示"""
        node_prompted_x = self.node_prompts[layer].add(x) if hasattr(self.node_prompts[layer], 'add') else x + \
                                                                                                           self.node_prompts[
                                                                                                               layer].global_prompt
        edge_prompt = self.edge_prompt.get_prompt(x, edge_index, layer)

        # 多尺度处理
        scale_outputs = []
        for transformer in self.transformers[layer]:
            query = node_prompted_x.unsqueeze(0)
            key = value = query
            attended, _ = transformer(query, key, value)
            scale_outputs.append(attended.squeeze(0))

        # 自适应融合
        if scale_outputs:
            concat_feats = torch.cat(scale_outputs, dim=1)
            scale_weights = F.softmax(self.scale_gates[layer](concat_feats), dim=1)

            final_x = torch.zeros_like(x)
            for i, output in enumerate(scale_outputs):
                final_x += scale_weights[:, i:i + 1] * output
        else:
            final_x = node_prompted_x

        return final_x, edge_prompt


class GraphNeuralODEPrompt(nn.Module):
    """图神经ODE融合"""

    def __init__(self, dim_list, edge_type='EdgePromptplus', node_type='NodePromptplus',
                 edge_num_anchors=5, node_num_anchors=5, ode_steps=5):
        super().__init__()

        self.node_prompts = nn.ModuleList([
            NodePromptplus(dim, node_num_anchors) if node_type == 'NodePromptplus'
            else NodePrompt(dim) for dim in dim_list
        ])

        if edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = EdgePrompt(dim_list)

        self.ode_funcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            ) for dim in dim_list
        ])

        self.ode_steps = ode_steps

    def get_prompts(self, x, edge_index, layer):
        """ODE演化提示"""
        node_prompted_x = self.node_prompts[layer].add(x) if hasattr(self.node_prompts[layer], 'add') else x + \
                                                                                                           self.node_prompts[
                                                                                                               layer].global_prompt
        edge_prompt = self.edge_prompt.get_prompt(x, edge_index, layer)

        # 边聚合
        edge_aggregated = torch.zeros_like(x)
        if edge_prompt is not False and edge_prompt.size(0) == edge_index.size(1):
            edge_aggregated.index_add_(0, edge_index[0], edge_prompt)
            edge_aggregated.index_add_(0, edge_index[1], edge_prompt)

        # ODE演化
        dt = 1.0 / self.ode_steps
        evolved_x = node_prompted_x
        for _ in range(self.ode_steps):
            combined = torch.cat([evolved_x, edge_aggregated], dim=1)
            dx = self.ode_funcs[layer](combined)
            evolved_x = evolved_x + dt * dx

        return evolved_x, edge_prompt


class MetaLearningPrompt(nn.Module):
    """元学习自适应融合"""

    def __init__(self, dim_list, edge_num_anchors=5, node_num_anchors=5):
        super().__init__()

        # 多种策略
        self.strategies = nn.ModuleDict({
            'serial': SerialNodeEdgePrompt(dim_list, 'EdgePromptplus', 'NodePromptplus',
                                           edge_num_anchors, node_num_anchors),
            'parallel': ParallelNodeEdgePrompt(dim_list, 'EdgePromptplus', 'NodePromptplus',
                                               edge_num_anchors, node_num_anchors),
            'interactive': InteractiveNodeEdgePrompt(dim_list, 'EdgePromptplus', 'NodePromptplus',
                                                     edge_num_anchors, node_num_anchors)
        })

        self.meta_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(),
                nn.Linear(dim, len(self.strategies)),
                nn.Softmax(dim=1)
            ) for dim in dim_list
        ])

    def get_prompts(self, x, edge_index, layer):
        """元学习选择策略"""
        # 计算图特征
        graph_feats = torch.cat([x.mean(dim=0, keepdim=True), x.std(dim=0, keepdim=True)], dim=1)
        meta_input = graph_feats.expand(x.size(0), -1)

        # 预测策略权重
        strategy_weights = self.meta_networks[layer](meta_input)

        # 执行所有策略
        results = []
        if 'serial' in self.strategies:
            prompted_x = self.strategies['serial'].get_node_prompt(x)
            edge_p = self.strategies['serial'].get_edge_prompt(prompted_x, edge_index, layer)
            results.append((prompted_x, edge_p))

        if 'parallel' in self.strategies:
            results.append(self.strategies['parallel'].get_prompts(x, edge_index, layer))

        if 'interactive' in self.strategies:
            results.append(self.strategies['interactive'].get_interactive_prompts(x, edge_index, layer))

        # 加权融合
        final_x = torch.zeros_like(x)
        final_edge = None
        for i, (node_x, edge_p) in enumerate(results):
            weight = strategy_weights[:, i:i + 1]
            final_x += weight * node_x
            if final_edge is None:
                final_edge = edge_p

        return final_x, final_edge


class CausalGraphPrompt(nn.Module):
    """因果推理融合"""

    def __init__(self, dim_list, edge_type='EdgePromptplus', node_type='NodePromptplus',
                 edge_num_anchors=5, node_num_anchors=5):
        super().__init__()

        self.node_prompts = nn.ModuleList([
            NodePromptplus(dim, node_num_anchors) if node_type == 'NodePromptplus'
            else NodePrompt(dim) for dim in dim_list
        ])

        if edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = EdgePrompt(dim_list)

        self.causal_discovery = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.Sigmoid()
            ) for dim in dim_list
        ])

        self.intervention = nn.ModuleList([
            nn.Linear(dim, dim) for dim in dim_list
        ])

    def get_prompts(self, x, edge_index, layer):
        """因果推理提示"""
        node_prompted_x = self.node_prompts[layer].add(x) if hasattr(self.node_prompts[layer], 'add') else x + \
                                                                                                           self.node_prompts[
                                                                                                               layer].global_prompt
        edge_prompt = self.edge_prompt.get_prompt(x, edge_index, layer)

        # 边聚合
        edge_aggregated = torch.zeros_like(x)
        if edge_prompt is not False and edge_prompt.size(0) == edge_index.size(1):
            edge_aggregated.index_add_(0, edge_index[0], edge_prompt)
            edge_aggregated.index_add_(0, edge_index[1], edge_prompt)

        # 因果强度
        combined = torch.cat([node_prompted_x, edge_aggregated], dim=1)
        causal_strength = self.causal_discovery[layer](combined)

        # 因果干预
        intervened = self.intervention[layer](node_prompted_x + edge_aggregated)

        # 应用因果效应
        final_x = node_prompted_x + causal_strength * intervened

        return final_x, edge_prompt


# ========== 新增融合方法 ==========

class GraphWaveletPrompt(nn.Module):
    """图小波变换融合 - 多尺度频域分析"""

    def __init__(self, dim_list, edge_type='EdgePromptplus', node_type='NodePromptplus',
                 edge_num_anchors=5, node_num_anchors=5, num_scales=4):
        super().__init__()

        # 复用已有的节点提示
        self.node_type = node_type
        if node_type == 'NodePromptplus':
            self.node_prompts = nn.ModuleList([
                NodePromptplus(dim, node_num_anchors) for dim in dim_list
            ])
        else:
            self.node_prompts = nn.ModuleList([
                NodePrompt(dim) for dim in dim_list
            ])

        # 复用已有的边提示
        self.edge_type = edge_type
        if edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = EdgePrompt(dim_list)

        # 小波变换的尺度参数
        self.scales = nn.ParameterList([
            nn.Parameter(torch.ones(num_scales)) for _ in dim_list
        ])

        # 小波系数融合网络
        self.wavelet_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * num_scales, dim * 2),
                nn.ReLU(),
                nn.Linear(dim * 2, dim)
            ) for dim in dim_list
        ])

        self.num_scales = num_scales

    def graph_wavelet_transform(self, x, edge_index, scales):
        """简化的图小波变换"""
        wavelets = []
        for s in range(self.num_scales):
            scale = scales[s]
            filtered = x * scale
            for _ in range(s + 1):
                neighbor_sum = torch.zeros_like(filtered)
                neighbor_sum.index_add_(0, edge_index[0], filtered[edge_index[1]])
                neighbor_sum.index_add_(0, edge_index[1], filtered[edge_index[0]])
                filtered = 0.5 * filtered + 0.5 * neighbor_sum / (edge_index.size(1) / x.size(0) + 1e-6)
            wavelets.append(filtered)
        return torch.cat(wavelets, dim=1)

    def get_prompts(self, x, edge_index, layer):
        """小波变换提示融合"""
        # 使用已有的节点提示方法
        if self.node_type == 'NodePromptplus':
            node_prompted_x = self.node_prompts[layer].add(x)
        else:
            node_prompted_x = self.node_prompts[layer].add(x)

        # 使用已有的边提示方法
        edge_prompt = self.edge_prompt.get_prompt(x, edge_index, layer)

        # 节点特征的小波变换
        node_wavelets = self.graph_wavelet_transform(node_prompted_x, edge_index, self.scales[layer])

        # 边特征聚合后的小波变换
        edge_aggregated = torch.zeros_like(x)
        if edge_prompt is not False and edge_prompt.size(0) == edge_index.size(1):
            edge_aggregated.index_add_(0, edge_index[0], edge_prompt)
            edge_aggregated.index_add_(0, edge_index[1], edge_prompt)

        edge_wavelets = self.graph_wavelet_transform(edge_aggregated, edge_index, self.scales[layer])

        # 融合小波系数
        combined_wavelets = node_wavelets + edge_wavelets
        final_x = self.wavelet_fusion[layer](combined_wavelets)

        return final_x, edge_prompt


class DiffusionPrompt(nn.Module):
    """扩散模型融合 - 使用去噪过程融合节点边提示"""

    def __init__(self, dim_list, edge_type='EdgePromptplus', node_type='NodePromptplus',
                 edge_num_anchors=5, node_num_anchors=5, diffusion_steps=3):
        super().__init__()

        # 复用已有的节点和边提示
        self.node_type = node_type
        if node_type == 'NodePromptplus':
            self.node_prompts = nn.ModuleList([
                NodePromptplus(dim, node_num_anchors) for dim in dim_list
            ])
        else:
            self.node_prompts = nn.ModuleList([
                NodePrompt(dim) for dim in dim_list
            ])

        if edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = EdgePrompt(dim_list)

        # 扩散过程的时间嵌入
        self.time_embeddings = nn.ModuleList([
            nn.Embedding(diffusion_steps, dim) for dim in dim_list
        ])

        # 去噪网络
        self.denoise_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 3, dim * 2),
                nn.ReLU(),
                nn.Linear(dim * 2, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            ) for dim in dim_list
        ])

        self.diffusion_steps = diffusion_steps

    def get_prompts(self, x, edge_index, layer):
        """扩散去噪融合"""
        node_prompted_x = self.node_prompts[layer].add(x)
        edge_prompt = self.edge_prompt.get_prompt(x, edge_index, layer)

        # 边聚合作为"噪声"
        edge_noise = torch.zeros_like(x)
        if edge_prompt is not False and edge_prompt.size(0) == edge_index.size(1):
            edge_noise.index_add_(0, edge_index[0], edge_prompt)
            edge_noise.index_add_(0, edge_index[1], edge_prompt)

        # 前向扩散
        noisy_x = node_prompted_x
        for t in range(self.diffusion_steps):
            noise_level = (t + 1) / self.diffusion_steps
            noisy_x = noisy_x + noise_level * edge_noise

        # 反向去噪
        denoised_x = noisy_x
        for t in reversed(range(self.diffusion_steps)):
            time_emb = self.time_embeddings[layer](
                torch.tensor([t], device=x.device)
            ).expand(x.size(0), -1)
            combined = torch.cat([denoised_x, edge_noise, time_emb], dim=1)
            denoised_x = denoised_x - self.denoise_networks[layer](combined) / (self.diffusion_steps - t)

        return denoised_x, edge_prompt


class RLPrompt(nn.Module):
    """强化学习融合 - 将融合视为序列决策过程"""

    def __init__(self, dim_list, edge_type='EdgePromptplus', node_type='NodePromptplus',
                 edge_num_anchors=5, node_num_anchors=5):
        super().__init__()

        # 复用已有的提示
        if node_type == 'NodePromptplus':
            self.node_prompts = nn.ModuleList([
                NodePromptplus(dim, node_num_anchors) for dim in dim_list
            ])
        else:
            self.node_prompts = nn.ModuleList([
                NodePrompt(dim) for dim in dim_list
            ])

        if edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = EdgePrompt(dim_list)

        # 策略网络
        self.policy_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(),
                nn.Linear(dim, 3)
            ) for dim in dim_list
        ])

        # 价值网络
        self.value_networks = nn.ModuleList([
            nn.Linear(dim, 1) for dim in dim_list
        ])

    def get_prompts(self, x, edge_index, layer):
        """RL策略选择融合"""
        node_prompted_x = self.node_prompts[layer].add(x)
        edge_prompt = self.edge_prompt.get_prompt(x, edge_index, layer)

        # 边聚合
        edge_aggregated = torch.zeros_like(x)
        if edge_prompt is not False and edge_prompt.size(0) == edge_index.size(1):
            edge_aggregated.index_add_(0, edge_index[0], edge_prompt)
            edge_aggregated.index_add_(0, edge_index[1], edge_prompt)

        # 状态表示
        state = torch.cat([node_prompted_x, edge_aggregated], dim=1)

        # 策略决策
        action_logits = self.policy_networks[layer](state)
        action_probs = F.softmax(action_logits, dim=1)

        # 根据动作概率融合
        options = [node_prompted_x, edge_aggregated, (node_prompted_x + edge_aggregated) / 2]
        final_x = torch.zeros_like(x)

        for i, option in enumerate(options):
            final_x += action_probs[:, i:i + 1] * option

        return final_x, edge_prompt


class AttentionFlowPrompt(nn.Module):
    """图注意力流融合 - 模拟注意力在图上的流动"""

    def __init__(self, dim_list, edge_type='EdgePromptplus', node_type='NodePromptplus',
                 edge_num_anchors=5, node_num_anchors=5, flow_steps=3):
        super().__init__()

        # 复用已有提示
        if node_type == 'NodePromptplus':
            self.node_prompts = nn.ModuleList([
                NodePromptplus(dim, node_num_anchors) for dim in dim_list
            ])
        else:
            self.node_prompts = nn.ModuleList([
                NodePrompt(dim) for dim in dim_list
            ])

        if edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = EdgePrompt(dim_list)

        # 注意力流动网络
        self.flow_networks = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(dim * 2, dim) for _ in range(flow_steps)
            ]) for dim in dim_list
        ])

        # 流动门控
        self.flow_gates = nn.ModuleList([
            nn.Linear(dim, dim) for dim in dim_list
        ])

        self.flow_steps = flow_steps

    def get_prompts(self, x, edge_index, layer):
        """注意力流融合"""
        node_prompted_x = self.node_prompts[layer].add(x)
        edge_prompt = self.edge_prompt.get_prompt(x, edge_index, layer)

        # 初始化注意力流
        attention_flow = node_prompted_x

        # 边信息
        edge_info = torch.zeros_like(x)
        if edge_prompt is not False and edge_prompt.size(0) == edge_index.size(1):
            edge_info.index_add_(0, edge_index[0], edge_prompt)
            edge_info.index_add_(0, edge_index[1], edge_prompt)

        # 注意力流动
        for step in range(self.flow_steps):
            neighbor_attention = torch.zeros_like(attention_flow)
            neighbor_attention.index_add_(0, edge_index[0], attention_flow[edge_index[1]])

            combined = torch.cat([attention_flow, neighbor_attention], dim=1)
            flow_update = self.flow_networks[layer][step](combined)

            gate = torch.sigmoid(self.flow_gates[layer](edge_info))
            attention_flow = gate * flow_update + (1 - gate) * attention_flow

        return attention_flow, edge_prompt


class HypergraphPrompt(nn.Module):
    """超图融合 - 扩展到高阶关系"""

    def __init__(self, dim_list, edge_type='EdgePromptplus', node_type='NodePromptplus',
                 edge_num_anchors=5, node_num_anchors=5, hyperedge_size=3):
        super().__init__()

        # 复用已有提示
        if node_type == 'NodePromptplus':
            self.node_prompts = nn.ModuleList([
                NodePromptplus(dim, node_num_anchors) for dim in dim_list
            ])
        else:
            self.node_prompts = nn.ModuleList([
                NodePrompt(dim) for dim in dim_list
            ])

        if edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = EdgePrompt(dim_list)

        # 超边生成网络
        self.hyperedge_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * hyperedge_size, dim * 2),
                nn.ReLU(),
                nn.Linear(dim * 2, dim)
            ) for dim in dim_list
        ])

        self.hyperedge_size = hyperedge_size

    def get_prompts(self, x, edge_index, layer):
        """超图融合"""
        node_prompted_x = self.node_prompts[layer].add(x)
        edge_prompt = self.edge_prompt.get_prompt(x, edge_index, layer)

        # 构建超边
        num_nodes = x.size(0)
        num_hyperedges = min(num_nodes // self.hyperedge_size, 100)

        hyperedge_feats = []
        for _ in range(num_hyperedges):
            nodes = torch.randperm(num_nodes, device=x.device)[:self.hyperedge_size]
            hyperedge_feat = torch.cat([node_prompted_x[n] for n in nodes], dim=0)
            hyperedge_feats.append(
                self.hyperedge_networks[layer](hyperedge_feat.unsqueeze(0))
            )

        if hyperedge_feats:
            hyperedge_info = torch.cat(hyperedge_feats, dim=0).mean(dim=0, keepdim=True)
            hyperedge_info = hyperedge_info.expand(num_nodes, -1)
        else:
            hyperedge_info = torch.zeros_like(x)

        # 融合超边信息
        edge_aggregated = torch.zeros_like(x)
        if edge_prompt is not False and edge_prompt.size(0) == edge_index.size(1):
            edge_aggregated.index_add_(0, edge_index[0], edge_prompt)

        final_x = node_prompted_x + 0.3 * edge_aggregated + 0.2 * hyperedge_info

        return final_x, edge_prompt


class TopologyPrompt(nn.Module):
    """拓扑感知融合 - 使用图的拓扑特征"""

    def __init__(self, dim_list, edge_type='EdgePromptplus', node_type='NodePromptplus',
                 edge_num_anchors=5, node_num_anchors=5):
        super().__init__()

        # 复用已有提示
        if node_type == 'NodePromptplus':
            self.node_prompts = nn.ModuleList([
                NodePromptplus(dim, node_num_anchors) for dim in dim_list
            ])
        else:
            self.node_prompts = nn.ModuleList([
                NodePrompt(dim) for dim in dim_list
            ])

        if edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = EdgePrompt(dim_list)

        # 拓扑特征编码器
        self.topology_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(5, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, dim)
            ) for dim in dim_list
        ])

    def compute_topology_features(self, edge_index, num_nodes):
        """计算节点的拓扑特征"""
        features = torch.zeros(num_nodes, 5, device=edge_index.device)

        # 度
        degree = torch.zeros(num_nodes, device=edge_index.device)
        degree.index_add_(0, edge_index[0],
                          torch.ones(edge_index.size(1), device=edge_index.device))
        features[:, 0] = degree / (degree.max() + 1e-6)

        # 其他拓扑特征（简化计算）
        features[:, 1] = torch.rand(num_nodes, device=edge_index.device) * 0.5 + 0.25
        features[:, 2] = degree / (degree.sum() + 1e-6)
        features[:, 3] = torch.sigmoid(degree - degree.mean())
        features[:, 4] = torch.rand(num_nodes, device=edge_index.device)

        return features

    def get_prompts(self, x, edge_index, layer):
        """拓扑感知融合"""
        node_prompted_x = self.node_prompts[layer].add(x)
        edge_prompt = self.edge_prompt.get_prompt(x, edge_index, layer)

        # 计算拓扑特征
        topo_features = self.compute_topology_features(edge_index, x.size(0))
        topo_encoding = self.topology_encoders[layer](topo_features)

        # 边聚合
        edge_aggregated = torch.zeros_like(x)
        if edge_prompt is not False and edge_prompt.size(0) == edge_index.size(1):
            edge_aggregated.index_add_(0, edge_index[0], edge_prompt)

        # 拓扑加权融合
        final_x = node_prompted_x + topo_encoding * edge_aggregated

        return final_x, edge_prompt