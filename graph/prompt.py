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