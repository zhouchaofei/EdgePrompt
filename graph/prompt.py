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
    """Simple node prompt that adds a learnable vector to node features"""
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
    """Attention-based node prompt with multiple learnable anchors"""
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
        # Calculate attention weights for each node
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)
        # Weighted combination of anchor prompts
        prompt = weights.mm(self.anchor_prompt)
        return x + prompt


class NodeEdgePrompt(nn.Module):
    """Fusion of node and edge prompts - Serial fusion strategy"""
    """方案一：串行融合 - 输入层节点提示 + 各层边提示"""

    def __init__(self, dim_list, edge_type='EdgePrompt', node_type='NodePrompt',
                 edge_num_anchors=5, node_num_anchors=5):
        super(NodeEdgePrompt, self).__init__()

        # Initialize node prompt (only for input layer)
        # 节点提示（仅用于输入层）
        self.node_type = node_type
        if node_type == 'NodePrompt':
            self.node_prompt = NodePrompt(dim_list[0])
        elif node_type == 'NodePromptplus':
            self.node_prompt = NodePromptplus(dim_list[0], node_num_anchors)
        else:
            self.node_prompt = None

        # Initialize edge prompt (for all layers)
        # 边提示（用于所有层）
        self.edge_type = edge_type
        if edge_type == 'EdgePrompt':
            self.edge_prompt = EdgePrompt(dim_list)
        elif edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = None

    def get_node_prompt(self, x):
        """Apply node prompt to input features"""
        """对输入特征应用节点提示"""
        if self.node_prompt is not None:
            return self.node_prompt.add(x)
        return x

    def get_edge_prompt(self, x, edge_index, layer):
        """Get edge prompt for specific layer"""
        """获取特定层的边提示"""
        if self.edge_prompt is not None:
            return self.edge_prompt.get_prompt(x, edge_index, layer)
        return False


class ParallelNodeEdgePrompt(nn.Module):
    """Parallel fusion of node and edge prompts"""
    """方案二：并行融合 - 每层同时使用节点和边提示"""

    def __init__(self, dim_list, edge_type='EdgePrompt', node_type='NodePrompt',
                 edge_num_anchors=5, node_num_anchors=5):
        super(ParallelNodeEdgePrompt, self).__init__()

        # Node prompts for all layers
        # 节点提示（所有层）
        self.node_type = node_type
        if node_type == 'NodePrompt':
            self.node_prompts = nn.ModuleList([NodePrompt(dim) for dim in dim_list])
        elif node_type == 'NodePromptplus':
            self.node_prompts = nn.ModuleList([NodePromptplus(dim, node_num_anchors) for dim in dim_list])
        else:
            self.node_prompts = None

        # Edge prompts for all layers
        # 边提示（所有层）
        self.edge_type = edge_type
        if edge_type == 'EdgePrompt':
            self.edge_prompt = EdgePrompt(dim_list)
        elif edge_type == 'EdgePromptplus':
            self.edge_prompt = EdgePromptplus(dim_list, edge_num_anchors)
        else:
            self.edge_prompt = None

        # Fusion weights
        # 融合权重
        self.fusion_weights = nn.ParameterList([nn.Parameter(torch.tensor([0.5, 0.5])) for _ in dim_list])

    def get_prompts(self, x, edge_index, layer):
        """Get both node and edge prompts for a specific layer"""
        """获取特定层的节点和边提示"""
        node_prompt_value = None
        edge_prompt_value = None

        # 应用节点提示
        if self.node_prompts is not None:
            x_with_prompt = self.node_prompts[layer].add(x)
            node_prompt_value = x_with_prompt - x  # Extract the prompt component

        # 获取边提示
        if self.edge_prompt is not None:
            edge_prompt_value = self.edge_prompt.get_prompt(x, edge_index, layer)

        return node_prompt_value, edge_prompt_value, self.fusion_weights[layer]