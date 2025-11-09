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