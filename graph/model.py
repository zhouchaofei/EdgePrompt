import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool


class GINConv(MessagePassing):
    def __init__(self, input_dim, hidden_dim):
        super(GINConv, self).__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.eps = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_prompt=False):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_prompt))
        return out

    def message(self, x_j, edge_attr):
        if edge_attr is not False:
            return F.relu(x_j + edge_attr)
        else:
            return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out


class GIN(nn.Module):
    def __init__(self, num_layer, input_dim, hidden_dim, drop_ratio=0.5):
        super(GIN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.hidden_dim = hidden_dim  # 添加：保存hidden_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GINConv(input_dim=input_dim, hidden_dim=hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for layer in range(num_layer - 1):
            self.convs.append(GINConv(input_dim=hidden_dim, hidden_dim=hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, data, prompt_type=None, prompt=False, pooling=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h_list = [x]

        # 新增：纯节点提示（已有的NodePrompt作为baseline）
        if prompt_type == 'NodePrompt':
            for layer in range(self.num_layer):
                # 只使用节点提示，不使用边提示
                prompted_x = prompt[layer].add(h_list[layer])
                x = self.convs[layer](prompted_x, edge_index, False)
                x = self.batch_norms[layer](x)
                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
                h_list.append(x)

        # 新增：NodePromptplus作为baseline
        elif prompt_type == 'NodePromptplus':
            for layer in range(self.num_layer):
                # 只使用节点提示plus，不使用边提示
                prompted_x = prompt[layer].add(h_list[layer])
                x = self.convs[layer](prompted_x, edge_index, False)
                x = self.batch_norms[layer](x)
                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
                h_list.append(x)

        # 原有的边提示方法
        elif prompt_type in ['EdgePrompt', 'EdgePromptplus']:
            for layer in range(self.num_layer):
                edge_prompt = prompt.get_prompt(h_list[layer], edge_index, layer)
                x = self.convs[layer](h_list[layer], edge_index, edge_prompt)
                x = self.batch_norms[layer](x)

                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

                h_list.append(x)

        # 无提示
        else:
            for layer in range(self.num_layer):
                x = self.convs[layer](h_list[layer], edge_index, False)
                x = self.batch_norms[layer](x)

                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

                h_list.append(x)

        node_emb = h_list[-1]

        if pooling == 'mean':
            graph_emb = global_mean_pool(node_emb, batch)
            return graph_emb

        return node_emb

    # ==========================================
    # 新增方法：获取节点嵌入（用于EdgePrediction预训练）
    # ==========================================
    def get_node_embeddings(self, data):
        """
        获取节点级嵌入（用于边预测等任务）

        Args:
            data: PyG Data对象
                - data.x: [N, input_dim] 节点特征
                - data.edge_index: [2, E] 边索引
                - data.batch: [N] batch索引（可选）

        Returns:
            node_embeddings: [N, hidden_dim] 节点嵌入

        使用场景：
            - EdgePrediction预训练任务
            - 需要节点级别表示的下游任务
            - Link prediction等边相关任务
        """
        x, edge_index = data.x, data.edge_index

        # 通过所有GNN层（不使用prompt）
        h = x
        for layer in range(self.num_layer):
            h = self.convs[layer](h, edge_index, edge_prompt=False)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # 最后一层
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                # 中间层
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        return h  # [N, hidden_dim]