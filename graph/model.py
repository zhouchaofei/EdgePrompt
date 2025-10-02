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

        # 方案一：串行融合
        if prompt_type == 'SerialNodeEdgePrompt':
            # 首先对输入应用节点提示
            x = prompt.get_node_prompt(x)
            h_list[0] = x

            # 然后在每层应用边提示
            for layer in range(self.num_layer):
                edge_prompt = prompt.get_edge_prompt(h_list[layer], edge_index, layer)
                x = self.convs[layer](h_list[layer], edge_index, edge_prompt)
                x = self.batch_norms[layer](x)

                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

                h_list.append(x)

        # 方案二：并行融合
        elif prompt_type == 'ParallelNodeEdgePrompt':
            for layer in range(self.num_layer):
                node_prompted_x, edge_prompt = prompt.get_prompts(h_list[layer], edge_index, layer)
                x = self.convs[layer](node_prompted_x, edge_index, edge_prompt)
                x = self.batch_norms[layer](x)

                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

                h_list.append(x)

        # 方案三：交互融合
        elif prompt_type == 'InteractiveNodeEdgePrompt':
            for layer in range(self.num_layer):
                interactive_x, edge_prompt = prompt.get_interactive_prompts(h_list[layer], edge_index, layer)
                x = self.convs[layer](interactive_x, edge_index, edge_prompt)
                x = self.batch_norms[layer](x)

                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

                h_list.append(x)

        # 方案四：互补学习融合
        elif prompt_type == 'ComplementaryNodeEdgePrompt':
            for layer in range(self.num_layer):
                final_x, edge_prompt, node_x, edge_x = prompt.get_complementary_prompts(
                    h_list[layer], edge_index, layer
                )
                x = self.convs[layer](final_x, edge_index, edge_prompt)
                x = self.batch_norms[layer](x)

                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

                h_list.append(x)

        # 方案五：对比学习融合
        elif prompt_type == 'ContrastiveNodeEdgePrompt':
            for layer in range(self.num_layer):
                final_x, edge_prompt, views = prompt.get_contrastive_prompts(
                    h_list[layer], edge_index, layer
                )
                x = self.convs[layer](final_x, edge_index, edge_prompt)
                x = self.batch_norms[layer](x)

                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

                h_list.append(x)

        # 方案六：图频域融合
        elif prompt_type == 'SpectralNodeEdgePrompt':
            for layer in range(self.num_layer):
                final_x, edge_prompt = prompt.get_spectral_prompts(
                    h_list[layer], edge_index, layer
                )
                x = self.convs[layer](final_x, edge_index, edge_prompt)
                x = self.batch_norms[layer](x)

                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

                h_list.append(x)

        # 新增：层次化图变换器
        elif prompt_type == 'HierarchicalGraphTransformerPrompt':
            for layer in range(self.num_layer):
                final_x, edge_prompt = prompt.get_prompts(h_list[layer], edge_index, layer)
                x = self.convs[layer](h_list[layer] if edge_prompt is False else final_x,
                                      edge_index, edge_prompt)
                x = self.batch_norms[layer](x)
                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
                h_list.append(x)

        # 新增：图神经ODE
        elif prompt_type == 'GraphNeuralODEPrompt':
            for layer in range(self.num_layer):
                final_x, edge_prompt = prompt.get_prompts(h_list[layer], edge_index, layer)
                x = self.convs[layer](final_x, edge_index, edge_prompt)
                x = self.batch_norms[layer](x)
                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
                h_list.append(x)

        # 新增：元学习
        elif prompt_type == 'MetaLearningPrompt':
            for layer in range(self.num_layer):
                final_x, edge_prompt = prompt.get_prompts(h_list[layer], edge_index, layer)
                x = self.convs[layer](final_x, edge_index, edge_prompt)
                x = self.batch_norms[layer](x)
                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
                h_list.append(x)

        # 新增：因果推理
        elif prompt_type == 'CausalGraphPrompt':
            for layer in range(self.num_layer):
                final_x, edge_prompt = prompt.get_prompts(h_list[layer], edge_index, layer)
                x = self.convs[layer](final_x, edge_index, edge_prompt)
                x = self.batch_norms[layer](x)
                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
                h_list.append(x)


        # 新增：图小波变换
        elif prompt_type == 'GraphWaveletPrompt':
            for layer in range(self.num_layer):
                final_x, edge_prompt = prompt.get_prompts(h_list[layer], edge_index, layer)
                x = self.convs[layer](final_x, edge_index, edge_prompt)
                x = self.batch_norms[layer](x)
                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
                h_list.append(x)

        # 新增：扩散模型
        elif prompt_type == 'DiffusionPrompt':
            for layer in range(self.num_layer):
                final_x, edge_prompt = prompt.get_prompts(h_list[layer], edge_index, layer)
                x = self.convs[layer](final_x, edge_index, edge_prompt)
                x = self.batch_norms[layer](x)
                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
                h_list.append(x)

        # 新增：强化学习
        elif prompt_type == 'RLPrompt':
            for layer in range(self.num_layer):
                final_x, edge_prompt = prompt.get_prompts(h_list[layer], edge_index, layer)
                x = self.convs[layer](final_x, edge_index, edge_prompt)
                x = self.batch_norms[layer](x)
                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
                h_list.append(x)

        # 新增：注意力流
        elif prompt_type == 'AttentionFlowPrompt':
            for layer in range(self.num_layer):
                final_x, edge_prompt = prompt.get_prompts(h_list[layer], edge_index, layer)
                x = self.convs[layer](final_x, edge_index, edge_prompt)
                x = self.batch_norms[layer](x)
                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
                h_list.append(x)

        # 新增：超图融合
        elif prompt_type == 'HypergraphPrompt':
            for layer in range(self.num_layer):
                final_x, edge_prompt = prompt.get_prompts(h_list[layer], edge_index, layer)
                x = self.convs[layer](final_x, edge_index, edge_prompt)
                x = self.batch_norms[layer](x)
                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
                h_list.append(x)

        # 新增：拓扑感知
        elif prompt_type == 'TopologyPrompt':
            for layer in range(self.num_layer):
                final_x, edge_prompt = prompt.get_prompts(h_list[layer], edge_index, layer)
                x = self.convs[layer](final_x, edge_index, edge_prompt)
                x = self.batch_norms[layer](x)
                if layer == self.num_layer - 1:
                    x = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
                h_list.append(x)

        # 新增：纯节点提示（已有的NodePrompt作为baseline）
        elif prompt_type == 'NodePrompt':
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