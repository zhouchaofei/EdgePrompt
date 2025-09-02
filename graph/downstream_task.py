import random
import logging
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn import metrics

from load_data import GraphDownstream, load_graph_data
from model import GIN
from prompt import EdgePrompt, EdgePromptplus, NodeEdgePrompt, ParallelNodeEdgePrompt
from logger import Logger


class GraphTask():
    def __init__(self, dataset_name, shots, gnn_type, num_layer, hidden_dim, device,
                 pretrain_task, prompt_type, num_prompts, logger, node_prompt_type=None,
                 node_num_prompts=5, fusion_type='serial'):
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.device = device
        self.pretrain_task = pretrain_task
        self.prompt_type = prompt_type
        self.num_prompts = num_prompts
        self.node_prompt_type = node_prompt_type
        self.node_num_prompts = node_num_prompts
        self.fusion_type = fusion_type
        self.logger = logger

        if dataset_name in ['ENZYMES', 'DD', 'NCI1', 'NCI109', 'Mutagenicity']:
            self.graph_list, self.input_dim, self.output_dim = load_graph_data(dataset_name, data_folder='./data')
            self.train_data, self.test_data = GraphDownstream(self.graph_list, shots, test_fraction=0.4)
        else:
            raise ValueError(
                'Error: invalid dataset name! Supported datasets: [ENZYMES, DD, NCI1, NCI109, Mutagenicity]')

        self.initialize_model()
        self.initialize_prompt()

    def initialize_model(self):
        if self.gnn_type == 'GIN':
            self.gnn = GIN(num_layer=self.num_layer, input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        else:
            raise ValueError(f"Error: invalid GNN type! Suppported GNNs: [GCN, GIN]")

        if self.pretrain_task is not None:
            pretrained_gnn_file = f'./pretrained_gnns/{self.dataset_name}_{self.pretrain_task}_{self.gnn_type}_5.pth'
            self.gnn.load_state_dict(torch.load(pretrained_gnn_file))

        print(self.gnn)
        self.gnn.to(self.device)

        self.classifier = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)

    def initialize_prompt(self):
        dim_list = [self.input_dim] + [self.hidden_dim] * (self.num_layer - 1)

        # Check if we're using fusion prompts
        if self.prompt_type == 'NodeEdgePrompt':
            # Serial fusion
            edge_type = 'EdgePrompt' if self.num_prompts == 1 else 'EdgePromptplus'
            node_type = self.node_prompt_type if self.node_prompt_type else 'NodePrompt'
            self.prompt = NodeEdgePrompt(
                dim_list=dim_list,
                edge_type=edge_type,
                node_type=node_type,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts
            ).to(self.device)

        elif self.prompt_type == 'ParallelNodeEdgePrompt':
            # Parallel fusion
            edge_type = 'EdgePrompt' if self.num_prompts == 1 else 'EdgePromptplus'
            node_type = self.node_prompt_type if self.node_prompt_type else 'NodePrompt'
            self.prompt = ParallelNodeEdgePrompt(
                dim_list=dim_list,
                edge_type=edge_type,
                node_type=node_type,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts
            ).to(self.device)

        elif self.prompt_type == 'EdgePrompt':
            self.prompt = EdgePrompt(dim_list=dim_list).to(self.device)

        elif self.prompt_type == 'EdgePromptplus':
            self.prompt = EdgePromptplus(dim_list=dim_list, num_anchors=self.num_prompts).to(self.device)

        else:
            self.prompt = None

    def train(self, batch_size, lr=0.001, decay=0, epochs=100):
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)
        learnable_parameters = list(self.classifier.parameters()) + list(self.prompt.parameters())
        optimizer = torch.optim.Adam(learnable_parameters, lr=lr, weight_decay=decay)

        best_test_accuracy = 0
        for epoch in range(1, 1 + epochs):
            total_loss = []
            self.gnn.train()
            for i, data in enumerate(train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                emb = self.gnn(data, self.prompt_type, self.prompt, pooling='mean')
                out = self.classifier(emb)
                loss = F.cross_entropy(out, data.y.squeeze())
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())
            train_loss = np.mean(total_loss)

            if epoch % 1 == 0:
                self.gnn.eval()
                pred_list = []
                label_list = []
                total_loss = []
                for i, data in enumerate(test_loader):
                    data = data.to(self.device)
                    emb = self.gnn(data, self.prompt_type, self.prompt, pooling='mean')
                    out = self.classifier(emb)
                    loss = F.cross_entropy(out, data.y.squeeze())
                    pred_list.extend(out.argmax(1).tolist())
                    label_list.extend(data.y.squeeze().tolist())
                    total_loss.append(loss.item())
                test_accuracy = metrics.accuracy_score(y_true=label_list, y_pred=pred_list)
                test_loss = np.mean(total_loss)

                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy

                log_info = ''.join(['| epoch: {:4d} '.format(epoch),
                                    '| train_loss: {:7.5f}'.format(train_loss),
                                    '| test_loss: {:7.5f}'.format(test_loss),
                                    '| test_accuracy: {:7.5f}'.format(test_accuracy),
                                    '| best_accuracy: {:7.5f} |'.format(best_test_accuracy)]
                                   )
                self.logger.info(log_info)

        return best_test_accuracy


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(args, seed):
    filename = f'log/{args.dataset_name}_{args.shots}_{args.pretrain_task}_{args.gnn_type}_{args.prompt_type}_{args.num_prompts}_{seed}.log'
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger = Logger(filename, formatter)
    set_random_seed(seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    task = GraphTask(
        args.dataset_name, args.shots, args.gnn_type, args.num_layer, args.hidden_dim, device,
        args.pretrain_task, args.prompt_type, args.num_prompts, logger,
        node_prompt_type=args.node_prompt_type, node_num_prompts=args.node_num_prompts,
        fusion_type=args.fusion_type
    )

    best_acc = task.train(args.batch_size, epochs=args.epochs)
    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downstream task: graph classification with prompt fusion')
    parser.add_argument('--dataset_name', type=str, default='NCI1', help='dataset name')
    parser.add_argument('--shots', type=int, default=50, help='number of shots (default: 50)')
    parser.add_argument('--gnn_type', type=str, default='GIN', help='gnn type')
    parser.add_argument('--num_layer', type=int, default=5, help='GNN layers (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim (default: 128)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--pretrain_task', type=str, default='GraphCL', help='pretrain task')

    # Prompt-related arguments
    parser.add_argument('--prompt_type', type=str, default='NodeEdgePrompt',
                        help='Prompt methods: EdgePrompt, EdgePromptplus, NodeEdgePrompt, ParallelNodeEdgePrompt')
    parser.add_argument('--num_prompts', type=int, default=5, help='num_prompts for edge prompt (default: 5)')
    parser.add_argument('--node_prompt_type', type=str, default='NodePrompt',
                        help='Node prompt type: NodePrompt or NodePromptplus')
    parser.add_argument('--node_num_prompts', type=int, default=5, help='num_prompts for node prompt (default: 5)')
    parser.add_argument('--fusion_type', type=str, default='serial',
                        help='Fusion type: serial or parallel')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200, help='epochs (default: 200)')
    args = parser.parse_args()

    # Run experiments with multiple seeds
    all_results = []
    for seed in range(5):
        acc = run(args, seed)
        all_results.append(acc)
        print(f"Seed {seed}: Best Test Accuracy = {acc:.4f}")

    print(f"\nFinal Results:")
    print(f"Mean Accuracy: {np.mean(all_results):.4f} ± {np.std(all_results):.4f}")

    # Save results
    with open(f'results_{args.prompt_type}.txt', 'a') as f:
        f.write(f"{args.dataset_name} {args.shots} {args.pretrain_task} {args.prompt_type}: ")
        f.write(f"{np.mean(all_results):.4f} ± {np.std(all_results):.4f}\n")