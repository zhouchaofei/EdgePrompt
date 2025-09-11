"""
修改后的下游任务模块，支持脑成像数据集和分子图数据集
"""
import random
import logging
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn import metrics

from load_data import GraphDownstream, load_graph_data, get_dataset_info
from model import GIN
from prompt import (EdgePrompt, EdgePromptplus, SerialNodeEdgePrompt,
                    ParallelNodeEdgePrompt, InteractiveNodeEdgePrompt)
from logger import Logger


class GraphTask():
    def __init__(self, dataset_name, shots, gnn_type, num_layer, hidden_dim, device,
                 pretrain_task, prompt_type, num_prompts, logger,
                 node_prompt_type=None, node_num_prompts=5, fusion_method='weighted'):
        self.dataset_name = dataset_name
        self.shots = shots
        self.gnn_type = gnn_type
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.device = device
        self.pretrain_task = pretrain_task
        self.prompt_type = prompt_type
        self.num_prompts = num_prompts
        self.node_prompt_type = node_prompt_type
        self.node_num_prompts = node_num_prompts
        self.fusion_method = fusion_method
        self.logger = logger

        # 加载数据集
        self.load_dataset()

        # 初始化模型和提示
        self.initialize_model()
        self.initialize_prompt()

    def load_dataset(self):
        """加载数据集，支持多种类型"""
        # 获取数据集信息
        dataset_info = get_dataset_info(self.dataset_name)
        if dataset_info:
            self.logger.info(f"数据集类型: {dataset_info.get('type', '未知')}")
            self.logger.info(f"任务: {dataset_info.get('task', '未知')}")
            self.logger.info(f"描述: {dataset_info.get('description', '无')}")

        # 支持的所有数据集
        supported_datasets = ['ENZYMES', 'DD', 'NCI1', 'NCI109', 'Mutagenicity',
                              'ABIDE', 'ABIDE_corr', 'ABIDE_dynamic', 'ABIDE_phase']

        if self.dataset_name in supported_datasets:
            self.graph_list, self.input_dim, self.output_dim = load_graph_data(
                self.dataset_name, data_folder='./data'
            )

            # 特殊处理脑成像数据集
            if self.dataset_name.startswith('ABIDE'):
                # 对于脑成像数据，可能需要调整shots参数
                if self.shots > 100:
                    self.logger.warning(f"ABIDE数据集样本有限，将shots从{self.shots}调整为50")
                    self.shots = min(self.shots, 50)

            # 划分训练和测试集
            self.train_data, self.test_data = GraphDownstream(
                self.graph_list, self.shots, test_fraction=0.4
            )

            self.logger.info(f"数据集 {self.dataset_name} 加载成功")
            self.logger.info(f"输入维度: {self.input_dim}, 输出类别数: {self.output_dim}")
            self.logger.info(f"训练集大小: {len(self.train_data)}, 测试集大小: {len(self.test_data)}")
        else:
            raise ValueError(
                f'错误: 无效的数据集名称！\n'
                f'支持的数据集: {supported_datasets}'
            )

    def initialize_model(self):
        """初始化GNN模型"""
        if self.gnn_type == 'GIN':
            self.gnn = GIN(num_layer=self.num_layer, input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        else:
            raise ValueError(f"错误: 无效的GNN类型！支持的GNN: [GIN]")

        # 加载预训练模型（如果有）
        if self.pretrain_task is not None:
            # 对于脑成像数据集，预训练模型路径可能不同
            if self.dataset_name.startswith('ABIDE'):
                pretrained_gnn_file = f'./pretrained_gnns/{self.dataset_name}_{self.pretrain_task}_{self.gnn_type}_{self.num_layer}.pth'
            else:
                pretrained_gnn_file = f'./pretrained_gnns/{self.dataset_name}_{self.pretrain_task}_{self.gnn_type}_{self.num_layer}.pth'

            if os.path.exists(pretrained_gnn_file):
                self.gnn.load_state_dict(torch.load(pretrained_gnn_file))
                self.logger.info(f"加载预训练模型: {pretrained_gnn_file}")
            else:
                self.logger.warning(f"预训练模型不存在: {pretrained_gnn_file}")

        self.logger.info(f"模型结构:\n{self.gnn}")
        self.gnn.to(self.device)
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)

    def initialize_prompt(self):
        """初始化提示模块"""
        dim_list = [self.input_dim] + [self.hidden_dim] * (self.num_layer - 1)

        if self.prompt_type == 'SerialNodeEdgePrompt':
            edge_type = 'EdgePrompt' if self.num_prompts == 1 else 'EdgePromptplus'
            node_type = self.node_prompt_type if self.node_prompt_type else 'NodePrompt'
            self.prompt = SerialNodeEdgePrompt(
                dim_list=dim_list,
                edge_type=edge_type,
                node_type=node_type,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts
            ).to(self.device)

        elif self.prompt_type == 'ParallelNodeEdgePrompt':
            edge_type = 'EdgePrompt' if self.num_prompts == 1 else 'EdgePromptplus'
            node_type = self.node_prompt_type if self.node_prompt_type else 'NodePrompt'
            self.prompt = ParallelNodeEdgePrompt(
                dim_list=dim_list,
                edge_type=edge_type,
                node_type=node_type,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts,
                fusion_method=self.fusion_method
            ).to(self.device)

        elif self.prompt_type == 'InteractiveNodeEdgePrompt':
            edge_type = 'EdgePrompt' if self.num_prompts == 1 else 'EdgePromptplus'
            node_type = self.node_prompt_type if self.node_prompt_type else 'NodePrompt'
            self.prompt = InteractiveNodeEdgePrompt(
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
            self.logger.info("不使用提示学习")

    def train(self, batch_size, lr=0.001, decay=0, epochs=100):
        """训练模型"""
        # 对于脑成像数据集，可能需要调整batch_size
        if self.dataset_name.startswith('ABIDE'):
            # ABIDE数据集通常图比较大，减小batch_size
            batch_size = max(batch_size, 32)
            self.logger.info(f"调整batch_size为: {batch_size}")

        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

        learnable_parameters = list(self.classifier.parameters())
        if self.prompt is not None:
            learnable_parameters += list(self.prompt.parameters())

        optimizer = torch.optim.Adam(learnable_parameters, lr=lr, weight_decay=decay)

        best_test_accuracy = 0
        best_test_f1 = 0
        patience_counter = 0
        early_stop_patience = 20  # 早停耐心值

        for epoch in range(1, 1 + epochs):
            # 训练阶段
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

            # 评估阶段
            if epoch % 1 == 0:
                self.gnn.eval()
                pred_list = []
                label_list = []
                total_loss = []

                with torch.no_grad():
                    for i, data in enumerate(test_loader):
                        data = data.to(self.device)
                        emb = self.gnn(data, self.prompt_type, self.prompt, pooling='mean')
                        out = self.classifier(emb)
                        loss = F.cross_entropy(out, data.y.squeeze())
                        pred_list.extend(out.argmax(1).tolist())
                        label_list.extend(data.y.squeeze().tolist())
                        total_loss.append(loss.item())

                test_accuracy = metrics.accuracy_score(y_true=label_list, y_pred=pred_list)
                test_f1 = metrics.f1_score(y_true=label_list, y_pred=pred_list, average='weighted')
                test_loss = np.mean(total_loss)

                # 更新最佳结果
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    best_test_f1 = test_f1
                    patience_counter = 0
                else:
                    patience_counter += 1

                # 记录日志
                log_info = ''.join(['| epoch: {:4d} '.format(epoch),
                                    '| train_loss: {:7.5f}'.format(train_loss),
                                    '| test_loss: {:7.5f}'.format(test_loss),
                                    '| test_accuracy: {:7.5f}'.format(test_accuracy),
                                    '| test_f1: {:7.5f}'.format(test_f1),
                                    '| best_accuracy: {:7.5f} |'.format(best_test_accuracy)]
                                   )
                self.logger.info(log_info)

                # 早停
                if patience_counter >= early_stop_patience:
                    self.logger.info(f"早停：{early_stop_patience}轮未改善")
                    break

        # 返回最佳结果
        return best_test_accuracy, best_test_f1


def set_random_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(args, seed):
    """运行单次实验"""
    # 创建日志文件名
    log_dir = os.path.join('log', args.dataset_name)
    os.makedirs(log_dir, exist_ok=True)

    filename = os.path.join(log_dir,
                            f'{args.shots}_{args.pretrain_task}_{args.gnn_type}_{args.prompt_type}_{args.num_prompts}_{seed}.log')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger = Logger(filename, formatter)

    # 设置随机种子
    set_random_seed(seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # 创建任务
    task = GraphTask(
        args.dataset_name, args.shots, args.gnn_type, args.num_layer, args.hidden_dim, device,
        args.pretrain_task, args.prompt_type, args.num_prompts, logger,
        node_prompt_type=args.node_prompt_type, node_num_prompts=args.node_num_prompts,
        fusion_method=args.fusion_method
    )

    # 训练
    best_acc, best_f1 = task.train(args.batch_size, epochs=args.epochs)
    return best_acc, best_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Node-Edge Prompt Fusion for Graph Classification')

    # 基础参数
    parser.add_argument('--dataset_name', type=str, default='ABIDE',
                        help='数据集名称 (ENZYMES, DD, NCI1, NCI109, Mutagenicity, ABIDE, ABIDE_dynamic, ABIDE_phase)')
    parser.add_argument('--shots', type=int, default=50, help='每类样本数 (default: 50)')
    parser.add_argument('--gnn_type', type=str, default='GIN', help='GNN类型')
    parser.add_argument('--num_layer', type=int, default=5, help='GNN层数 (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度 (default: 128)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU设备ID (default: 0)')
    parser.add_argument('--pretrain_task', type=str, default=None, help='预训练任务 (可选)')

    # 提示相关参数
    parser.add_argument('--prompt_type', type=str, default='SerialNodeEdgePrompt',
                        choices=['EdgePrompt', 'EdgePromptplus', 'SerialNodeEdgePrompt',
                                 'ParallelNodeEdgePrompt', 'InteractiveNodeEdgePrompt'],
                        help='提示融合方法')
    parser.add_argument('--num_prompts', type=int, default=5, help='边提示锚点数 (default: 5)')
    parser.add_argument('--node_prompt_type', type=str, default='NodePrompt',
                        choices=['NodePrompt', 'NodePromptplus'],
                        help='节点提示类型')
    parser.add_argument('--node_num_prompts', type=int, default=5, help='节点提示锚点数 (default: 5)')
    parser.add_argument('--fusion_method', type=str, default='weighted',
                        choices=['weighted', 'gated'],
                        help='并行策略的融合方法')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小 (default: 256)')
    parser.add_argument('--epochs', type=int, default=400, help='训练轮数 (default: 400)')

    args = parser.parse_args()

    # 运行实验
    all_results_acc = []
    all_results_f1 = []

    for seed in range(5):
        acc, f1 = run(args, seed)
        all_results_acc.append(acc)
        all_results_f1.append(f1)
        print(f"Seed {seed}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

    print(f"\n最终结果 ({args.dataset_name} - {args.prompt_type}):")
    print(f"平均准确率: {np.mean(all_results_acc):.4f} ± {np.std(all_results_acc):.4f}")
    print(f"平均F1分数: {np.mean(all_results_f1):.4f} ± {np.std(all_results_f1):.4f}")

    # 保存结果
    result_dir = os.path.join('results', args.dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, f'{args.prompt_type}_results.txt'), 'a') as f:
        f.write(f"{args.dataset_name} {args.shots} {args.pretrain_task} {args.prompt_type}: ")
        f.write(f"Acc={np.mean(all_results_acc):.4f}±{np.std(all_results_acc):.4f}, ")
        f.write(f"F1={np.mean(all_results_f1):.4f}±{np.std(all_results_f1):.4f}\n")