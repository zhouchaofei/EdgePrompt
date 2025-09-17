"""
修改后的下游任务模块，支持脑成像数据集（ABIDE, MDD, ADHD）和分子图数据集
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
                    ParallelNodeEdgePrompt, InteractiveNodeEdgePrompt, ComplementaryNodeEdgePrompt,
                    ContrastiveNodeEdgePrompt, SpectralNodeEdgePrompt)
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

        # 支持的所有数据集（更新列表）
        supported_datasets = [
            # 分子图数据集
            'ENZYMES', 'DD', 'NCI1', 'NCI109', 'Mutagenicity',
            # ABIDE数据集
            'ABIDE', 'ABIDE_corr', 'ABIDE_dynamic', 'ABIDE_phase',
            # MDD数据集
            'MDD', 'MDD_corr', 'MDD_dynamic', 'MDD_phase',
            # ADHD数据集
            'ADHD', 'ADHD_corr', 'ADHD_dynamic', 'ADHD_phase'
        ]

        if self.dataset_name in supported_datasets:
            self.graph_list, self.input_dim, self.output_dim = load_graph_data(
                self.dataset_name, data_folder='./data'
            )

            # 特殊处理脑成像数据集
            if self.dataset_name.startswith(('ABIDE', 'MDD', 'ADHD')):
                # 对于脑成像数据，可能需要调整shots参数
                max_shots_per_class = 50  # 默认最大值

                # 根据不同数据集调整
                if self.dataset_name.startswith('MDD'):
                    max_shots_per_class = 30  # MDD数据可能较少
                elif self.dataset_name.startswith('ADHD'):
                    max_shots_per_class = 40  # ADHD数据中等

                if self.shots > max_shots_per_class:
                    self.logger.warning(
                        f"{self.dataset_name}数据集样本有限，"
                        f"将shots从{self.shots}调整为{max_shots_per_class}"
                    )
                    self.shots = min(self.shots, max_shots_per_class)

                # 检查数据是否成功加载
                if not self.graph_list or len(self.graph_list) == 0:
                    self.logger.error(f"数据集 {self.dataset_name} 加载失败或为空")
                    raise ValueError(f"无法加载数据集 {self.dataset_name}")

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
            if self.dataset_name.startswith(('ABIDE', 'MDD', 'ADHD')):
                # 提取基础数据集名称（去除后缀）
                base_dataset = self.dataset_name.split('_')[0]

                # 根据预训练任务选择模型路径
                if self.pretrain_task == 'GraphMAE':
                    # GraphMAE跨疾病预训练模型
                    # 例如：在MDD上训练用于ABIDE，或在ABIDE上训练用于MDD
                    if base_dataset == 'ABIDE':
                        pretrained_gnn_file = f'./pretrained_models/graphmae/graphmae_MDD_for_ABIDE_correlation_matrix.pth'
                    elif base_dataset == 'MDD':
                        pretrained_gnn_file = f'./pretrained_models/graphmae/graphmae_ABIDE_for_MDD_correlation_matrix.pth'
                    else:
                        pretrained_gnn_file = f'./pretrained_models/graphmae/{base_dataset}_graphmae.pth'

                elif self.pretrain_task == 'EdgePrediction':
                    # Edge Prediction跨疾病预训练模型
                    if base_dataset == 'ABIDE':
                        pretrained_gnn_file = f'./pretrained_models/edge_prediction/edge_prediction_MDD_for_ABIDE_correlation_matrix.pth'
                    elif base_dataset == 'MDD':
                        pretrained_gnn_file = f'./pretrained_models/edge_prediction/edge_prediction_ABIDE_for_MDD_correlation_matrix.pth'
                    else:
                        pretrained_gnn_file = f'./pretrained_models/edge_prediction/{base_dataset}_edge_prediction.pth'

                else:
                    # 其他预训练任务
                    pretrained_gnn_file = f'./pretrained_gnns/{base_dataset}_{self.pretrain_task}_{self.gnn_type}_{self.num_layer}.pth'
            else:
                pretrained_gnn_file = f'./pretrained_gnns/{self.dataset_name}_{self.pretrain_task}_{self.gnn_type}_{self.num_layer}.pth'

            if os.path.exists(pretrained_gnn_file):
                checkpoint = torch.load(pretrained_gnn_file, map_location=self.device)

                # 根据不同的预训练模型格式加载权重
                if 'model_state_dict' in checkpoint:
                    # GraphMAE和EdgePrediction格式
                    if self.pretrain_task == 'GraphMAE':
                        # GraphMAE的encoder权重
                        self.gnn.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        self.logger.info(f"加载GraphMAE预训练模型: {pretrained_gnn_file}")
                    elif self.pretrain_task == 'EdgePrediction':
                        # EdgePrediction的encoder权重
                        # 需要从EdgePredictionModel中提取encoder部分
                        encoder_state_dict = {}
                        for k, v in checkpoint['model_state_dict'].items():
                            if k.startswith('encoder.'):
                                # 移除'encoder.'前缀
                                new_k = k.replace('encoder.', '')
                                encoder_state_dict[new_k] = v
                        self.gnn.load_state_dict(encoder_state_dict, strict=False)
                        self.logger.info(f"加载EdgePrediction预训练模型: {pretrained_gnn_file}")
                else:
                    # 原始格式
                    self.gnn.load_state_dict(checkpoint)
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

        # 方案四：互补学习融合
        elif self.prompt_type == 'ComplementaryNodeEdgePrompt':
            edge_type = 'EdgePrompt' if self.num_prompts == 1 else 'EdgePromptplus'
            node_type = self.node_prompt_type if self.node_prompt_type else 'NodePrompt'
            self.prompt = ComplementaryNodeEdgePrompt(
                dim_list=dim_list,
                edge_type=edge_type,
                node_type=node_type,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts,
                recon_weight=0.1,  # 可以从args传入
                link_pred_weight=0.1  # 可以从args传入
            ).to(self.device)

        # 方案五：对比学习融合
        elif self.prompt_type == 'ContrastiveNodeEdgePrompt':
            edge_type = 'EdgePrompt' if self.num_prompts == 1 else 'EdgePromptplus'
            node_type = self.node_prompt_type if self.node_prompt_type else 'NodePrompt'
            self.prompt = ContrastiveNodeEdgePrompt(
                dim_list=dim_list,
                edge_type=edge_type,
                node_type=node_type,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts,
                temperature=0.5,  # 可以从args传入
                contrast_weight=0.1  # 可以从args传入
            ).to(self.device)

        # 方案六：图频域融合
        elif self.prompt_type == 'SpectralNodeEdgePrompt':
            edge_type = 'EdgePrompt' if self.num_prompts == 1 else 'EdgePromptplus'
            node_type = self.node_prompt_type if self.node_prompt_type else 'NodePrompt'
            self.prompt = SpectralNodeEdgePrompt(
                dim_list=dim_list,
                edge_type=edge_type,
                node_type=node_type,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts,
                num_filters=8  # 可以从args传入
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
        if self.dataset_name.startswith(('ABIDE', 'MDD', 'ADHD')):
            # 脑成像数据集通常图比较大，使用较小的batch_size
            if batch_size > 64:
                batch_size = 32
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
        early_stop_patience = 20

        for epoch in range(1, 1 + epochs):
            # 训练阶段
            total_loss = []
            total_aux_loss = []  # 新增：辅助损失
            self.gnn.train()

            for i, data in enumerate(train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()

                # 计算主任务损失
                emb = self.gnn(data, self.prompt_type, self.prompt, pooling='mean')
                out = self.classifier(emb)
                main_loss = F.cross_entropy(out, data.y.squeeze())

                # 计算辅助损失（如果是互补学习）
                if self.prompt_type == 'ComplementaryNodeEdgePrompt':
                    # 获取中间结果
                    x = data.x
                    h_list = [x]
                    aux_loss_total = 0

                    for layer in range(min(1, self.gnn.num_layer)):  # 只计算第一层的辅助损失
                        final_x, edge_prompt, node_x, edge_x = \
                            self.prompt.get_complementary_prompts(
                                h_list[layer], data.edge_index, layer
                            )
                        aux_losses = self.prompt.compute_auxiliary_losses(
                            h_list[layer], data.edge_index, layer, node_x, edge_x
                        )
                        if aux_losses:
                            aux_loss_total += sum(aux_losses.values())

                    loss = main_loss + aux_loss_total
                    if aux_loss_total > 0:
                        total_aux_loss.append(aux_loss_total.item())

                # 计算对比损失（如果是对比学习）
                elif self.prompt_type == 'ContrastiveNodeEdgePrompt':
                    contrast_loss_total = 0
                    x = data.x
                    h_list = [x]

                    for layer in range(min(1, self.gnn.num_layer)):  # 只计算第一层的对比损失
                        final_x, edge_prompt, views = \
                            self.prompt.get_contrastive_prompts(
                                h_list[layer], data.edge_index, layer
                            )
                        contrast_loss = self.prompt.compute_contrastive_loss(
                            views, layer, data.batch
                        )
                        contrast_loss_total += contrast_loss

                    loss = main_loss + contrast_loss_total
                    if contrast_loss_total > 0:
                        total_aux_loss.append(contrast_loss_total.item())
                else:
                    loss = main_loss

                loss.backward()
                optimizer.step()
                total_loss.append(main_loss.item())

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
    parser.add_argument('--dataset_name', type=str, default='MDD',
                        help='数据集名称 (支持: ENZYMES, DD, NCI1, NCI109, Mutagenicity, '
                             'ABIDE, ABIDE_dynamic, ABIDE_phase, '
                             'MDD, MDD_dynamic, MDD_phase, '
                             'ADHD, ADHD_dynamic, ADHD_phase)')
    parser.add_argument('--shots', type=int, default=30, help='每类样本数 (default: 30)')
    parser.add_argument('--gnn_type', type=str, default='GIN', help='GNN类型')
    parser.add_argument('--num_layer', type=int, default=5, help='GNN层数 (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度 (default: 128)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU设备ID (default: 0)')
    parser.add_argument('--pretrain_task', type=str, default=None, help='预训练任务 (可选)')

    # 提示相关参数
    parser.add_argument('--prompt_type', type=str, default='SerialNodeEdgePrompt',
                        choices=['EdgePrompt', 'EdgePromptplus', 'SerialNodeEdgePrompt',
                                 'ParallelNodeEdgePrompt', 'InteractiveNodeEdgePrompt',
                                 'ComplementaryNodeEdgePrompt', 'ContrastiveNodeEdgePrompt',
                                 'SpectralNodeEdgePrompt'],
                        help='提示融合方法')

    # 新增参数
    parser.add_argument('--recon_weight', type=float, default=0.1,
                        help='互补学习中重构损失权重')
    parser.add_argument('--link_pred_weight', type=float, default=0.1,
                        help='互补学习中链接预测损失权重')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='对比学习温度参数')
    parser.add_argument('--contrast_weight', type=float, default=0.1,
                        help='对比学习损失权重')
    # 谱域融合参数
    parser.add_argument('--num_filters', type=int, default=8,
                        help='谱域融合的滤波器数量')

    parser.add_argument('--num_prompts', type=int, default=5, help='边提示锚点数 (default: 5)')
    parser.add_argument('--node_prompt_type', type=str, default='NodePrompt',
                        choices=['NodePrompt', 'NodePromptplus'],
                        help='节点提示类型')
    parser.add_argument('--node_num_prompts', type=int, default=5, help='节点提示锚点数 (default: 5)')
    parser.add_argument('--fusion_method', type=str, default='weighted',
                        choices=['weighted', 'gated'],
                        help='并行策略的融合方法')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小 (default: 32)')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数 (default: 200)')

    args = parser.parse_args()

    # 打印实验配置
    print("\n" + "="*60)
    print("实验配置")
    print("="*60)
    for key, value in vars(args).items():
        print(f"{key:20s}: {value}")
    print("="*60 + "\n")

    # 运行实验
    all_results_acc = []
    all_results_f1 = []

    for seed in range(5):
        print(f"\n运行 Seed {seed}...")
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