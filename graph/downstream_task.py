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
                    ContrastiveNodeEdgePrompt, SpectralNodeEdgePrompt, HierarchicalGraphTransformerPrompt,
                    GraphNeuralODEPrompt, MetaLearningPrompt, CausalGraphPrompt, GraphWaveletPrompt, DiffusionPrompt,
                    RLPrompt, AttentionFlowPrompt, HypergraphPrompt, NodePrompt, TopologyPrompt, NodePromptplus)
from logger import Logger


class GraphTask():
    def __init__(self, dataset_name, shots, gnn_type, num_layer, hidden_dim, device,
                 pretrain_task, prompt_type, num_prompts, logger,
                 node_prompt_type=None, node_num_prompts=5, fusion_method='weighted',
                 graph_method='correlation_matrix', pretrain_source='auto'):  # 添加graph_method参数
        self.dataset_name = dataset_name
        self.graph_method = graph_method  # 新增
        self.shots = shots
        self.gnn_type = gnn_type
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.device = device
        self.pretrain_task = pretrain_task
        self.pretrain_source = pretrain_source  # 新增
        self.prompt_type = prompt_type
        self.num_prompts = num_prompts
        self.node_prompt_type = node_prompt_type
        self.node_num_prompts = node_num_prompts
        self.fusion_method = fusion_method
        self.logger = logger

        # 日志记录
        self.logger.info("=" * 60)
        self.logger.info("初始化GraphTask")
        self.logger.info("=" * 60)
        self.logger.info(f"数据集: {dataset_name}")
        self.logger.info(f"图构建方法: {graph_method}")
        self.logger.info(f"Few-shot数量: {shots}")
        self.logger.info(f"GNN类型: {gnn_type}")
        self.logger.info(f"层数: {num_layer}")
        self.logger.info(f"隐藏维度: {hidden_dim}")
        self.logger.info(f"预训练任务: {pretrain_task if pretrain_task else '无'}")
        self.logger.info(f"Prompt类型: {prompt_type}")
        self.logger.info(f"预训练源: {pretrain_source}")
        self.logger.info(f"设备: {device}")

        # 加载数据集
        self.load_dataset()

        # 初始化模型和提示
        self.initialize_model()
        self.initialize_prompt()

    def load_dataset(self):
        """加载数据集"""

        # 获取数据集信息
        dataset_info = get_dataset_info(self.dataset_name)
        if dataset_info:
            self.logger.info(f"数据集类型: {dataset_info.get('type', '未知')}")
            self.logger.info(f"任务: {dataset_info.get('task', '未知')}")
            self.logger.info(f"描述: {dataset_info.get('description', '无')}")

        # 支持的所有数据集
        supported_datasets = [
            # 分子图数据集
            'ENZYMES', 'DD', 'NCI1', 'NCI109', 'Mutagenicity',
            # 脑成像数据集 - 原始名称
            'ABIDE', 'MDD', 'ADHD'
        ]

        # 特殊处理脑成像数据集
        if self.dataset_name in ['ABIDE', 'MDD', 'ADHD']:
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

            # 根据graph_method构建实际的数据集名称
            if hasattr(self, 'graph_method'):
                if self.graph_method == 'correlation_matrix':
                    dataset_name_to_load = f"{self.dataset_name}_corr"
                elif self.graph_method == 'dynamic_connectivity':
                    dataset_name_to_load = f"{self.dataset_name}_dynamic"
                elif self.graph_method == 'phase_synchronization':
                    dataset_name_to_load = f"{self.dataset_name}_phase"
                else:
                    dataset_name_to_load = self.dataset_name
            else:
                # 如果没有graph_method属性，使用默认
                dataset_name_to_load = self.dataset_name

        else:
            # 分子图数据集直接使用原始名称
            dataset_name_to_load = self.dataset_name

        # 使用修改后的数据集名称加载数据
        self.graph_list, self.input_dim, self.output_dim = load_graph_data(
            dataset_name_to_load, data_folder='./data'
        )

        # 检查数据是否加载成功
        if not self.graph_list or len(self.graph_list) == 0:
            self.logger.error(f"数据集 {self.dataset_name} 加载失败或为空")
            raise ValueError(f"无法加载数据集 {self.dataset_name}")

        # 划分训练和测试集（这个由GraphDownstream处理）
        self.train_data, self.test_data = GraphDownstream(
            self.graph_list, self.shots, test_fraction=0.4
        )

        self.logger.info(f"数据集 {self.dataset_name} 加载成功")
        self.logger.info(f"输入维度: {self.input_dim}, 输出类别数: {self.output_dim}")
        self.logger.info(f"训练集大小: {len(self.train_data)}, 测试集大小: {len(self.test_data)}")

    def initialize_model(self):
        """初始化GNN模型并加载预训练权重"""

        # 初始化基础GNN模型
        self.logger.info(f"初始化{self.gnn_type}模型...")
        if self.gnn_type == 'GIN':
            self.gnn = GIN(
                num_layer=self.num_layer,
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim
            )
        else:
            raise ValueError(f"错误: 无效的GNN类型 {self.gnn_type}！")

        # 处理预训练模型加载
        if self.pretrain_task and self.pretrain_task != 'None' and self.pretrain_task != '':
            pretrained_gnn_file = self.get_pretrain_model_path()

            if pretrained_gnn_file and os.path.exists(pretrained_gnn_file):
                self.logger.info(f"正在加载预训练模型: {pretrained_gnn_file}")

                try:
                    # 加载checkpoint
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
                            # 其他格式，尝试直接加载
                            self.gnn.load_state_dict(checkpoint['model_state_dict'], strict=False)
                            self.logger.info(f"加载预训练模型: {pretrained_gnn_file}")

                    elif 'state_dict' in checkpoint:
                        # 分子图预训练模型格式（GraphCL, SimGRACE等）
                        state_dict = checkpoint['state_dict']

                        # 处理可能的模块前缀问题
                        new_state_dict = {}
                        for k, v in state_dict.items():
                            # 移除可能的module.前缀
                            if k.startswith('module.'):
                                new_k = k.replace('module.', '')
                            else:
                                new_k = k
                            new_state_dict[new_k] = v

                        self.gnn.load_state_dict(new_state_dict, strict=False)
                        self.logger.info(f"加载分子图预训练模型: {pretrained_gnn_file}")

                    else:
                        # 原始格式
                        self.gnn.load_state_dict(checkpoint, strict=False)
                        self.logger.info(f"加载预训练模型: {pretrained_gnn_file}")

                    self.logger.info("预训练权重加载成功！")

                except Exception as e:
                    self.logger.warning(f"加载预训练模型失败: {e}")
                    self.logger.warning("将从头开始训练...")
            else:
                self.logger.warning(f"预训练模型不存在: {pretrained_gnn_file}")
                self.logger.warning("将从头开始训练...")
        else:
            self.logger.info("不使用预训练模型，从头开始训练")

        # 多GPU支持
        # if torch.cuda.device_count() > 1:
        #     self.logger.info(f"检测到{torch.cuda.device_count()}个GPU，启用数据并行")
        #     self.gnn = nn.DataParallel(self.gnn)

        # 使用单GPU
        self.gnn.to(self.device)
        self.logger.info(f"使用设备: {self.device}")

        # 初始化分类器
        self.logger.info("初始化分类器...")
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

        # if torch.cuda.device_count() > 1:
        #     self.classifier = nn.DataParallel(self.classifier)

        self.classifier.to(self.device)

        # 打印模型信息
        total_params = sum(p.numel() for p in self.gnn.parameters())
        trainable_params = sum(p.numel() for p in self.gnn.parameters() if p.requires_grad)
        self.logger.info(f"模型结构:\n{self.gnn}")
        self.logger.info(f"模型参数统计:")
        self.logger.info(f"  总参数量: {total_params:,}")
        self.logger.info(f"  可训练参数量: {trainable_params:,}")

    def get_pretrain_model_path(self):
        """根据数据集和预训练任务获取模型路径"""

        # 脑成像数据集的预训练模型路径
        if self.dataset_name in ['ABIDE', 'MDD', 'ADHD']:
            pretrain_dir = './pretrained_models'

            if self.pretrain_task == 'GraphMAE':
                graph_method = getattr(self, 'graph_method', 'correlation_matrix')

                # 获取预训练源设置
                pretrain_source = getattr(self, 'pretrain_source', 'auto')

                if pretrain_source == 'same':
                    # 明确使用同疾病预训练
                    filename = f"graphmae_{self.dataset_name}_for_{self.dataset_name}_{graph_method}.pth"
                    self.logger.info(f"使用同疾病预训练: {self.dataset_name} -> {self.dataset_name}")

                elif pretrain_source == 'cross':
                    # 明确使用跨疾病预训练
                    source = 'ABIDE' if self.dataset_name == 'MDD' else 'MDD'
                    filename = f"graphmae_{source}_for_{self.dataset_name}_{graph_method}.pth"
                    self.logger.info(f"使用跨疾病预训练: {source} -> {self.dataset_name}")

                elif pretrain_source == 'auto':
                    # 自动选择（优先同疾病）
                    same_filename = f"graphmae_{self.dataset_name}_for_{self.dataset_name}_{graph_method}.pth"
                    same_path = os.path.join(pretrain_dir, 'graphmae', same_filename)

                    if os.path.exists(same_path):
                        filename = same_filename
                        self.logger.info(f"自动选择: 找到同疾病预训练 {self.dataset_name} -> {self.dataset_name}")
                    else:
                        source = 'ABIDE' if self.dataset_name == 'MDD' else 'MDD'
                        filename = f"graphmae_{source}_for_{self.dataset_name}_{graph_method}.pth"
                        self.logger.info(f"自动选择: 使用跨疾病预训练 {source} -> {self.dataset_name}")

                elif pretrain_source in ['ABIDE', 'MDD', 'ADHD']:
                    # 直接指定源数据集
                    filename = f"graphmae_{pretrain_source}_for_{self.dataset_name}_{graph_method}.pth"
                    self.logger.info(f"使用指定源预训练: {pretrain_source} -> {self.dataset_name}")

                else:
                    self.logger.warning(f"未知的pretrain_source: {pretrain_source}，使用auto模式")
                    pretrain_source = 'auto'
                    # 递归调用auto逻辑
                    return self.get_pretrain_model_path()

                path = os.path.join(pretrain_dir, 'graphmae', filename)

            elif self.pretrain_task == 'EdgePrediction':
                # EdgePrediction的逻辑相同
                graph_method = getattr(self, 'graph_method', 'correlation_matrix')
                pretrain_source = getattr(self, 'pretrain_source', 'auto')

                if pretrain_source == 'same':
                    filename = f"edge_prediction_{self.dataset_name}_for_{self.dataset_name}_{graph_method}.pth"
                    self.logger.info(f"使用同疾病预训练: {self.dataset_name} -> {self.dataset_name}")

                elif pretrain_source == 'cross':
                    source = 'ABIDE' if self.dataset_name == 'MDD' else 'MDD'
                    filename = f"edge_prediction_{source}_for_{self.dataset_name}_{graph_method}.pth"
                    self.logger.info(f"使用跨疾病预训练: {source} -> {self.dataset_name}")

                elif pretrain_source == 'auto':
                    same_filename = f"edge_prediction_{self.dataset_name}_for_{self.dataset_name}_{graph_method}.pth"
                    same_path = os.path.join(pretrain_dir, 'edge_prediction', same_filename)

                    if os.path.exists(same_path):
                        filename = same_filename
                        self.logger.info(f"自动选择: 找到同疾病预训练 {self.dataset_name} -> {self.dataset_name}")
                    else:
                        source = 'ABIDE' if self.dataset_name == 'MDD' else 'MDD'
                        filename = f"edge_prediction_{source}_for_{self.dataset_name}_{graph_method}.pth"
                        self.logger.info(f"自动选择: 使用跨疾病预训练 {source} -> {self.dataset_name}")

                elif pretrain_source in ['ABIDE', 'MDD', 'ADHD']:
                    filename = f"edge_prediction_{pretrain_source}_for_{self.dataset_name}_{graph_method}.pth"
                    self.logger.info(f"使用指定源预训练: {pretrain_source} -> {self.dataset_name}")

                else:
                    self.logger.warning(f"未知的pretrain_source: {pretrain_source}，使用auto模式")
                    return self.get_pretrain_model_path()

                path = os.path.join(pretrain_dir, 'edge_prediction', filename)

            else:
                # 其他预训练任务
                filename = f"{self.pretrain_task}.pth"
                path = os.path.join(pretrain_dir, filename)

            self.logger.info(f"预训练模型路径: {path}")
            return path

        # 分子图数据集的预训练模型路径（保持不变）
        elif self.dataset_name in ['ENZYMES', 'DD', 'NCI1', 'NCI109', 'Mutagenicity']:
            pretrain_dir = './pretrained_gnns'

            if self.pretrain_task and self.pretrain_task != 'None':
                filename = f"{self.dataset_name}_{self.pretrain_task}_GIN_5.pth"
                path = os.path.join(pretrain_dir, filename)
                self.logger.info(f"分子图预训练模型路径: {path}")
                return path

        return None


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
        # 新增方法
        elif self.prompt_type == 'HierarchicalGraphTransformerPrompt':
            self.prompt = HierarchicalGraphTransformerPrompt(
                dim_list=dim_list,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts
            ).to(self.device)

        elif self.prompt_type == 'GraphNeuralODEPrompt':
            self.prompt = GraphNeuralODEPrompt(
                dim_list=dim_list,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts,
                ode_steps=5  # 可从args传入
            ).to(self.device)

        elif self.prompt_type == 'MetaLearningPrompt':
            self.prompt = MetaLearningPrompt(
                dim_list=dim_list,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts
            ).to(self.device)

        elif self.prompt_type == 'CausalGraphPrompt':
            self.prompt = CausalGraphPrompt(
                dim_list=dim_list,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts
            ).to(self.device)

        # 新增方法初始化
        elif self.prompt_type == 'GraphWaveletPrompt':
            self.prompt = GraphWaveletPrompt(
                dim_list=dim_list,
                edge_type='EdgePromptplus',
                node_type='NodePromptplus',
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts,
                num_scales=4
            ).to(self.device)

        elif self.prompt_type == 'DiffusionPrompt':
            self.prompt = DiffusionPrompt(
                dim_list=dim_list,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts,
                diffusion_steps=3
            ).to(self.device)

        elif self.prompt_type == 'RLPrompt':
            self.prompt = RLPrompt(
                dim_list=dim_list,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts
            ).to(self.device)

        elif self.prompt_type == 'AttentionFlowPrompt':
            self.prompt = AttentionFlowPrompt(
                dim_list=dim_list,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts,
                flow_steps=3
            ).to(self.device)

        elif self.prompt_type == 'HypergraphPrompt':
            self.prompt = HypergraphPrompt(
                dim_list=dim_list,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts,
                hyperedge_size=3
            ).to(self.device)

        elif self.prompt_type == 'TopologyPrompt':
            self.prompt = TopologyPrompt(
                dim_list=dim_list,
                edge_num_anchors=self.num_prompts,
                node_num_anchors=self.node_num_prompts
            ).to(self.device)

        # 纯节点提示baseline
        elif self.prompt_type == 'NodePrompt':
            # 为每一层创建NodePrompt
            self.prompt = nn.ModuleList([
                NodePrompt(dim) for dim in dim_list
            ]).to(self.device)

        elif self.prompt_type == 'NodePromptplus':
            # 为每一层创建NodePromptplus
            self.prompt = nn.ModuleList([
                NodePromptplus(dim, self.node_num_prompts) for dim in dim_list
            ]).to(self.device)

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


class SF_DPL_Task:
    """SF-DPL下游任务（完整优化版）"""

    def __init__(self,
                 dataset_name,
                 shots,
                 device,
                 logger,
                 graph_method='correlation_matrix',
                 pretrain_task='GraphMAE',
                 pretrain_source='same',
                 num_layer=5,
                 hidden_dim=128,
                 drop_ratio=0.3,
                 num_prompts=5,
                 epochs=100,
                 lr=0.001):

        self.dataset_name = dataset_name
        self.shots = shots
        self.device = device
        self.logger = logger
        self.graph_method = graph_method
        self.pretrain_task = pretrain_task
        self.pretrain_source = pretrain_source

        # 保存超参数
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.drop_ratio = drop_ratio
        self.num_prompts = num_prompts
        self.epochs = epochs
        self.lr = lr

        # 加载双流数据
        self.load_dual_stream_data()

        # ⭐ 关键修改：动态获取输入维度
        sample_data = self.dual_stream_data[0]
        struct_input_dim = sample_data[1].x.shape[1]  # 结构流
        func_input_dim = sample_data[0].x.shape[1]  # 功能流

        self.logger.info(f"检测到输入特征维度:")
        self.logger.info(f"  结构流: {struct_input_dim}")
        self.logger.info(f"  功能流: {func_input_dim}")

        # 初始化SF-DPL模型
        from sf_dpl_model import SF_DPL

        self.model = SF_DPL(
            num_layer=num_layer,
            struct_input_dim=struct_input_dim,
            func_input_dim=func_input_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            drop_ratio=drop_ratio,
            num_prompts=num_prompts
        ).to(device)

        # 加载预训练权重
        self.load_pretrained_weights()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"模型参数统计:")
        self.logger.info(f"  总参数量: {total_params:,}")
        self.logger.info(f"  可训练参数: {trainable_params:,}")

    def load_pretrained_weights(self):
        """加载预训练权重到双流编码器"""
        if self.pretrain_task is None or self.pretrain_task == 'None':
            self.logger.info("不使用预训练模型")
            return

        # 确定预训练模型路径
        if self.pretrain_source == 'same':
            source = self.dataset_name
            target = self.dataset_name
        elif self.pretrain_source == 'cross':
            source = 'MDD' if self.dataset_name == 'ABIDE' else 'ABIDE'
            target = self.dataset_name
        else:
            source = self.pretrain_source
            target = self.dataset_name

        # ⭐ 关键：使用双流预训练模型
        if self.pretrain_task == 'GraphMAE':
            pretrain_dir = './pretrained_models/dual_stream_temporal'
            pretrain_file = f'graphmae_{source}_for_{target}.pth'
        elif self.pretrain_task == 'EdgePrediction':
            pretrain_dir = './pretrained_models/dual_stream_temporal'
            pretrain_file = f'edge_prediction_{source}_for_{target}.pth'
        else:
            self.logger.warning(f"未知的预训练任务: {self.pretrain_task}")
            return

        pretrain_path = os.path.join(pretrain_dir, pretrain_file)

        if not os.path.exists(pretrain_path):
            self.logger.warning(f"预训练模型不存在: {pretrain_path}")
            return

        self.logger.info(f"加载预训练模型: {pretrain_path}")

        try:
            checkpoint = torch.load(pretrain_path, map_location=self.device)

            # 提取GNN权重
            if 'gnn' in checkpoint:
                gnn_state_dict = checkpoint['gnn']
            elif 'model_state_dict' in checkpoint:
                gnn_state_dict = checkpoint['model_state_dict']
            else:
                gnn_state_dict = checkpoint

            # ⭐ 关键：跳过第一层（输入层）的权重
            filtered_state = {}
            for k, v in gnn_state_dict.items():
                # 跳过第一层的权重
                if 'convs.0.mlp.0' in k:
                    self.logger.info(f"跳过第一层权重: {k} (维度不匹配)")
                    continue
                filtered_state[k] = v

            # 加载到结构编码器（strict=False允许部分加载）
            if self.model.struct_encoder:
                self.model.struct_encoder.load_state_dict(filtered_state, strict=False)
                self.logger.info("结构编码器加载预训练权重完成")

            # 加载到功能编码器
            if self.model.func_encoder:
                self.model.func_encoder.load_state_dict(filtered_state, strict=False)
                self.logger.info("功能编码器加载预训练权重完成")

        except Exception as e:
            self.logger.warning(f"加载预训练权重失败: {e}")
            self.logger.warning("将继续训练...")

    def load_dual_stream_data(self):
        """加载双流数据"""
        if self.dataset_name == 'ABIDE':
            data_path = './data/ABIDE/processed/abide_dual_stream_temporal_78.pt'

            if not os.path.exists(data_path):
                self.logger.error(f"数据文件不存在: {data_path}")
                self.logger.info("请先运行数据处理脚本生成双流数据")
                raise FileNotFoundError(f"数据文件不存在: {data_path}")

            self.dual_stream_data = torch.load(data_path)

        elif self.dataset_name == 'MDD':
            data_path = './data/REST-meta-MDD/processed/mdd_dual_stream_temporal_150.pt'

            if not os.path.exists(data_path):
                self.logger.error(f"数据文件不存在: {data_path}")
                raise FileNotFoundError(f"数据文件不存在: {data_path}")

            self.dual_stream_data = torch.load(data_path)

        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")

        # Few-shot划分
        labels = [d[0].y.item() for d in self.dual_stream_data]
        from collections import Counter
        label_counts = Counter(labels)

        train_idx = []
        test_idx = []

        for label in label_counts:
            indices = [i for i, l in enumerate(labels) if l == label]
            random.shuffle(indices)
            train_idx.extend(indices[:self.shots])
            test_idx.extend(indices[self.shots:])

        self.train_data = [self.dual_stream_data[i] for i in train_idx]
        self.test_data = [self.dual_stream_data[i] for i in test_idx]

        self.logger.info(f"数据集: {self.dataset_name}")
        self.logger.info(f"训练样本: {len(self.train_data)}")
        self.logger.info(f"测试样本: {len(self.test_data)}")

    def train_epoch(self):
        """训练一个epoch（改进版）"""
        from torch_geometric.data import Batch
        from sf_dpl_model import train_sf_dpl_one_epoch

        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_ortho_loss = 0
        num_batches = 0

        batch_size = 32
        random.shuffle(self.train_data)

        for i in range(0, len(self.train_data), batch_size):
            batch_data = self.train_data[i:i + batch_size]

            func_list = [d[0] for d in batch_data]
            struct_list = [d[1] for d in batch_data]

            func_batch = Batch.from_data_list(func_list).to(self.device)
            struct_batch = Batch.from_data_list(struct_list).to(self.device)

            # 检查输入数据
            if torch.isnan(func_batch.x).any() or torch.isnan(struct_batch.x).any():
                self.logger.warning("检测到输入数据中有nan，跳过该batch")
                continue

            self.optimizer.zero_grad()
            output, ortho_loss = self.model(struct_batch, func_batch)

            # 检查输出
            if torch.isnan(output).any():
                self.logger.warning("检测到输出中有nan，跳过该batch")
                continue

            ce_loss = F.cross_entropy(output, func_batch.y)
            loss = ce_loss + ortho_loss

            # 检查损失
            if torch.isnan(loss):
                self.logger.warning("检测到损失为nan，跳过该batch")
                continue

            loss.backward()

            # ⭐ 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_ortho_loss += ortho_loss.item()
            num_batches += 1

        if num_batches == 0:
            return 0, 0, 0

        return (total_loss / num_batches,
                total_ce_loss / num_batches,
                total_ortho_loss / num_batches)

    def evaluate(self):
        """评估（返回多个指标）"""
        from torch_geometric.data import Batch
        from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                     precision_score, recall_score)

        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            batch_size = 32
            for i in range(0, len(self.test_data), batch_size):
                batch_data = self.test_data[i:i + batch_size]

                func_list = [d[0] for d in batch_data]
                struct_list = [d[1] for d in batch_data]

                func_batch = Batch.from_data_list(func_list).to(self.device)
                struct_batch = Batch.from_data_list(struct_list).to(self.device)

                output, ortho_loss = self.model(struct_batch, func_batch)

                # 检查nan
                if torch.isnan(output).any():
                    continue

                loss = F.cross_entropy(output, func_batch.y) + ortho_loss
                total_loss += loss.item()
                num_batches += 1

                probs = F.softmax(output, dim=1)
                preds = output.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(func_batch.y.cpu().numpy())

        # 计算各种指标
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        metrics = {}
        metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0
        metrics['accuracy'] = accuracy_score(all_labels, all_preds)
        metrics['f1'] = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        metrics['precision'] = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        metrics['recall'] = recall_score(all_labels, all_preds, average='binary', zero_division=0)

        # AUC（需要至少两个类别）
        if len(np.unique(all_labels)) > 1:
            try:
                metrics['auc'] = roc_auc_score(all_labels, all_probs)
            except:
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0

        return metrics

    def run(self, epochs=None):
        """运行训练和评估"""
        if epochs is None:
            epochs = self.epochs

        best_acc = 0
        best_f1 = 0
        best_epoch = 0
        patience = 0
        max_patience = 20

        self.logger.info("\n" + "=" * 80)
        self.logger.info("开始训练SF-DPL模型")
        self.logger.info("=" * 80)

        for epoch in range(epochs):
            # 训练
            train_loss, train_ce_loss, train_ortho_loss = self.train_epoch()

            # 评估
            if epoch % 1 == 0:
                test_metrics = self.evaluate()

                # 更新最佳结果
                if test_metrics['accuracy'] > best_acc:
                    best_acc = test_metrics['accuracy']
                    best_epoch = epoch
                    patience = 0

                if test_metrics['f1'] > best_f1:
                    best_f1 = test_metrics['f1']
                else:
                    patience += 1

                # 格式化日志输出
                log_info = ''.join([
                    '| epoch: {:4d} '.format(epoch),
                    '| train_loss: {:7.5f} '.format(train_loss),
                    '| train_ce: {:7.5f} '.format(train_ce_loss),
                    '| train_ortho: {:7.5f} '.format(train_ortho_loss),
                    '| test_loss: {:7.5f} '.format(test_metrics['loss']),
                    '| test_acc: {:7.5f} '.format(test_metrics['accuracy']),
                    '| test_f1: {:7.5f} '.format(test_metrics['f1']),
                    '| test_auc: {:7.5f} '.format(test_metrics['auc']),
                    '| best_acc: {:7.5f} |'.format(best_acc)
                ])

                self.logger.info(log_info)

                # 早停
                if patience >= max_patience:
                    self.logger.info(f"早停：{max_patience}轮未改善")
                    break

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"训练完成！最佳准确率: {best_acc:.4f} (Epoch {best_epoch})")
        self.logger.info(f"最佳F1分数: {best_f1:.4f}")
        self.logger.info("=" * 80)

        return best_acc, best_f1

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
        fusion_method=args.fusion_method, graph_method=args.graph_method, pretrain_source=args.pretrain_source
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
    parser.add_argument('--graph_method', type=str, default='correlation_matrix',
                        choices=['correlation_matrix', 'dynamic_connectivity', 'phase_synchronization'],
                        help='图构建方法（仅用于脑成像数据）')
    parser.add_argument('--shots', type=int, default=30, help='每类样本数 (default: 30)')
    parser.add_argument('--gnn_type', type=str, default='GIN', help='GNN类型')
    parser.add_argument('--num_layer', type=int, default=5, help='GNN层数 (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度 (default: 128)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU设备ID (default: 0)')
    parser.add_argument('--pretrain_task', type=str, default=None, help='预训练任务 (可选)')

    # 新增参数：预训练源控制
    parser.add_argument('--pretrain_source', type=str, default='auto',
                        choices=['same', 'cross', 'auto', 'ABIDE', 'MDD', 'ADHD'],
                        help='预训练模型来源: same(同疾病), cross(跨疾病), auto(自动选择), 或指定具体数据集')

    # 提示相关参数
    parser.add_argument('--prompt_type', type=str, default=None,
                        choices=['EdgePrompt', 'EdgePromptplus',
                                 'NodePrompt', 'NodePromptplus',  # 添加纯节点baseline
                                 'SerialNodeEdgePrompt', 'ParallelNodeEdgePrompt',
                                 'InteractiveNodeEdgePrompt', 'ComplementaryNodeEdgePrompt',
                                 'ContrastiveNodeEdgePrompt', 'SpectralNodeEdgePrompt',
                                 'HierarchicalGraphTransformerPrompt', 'GraphNeuralODEPrompt',
                                 'MetaLearningPrompt', 'CausalGraphPrompt',
                                 'GraphWaveletPrompt', 'DiffusionPrompt',  # 新增方法
                                 'RLPrompt', 'AttentionFlowPrompt',
                                 'HypergraphPrompt', 'TopologyPrompt'],
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

    # 添加SF-DPL特定参数
    parser.add_argument('--use_sf_dpl', action='store_true', help='使用SF-DPL方法')
    parser.add_argument('--drop_ratio', type=float, default=0.3, help='Dropout比率')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    # 添加多seed支持
    parser.add_argument('--num_seeds', type=int, default=5, help='运行多少个不同的随机种子')
    parser.add_argument('--start_seed', type=int, default=0, help='起始随机种子')

    args = parser.parse_args()

    if args.use_sf_dpl:
        # 打印实验配置
        print("\n" + "=" * 80)
        print("SF-DPL实验配置")
        print("=" * 80)
        for key, value in vars(args).items():
            print(f"{key:20s}: {value}")
        print("=" * 80 + "\n")

        # 运行多个seed
        all_results_acc = []
        all_results_f1 = []

        for seed_idx in range(args.num_seeds):
            seed = args.start_seed + seed_idx

            print(f"\n{'=' * 80}")
            print(f"运行 Seed {seed} ({seed_idx + 1}/{args.num_seeds})")
            print(f"{'=' * 80}")

            # 设置随机种子
            set_random_seed(seed)

            # 创建logger
            log_dir = os.path.join('log', args.dataset_name, 'sf_dpl')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir,
                                    f'{args.dataset_name}_{args.shots}shot_{args.pretrain_source}_{seed}.log')

            formatter = logging.Formatter('%(asctime)s - %(message)s')
            logger = Logger(log_file, formatter)

            # ⭐ 创建SF_DPL任务
            task = SF_DPL_Task(
                args.dataset_name,
                args.shots,
                torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'),
                logger,
                graph_method=args.graph_method,
                pretrain_task=args.pretrain_task,
                pretrain_source=args.pretrain_source,
                num_layer=args.num_layer,
                hidden_dim=args.hidden_dim,
                drop_ratio=args.drop_ratio,
                num_prompts=args.num_prompts,
                epochs=args.epochs,
                lr=args.lr
            )

            # 训练
            acc, f1 = task.run(epochs=args.epochs)
            all_results_acc.append(acc)
            all_results_f1.append(f1)

            print(f"\nSeed {seed} 结果: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

        # 计算统计结果
        mean_acc = np.mean(all_results_acc)
        std_acc = np.std(all_results_acc)
        mean_f1 = np.mean(all_results_f1)
        std_f1 = np.std(all_results_f1)

        # 打印最终结果
        print(f"\n{'=' * 80}")
        print(f"最终结果 ({args.dataset_name} - SF-DPL - {args.shots}shot - {args.pretrain_source})")
        print(f"{'=' * 80}")
        print(f"准确率: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"F1分数: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"{'=' * 80}\n")

        # 保存结果到文件
        result_dir = os.path.join('results', args.dataset_name)
        os.makedirs(result_dir, exist_ok=True)

        result_file = os.path.join(result_dir, 'sf_dpl_results.txt')
        with open(result_file, 'a') as f:
            f.write(f"\n{'-' * 80}\n")
            f.write(f"Dataset: {args.dataset_name}\n")
            f.write(f"Shots: {args.shots}\n")
            f.write(f"Pretrain: {args.pretrain_task} ({args.pretrain_source})\n")
            f.write(f"Hidden_dim: {args.hidden_dim}, Num_prompts: {args.num_prompts}\n")
            f.write(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
            f.write(f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f}\n")
            f.write(f"Individual results: {[f'{acc:.4f}' for acc in all_results_acc]}\n")
    else:
        # 打印实验配置
        print("\n" + "=" * 60)
        print("实验配置")
        print("=" * 60)
        for key, value in vars(args).items():
            print(f"{key:20s}: {value}")
        print("=" * 60 + "\n")

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