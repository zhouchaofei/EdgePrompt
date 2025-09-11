"""
修改后的数据加载模块，支持ABIDE等脑成像数据集和原有的分子图数据集
"""
import random
import torch
import os
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Flickr, TUDataset
from abide_data import load_abide_data, ABIDEDataProcessor


def load_graph_data(dataset_name, data_folder):
    """
    加载图数据集，支持分子图和脑成像数据集

    Args:
        dataset_name: 数据集名称
        data_folder: 数据保存路径

    Returns:
        dataset: 数据集
        input_dim: 输入维度
        output_dim: 输出维度
    """
    # 原有的分子图数据集
    if dataset_name in ['ENZYMES', 'DD', 'NCI1', 'NCI109', 'Mutagenicity']:
        dataset = TUDataset(f'{data_folder}/TUDataset/', name=dataset_name)
        input_dim = dataset.num_features
        output_dim = dataset.num_classes
        return dataset, input_dim, output_dim

    # 脑成像数据集
    elif dataset_name.startswith('ABIDE'):
        # 支持不同的图构建方法
        # ABIDE_corr: 相关矩阵方法
        # ABIDE_dynamic: 动态连接性方法
        # ABIDE_phase: 相位同步方法
        if dataset_name == 'ABIDE' or dataset_name == 'ABIDE_corr':
            graph_method = 'correlation_matrix'
        elif dataset_name == 'ABIDE_dynamic':
            graph_method = 'dynamic_connectivity'
        elif dataset_name == 'ABIDE_phase':
            graph_method = 'phase_synchronization'
        else:
            graph_method = 'correlation_matrix'

        graph_list, input_dim, output_dim = load_abide_data(
            data_folder=data_folder,
            graph_method=graph_method
        )
        return graph_list, input_dim, output_dim

    # 未来支持的其他脑成像数据集
    elif dataset_name == 'MDD':
        # MDD数据集处理（待实现）
        raise NotImplementedError(f'MDD数据集支持即将推出')

    elif dataset_name == 'ADHD':
        # ADHD数据集处理（待实现）
        raise NotImplementedError(f'ADHD数据集支持即将推出')

    else:
        raise ValueError(
            f'错误：无效的数据集名称！\n'
            f'支持的分子图数据集: [ENZYMES, DD, NCI1, NCI109, Mutagenicity]\n'
            f'支持的脑成像数据集: [ABIDE, ABIDE_corr, ABIDE_dynamic, ABIDE_phase]\n'
            f'即将支持: [MDD, ADHD]'
        )


def GraphDownstream(data, shots=5, test_fraction=0.4):
    """
    准备下游任务的训练和测试数据（few-shot learning）

    Args:
        data: 图数据列表或数据集
        shots: 每个类别的样本数
        test_fraction: 测试集比例

    Returns:
        train_data: 训练数据
        test_data: 测试数据
    """
    # 检查数据类型
    if isinstance(data, list):
        # 处理列表形式的数据（如ABIDE）
        return GraphDownstreamFromList(data, shots, test_fraction)
    else:
        # 处理TUDataset形式的数据（原有数据集）
        return GraphDownstreamFromDataset(data, shots, test_fraction)


def GraphDownstreamFromList(graph_list, shots=5, test_fraction=0.4):
    """
    从图列表准备下游任务数据

    Args:
        graph_list: 图数据列表
        shots: 每个类别的样本数
        test_fraction: 测试集比例

    Returns:
        train_data: 训练数据
        test_data: 测试数据
    """
    # 获取所有标签
    labels = torch.cat([g.y for g in graph_list])
    num_classes = labels.max().item() + 1

    # 按类别分组
    train_indices = []
    for c in range(num_classes):
        # 找到属于类别c的所有索引
        class_indices = torch.where(labels == c)[0].tolist()

        if len(class_indices) < shots:
            # 如果样本数少于shots，全部用于训练
            train_indices.extend(class_indices)
        else:
            # 随机选择shots个样本
            selected = random.sample(class_indices, k=shots)
            train_indices.extend(selected)

    # 创建测试集索引（排除训练集）
    all_indices = list(range(len(graph_list)))
    test_indices = [i for i in all_indices if i not in train_indices]

    # 如果需要，减少测试集大小
    n_test = int(len(graph_list) * test_fraction)
    if len(test_indices) > n_test:
        test_indices = random.sample(test_indices, k=n_test)

    # 创建训练和测试数据
    train_data = [graph_list[i] for i in train_indices]
    test_data = [graph_list[i] for i in test_indices]

    print(f"数据集划分完成：训练集 {len(train_data)} 个图，测试集 {len(test_data)} 个图")

    return train_data, test_data


def GraphDownstreamFromDataset(data, shots=5, test_fraction=0.4):
    """
    从TUDataset准备下游任务数据（原有函数）
    """
    num_classes = data.y.max().item() + 1
    graph_list = []

    for c in range(num_classes):
        indices = torch.where(data.y.squeeze() == c)[0].tolist()
        if len(indices) < shots:
            graph_list.extend(indices)
        else:
            graph_list.extend(random.sample(indices, k=shots))

    random_graph_list = random.sample(range(len(data)), k=len(data))
    for graph in graph_list:
        random_graph_list.remove(graph)

    train_data = [data[g] for g in graph_list]
    test_data = [data[g] for g in random_graph_list[:int(test_fraction * len(data))]]

    return train_data, test_data


def get_dataset_info(dataset_name):
    """
    获取数据集的基本信息

    Args:
        dataset_name: 数据集名称

    Returns:
        info: 数据集信息字典
    """
    info = {}

    if dataset_name.startswith('ABIDE'):
        info['type'] = '脑成像数据'
        info['task'] = '自闭症谱系障碍分类'
        info['classes'] = 2
        info['description'] = 'ABIDE I 数据集，包含控制组和ASD患者的静息态fMRI数据'

        if 'dynamic' in dataset_name:
            info['graph_method'] = '动态连接性'
        elif 'phase' in dataset_name:
            info['graph_method'] = '相位同步'
        else:
            info['graph_method'] = '相关矩阵'

    elif dataset_name in ['ENZYMES', 'DD', 'NCI1', 'NCI109', 'Mutagenicity']:
        info['type'] = '分子图数据'
        info['task'] = '图分类'

        if dataset_name == 'ENZYMES':
            info['classes'] = 6
            info['description'] = '酶分子分类数据集'
        elif dataset_name == 'DD':
            info['classes'] = 2
            info['description'] = '蛋白质结构数据集'
        elif dataset_name in ['NCI1', 'NCI109']:
            info['classes'] = 2
            info['description'] = '化合物抗癌活性数据集'
        elif dataset_name == 'Mutagenicity':
            info['classes'] = 2
            info['description'] = '化合物致突变性数据集'

    return info