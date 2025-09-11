"""
ABIDE数据集测试脚本
用于测试数据下载、处理和训练流程
"""
import os
import sys
import torch
import argparse
from abide_data import ABIDEDataProcessor, load_abide_data
from load_data import load_graph_data, GraphDownstream


def test_download_and_process():
    """测试数据下载和处理"""
    print("=" * 60)
    print("测试1: ABIDE数据下载和处理")
    print("=" * 60)

    # 创建处理器
    processor = ABIDEDataProcessor(
        data_folder='./data',
        pipeline='cpac',
        atlas='ho',
        connectivity_kind='correlation',
        threshold=0.3
    )

    # 测试下载少量数据
    print("\n1. 下载5个被试的数据进行测试...")
    data = processor.download_data(n_subjects=5)
    print(f"   成功下载！")

    # 测试不同的图构建方法
    methods = ['correlation_matrix', 'dynamic_connectivity', 'phase_synchronization']

    for method in methods:
        print(f"\n2. 测试图构建方法: {method}")
        try:
            graph_list = processor.process_and_save(n_subjects=5, graph_method=method)
            print(f"   成功构建 {len(graph_list)} 个脑功能图")

            if graph_list:
                sample_graph = graph_list[0]
                print(f"   示例图信息:")
                print(f"   - 节点数: {sample_graph.x.shape[0]}")
                print(f"   - 节点特征维度: {sample_graph.x.shape[1]}")
                print(f"   - 边数: {sample_graph.edge_index.shape[1]}")
                print(f"   - 标签: {sample_graph.y.item()}")
        except Exception as e:
            print(f"   错误: {e}")

    print("\n测试1完成！\n")


def test_data_loading():
    """测试数据加载接口"""
    print("=" * 60)
    print("测试2: 数据加载接口")
    print("=" * 60)

    # 测试ABIDE数据集加载
    print("\n1. 测试ABIDE数据集加载...")
    try:
        graph_list, input_dim, output_dim = load_graph_data('ABIDE', './data')
        print(f"   成功加载 {len(graph_list)} 个图")
        print(f"   输入维度: {input_dim}")
        print(f"   输出类别数: {output_dim}")
    except Exception as e:
        print(f"   错误: {e}")

    # 测试数据集划分
    print("\n2. 测试数据集划分 (few-shot)...")
    if graph_list:
        shots_list = [5, 10, 20]
        for shots in shots_list:
            train_data, test_data = GraphDownstream(graph_list, shots=shots, test_fraction=0.4)
            print(f"   shots={shots}: 训练集 {len(train_data)} 个, 测试集 {len(test_data)} 个")

    print("\n测试2完成！\n")


def test_training():
    """测试训练流程"""
    print("=" * 60)
    print("测试3: 训练流程")
    print("=" * 60)

    from downstream_task import GraphTask, set_random_seed
    from logger import Logger
    import logging

    # 设置参数
    set_random_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 创建logger
    os.makedirs('test_logs', exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger = Logger('test_logs/test_training.log', formatter)

    # 测试不同的提示方法
    prompt_types = ['EdgePrompt', 'EdgePromptplus', 'SerialNodeEdgePrompt']

    for prompt_type in prompt_types:
        print(f"\n测试提示方法: {prompt_type}")
        try:
            task = GraphTask(
                dataset_name='ABIDE',
                shots=5,
                gnn_type='GIN',
                num_layer=3,  # 减少层数以加快测试
                hidden_dim=64,  # 减少隐藏维度以加快测试
                device=device,
                pretrain_task=None,
                prompt_type=prompt_type,
                num_prompts=3,
                logger=logger,
                node_prompt_type='NodePrompt',
                node_num_prompts=3,
                fusion_method='weighted'
            )

            # 训练几个epoch测试
            best_acc, best_f1 = task.train(
                batch_size=16,
                lr=0.001,
                epochs=30  # 减少训练轮数以加快测试
            )

            print(f"   训练完成！最佳准确率: {best_acc:.4f}, F1分数: {best_f1:.4f}")

        except Exception as e:
            print(f"   错误: {e}")

    print("\n测试3完成！\n")


def test_compatibility():
    """测试与原有数据集的兼容性"""
    print("=" * 60)
    print("测试4: 原有数据集兼容性")
    print("=" * 60)

    # 测试原有的分子图数据集
    molecular_datasets = ['ENZYMES', 'NCI1']

    for dataset in molecular_datasets:
        print(f"\n测试数据集: {dataset}")
        try:
            graph_list, input_dim, output_dim = load_graph_data(dataset, './data')
            print(f"   成功加载！")
            print(f"   图数量: {len(graph_list) if hasattr(graph_list, '__len__') else graph_list.len()}")
            print(f"   输入维度: {input_dim}")
            print(f"   输出类别数: {output_dim}")
        except Exception as e:
            print(f"   错误: {e}")

    print("\n测试4完成！\n")


def main():
    parser = argparse.ArgumentParser(description='ABIDE数据集测试脚本')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'download', 'load', 'train', 'compat'],
                        help='选择测试内容')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ABIDE数据集集成测试")
    print("=" * 60 + "\n")

    if args.test == 'all':
        test_download_and_process()
        test_data_loading()
        test_training()
        test_compatibility()
    elif args.test == 'download':
        test_download_and_process()
    elif args.test == 'load':
        test_data_loading()
    elif args.test == 'train':
        test_training()
    elif args.test == 'compat':
        test_compatibility()

    print("\n所有测试完成！")


if __name__ == '__main__':
    main()