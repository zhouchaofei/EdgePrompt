"""
跨疾病Edge Prediction预训练脚本
实现Edge Prediction在ABIDE和MDD数据集上的跨疾病预训练
"""
import os
import torch
import numpy as np
import random
import argparse
from torch_geometric.data import DataLoader

from edge_prediction_pretrain import EdgePredictionModel, EdgePredictionTrainer
from abide_data import load_abide_data
from mdd_data import load_mdd_data


def set_random_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_data(graph_list, train_ratio=0.8):
    """划分训练集和验证集"""
    n = len(graph_list)
    indices = list(range(n))
    random.shuffle(indices)

    train_size = int(n * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data = [graph_list[i] for i in train_indices]
    val_data = [graph_list[i] for i in val_indices]

    return train_data, val_data


def pretrain_edge_prediction(source_dataset, target_dataset, args):
    """
    使用Edge Prediction在源数据集上预训练，用于目标数据集

    Args:
        source_dataset: 源数据集名称 ('ABIDE' 或 'MDD')
        target_dataset: 目标数据集名称 ('ABIDE' 或 'MDD')
        args: 参数
    """
    print(f"\n{'='*60}")
    print(f"Edge Prediction: 在 {source_dataset} 上预训练，用于 {target_dataset}")
    print(f"{'='*60}")

    # 加载源数据集
    print(f"加载 {source_dataset} 数据集...")
    if source_dataset == 'ABIDE':
        graph_list, input_dim, _ = load_abide_data(
            data_folder=args.data_folder,
            graph_method=args.graph_method
        )
    elif source_dataset == 'MDD':
        graph_list, input_dim, _ = load_mdd_data(
            data_folder=args.data_folder,
            graph_method=args.graph_method
        )
    else:
        raise ValueError(f"未知数据集: {source_dataset}")

    if not graph_list or len(graph_list) == 0:
        print(f"错误：{source_dataset} 数据集为空！")
        return None

    print(f"数据集大小: {len(graph_list)}")
    print(f"输入特征维度: {input_dim}")

    # 划分训练集和验证集
    train_data, val_data = split_data(graph_list, train_ratio=0.8)
    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    # 创建Edge Prediction模型
    model = EdgePredictionModel(
        num_layer=args.num_layer,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        drop_ratio=args.drop_ratio
    )

    print(f"\n模型参数:")
    print(f"  GNN层数: {args.num_layer}")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  Dropout比率: {args.drop_ratio}")

    # 设置设备
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建训练器
    trainer = EdgePredictionTrainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 设置保存路径
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(
        args.save_dir,
        f'edge_prediction_{source_dataset}_for_{target_dataset}_{args.graph_method}.pth'
    )

    # 训练
    print(f"\n开始训练...")
    best_auc = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=save_path,
        verbose=True
    )

    print(f"\n训练完成！最佳验证AUC: {best_auc:.4f}")
    print(f"模型保存至: {save_path}")

    return save_path


def main():
    parser = argparse.ArgumentParser(description='Edge Prediction跨疾病预训练')

    parser.add_argument('--source', type=str, required=True,
                       choices=['ABIDE', 'MDD', 'ADHD'],
                       help='源数据集（用于训练）')
    parser.add_argument('--target', type=str, required=True,
                       choices=['ABIDE', 'MDD', 'ADHD'],
                       help='目标数据集（用于下游任务）')

    # 数据相关参数
    parser.add_argument('--data_folder', type=str, default='./data',
                       help='数据文件夹路径')
    parser.add_argument('--graph_method', type=str, default='correlation_matrix',
                       choices=['correlation_matrix', 'dynamic_connectivity', 'phase_synchronization'],
                       help='图构建方法')

    # 模型相关参数
    parser.add_argument('--num_layer', type=int, default=5, help='GNN层数')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--drop_ratio', type=float, default=0.0, help='Dropout比率')

    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')

    # 其他参数
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./pretrained_models/edge_prediction',
                       help='模型保存目录')

    args = parser.parse_args()

    # 设置随机种子
    set_random_seed(args.seed)

    pretrain_edge_prediction(args.source, args.target, args)

    # 执行跨疾病Edge Prediction预训练
    # 1. 在ABIDE上训练，用于MDD
    # pretrain_edge_prediction('ABIDE', 'MDD', args)

    # 2. 在MDD上训练，用于ABIDE
    # pretrain_edge_prediction('MDD', 'ABIDE', args)

    # 3. 在ABIDE上训练，用于ABIDE
    # pretrain_edge_prediction('ABIDE', 'ABIDE', args)

    # 4. 在MDD上训练，用于MDD
    # pretrain_edge_prediction('MDD', 'MDD', args)


if __name__ == '__main__':
    main()