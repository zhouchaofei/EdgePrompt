"""
SF-DPL完整实验脚本
支持同数据集和跨数据集预训练实验
"""
import os
import torch
import numpy as np
import random
import argparse
import logging
from datetime import datetime
from logger import Logger


def set_random_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_experiment(args, seed):
    """运行单次实验"""
    # 设置随机种子
    set_random_seed(seed)

    # 创建日志目录
    log_dir = os.path.join('log', args.dataset, 'sf_dpl')
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件
    log_file = os.path.join(
        log_dir,
        f'{args.dataset}_{args.shots}shot_{args.pretrain_source}_{seed}.log'
    )

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger = Logger(log_file, formatter)

    logger.info("=" * 60)
    logger.info("SF-DPL实验配置")
    logger.info("=" * 60)
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"Few-shot数量: {args.shots}")
    logger.info(f"预训练任务: {args.pretrain_task}")
    logger.info(f"预训练源: {args.pretrain_source}")
    logger.info(f"图构建方法: {args.graph_method}")
    logger.info(f"随机种子: {seed}")
    logger.info(f"设备: cuda:{args.gpu}")
    logger.info("=" * 60)

    # 创建SF-DPL任务
    from downstream_task import SF_DPL_Task

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    task = SF_DPL_Task(
        dataset_name=args.dataset,
        shots=args.shots,
        device=device,
        logger=logger,
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

    # 运行实验
    acc, f1 = task.run(epochs=args.epochs)

    logger.info("=" * 60)
    logger.info(f"实验完成 (Seed {seed})")
    logger.info(f"准确率: {acc:.4f}")
    logger.info(f"F1分数: {f1:.4f}")
    logger.info("=" * 60)

    return acc, f1


def main():
    parser = argparse.ArgumentParser(description='SF-DPL完整实验')

    # 数据集配置
    parser.add_argument('--dataset', type=str, default='ABIDE',
                        choices=['ABIDE', 'MDD'],
                        help='数据集名称')
    parser.add_argument('--shots', type=int, default=5,
                        help='Few-shot样本数')
    parser.add_argument('--graph_method', type=str, default='correlation_matrix',
                        help='图构建方法')

    # 预训练配置
    parser.add_argument('--pretrain_task', type=str, default='GraphMAE',
                        choices=['GraphMAE', 'EdgePrediction', 'None'],
                        help='预训练任务')
    parser.add_argument('--pretrain_source', type=str, default='same',
                        choices=['same', 'cross'],
                        help='预训练源: same(同数据集), cross(跨数据集)')

    # 模型配置
    parser.add_argument('--num_layer', type=int, default=5,
                        help='GNN层数')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='隐藏层维度')
    parser.add_argument('--drop_ratio', type=float, default=0.3,
                        help='Dropout比率')
    parser.add_argument('--num_prompts', type=int, default=5,
                        help='Prompt数量')

    # 训练配置
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')

    # 实验配置
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='运行多少个随机种子')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='起始随机种子')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备ID')

    args = parser.parse_args()

    # 打印实验配置
    print("\\n" + "=" * 80)
    print("SF-DPL实验配置")
    print("=" * 80)
    for key, value in vars(args).items():
        print(f"{key:20s}: {value}")
    print("=" * 80 + "\\n")

    # 运行多个种子的实验
    all_results_acc = []
    all_results_f1 = []

    for seed_idx in range(args.num_seeds):
        seed = args.start_seed + seed_idx

        print(f"\\n{'=' * 80}")
        print(f"运行 Seed {seed} ({seed_idx + 1}/{args.num_seeds})")
        print(f"{'=' * 80}")

        acc, f1 = run_single_experiment(args, seed)
        all_results_acc.append(acc)
        all_results_f1.append(f1)

        print(f"\\nSeed {seed} 结果: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

    # 计算统计结果
    mean_acc = np.mean(all_results_acc)
    std_acc = np.std(all_results_acc)
    mean_f1 = np.mean(all_results_f1)
    std_f1 = np.std(all_results_f1)

    # 打印最终结果
    print(f"\\n{'=' * 80}")
    print(f"最终结果 ({args.dataset} - SF-DPL - {args.shots}shot - {args.pretrain_source})")
    print(f"{'=' * 80}")
    print(f"准确率: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"F1分数: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"{'=' * 80}\\n")

    # 保存结果到文件
    result_dir = os.path.join('results', args.dataset)
    os.makedirs(result_dir, exist_ok=True)

    result_file = os.path.join(result_dir, 'sf_dpl_results.txt')
    with open(result_file, 'a') as f:
        f.write(f"\\n{'-' * 80}\\n")
        f.write(f"Dataset: {args.dataset}\\n")
        f.write(f"Shots: {args.shots}\\n")
        f.write(f"Pretrain: {args.pretrain_task} ({args.pretrain_source})\\n")
        f.write(f"Hidden_dim: {args.hidden_dim}, Num_prompts: {args.num_prompts}\\n")
        f.write(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\\n")
        f.write(f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f}\\n")
        f.write(f"Individual results: {[f'{acc:.4f}' for acc in all_results_acc]}\\n")
        f.write(f"Timestamp: {datetime.now()}\\n")


if __name__ == '__main__':
    main()