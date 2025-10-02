"""
主实验脚本
执行所有实验并生成结果
"""
import argparse
import os
import torch
import numpy as np
import random
from datetime import datetime
import pandas as pd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()

    # 实验设置
    parser.add_argument('--dataset', type=str, default='ABIDE',
                        choices=['ABIDE', 'MDD', 'ADHD'])
    parser.add_argument('--shots', type=int, default=5,
                        help='Few-shot数量')
    parser.add_argument('--graph_method', type=str, default='correlation_matrix',
                        choices=['correlation_matrix', 'dynamic_connectivity', 'phase_synchronization'])

    # 预训练设置
    parser.add_argument('--pretrain', type=str, default='same',
                        choices=['same', 'cross', 'none'],
                        help='same: 同数据集预训练, cross: 跨数据集, none: 不预训练')

    # 模型设置
    parser.add_argument('--method', type=str, default='SF-DPL',
                        choices=['SF-DPL', 'EdgePrompt', 'NodePrompt', 'FineTuning', 'SVM'])

    args = parser.parse_args()

    print(f"实验开始: {datetime.now()}")
    print(f"配置: {args}")

    # 1. 数据准备
    print("\n步骤1: 准备数据...")
    if args.dataset == 'ABIDE':
        from abide_data import ABIDEDataProcessor
        processor = ABIDEDataProcessor()
        processor.process_dual_stream()
    elif args.dataset == 'MDD':
        from mdd_data import MDDDataProcessor
        processor = MDDDataProcessor()
        processor.process_dual_stream()

    # 2. 预训练（如果需要）
    if args.pretrain != 'none':
        print("\n步骤2: 预训练...")

        if args.pretrain == 'same':
            # 同数据集预训练
            os.system(f"""
                python cross_disease_edge_prediction.py \\
                    --data_folder ./data \\
                    --graph_method {args.graph_method} \\
                    --epochs 100
            """)
        else:
            # 跨数据集预训练
            source = 'MDD' if args.dataset == 'ABIDE' else 'ABIDE'
            os.system(f"""
                python cross_disease_pretrain.py \\
                    --source {source} \\
                    --target {args.dataset} \\
                    --graph_method {args.graph_method} \\
                    --epochs 100
            """)

    # 3. 下游任务
    print("\n步骤3: 下游任务...")

    # 运行实验
    if args.method == 'SF-DPL':
        from downstream_task import SF_DPL_Task
        from logger import Logger

        logger = Logger('sf_dpl.log', None)
        task = SF_DPL_Task(args.dataset, args.shots, 'cuda', logger)
        acc = task.run(epochs=100)

        print(f"\nSF-DPL 准确率: {acc:.4f}")

    # 4. 保存结果
    results = {
        'dataset': args.dataset,
        'shots': args.shots,
        'method': args.method,
        'accuracy': acc,
        'timestamp': datetime.now()
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv('experiment_results.csv', mode='a', header=False)

    print(f"\n实验完成: {datetime.now()}")


if __name__ == '__main__':
    main()