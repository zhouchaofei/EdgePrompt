"""
脑成像数据集实验运行脚本
支持ABIDE、MDD、ADHD200数据集的单个和批量实验
"""
import os
import subprocess
import json
import argparse
from datetime import datetime
import numpy as np

# 实验配置模板
EXPERIMENT_TEMPLATES = {
    # MDD数据集实验配置
    'mdd_experiments': {
        'mdd_baseline': {
            'dataset_name': 'MDD',
            'prompt_type': 'EdgePrompt',
            'shots': 30,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
        'mdd_edgepromptplus': {
            'dataset_name': 'MDD',
            'prompt_type': 'EdgePromptplus',
            'num_prompts': 5,
            'shots': 30,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
        'mdd_serial': {
            'dataset_name': 'MDD',
            'prompt_type': 'SerialNodeEdgePrompt',
            'num_prompts': 5,
            'node_num_prompts': 5,
            'shots': 30,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
        'mdd_parallel': {
            'dataset_name': 'MDD',
            'prompt_type': 'ParallelNodeEdgePrompt',
            'num_prompts': 5,
            'node_num_prompts': 5,
            'fusion_method': 'weighted',
            'shots': 30,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
        'mdd_interactive': {
            'dataset_name': 'MDD',
            'prompt_type': 'InteractiveNodeEdgePrompt',
            'num_prompts': 5,
            'node_num_prompts': 5,
            'shots': 30,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
        'mdd_dynamic': {
            'dataset_name': 'MDD_dynamic',
            'prompt_type': 'SerialNodeEdgePrompt',
            'num_prompts': 5,
            'node_num_prompts': 5,
            'shots': 30,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
    },

    # ADHD数据集实验配置
    'adhd_experiments': {
        'adhd_baseline': {
            'dataset_name': 'ADHD',
            'prompt_type': 'EdgePrompt',
            'shots': 40,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
        'adhd_edgepromptplus': {
            'dataset_name': 'ADHD',
            'prompt_type': 'EdgePromptplus',
            'num_prompts': 5,
            'shots': 40,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
        'adhd_serial': {
            'dataset_name': 'ADHD',
            'prompt_type': 'SerialNodeEdgePrompt',
            'num_prompts': 5,
            'node_num_prompts': 5,
            'shots': 40,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
        'adhd_parallel': {
            'dataset_name': 'ADHD',
            'prompt_type': 'ParallelNodeEdgePrompt',
            'num_prompts': 5,
            'node_num_prompts': 5,
            'fusion_method': 'weighted',
            'shots': 40,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
        'adhd_interactive': {
            'dataset_name': 'ADHD',
            'prompt_type': 'InteractiveNodeEdgePrompt',
            'num_prompts': 5,
            'node_num_prompts': 5,
            'shots': 40,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
        'adhd_dynamic': {
            'dataset_name': 'ADHD_dynamic',
            'prompt_type': 'SerialNodeEdgePrompt',
            'num_prompts': 5,
            'node_num_prompts': 5,
            'shots': 40,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
    },

    # ABIDE数据集实验配置（更新版）
    'abide_experiments': {
        'abide_baseline': {
            'dataset_name': 'ABIDE',
            'prompt_type': 'EdgePrompt',
            'shots': 50,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
        'abide_serial': {
            'dataset_name': 'ABIDE',
            'prompt_type': 'SerialNodeEdgePrompt',
            'num_prompts': 5,
            'node_num_prompts': 5,
            'shots': 50,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
        'abide_parallel': {
            'dataset_name': 'ABIDE',
            'prompt_type': 'ParallelNodeEdgePrompt',
            'num_prompts': 5,
            'node_num_prompts': 5,
            'fusion_method': 'weighted',
            'shots': 50,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
        'abide_interactive': {
            'dataset_name': 'ABIDE',
            'prompt_type': 'InteractiveNodeEdgePrompt',
            'num_prompts': 5,
            'node_num_prompts': 5,
            'shots': 50,
            'epochs': 200,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
    },

    # 对比实验：所有数据集最佳方法
    'comparison_experiments': {
        'abide_best': {
            'dataset_name': 'ABIDE',
            'prompt_type': 'SerialNodeEdgePrompt',
            'num_prompts': 5,
            'node_num_prompts': 5,
            'shots': 50,
            'epochs': 300,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
        'mdd_best': {
            'dataset_name': 'MDD',
            'prompt_type': 'SerialNodeEdgePrompt',
            'num_prompts': 5,
            'node_num_prompts': 5,
            'shots': 30,
            'epochs': 300,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
        'adhd_best': {
            'dataset_name': 'ADHD',
            'prompt_type': 'SerialNodeEdgePrompt',
            'num_prompts': 5,
            'node_num_prompts': 5,
            'shots': 40,
            'epochs': 300,
            'batch_size': 32,
            'hidden_dim': 128,
            'num_layer': 5,
            'gpu_id': 0
        },
    }
}


def run_single_experiment(config, seeds=5, verbose=True):
    """
    运行单个实验配置

    Args:
        config: 实验配置字典
        seeds: 随机种子数量
        verbose: 是否显示详细输出

    Returns:
        results: 实验结果
    """
    all_acc = []
    all_f1 = []

    print(f"\n{'=' * 60}")
    print(f"运行实验配置:")
    print(json.dumps(config, indent=2))
    print(f"{'=' * 60}\n")

    for seed in range(seeds):
        print(f"\nSeed {seed}:")

        # 构建命令
        cmd = ['python', 'downstream_task.py']
        for key, value in config.items():
            cmd.append(f'--{key}')
            cmd.append(str(value))

        # 运行命令
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # 解析结果
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'best_accuracy' in line:
                        # 提取准确率
                        acc = float(line.split('best_accuracy:')[1].strip().split()[0])
                        all_acc.append(acc)
                        print(f"  准确率: {acc:.4f}")
                    if 'best_f1' in line or 'F1' in line:
                        # 提取F1分数
                        try:
                            f1 = float(line.split(':')[-1].strip().split()[0])
                            all_f1.append(f1)
                            print(f"  F1分数: {f1:.4f}")
                        except:
                            pass

                if verbose:
                    print(result.stdout[-500:])  # 显示最后500个字符
            else:
                print(f"  实验失败: {result.stderr[:200]}")

        except Exception as e:
            print(f"  运行出错: {e}")

    # 计算统计结果
    results = {
        'config': config,
        'accuracy': {
            'mean': np.mean(all_acc) if all_acc else 0,
            'std': np.std(all_acc) if all_acc else 0,
            'values': all_acc
        },
        'f1': {
            'mean': np.mean(all_f1) if all_f1 else 0,
            'std': np.std(all_f1) if all_f1 else 0,
            'values': all_f1
        }
    }

    print(f"\n最终结果:")
    print(f"准确率: {results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}")
    if all_f1:
        print(f"F1分数: {results['f1']['mean']:.4f} ± {results['f1']['std']:.4f}")

    return results


def run_batch_experiments(experiment_group, seeds=5):
    """
    批量运行实验组

    Args:
        experiment_group: 实验组名称
        seeds: 随机种子数量

    Returns:
        all_results: 所有实验结果
    """
    if experiment_group not in EXPERIMENT_TEMPLATES:
        print(f"错误：未找到实验组 {experiment_group}")
        print(f"可用的实验组: {list(EXPERIMENT_TEMPLATES.keys())}")
        return None

    experiments = EXPERIMENT_TEMPLATES[experiment_group]
    all_results = {}

    print(f"\n开始运行实验组: {experiment_group}")
    print(f"共 {len(experiments)} 个实验配置\n")

    for exp_name, exp_config in experiments.items():
        print(f"\n运行实验: {exp_name}")
        results = run_single_experiment(exp_config, seeds=seeds, verbose=False)
        all_results[exp_name] = results

        # 保存中间结果
        save_results(all_results, f'{experiment_group}_results.json')

    # 打印汇总
    print_summary(all_results)

    return all_results


def save_results(results, filename):
    """保存结果到JSON文件"""
    os.makedirs('experiment_results', exist_ok=True)
    filepath = os.path.join('experiment_results', filename)

    # 转换numpy类型为Python类型
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    results_converted = convert_numpy(results)

    with open(filepath, 'w') as f:
        json.dump(results_converted, f, indent=2)

    print(f"结果已保存到: {filepath}")


def print_summary(results):
    """打印结果汇总"""
    print("\n" + "=" * 80)
    print("实验结果汇总")
    print("=" * 80)

    # 按数据集分组
    datasets = {}
    for exp_name, result in results.items():
        dataset = result['config']['dataset_name']
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append((exp_name, result))

    # 打印每个数据集的结果
    for dataset, experiments in datasets.items():
        print(f"\n{dataset} 数据集结果:")
        print("-" * 40)

        # 按准确率排序
        experiments.sort(key=lambda x: x[1]['accuracy']['mean'], reverse=True)

        for exp_name, result in experiments:
            prompt_type = result['config'].get('prompt_type', 'Unknown')
            acc = result['accuracy']
            f1 = result['f1']

            print(f"  {exp_name:25s} ({prompt_type:25s}): ")
            print(f"    准确率: {acc['mean']:6.4f} ± {acc['std']:6.4f}")
            if f1['mean'] > 0:
                print(f"    F1分数: {f1['mean']:6.4f} ± {f1['std']:6.4f}")

        # 找出最佳方法
        best_exp = experiments[0]
        print(f"\n  最佳方法: {best_exp[0]} (准确率: {best_exp[1]['accuracy']['mean']:.4f})")


def main():
    parser = argparse.ArgumentParser(description='脑成像数据集实验脚本')

    # 实验模式
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'batch', 'all'],
                        help='运行模式: single(单个实验), batch(批量实验), all(所有实验)')

    # 单个实验参数
    parser.add_argument('--dataset', type=str, default='MDD',
                        choices=['ABIDE', 'MDD', 'ADHD', 'MDD_dynamic', 'ADHD_dynamic'],
                        help='数据集名称')
    parser.add_argument('--prompt_type', type=str, default='SerialNodeEdgePrompt',
                        choices=['EdgePrompt', 'EdgePromptplus', 'SerialNodeEdgePrompt',
                                 'ParallelNodeEdgePrompt', 'InteractiveNodeEdgePrompt'],
                        help='提示方法')
    parser.add_argument('--shots', type=int, default=30,
                        help='每类样本数')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='隐藏层维度')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='GNN层数')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU设备ID')

    # 批量实验参数
    parser.add_argument('--exp_group', type=str, default='mdd_experiments',
                        choices=list(EXPERIMENT_TEMPLATES.keys()),
                        help='实验组名称')

    # 其他参数
    parser.add_argument('--seeds', type=int, default=5,
                        help='随机种子数量')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用的实验配置')

    args = parser.parse_args()

    if args.list:
        print("\n可用的实验组:")
        for group_name, experiments in EXPERIMENT_TEMPLATES.items():
            print(f"\n{group_name}:")
            for exp_name in experiments:
                print(f"  - {exp_name}")

    elif args.mode == 'single':
        # 运行单个实验
        config = {
            'dataset_name': args.dataset,
            'prompt_type': args.prompt_type,
            'shots': args.shots,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'hidden_dim': args.hidden_dim,
            'num_layer': args.num_layer,
            'gpu_id': args.gpu_id
        }

        # 添加额外参数
        if args.prompt_type in ['EdgePromptplus', 'SerialNodeEdgePrompt',
                                'ParallelNodeEdgePrompt', 'InteractiveNodeEdgePrompt']:
            config['num_prompts'] = 5
            config['node_num_prompts'] = 5

        if args.prompt_type == 'ParallelNodeEdgePrompt':
            config['fusion_method'] = 'weighted'

        results = run_single_experiment(config, seeds=args.seeds)
        save_results({f'{args.dataset}_{args.prompt_type}': results},
                     f'{args.dataset}_{args.prompt_type}_results.json')

    elif args.mode == 'batch':
        # 运行批量实验
        results = run_batch_experiments(args.exp_group, seeds=args.seeds)

    elif args.mode == 'all':
        # 运行所有实验
        all_results = {}
        for group_name in EXPERIMENT_TEMPLATES:
            print(f"\n\n{'#' * 80}")
            print(f"运行实验组: {group_name}")
            print(f"{'#' * 80}")

            group_results = run_batch_experiments(group_name, seeds=args.seeds)
            all_results.update(group_results)

        # 保存所有结果
        save_results(all_results, 'all_experiments_results.json')
        print_summary(all_results)


if __name__ == '__main__':
    main()