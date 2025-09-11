"""
实验运行脚本
用于批量运行不同配置的实验
"""
import os
import subprocess
import json
from datetime import datetime

# 实验配置
EXPERIMENTS = {
    # ABIDE数据集实验
    'abide_baseline': {
        'dataset_name': 'ABIDE',
        'prompt_type': 'EdgePrompt',
        'shots': 50,
        'epochs': 200,
        'batch_size': 64,
        'gpu_id': 2
    },

    'abide_edgepromptplus': {
        'dataset_name': 'ABIDE',
        'prompt_type': 'EdgePromptplus',
        'num_prompts': 5,
        'shots': 50,
        'epochs': 200,
        'batch_size': 64,
        'gpu_id': 2
    },

    'abide_serial': {
        'dataset_name': 'ABIDE',
        'prompt_type': 'SerialNodeEdgePrompt',
        'num_prompts': 5,
        'node_num_prompts': 5,
        'shots': 50,
        'epochs': 200,
        'batch_size': 64,
        'gpu_id': 2
    },

    'abide_parallel': {
        'dataset_name': 'ABIDE',
        'prompt_type': 'ParallelNodeEdgePrompt',
        'num_prompts': 5,
        'node_num_prompts': 5,
        'fusion_method': 'weighted',
        'shots': 50,
        'epochs': 200,
        'batch_size': 64,
        'gpu_id': 2
    },

    'abide_interactive': {
        'dataset_name': 'ABIDE',
        'prompt_type': 'InteractiveNodeEdgePrompt',
        'num_prompts': 5,
        'node_num_prompts': 5,
        'shots': 50,
        'epochs': 200,
        'batch_size': 64,
        'gpu_id': 2
    },

    # 不同图构建方法的实验
    'abide_dynamic': {
        'dataset_name': 'ABIDE_dynamic',
        'prompt_type': 'SerialNodeEdgePrompt',
        'num_prompts': 5,
        'node_num_prompts': 5,
        'shots': 50,
        'epochs': 200,
        'batch_size': 64,
        'gpu_id': 2
    },

    'abide_phase': {
        'dataset_name': 'ABIDE_phase',
        'prompt_type': 'SerialNodeEdgePrompt',
        'num_prompts': 5,
        'node_num_prompts': 5,
        'shots': 50,
        'epochs': 200,
        'batch_size': 64,
        'gpu_id': 2
    },

    # 原有数据集的对比实验
    'enzymes_baseline': {
        'dataset_name': 'ENZYMES',
        'prompt_type': 'EdgePrompt',
        'shots': 50,
        'epochs': 400,
        'batch_size': 64,
        'gpu_id': 2
    },

    'nci1_baseline': {
        'dataset_name': 'NCI1',
        'prompt_type': 'EdgePrompt',
        'shots': 50,
        'epochs': 400,
        'batch_size': 64,
        'gpu_id': 2
    }
}


def run_experiment(name, config):
    """
    运行单个实验

    Args:
        name: 实验名称
        config: 实验配置
    """
    print(f"\n{'=' * 60}")
    print(f"运行实验: {name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置: {json.dumps(config, indent=2)}")
    print(f"{'=' * 60}\n")

    # 构建命令
    cmd = ['python', 'downstream_task.py']
    for key, value in config.items():
        cmd.append(f'--{key}')
        cmd.append(str(value))

    # 运行命令
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        # 保存输出
        output_dir = os.path.join('experiment_outputs', name)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'stdout.txt'), 'w') as f:
            f.write(result.stdout)

        with open(os.path.join(output_dir, 'stderr.txt'), 'w') as f:
            f.write(result.stderr)

        # 检查是否成功
        if result.returncode == 0:
            print(f"✓ 实验 {name} 完成")

            # 提取结果
            lines = result.stdout.split('\n')
            for line in lines:
                if '平均准确率' in line or '平均F1分数' in line:
                    print(f"  {line.strip()}")
        else:
            print(f"✗ 实验 {name} 失败")
            print(f"  错误信息: {result.stderr[:200]}")

    except Exception as e:
        print(f"✗ 运行实验 {name} 时出错: {e}")


def run_all_experiments(experiment_names=None):
    """
    运行所有或指定的实验

    Args:
        experiment_names: 要运行的实验名称列表，None表示运行所有
    """
    if experiment_names is None:
        experiment_names = EXPERIMENTS.keys()

    print(f"\n开始运行 {len(experiment_names)} 个实验")

    for name in experiment_names:
        if name in EXPERIMENTS:
            run_experiment(name, EXPERIMENTS[name])
        else:
            print(f"警告: 未找到实验配置 '{name}'")

    print(f"\n所有实验完成！")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='批量运行实验')
    parser.add_argument('--experiments', nargs='+',
                        help='要运行的实验名称列表，不指定则运行所有')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用的实验')

    args = parser.parse_args()

    if args.list:
        print("\n可用的实验配置:")
        for name, config in EXPERIMENTS.items():
            print(f"\n{name}:")
            print(f"  数据集: {config.get('dataset_name')}")
            print(f"  提示方法: {config.get('prompt_type')}")
    else:
        run_all_experiments(args.experiments)


if __name__ == '__main__':
    main()