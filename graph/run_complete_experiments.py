#!/usr/bin/env python
"""
完整的实验执行脚本
包含：预训练、prompt tuning、结果记录和可视化
支持pretrain_source参数控制
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np


class CompleteExperimentPipeline:
    def __init__(self):
        self.start_time = datetime.now()
        self.experiment_log = []
        self.base_dir = './experiments'
        os.makedirs(self.base_dir, exist_ok=True)

    def step1_pretrain_models(self):
        """步骤1: 运行所有预训练"""
        print("=" * 80)
        print("步骤1: 预训练模型")
        print("=" * 80)

        # 1.1 GraphMAE预训练（包括同疾病和跨疾病）
        print("\n1.1 运行GraphMAE预训练...")

        # 跨疾病预训练
        for graph_method in ['correlation_matrix', 'dynamic_connectivity', 'phase_synchronization']:
            cmd = f"""python cross_disease_pretrain.py \
                --data_folder ./data \
                --graph_method {graph_method} \
                --num_layer 5 \
                --hidden_dim 128 \
                --mask_rate 0.5 \
                --batch_size 32 \
                --epochs 200 \
                --lr 0.0005 \
                --gpu_id 0 \
                --drop_ratio 0.1 \
                --save_dir ./pretrained_models/graphmae"""

            print(f"Training GraphMAE (cross-disease) with {graph_method}...")
            os.system(cmd)

        # 1.2 EdgePrediction预训练
        print("\n1.2 运行EdgePrediction预训练...")
        for graph_method in ['correlation_matrix', 'dynamic_connectivity', 'phase_synchronization']:
            cmd = f"""python cross_disease_edge_prediction.py \
                --data_folder ./data \
                --graph_method {graph_method} \
                --num_layer 5 \
                --hidden_dim 128 \
                --batch_size 32 \
                --epochs 200 \
                --lr 0.001 \
                --gpu_id 0 \
                --save_dir ./pretrained_models/edge_prediction"""

            print(f"Training EdgePrediction with {graph_method}...")
            os.system(cmd)

        print("\n预训练完成！")

    def generate_all_experiments(self):
        """生成所有实验组合"""
        experiments = []

        # ========== 脑成像数据集实验 ==========
        # brain_datasets = ['ABIDE', 'MDD']
        # graph_methods = ['correlation_matrix', 'dynamic_connectivity', 'phase_synchronization']

        brain_datasets = ['ABIDE']
        graph_methods = ['correlation_matrix']

        # 20种prompt方法
        prompt_types = [
            'EdgePrompt', 'EdgePromptplus', 'NodePrompt', 'NodePromptplus',
            'SerialNodeEdgePrompt', 'ParallelNodeEdgePrompt', 'InteractiveNodeEdgePrompt',
            'ComplementaryNodeEdgePrompt', 'ContrastiveNodeEdgePrompt', 'SpectralNodeEdgePrompt',
            'HierarchicalGraphTransformerPrompt', 'GraphNeuralODEPrompt', 'MetaLearningPrompt',
            'CausalGraphPrompt', 'GraphWaveletPrompt', 'DiffusionPrompt',
            'RLPrompt', 'AttentionFlowPrompt', 'HypergraphPrompt', 'TopologyPrompt'
        ]

        # 预训练配置：任务和源的组合
        pretrain_configs = [
            {'task': 'None', 'source': None},  # 无预训练
            {'task': 'GraphMAE', 'source': 'same'},  # 同疾病GraphMAE
            # {'task': 'GraphMAE', 'source': 'cross'},  # 跨疾病GraphMAE
            # {'task': 'GraphMAE', 'source': 'auto'},  # 自动选择GraphMAE
            {'task': 'EdgePrediction', 'source': 'same'},  # 同疾病EdgePrediction
            # {'task': 'EdgePrediction', 'source': 'cross'},  # 跨疾病EdgePrediction
            # {'task': 'EdgePrediction', 'source': 'auto'},  # 自动选择EdgePrediction
        ]

        # 生成脑成像实验
        for dataset in brain_datasets:
            for graph_method in graph_methods:
                for prompt_type in prompt_types:
                    for pretrain_config in pretrain_configs:
                        exp = {
                            'dataset_name': dataset,
                            'graph_method': graph_method,
                            'prompt_type': prompt_type,
                            'pretrain_task': pretrain_config['task'],
                            'pretrain_source': pretrain_config['source'] if pretrain_config['source'] else 'auto',
                            'shots': 30,
                            'batch_size': 32,
                            'epochs': 200,
                            'num_layer': 5,
                            'hidden_dim': 128,
                            'gpu_id': 0
                        }

                        # 添加特定prompt的参数
                        exp = self.add_prompt_specific_params(exp)
                        experiments.append(exp)

        # ========== 分子图数据集实验 ==========
        molecular_configs = {
            'ENZYMES': ['None', 'EdgePredGPPT', 'EdgePredGraphPrompt', 'GraphCL', 'SimGRACE'],
            'DD': ['None', 'EdgePredGPPT', 'EdgePredGraphPrompt', 'GraphCL', 'SimGRACE'],
            'NCI1': ['None', 'EdgePredGPPT', 'EdgePredGraphPrompt', 'GraphCL', 'SimGRACE'],
            'NCI109': ['None', 'EdgePredGPPT', 'EdgePredGraphPrompt', 'GraphCL', 'SimGRACE'],
            'Mutagenicity': ['None', 'EdgePredGPPT', 'EdgePredGraphPrompt', 'GraphCL', 'SimGRACE']
        }

        # 分子图只使用前10种prompt方法（减少实验量）
        molecular_prompts = prompt_types[:10]

        for dataset, pretrain_options in molecular_configs.items():
            for prompt_type in prompt_types:
                for pretrain in pretrain_options:
                    exp = {
                        'dataset_name': dataset,
                        'prompt_type': prompt_type,
                        'pretrain_task': pretrain if pretrain != 'None' else '',
                        'shots': 50,
                        'batch_size': 128,
                        'epochs': 200,
                        'num_layer': 5,
                        'hidden_dim': 128,
                        'gpu_id': 0
                    }
                    exp = self.add_prompt_specific_params(exp)
                    # experiments.append(exp)

        return experiments

    def add_prompt_specific_params(self, exp):
        """根据prompt类型添加特定参数"""
        prompt_type = exp['prompt_type']

        # 融合方法的通用参数
        if 'NodeEdge' in prompt_type:
            exp['num_prompts'] = 5
            exp['node_num_prompts'] = 5

        # 特定方法的参数
        if prompt_type == 'ContrastiveNodeEdgePrompt':
            exp['temperature'] = 0.5
            exp['contrast_weight'] = 0.1
        elif prompt_type == 'ComplementaryNodeEdgePrompt':
            exp['recon_weight'] = 0.1
            exp['link_pred_weight'] = 0.1
        elif prompt_type == 'SpectralNodeEdgePrompt':
            exp['num_filters'] = 8
        elif prompt_type == 'GraphNeuralODEPrompt':
            exp['ode_steps'] = 5
        elif prompt_type == 'GraphWaveletPrompt':
            exp['num_scales'] = 4
        elif prompt_type == 'DiffusionPrompt':
            exp['diffusion_steps'] = 5
        elif prompt_type == 'AttentionFlowPrompt':
            exp['flow_steps'] = 3
        elif prompt_type == 'HypergraphPrompt':
            exp['hyperedge_size'] = 4
        elif prompt_type == 'ParallelNodeEdgePrompt':
            exp['fusion_method'] = 'weighted'

        return exp

    def build_command(self, exp):
        """构建实验命令"""
        cmd = "python downstream_task.py"

        for key, value in exp.items():
            # 跳过空值和None
            if value is not None and value != '' and value != 'None':
                cmd += f" --{key} {value}"

        return cmd

    def step2_run_prompt_experiments(self):
        """步骤2: 运行所有Prompt Tuning实验"""
        print("\n" + "=" * 80)
        print("步骤2: Prompt Tuning实验")
        print("=" * 80)

        # 生成所有实验组合
        experiments = self.generate_all_experiments()

        total = len(experiments)
        print(f"\n总共 {total} 个实验需要运行")

        # 可以选择运行部分实验进行测试
        # experiments = experiments[:10]  # 只运行前10个实验进行测试

        for idx, exp in enumerate(experiments, 1):
            print(f"\n[{idx}/{total}] 运行实验:")
            print(f"  数据集: {exp['dataset_name']}")
            print(f"  图方法: {exp.get('graph_method', 'N/A')}")
            print(f"  Prompt: {exp['prompt_type']}")
            print(f"  预训练: {exp.get('pretrain_task', 'None')}")
            print(f"  预训练源: {exp.get('pretrain_source', 'N/A')}")

            # 构建命令
            cmd = self.build_command(exp)
            print(f"  命令: {cmd}")

            # 记录开始时间
            exp_start = time.time()

            # 运行实验
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)  # 1小时超时
                success = result.returncode == 0
            except subprocess.TimeoutExpired:
                print(f"  实验超时！")
                success = False
            except Exception as e:
                print(f"  实验失败: {e}")
                success = False

            # 记录结束时间
            exp_time = time.time() - exp_start

            # 保存实验记录
            self.experiment_log.append({
                'index': idx,
                'config': exp,
                'command': cmd,
                'duration': exp_time,
                'success': success,
                'timestamp': datetime.now().isoformat()
            })

            # 定期保存日志
            if idx % 10 == 0:
                self.save_experiment_log()

            # 对于内存密集型实验，等待GPU清理
            if exp.get('prompt_type') in ['DiffusionPrompt', 'HierarchicalGraphTransformerPrompt',
                                          'GraphWaveletPrompt', 'AttentionFlowPrompt']:
                print("  等待GPU内存释放...")
                time.sleep(10)
            else:
                time.sleep(2)  # 一般实验间隔

    def save_experiment_log(self):
        """保存实验日志"""
        log_file = os.path.join(self.base_dir, 'experiment_log.json')
        with open(log_file, 'w') as f:
            json.dump(self.experiment_log, f, indent=2)

        # 同时保存CSV格式便于分析
        if self.experiment_log:
            df_data = []
            for exp in self.experiment_log:
                row = {
                    'index': exp['index'],
                    'dataset': exp['config'].get('dataset_name', ''),
                    'graph_method': exp['config'].get('graph_method', ''),
                    'prompt_type': exp['config'].get('prompt_type', ''),
                    'pretrain_task': exp['config'].get('pretrain_task', ''),
                    'pretrain_source': exp['config'].get('pretrain_source', ''),
                    'duration': exp['duration'],
                    'success': exp.get('success', False),
                    'timestamp': exp['timestamp']
                }
                df_data.append(row)

            df = pd.DataFrame(df_data)
            df.to_csv(os.path.join(self.base_dir, 'experiment_summary.csv'), index=False)

    def step3_analyze_results(self):
        """步骤3: 分析结果并生成报告"""
        print("\n" + "=" * 80)
        print("步骤3: 结果分析和可视化")
        print("=" * 80)

        # 运行结果分析脚本
        os.system("python experiment_tracker.py")

        # 生成额外的统计报告
        self.generate_summary_report()

    def generate_summary_report(self):
        """生成总结报告"""
        report_path = os.path.join(self.base_dir, 'pipeline_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("完整实验流程总结报告\n")
            f.write(f"生成时间: {datetime.now()}\n")
            f.write(f"总耗时: {datetime.now() - self.start_time}\n")
            f.write("=" * 80 + "\n\n")

            if self.experiment_log:
                f.write(f"总实验数: {len(self.experiment_log)}\n")
                success_count = sum(1 for e in self.experiment_log if e.get('success', False))
                f.write(f"成功实验数: {success_count}\n")
                f.write(f"失败实验数: {len(self.experiment_log) - success_count}\n")

                avg_time = np.mean([e['duration'] for e in self.experiment_log])
                f.write(f"平均实验耗时: {avg_time:.2f}秒\n\n")

                # 按数据集统计
                f.write("按数据集统计:\n")
                dataset_counts = {}
                for exp in self.experiment_log:
                    dataset = exp['config'].get('dataset_name', 'Unknown')
                    dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

                for dataset, count in dataset_counts.items():
                    f.write(f"  {dataset}: {count}个实验\n")

                f.write("\n按Prompt类型统计:\n")
                prompt_counts = {}
                for exp in self.experiment_log:
                    prompt = exp['config'].get('prompt_type', 'Unknown')
                    prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1

                for prompt, count in sorted(prompt_counts.items()):
                    f.write(f"  {prompt}: {count}个实验\n")

        print(f"\n流程报告已保存到: {report_path}")

    def run_pipeline(self, skip_pretrain=False, test_mode=False):
        """运行完整的实验流程

        Args:
            skip_pretrain: 是否跳过预训练步骤
            test_mode: 测试模式，只运行少量实验
        """
        print("=" * 80)
        print("开始完整实验流程")
        print(f"模式: {'测试' if test_mode else '完整'}")
        print("=" * 80)

        if not skip_pretrain:
            self.step1_pretrain_models()
        else:
            print("跳过预训练步骤（使用已有模型）")

        self.step2_run_prompt_experiments()
        self.step3_analyze_results()

        print("\n" + "=" * 80)
        print("所有实验完成！")
        print(f"总耗时: {datetime.now() - self.start_time}")
        print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='完整实验流程管理')
    parser.add_argument('--skip_pretrain', action='store_true',
                        help='跳过预训练步骤')
    parser.add_argument('--test_mode', action='store_true',
                        help='测试模式，只运行少量实验')
    args = parser.parse_args()

    # 创建实验管道
    pipeline = CompleteExperimentPipeline()

    # 运行完整流程
    pipeline.run_pipeline(skip_pretrain=args.skip_pretrain, test_mode=args.test_mode)