import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from pathlib import Path


class EnhancedExperimentTracker:
    def __init__(self, log_dir='./log', output_dir='./experiments/analysis'):
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_single_log(self, log_file):
        """解析单个日志文件"""
        results = {
            'file_path': str(log_file),
            'file_name': log_file.name,
            'accuracies': [],
            'f1_scores': [],
            'losses': [],
            'best_accuracy': 0,
            'best_f1': 0,
            'final_accuracy': 0,
            'final_f1': 0
        }

        # 从文件名提取配置信息
        # 格式: dataset_shots_pretrain_prompt_seed_timestamp.log
        parts = log_file.stem.split('_')
        if len(parts) >= 4:
            results['dataset'] = parts[0]
            results['shots'] = int(parts[1]) if parts[1].isdigit() else parts[1]
            results['pretrain'] = parts[2]
            results['prompt_type'] = parts[3]

        # 解析文件内容
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

                # 提取准确率
                acc_pattern = r'test_accuracy:\s*([\d.]+)'
                accuracies = re.findall(acc_pattern, content)
                results['accuracies'] = [float(a) for a in accuracies]

                # 提取F1分数
                f1_pattern = r'test_f1:\s*([\d.]+)'
                f1_scores = re.findall(f1_pattern, content)
                results['f1_scores'] = [float(f) for f in f1_scores]

                # 提取损失
                loss_pattern = r'test_loss:\s*([\d.]+)'
                losses = re.findall(loss_pattern, content)
                results['losses'] = [float(l) for l in losses]

                # 计算最佳和最终结果
                if results['accuracies']:
                    results['best_accuracy'] = max(results['accuracies'])
                    results['final_accuracy'] = results['accuracies'][-1]

                if results['f1_scores']:
                    results['best_f1'] = max(results['f1_scores'])
                    results['final_f1'] = results['f1_scores'][-1]

        except Exception as e:
            print(f"Error parsing {log_file}: {e}")

        return results

    def collect_all_logs(self):
        """收集所有日志文件的结果"""
        all_results = []

        # 遍历所有.log文件
        for log_file in self.log_dir.glob('**/*.log'):
            result = self.parse_single_log(log_file)
            all_results.append(result)

        # 转换为DataFrame
        df = pd.DataFrame(all_results)

        # 保存原始数据
        df.to_csv(self.output_dir / 'all_results.csv', index=False)

        print(f"收集了 {len(df)} 个实验结果")

        return df

    def create_visualizations(self, df):
        """创建所有可视化图表"""

        # 设置绘图风格
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (20, 15)

        fig = plt.figure(figsize=(24, 20))

        # 1. Prompt方法性能比较（箱线图）
        ax1 = plt.subplot(3, 3, 1)
        if 'prompt_type' in df.columns and 'best_accuracy' in df.columns:
            prompt_data = df[df['prompt_type'].notna()]
            if not prompt_data.empty:
                sns.boxplot(data=prompt_data, y='prompt_type', x='best_accuracy', ax=ax1)
                ax1.set_title('Prompt方法性能分布', fontsize=12, fontweight='bold')
                ax1.set_xlabel('最佳准确率')
                ax1.set_ylabel('Prompt类型')

        # 2. 预训练效果热力图
        ax2 = plt.subplot(3, 3, 2)
        if all(col in df.columns for col in ['prompt_type', 'pretrain', 'best_accuracy']):
            pivot_data = df.pivot_table(
                values='best_accuracy',
                index='prompt_type',
                columns='pretrain',
                aggfunc='mean'
            )
            if not pivot_data.empty:
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2)
                ax2.set_title('预训练策略效果热力图', fontsize=12, fontweight='bold')

        # 3. 数据集性能对比
        ax3 = plt.subplot(3, 3, 3)
        if 'dataset' in df.columns:
            dataset_perf = df.groupby('dataset')['best_accuracy'].agg(['mean', 'std'])
            dataset_perf.plot(kind='bar', y='mean', yerr='std', ax=ax3, legend=False)
            ax3.set_title('各数据集平均性能', fontsize=12, fontweight='bold')
            ax3.set_xlabel('数据集')
            ax3.set_ylabel('平均准确率')
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)

        # 4. Shot Learning曲线
        ax4 = plt.subplot(3, 3, 4)
        if 'shots' in df.columns:
            shot_data = df[df['shots'].apply(lambda x: isinstance(x, (int, float)))]
            if not shot_data.empty:
                shot_perf = shot_data.groupby('shots')['best_accuracy'].agg(['mean', 'std'])
                ax4.errorbar(shot_perf.index, shot_perf['mean'], yerr=shot_perf['std'],
                             marker='o', markersize=8, linewidth=2, capsize=5)
                ax4.set_title('Few-shot Learning性能曲线', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Shots数量')
                ax4.set_ylabel('平均准确率')
                ax4.grid(True, alpha=0.3)

        # 5. Top 15最佳配置
        ax5 = plt.subplot(3, 3, 5)
        top_n = 15
        top_configs = df.nlargest(top_n, 'best_accuracy')
        if not top_configs.empty:
            labels = [f"{row.get('dataset', 'NA')[:4]}-{row.get('prompt_type', 'NA')[:8]}"
                      for _, row in top_configs.iterrows()]
            y_pos = np.arange(len(labels))
            ax5.barh(y_pos, top_configs['best_accuracy'].values, color='steelblue')
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(labels, fontsize=8)
            ax5.set_xlabel('准确率')
            ax5.set_title(f'Top {top_n} 最佳配置', fontsize=12, fontweight='bold')
            ax5.set_xlim([top_configs['best_accuracy'].min() * 0.95,
                          top_configs['best_accuracy'].max() * 1.02])

        # 6. 脑成像 vs 分子图性能
        ax6 = plt.subplot(3, 3, 6)
        brain_datasets = ['ABIDE', 'MDD', 'ADHD']
        molecular_datasets = ['ENZYMES', 'DD', 'NCI1', 'NCI109', 'Mutagenicity']

        brain_data = df[df['dataset'].isin(brain_datasets)]
        molecular_data = df[df['dataset'].isin(molecular_datasets)]

        if not brain_data.empty or not molecular_data.empty:
            comparison_data = []
            if not brain_data.empty:
                comparison_data.append(('脑成像', brain_data['best_accuracy'].values))
            if not molecular_data.empty:
                comparison_data.append(('分子图', molecular_data['best_accuracy'].values))

            if comparison_data:
                ax6.violinplot([d[1] for d in comparison_data],
                               positions=range(len(comparison_data)),
                               showmeans=True, showmedians=True)
                ax6.set_xticks(range(len(comparison_data)))
                ax6.set_xticklabels([d[0] for d in comparison_data])
                ax6.set_ylabel('准确率分布')
                ax6.set_title('脑成像 vs 分子图数据集性能', fontsize=12, fontweight='bold')

        # 7. 训练曲线（随机选择几个实验）
        ax7 = plt.subplot(3, 3, 7)
        sample_experiments = df.sample(min(5, len(df)))
        for _, exp in sample_experiments.iterrows():
            if exp['accuracies'] and len(exp['accuracies']) > 1:
                epochs = range(1, len(exp['accuracies']) + 1)
                label = f"{exp.get('dataset', 'NA')[:4]}-{exp.get('prompt_type', 'NA')[:8]}"
                ax7.plot(epochs, exp['accuracies'], label=label, alpha=0.7)
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('准确率')
        ax7.set_title('训练曲线示例', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=8, loc='best')
        ax7.grid(True, alpha=0.3)

        # 8. F1分数分布
        ax8 = plt.subplot(3, 3, 8)
        if 'best_f1' in df.columns:
            f1_data = df[df['best_f1'] > 0]
            if not f1_data.empty:
                ax8.hist(f1_data['best_f1'], bins=30, edgecolor='black', alpha=0.7)
                ax8.axvline(f1_data['best_f1'].mean(), color='red',
                            linestyle='--', linewidth=2, label=f'平均值: {f1_data["best_f1"].mean():.3f}')
                ax8.set_xlabel('F1分数')
                ax8.set_ylabel('频数')
                ax8.set_title('F1分数分布', fontsize=12, fontweight='bold')
                ax8.legend()

        # 9. 图构建方法比较（仅脑成像数据）
        ax9 = plt.subplot(3, 3, 9)
        # 尝试从文件名或其他字段推断图构建方法
        brain_data = df[df['dataset'].isin(brain_datasets)]
        if not brain_data.empty:
            # 创建一个简单的对比
            methods = ['correlation', 'dynamic', 'phase']
            method_accs = []
            for method in methods:
                method_data = brain_data[brain_data['file_name'].str.contains(method, case=False, na=False)]
                if not method_data.empty:
                    method_accs.append(method_data['best_accuracy'].mean())
                else:
                    method_accs.append(0)

            if any(acc > 0 for acc in method_accs):
                ax9.bar(methods, method_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                ax9.set_xlabel('图构建方法')
                ax9.set_ylabel('平均准确率')
                ax9.set_title('图构建方法比较（脑成像数据）', fontsize=12, fontweight='bold')

        plt.suptitle('实验结果综合分析', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        # 保存图表
        fig_path = self.output_dir / 'experiment_analysis.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"可视化图表已保存到: {fig_path}")

        plt.show()

    def generate_detailed_report(self, df):
        """生成详细的文本报告"""
        report_path = self.output_dir / 'detailed_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("实验结果详细分析报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")

            # 1. 总体统计
            f.write("1. 总体统计\n")
            f.write("-" * 50 + "\n")
            f.write(f"总实验数: {len(df)}\n")
            f.write(f"平均准确率: {df['best_accuracy'].mean():.4f} ± {df['best_accuracy'].std():.4f}\n")
            f.write(f"最高准确率: {df['best_accuracy'].max():.4f}\n")
            f.write(f"最低准确率: {df['best_accuracy'].min():.4f}\n")

            if 'best_f1' in df.columns:
                f.write(f"平均F1分数: {df['best_f1'].mean():.4f} ± {df['best_f1'].std():.4f}\n")
            f.write("\n")

            # 2. 最佳配置
            f.write("2. 最佳配置（Top 10）\n")
            f.write("-" * 50 + "\n")
            top10 = df.nlargest(10, 'best_accuracy')
            for idx, row in top10.iterrows():
                f.write(f"排名 {idx + 1}:\n")
                f.write(f"  数据集: {row.get('dataset', 'NA')}\n")
                f.write(f"  Prompt: {row.get('prompt_type', 'NA')}\n")
                f.write(f"  预训练: {row.get('pretrain', 'NA')}\n")
                f.write(f"  Shots: {row.get('shots', 'NA')}\n")
                f.write(f"  准确率: {row['best_accuracy']:.4f}\n")
                f.write(f"  F1分数: {row.get('best_f1', 0):.4f}\n\n")

            # 3. Prompt方法排名
            f.write("3. Prompt方法性能排名\n")
            f.write("-" * 50 + "\n")
            if 'prompt_type' in df.columns:
                prompt_ranking = df.groupby('prompt_type')['best_accuracy'].agg(['mean', 'std', 'max', 'count'])
                prompt_ranking = prompt_ranking.sort_values('mean', ascending=False)
                f.write(prompt_ranking.to_string())
                f.write("\n\n")

            # 4. 预训练策略分析
            f.write("4. 预训练策略效果\n")
            f.write("-" * 50 + "\n")
            if 'pretrain' in df.columns:
                pretrain_stats = df.groupby('pretrain')['best_accuracy'].agg(['mean', 'std', 'max', 'count'])
                f.write(pretrain_stats.to_string())
                f.write("\n\n")

            # 5. 数据集性能
            f.write("5. 各数据集性能统计\n")
            f.write("-" * 50 + "\n")
            if 'dataset' in df.columns:
                dataset_stats = df.groupby('dataset')['best_accuracy'].agg(['mean', 'std', 'max', 'min', 'count'])
                f.write(dataset_stats.to_string())
                f.write("\n\n")

            # 6. 推荐配置
            f.write("6. 推荐配置\n")
            f.write("-" * 50 + "\n")

            # 为每个数据集推荐最佳配置
            if 'dataset' in df.columns:
                for dataset in df['dataset'].unique():
                    if pd.notna(dataset):
                        dataset_df = df[df['dataset'] == dataset]
                        if not dataset_df.empty:
                            best = dataset_df.loc[dataset_df['best_accuracy'].idxmax()]
                            f.write(f"\n{dataset}数据集推荐配置:\n")
                            f.write(f"  Prompt方法: {best.get('prompt_type', 'NA')}\n")
                            f.write(f"  预训练: {best.get('pretrain', 'NA')}\n")
                            f.write(f"  准确率: {best['best_accuracy']:.4f}\n")

        print(f"详细报告已保存到: {report_path}")

        return report_path

    def run_analysis(self):
        """运行完整的分析流程"""
        print("=" * 50)
        print("开始分析实验结果")
        print("=" * 50)

        # 1. 收集数据
        print("\n1. 收集日志文件...")
        df = self.collect_all_logs()

        if df.empty:
            print("警告: 没有找到任何日志文件！")
            return

        # 2. 创建可视化
        print("\n2. 生成可视化图表...")
        self.create_visualizations(df)

        # 3. 生成报告
        print("\n3. 生成详细报告...")
        report_path = self.generate_detailed_report(df)

        print("\n" + "=" * 50)
        print("分析完成！")
        print(f"结果保存在: {self.output_dir}")
        print("=" * 50)


if __name__ == "__main__":
    # 创建追踪器并运行分析
    tracker = EnhancedExperimentTracker()
    tracker.run_analysis()