"""
结果分析和可视化工具
用于分析增强实验的结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_results(csv_file):
    """加载实验结果CSV文件"""
    df = pd.read_csv(csv_file)
    return df


def analyze_by_method(df):
    """按构图方法分析结果"""
    # 提取方法名称（去掉参数和阈值信息）
    df['base_method'] = df['config_name'].apply(lambda x: x.split('_')[0])

    # 对每个分类器和方法计算平均性能
    classifiers = ['LogisticRegression', 'SVM', 'RandomForest']
    methods = df['base_method'].unique()

    results = []
    for method in methods:
        method_df = df[df['base_method'] == method]
        for clf in classifiers:
            bal_acc_col = f'{clf}_balanced_accuracy_mean'
            auc_col = f'{clf}_auc_mean'

            if bal_acc_col in method_df.columns:
                results.append({
                    'Method': method,
                    'Classifier': clf,
                    'Balanced_Acc': method_df[bal_acc_col].mean(),
                    'Balanced_Acc_Std': method_df[bal_acc_col].std(),
                    'AUC': method_df[auc_col].mean(),
                    'AUC_Std': method_df[auc_col].std(),
                    'N_Configs': len(method_df)
                })

    return pd.DataFrame(results)


def analyze_by_threshold(df):
    """按阈值策略分析结果"""

    # 提取阈值策略
    def extract_threshold(config_name):
        if 'none' in config_name:
            return 'none'
        elif 'top_k_8' in config_name:
            return 'top_k_8'
        elif 'top_k_12' in config_name:
            return 'top_k_12'
        else:
            return 'none'

    df['threshold'] = df['config_name'].apply(extract_threshold)

    classifiers = ['LogisticRegression', 'SVM', 'RandomForest']
    thresholds = df['threshold'].unique()

    results = []
    for threshold in thresholds:
        threshold_df = df[df['threshold'] == threshold]
        for clf in classifiers:
            bal_acc_col = f'{clf}_balanced_accuracy_mean'

            if bal_acc_col in threshold_df.columns:
                results.append({
                    'Threshold': threshold,
                    'Classifier': clf,
                    'Balanced_Acc': threshold_df[bal_acc_col].mean(),
                    'Balanced_Acc_Std': threshold_df[bal_acc_col].std(),
                    'Mean_Sparsity': threshold_df['sparsity'].mean()
                })

    return pd.DataFrame(results)


def find_best_configs(df, top_n=10):
    """找出最佳配置"""
    classifiers = ['LogisticRegression', 'SVM', 'RandomForest']

    best_configs = []

    for clf in classifiers:
        bal_acc_col = f'{clf}_balanced_accuracy_mean'
        auc_col = f'{clf}_auc_mean'
        f1_col = f'{clf}_f1_mean'

        if bal_acc_col in df.columns:
            # 按balanced accuracy排序
            sorted_df = df.sort_values(bal_acc_col, ascending=False).head(top_n)

            for _, row in sorted_df.iterrows():
                best_configs.append({
                    'Classifier': clf,
                    'Config': row['config_name'],
                    'Method': row['method'],
                    'Threshold': row['threshold'],
                    'Balanced_Acc': row[bal_acc_col],
                    'AUC': row[auc_col],
                    'F1': row[f1_col],
                    'Sparsity': row['sparsity']
                })

    return pd.DataFrame(best_configs)


def plot_method_comparison(method_df, save_path=None):
    """绘制不同方法的性能比较"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Balanced Accuracy比较
    pivot_bal = method_df.pivot(index='Method', columns='Classifier', values='Balanced_Acc')
    pivot_bal.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Balanced Accuracy by Method and Classifier')
    axes[0].set_xlabel('Method')
    axes[0].set_ylabel('Balanced Accuracy')
    axes[0].legend(title='Classifier')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.4, 0.8])

    # AUC比较
    pivot_auc = method_df.pivot(index='Method', columns='Classifier', values='AUC')
    pivot_auc.plot(kind='bar', ax=axes[1])
    axes[1].set_title('AUC by Method and Classifier')
    axes[1].set_xlabel('Method')
    axes[1].set_ylabel('AUC')
    axes[1].legend(title='Classifier')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.4, 0.8])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_threshold_impact(threshold_df, save_path=None):
    """绘制阈值策略的影响"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Balanced Accuracy vs Threshold
    pivot_bal = threshold_df.pivot(index='Threshold', columns='Classifier', values='Balanced_Acc')
    pivot_bal.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Impact of Threshold Strategy on Balanced Accuracy')
    axes[0].set_xlabel('Threshold Strategy')
    axes[0].set_ylabel('Balanced Accuracy')
    axes[0].legend(title='Classifier')
    axes[0].grid(True, alpha=0.3)

    # Sparsity vs Threshold
    sparsity_by_threshold = threshold_df.groupby('Threshold')['Mean_Sparsity'].mean()
    sparsity_by_threshold.plot(kind='bar', ax=axes[1], color='steelblue')
    axes[1].set_title('Network Sparsity by Threshold Strategy')
    axes[1].set_xlabel('Threshold Strategy')
    axes[1].set_ylabel('Mean Sparsity')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_top_configs_heatmap(best_configs_df, save_path=None):
    """绘制最佳配置的热力图"""
    # 创建配置-指标矩阵
    pivot = best_configs_df.pivot_table(
        index='Config',
        columns='Classifier',
        values='Balanced_Acc',
        aggfunc='mean'
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='coolwarm', center=0.6,
                cbar_kws={'label': 'Balanced Accuracy'})
    plt.title('Top Configurations Performance Heatmap')
    plt.xlabel('Classifier')
    plt.ylabel('Configuration')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def generate_summary_report(df, output_file=None):
    """生成详细的文本报告"""
    lines = []
    lines.append("=" * 80)
    lines.append("ENHANCED FC EXPERIMENT - SUMMARY REPORT")
    lines.append("=" * 80)
    lines.append("")

    # 1. 总体统计
    lines.append("1. OVERALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total configurations tested: {len(df)}")
    lines.append(f"Mean sparsity: {df['sparsity'].mean():.4f} ± {df['sparsity'].std():.4f}")
    lines.append("")

    # 2. 按方法的最佳性能
    lines.append("2. BEST PERFORMANCE BY METHOD")
    lines.append("-" * 40)

    df['base_method'] = df['config_name'].apply(lambda x: x.split('_')[0])
    methods = df['base_method'].unique()

    for method in sorted(methods):
        method_df = df[df['base_method'] == method]
        lines.append(f"\n{method.upper()}:")

        for clf in ['LogisticRegression', 'SVM', 'RandomForest']:
            bal_acc_col = f'{clf}_balanced_accuracy_mean'
            if bal_acc_col in method_df.columns:
                best_bal_acc = method_df[bal_acc_col].max()
                best_idx = method_df[bal_acc_col].idxmax()
                best_config = method_df.loc[best_idx, 'config_name']
                lines.append(f"  {clf}: {best_bal_acc:.4f} ({best_config})")

    lines.append("")

    # 3. 全局最佳配置
    lines.append("3. GLOBAL TOP 5 CONFIGURATIONS")
    lines.append("-" * 40)

    for clf in ['LogisticRegression', 'SVM', 'RandomForest']:
        bal_acc_col = f'{clf}_balanced_accuracy_mean'
        auc_col = f'{clf}_auc_mean'

        if bal_acc_col in df.columns:
            lines.append(f"\n{clf}:")
            top5 = df.nlargest(5, bal_acc_col)

            for i, (_, row) in enumerate(top5.iterrows(), 1):
                lines.append(f"  {i}. {row['config_name']}")
                lines.append(f"     Balanced Acc: {row[bal_acc_col]:.4f} ± {row[f'{clf}_balanced_accuracy_std']:.4f}")
                lines.append(f"     AUC: {row[auc_col]:.4f} ± {row[f'{clf}_auc_std']:.4f}")
                lines.append(f"     Sparsity: {row['sparsity']:.4f}")

    lines.append("")

    # 4. 阈值策略影响
    lines.append("4. IMPACT OF THRESHOLD STRATEGIES")
    lines.append("-" * 40)

    def extract_threshold(config_name):
        if 'none' in config_name:
            return 'none'
        elif 'top_k_8' in config_name:
            return 'top_k_8'
        elif 'top_k_12' in config_name:
            return 'top_k_12'
        else:
            return 'none'

    df['threshold'] = df['config_name'].apply(extract_threshold)

    for threshold in ['none', 'top_k_8', 'top_k_12']:
        threshold_df = df[df['threshold'] == threshold]
        if len(threshold_df) > 0:
            lines.append(f"\n{threshold}:")
            lines.append(f"  Mean sparsity: {threshold_df['sparsity'].mean():.4f}")

            for clf in ['LogisticRegression', 'SVM', 'RandomForest']:
                bal_acc_col = f'{clf}_balanced_accuracy_mean'
                if bal_acc_col in threshold_df.columns:
                    mean_bal_acc = threshold_df[bal_acc_col].mean()
                    lines.append(f"  {clf} mean balanced acc: {mean_bal_acc:.4f}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report = "\n".join(lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_file}")

    print(report)

    return report


def main():
    parser = argparse.ArgumentParser(description='Analyze enhanced experiment results')

    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to CSV results file')
    parser.add_argument('--output_dir', type=str, default='./results/analysis',
                        help='Directory to save analysis outputs')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载结果
    print(f"Loading results from: {args.results_file}")
    df = load_results(args.results_file)
    print(f"Loaded {len(df)} configurations")

    # 分析
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    # 1. 按方法分析
    print("\n1. Analysis by Method:")
    method_df = analyze_by_method(df)
    print(method_df.to_string(index=False))
    method_df.to_csv(output_dir / 'analysis_by_method.csv', index=False)

    # 2. 按阈值分析
    print("\n2. Analysis by Threshold:")
    threshold_df = analyze_by_threshold(df)
    print(threshold_df.to_string(index=False))
    threshold_df.to_csv(output_dir / 'analysis_by_threshold.csv', index=False)

    # 3. 最佳配置
    print("\n3. Top 10 Configurations:")
    best_configs_df = find_best_configs(df, top_n=10)
    print(best_configs_df.head(10).to_string(index=False))
    best_configs_df.to_csv(output_dir / 'best_configs.csv', index=False)

    # 4. 生成报告
    print("\n4. Generating summary report...")
    report_file = output_dir / 'summary_report.txt'
    generate_summary_report(df, report_file)

    # 5. 生成图表
    if not args.no_plots:
        print("\n5. Generating plots...")

        # 方法比较图
        plot_method_comparison(method_df, output_dir / 'method_comparison.png')

        # 阈值影响图
        plot_threshold_impact(threshold_df, output_dir / 'threshold_impact.png')

        # 最佳配置热力图
        plot_top_configs_heatmap(best_configs_df.head(20),
                                 output_dir / 'top_configs_heatmap.png')

    print("\n✅ Analysis completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()