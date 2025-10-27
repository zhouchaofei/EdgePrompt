"""
主执行脚本：运行完整的增强版FC构图实验
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import time
from datetime import datetime


def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'numpy',
        'pandas',
        'sklearn',
        'scipy',
        'matplotlib',
        'seaborn',
        'nilearn',
        'tqdm'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print("Error: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def prepare_data(dataset, data_folder='./data'):
    """准备数据"""
    print(f"\n{'=' * 60}")
    print(f"Preparing {dataset} data...")
    print(f"{'=' * 60}")

    if dataset == 'ABIDE':
        from abide_data_baseline import ABIDEBaselineProcessor
        processor = ABIDEBaselineProcessor(
            data_folder=data_folder,
            pipeline='cpac',
            atlas='aal'
        )
        # 检查是否已有数据
        baseline_path = os.path.join(data_folder, 'ABIDE', 'baseline')
        data_file = os.path.join(baseline_path, 'abide_aal_pearson_fc_normalized.npz')

        if os.path.exists(data_file):
            print(f"Data already exists: {data_file}")
            return True

        # 处理数据
        save_path = processor.process_and_save(
            n_subjects=None,  # 使用所有数据
            fc_method='pearson',
            apply_zscore=True
        )
        return save_path is not None

    elif dataset == 'MDD':
        from mdd_data_baseline import MDDBaselineProcessor
        processor = MDDBaselineProcessor(data_folder=data_folder)

        # 检查是否已有数据
        baseline_path = os.path.join(data_folder, 'REST-meta-MDD', 'baseline')
        data_file = os.path.join(baseline_path, 'mdd_aal116_pearson_fc_normalized.npz')

        if os.path.exists(data_file):
            print(f"Data already exists: {data_file}")
            return True

        # 处理数据
        save_path = processor.process_and_save(
            fc_method='pearson',
            apply_zscore=True
        )
        return save_path is not None

    return False


def run_enhanced_experiment(dataset, data_folder='./data', quick_test=False):
    """运行增强实验"""
    print(f"\n{'=' * 60}")
    print(f"Running enhanced experiment for {dataset}...")
    print(f"{'=' * 60}")

    from run_enhanced_experiment import run_enhanced_experiment as run_exp

    # 设置参数
    n_folds = 5
    n_repeats = 2 if quick_test else 10
    save_dir = './results/enhanced'

    # 运行实验
    results = run_exp(
        dataset_name=dataset,
        data_folder=data_folder,
        n_folds=n_folds,
        n_repeats=n_repeats,
        save_dir=save_dir,
        quick_test=quick_test
    )

    return results


def analyze_results(dataset, results_dir='./results/enhanced'):
    """分析实验结果"""
    print(f"\n{'=' * 60}")
    print(f"Analyzing results for {dataset}...")
    print(f"{'=' * 60}")

    # 找最新的结果文件
    pattern = f"{dataset}_enhanced_results_*.csv"
    result_files = list(Path(results_dir).glob(pattern))

    if not result_files:
        print(f"No results found for {dataset}")
        return False

    latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing: {latest_result}")

    from analyze_results import (
        load_results,
        analyze_by_method,
        analyze_by_threshold,
        find_best_configs,
        generate_summary_report
    )

    # 加载和分析
    df = load_results(latest_result)

    # 创建分析目录
    analysis_dir = Path(results_dir) / f'analysis_{dataset}'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 生成分析报告
    report_file = analysis_dir / 'summary_report.txt'
    generate_summary_report(df, report_file)

    # 保存分析结果
    method_df = analyze_by_method(df)
    method_df.to_csv(analysis_dir / 'analysis_by_method.csv', index=False)

    threshold_df = analyze_by_threshold(df)
    threshold_df.to_csv(analysis_dir / 'analysis_by_threshold.csv', index=False)

    best_configs_df = find_best_configs(df, top_n=10)
    best_configs_df.to_csv(analysis_dir / 'best_configs.csv', index=False)

    print(f"\nAnalysis saved to: {analysis_dir}")

    return True


def run_comparison_analysis(results_dir='./results/enhanced'):
    """运行跨数据集比较分析"""
    print(f"\n{'=' * 60}")
    print("Running cross-dataset comparison...")
    print(f"{'=' * 60}")

    import pandas as pd

    # 找两个数据集的结果
    abide_files = list(Path(results_dir).glob("ABIDE_enhanced_results_*.csv"))
    mdd_files = list(Path(results_dir).glob("MDD_enhanced_results_*.csv"))

    if not abide_files or not mdd_files:
        print("Need both ABIDE and MDD results for comparison")
        return

    # 加载最新结果
    abide_df = pd.read_csv(max(abide_files, key=lambda p: p.stat().st_mtime))
    mdd_df = pd.read_csv(max(mdd_files, key=lambda p: p.stat().st_mtime))

    # 比较最佳配置
    print("\n" + "=" * 60)
    print("CROSS-DATASET COMPARISON")
    print("=" * 60)

    classifiers = ['LogisticRegression', 'SVM', 'RandomForest']

    for clf in classifiers:
        print(f"\n{clf}:")

        bal_acc_col = f'{clf}_balanced_accuracy_mean'

        # ABIDE最佳
        if bal_acc_col in abide_df.columns:
            best_idx = abide_df[bal_acc_col].idxmax()
            best_config = abide_df.loc[best_idx, 'config_name']
            best_score = abide_df.loc[best_idx, bal_acc_col]
            print(f"  ABIDE best: {best_score:.4f} ({best_config})")

        # MDD最佳
        if bal_acc_col in mdd_df.columns:
            best_idx = mdd_df[bal_acc_col].idxmax()
            best_config = mdd_df.loc[best_idx, 'config_name']
            best_score = mdd_df.loc[best_idx, bal_acc_col]
            print(f"  MDD best: {best_score:.4f} ({best_config})")


def main():
    parser = argparse.ArgumentParser(
        description='Run complete enhanced FC construction experiments'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['ABIDE', 'MDD', 'both'],
        default='both',
        help='Dataset(s) to process'
    )

    parser.add_argument(
        '--data_folder',
        type=str,
        default='./data',
        help='Root folder for datasets'
    )

    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Quick test mode (fewer configs and repeats)'
    )

    parser.add_argument(
        '--skip_data_prep',
        action='store_true',
        help='Skip data preparation step'
    )

    parser.add_argument(
        '--skip_experiment',
        action='store_true',
        help='Skip experiment and only run analysis'
    )

    args = parser.parse_args()

    # 检查依赖
    if not check_dependencies():
        return 1

    # 确定要处理的数据集
    if args.dataset == 'both':
        datasets = ['ABIDE', 'MDD']
    else:
        datasets = [args.dataset]

    print(f"\n{'=' * 80}")
    print("ENHANCED FC CONSTRUCTION EXPERIMENTS")
    print(f"{'=' * 80}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Mode: {'Quick Test' if args.quick_test else 'Full Experiment'}")
    print(f"{'=' * 80}")

    # 对每个数据集执行流程
    for dataset in datasets:
        print(f"\n{'#' * 80}")
        print(f"Processing {dataset} Dataset")
        print(f"{'#' * 80}")

        # Step 1: 准备数据
        if not args.skip_data_prep:
            success = prepare_data(dataset, args.data_folder)
            if not success:
                print(f"Error: Failed to prepare {dataset} data")
                continue

        # Step 2: 运行实验
        if not args.skip_experiment:
            results = run_enhanced_experiment(
                dataset,
                args.data_folder,
                args.quick_test
            )
            if results is None:
                print(f"Error: Experiment failed for {dataset}")
                continue

        # Step 3: 分析结果
        success = analyze_results(dataset)
        if not success:
            print(f"Warning: Analysis failed for {dataset}")

    # Step 4: 跨数据集比较（如果运行了两个数据集）
    if len(datasets) > 1:
        run_comparison_analysis()

    print(f"\n{'=' * 80}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Results saved in: ./results/enhanced/")
    print(f"{'=' * 80}")

    return 0


if __name__ == '__main__':
    sys.exit(main())