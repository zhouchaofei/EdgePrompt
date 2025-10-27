"""
增强版实验：支持多种FC构图方式 + 阈值策略
- Pearson, Ledoit-Wolf, Ridge, Graphical Lasso
- 阈值策略：none, top-k per node
- 传统机器学习分类器：Logistic, SVM, RandomForest
- 5-fold stratified CV × 10 repeats
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')

import argparse
import logging
from datetime import datetime
import pickle
from tqdm import tqdm

# 导入FC构建模块
from fc_construction import compute_fc_matrices_enhanced, generate_fc_configs


def setup_logger(dataset_name, save_dir='./results/enhanced'):
    """设置日志记录器"""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'{dataset_name}_enhanced_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def load_timeseries_data(dataset_name, data_folder='./data'):
    """
    加载时间序列数据（不是FC矩阵）

    Returns:
        timeseries_list: 时间序列列表
        labels: 标签
        meta: 元信息
    """
    if dataset_name == 'ABIDE':
        # 需要重新加载时间序列数据
        from abide_data_baseline import ABIDEBaselineProcessor
        processor = ABIDEBaselineProcessor(data_folder=data_folder)
        timeseries_list, labels, subject_ids, site_ids = processor.download_and_extract(
            n_subjects=None, apply_zscore=True
        )
        meta = {
            'dataset': 'ABIDE',
            'n_subjects': len(labels),
            'n_rois': timeseries_list[0].shape[1] if len(timeseries_list) > 0 else 116
        }

    elif dataset_name == 'MDD':
        from mdd_data_baseline import MDDBaselineProcessor
        processor = MDDBaselineProcessor(data_folder=data_folder)
        timeseries_list, labels, subject_ids, site_ids = processor.load_roi_signals(
            apply_zscore=True
        )
        meta = {
            'dataset': 'MDD',
            'n_subjects': len(labels),
            'n_rois': 116
        }

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return timeseries_list, labels, meta


def flatten_fc_upper_triangle(fc_matrices):
    """展开FC矩阵的上三角"""
    n_subjects = fc_matrices.shape[0]
    n_rois = fc_matrices.shape[1]

    triu_indices = np.triu_indices(n_rois, k=1)
    X = np.array([fc[triu_indices] for fc in fc_matrices])

    return X


def get_classifiers():
    """获取分类器字典"""
    classifiers = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    }

    return classifiers


def evaluate_single_fold(clf, X_train, y_train, X_test, y_test):
    """在单个fold上训练和评估"""
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练
    clf.fit(X_train_scaled, y_train)

    # 预测
    y_pred = clf.predict(X_test_scaled)

    # 概率预测
    if hasattr(clf, 'predict_proba'):
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = y_pred

    # 计算指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
    }

    return metrics


def run_cv_for_config(X, y, clf_name, clf, n_folds=5, n_repeats=10):
    """对单个配置运行交叉验证"""
    all_results = {
        'accuracy': [],
        'balanced_accuracy': [],
        'f1': [],
        'auc': []
    }

    for repeat in range(n_repeats):
        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=42 + repeat
        )

        repeat_results = {
            'accuracy': [],
            'balanced_accuracy': [],
            'f1': [],
            'auc': []
        }

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            metrics = evaluate_single_fold(clf, X_train, y_train, X_test, y_test)

            for metric, value in metrics.items():
                repeat_results[metric].append(value)

        # 记录本次重复的平均结果
        for metric in all_results.keys():
            mean_value = np.mean(repeat_results[metric])
            all_results[metric].append(mean_value)

    # 计算最终统计
    final_results = {}
    for metric in all_results.keys():
        final_results[f'{metric}_mean'] = np.mean(all_results[metric])
        final_results[f'{metric}_std'] = np.std(all_results[metric])

    return final_results


def run_enhanced_experiment(
    dataset_name,
    data_folder='./data',
    n_folds=5,
    n_repeats=10,
    save_dir='./results/enhanced',
    quick_test=False
):
    """
    运行增强版实验

    Args:
        quick_test: 如果为True，只测试部分配置用于调试
    """
    # 设置日志
    logger = setup_logger(dataset_name, save_dir)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"ENHANCED EXPERIMENT: Multiple FC Construction Methods")
    logger.info(f"{'=' * 80}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Cross-validation: {n_folds}-fold, {n_repeats} repeats")
    logger.info(f"{'=' * 80}\n")

    # 1. 加载时间序列数据
    logger.info("1. Loading time series data...")
    timeseries_list, labels, meta = load_timeseries_data(dataset_name, data_folder)

    logger.info(f"   Loaded {len(labels)} subjects")
    logger.info(f"   Number of ROIs: {meta['n_rois']}")

    # 打印标签分布
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"\n   Label distribution:")
    for u, c in zip(unique, counts):
        logger.info(f"     Class {u}: {c} subjects ({c/len(labels)*100:.1f}%)")

    # 2. 生成所有FC配置
    fc_configs = generate_fc_configs()

    if quick_test:
        # 快速测试模式：只测试几个配置
        fc_configs = [
            {'method': 'pearson', 'threshold_strategy': 'none', 'name': 'pearson_none'},
            {'method': 'ledoit_wolf', 'threshold_strategy': 'none', 'name': 'ledoit_wolf_none'},
            {'method': 'graphical_lasso', 'graphical_alpha': 0.05,
             'threshold_strategy': 'top_k', 'threshold_k': 8,
             'name': 'graphical_lasso_alpha0.05_top_k_8'}
        ]
        n_repeats = 2  # 减少重复次数

    logger.info(f"\n2. Testing {len(fc_configs)} FC configurations")

    # 获取分类器
    classifiers = get_classifiers()

    # 存储所有结果
    all_results = []

    # 3. 对每个FC配置进行实验
    for fc_idx, fc_config in enumerate(fc_configs):
        config_name = fc_config.pop('name')

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Configuration {fc_idx + 1}/{len(fc_configs)}: {config_name}")
        logger.info(f"{'=' * 60}")

        # 构建FC矩阵
        logger.info("Building FC matrices...")

        # 提取方法参数
        method = fc_config.pop('method')
        threshold_strategy = fc_config.pop('threshold_strategy')
        threshold_k = fc_config.pop('threshold_k', None)

        # 计算FC矩阵
        fc_matrices, sparsity_rates = compute_fc_matrices_enhanced(
            timeseries_list=timeseries_list,
            method=method,
            threshold_strategy=threshold_strategy,
            threshold_k=threshold_k if threshold_k else 8,
            **fc_config  # 传递其他参数如ridge_lambda, graphical_alpha
        )

        logger.info(f"Mean sparsity: {sparsity_rates.mean():.4f}")

        # Flatten为特征向量
        X = flatten_fc_upper_triangle(fc_matrices)
        logger.info(f"Feature matrix shape: {X.shape}")

        # 对每个分类器运行CV
        config_results = {
            'config_name': config_name,
            'method': method,
            'threshold': threshold_strategy,
            'sparsity': sparsity_rates.mean()
        }

        for clf_name, clf in classifiers.items():
            logger.info(f"\n  Testing {clf_name}...")

            results = run_cv_for_config(
                X, labels, clf_name, clf, n_folds, n_repeats
            )

            # 保存结果
            for metric, value in results.items():
                config_results[f'{clf_name}_{metric}'] = value

            # 打印主要指标
            logger.info(
                f"    Balanced Acc: {results['balanced_accuracy_mean']:.4f} ± "
                f"{results['balanced_accuracy_std']:.4f}"
            )
            logger.info(
                f"    AUC: {results['auc_mean']:.4f} ± {results['auc_std']:.4f}"
            )

        all_results.append(config_results)

    # 4. 保存结果
    logger.info(f"\n{'=' * 60}")
    logger.info("Saving results...")

    # 转换为DataFrame
    df_results = pd.DataFrame(all_results)

    # 保存CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = os.path.join(save_dir, f'{dataset_name}_enhanced_results_{timestamp}.csv')
    df_results.to_csv(csv_file, index=False, float_format='%.4f')
    logger.info(f"Results saved to: {csv_file}")

    # 保存详细结果
    pickle_file = os.path.join(save_dir, f'{dataset_name}_enhanced_detailed_{timestamp}.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump({
            'results': all_results,
            'meta': meta,
            'n_folds': n_folds,
            'n_repeats': n_repeats
        }, f)
    logger.info(f"Detailed results saved to: {pickle_file}")

    # 5. 打印最佳结果摘要
    print_best_results(df_results, logger)

    return df_results


def print_best_results(df_results, logger):
    """打印最佳结果摘要"""
    logger.info(f"\n{'=' * 80}")
    logger.info("BEST RESULTS SUMMARY")
    logger.info(f"{'=' * 80}")

    # 对每个分类器找最佳配置
    classifiers = ['LogisticRegression', 'SVM', 'RandomForest']

    for clf in classifiers:
        logger.info(f"\n{clf}:")

        # 按balanced accuracy排序
        metric_col = f'{clf}_balanced_accuracy_mean'
        if metric_col in df_results.columns:
            best_idx = df_results[metric_col].idxmax()
            best_row = df_results.iloc[best_idx]

            logger.info(f"  Best config: {best_row['config_name']}")
            logger.info(f"  Balanced Acc: {best_row[metric_col]:.4f} ± "
                       f"{best_row[f'{clf}_balanced_accuracy_std']:.4f}")
            logger.info(f"  AUC: {best_row[f'{clf}_auc_mean']:.4f} ± "
                       f"{best_row[f'{clf}_auc_std']:.4f}")
            logger.info(f"  F1: {best_row[f'{clf}_f1_mean']:.4f} ± "
                       f"{best_row[f'{clf}_f1_std']:.4f}")
            logger.info(f"  Sparsity: {best_row['sparsity']:.4f}")

    # 找全局最佳
    logger.info(f"\n{'=' * 60}")
    logger.info("GLOBAL BEST:")

    best_value = 0
    best_config = None
    best_clf = None

    for clf in classifiers:
        metric_col = f'{clf}_balanced_accuracy_mean'
        if metric_col in df_results.columns:
            max_val = df_results[metric_col].max()
            if max_val > best_value:
                best_value = max_val
                best_idx = df_results[metric_col].idxmax()
                best_config = df_results.iloc[best_idx]['config_name']
                best_clf = clf

    if best_config:
        logger.info(f"  Classifier: {best_clf}")
        logger.info(f"  Config: {best_config}")
        logger.info(f"  Balanced Accuracy: {best_value:.4f}")

    logger.info(f"{'=' * 80}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Enhanced experiment with multiple FC construction methods'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['ABIDE', 'MDD'],
        help='Dataset name'
    )

    parser.add_argument(
        '--data_folder',
        type=str,
        default='./data',
        help='Root folder for datasets'
    )

    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of CV folds'
    )

    parser.add_argument(
        '--n_repeats',
        type=int,
        default=10,
        help='Number of CV repeats'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='./results/enhanced',
        help='Directory to save results'
    )

    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Quick test mode (fewer configs and repeats)'
    )

    args = parser.parse_args()

    # 运行实验
    results = run_enhanced_experiment(
        dataset_name=args.dataset,
        data_folder=args.data_folder,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        save_dir=args.save_dir,
        quick_test=args.quick_test
    )

    print("\n✅ Enhanced experiment completed successfully!")


if __name__ == '__main__':
    main()