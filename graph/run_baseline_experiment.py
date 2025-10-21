"""
Baseline实验：功能连接 + 传统机器学习分类
使用npz格式的FC矩阵数据

特点：
- 读取预处理好的FC矩阵（不需要从时间序列重新计算）
- Flatten上三角为特征向量
- Logistic / SVM / RandomForest分类
- 5-fold stratified CV × 10 repeats
- 评估指标：Accuracy / Balanced Accuracy / AUC / F1
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
    f1_score,
    classification_report,
    confusion_matrix
)
import warnings

warnings.filterwarnings('ignore')

import argparse
import logging
from datetime import datetime


def setup_logger(dataset_name, save_dir='./results/baseline'):
    """设置日志记录器"""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'{dataset_name}_baseline_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def load_fc_data(dataset_name, data_folder='./data'):
    """
    加载FC矩阵数据

    Args:
        dataset_name: 'ABIDE' or 'MDD'
        data_folder: 数据根目录

    Returns:
        fc_matrices: [N, N_ROI, N_ROI]
        labels: [N]
        meta: 元信息字典
    """
    if dataset_name == 'ABIDE':
        from abide_data_baseline import load_abide_baseline
        fc_matrices, labels, subject_ids, site_ids, meta = load_abide_baseline(data_folder)

    elif dataset_name == 'MDD':
        from mdd_data_baseline import load_mdd_baseline
        fc_matrices, labels, subject_ids, site_ids, meta = load_mdd_baseline(data_folder)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return fc_matrices, labels, meta


def flatten_fc_upper_triangle(fc_matrices):
    """
    展开FC矩阵的上三角（不包含对角线）

    Args:
        fc_matrices: [N_subjects, N_ROI, N_ROI]

    Returns:
        X: [N_subjects, N_features]
    """
    n_subjects = fc_matrices.shape[0]
    n_rois = fc_matrices.shape[1]

    # 上三角索引（不包含对角线）
    triu_indices = np.triu_indices(n_rois, k=1)

    # 提取上三角
    X = np.array([fc[triu_indices] for fc in fc_matrices])

    n_features = X.shape[1]
    expected_features = n_rois * (n_rois - 1) // 2

    assert n_features == expected_features, \
        f"Feature count mismatch: {n_features} vs {expected_features}"

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
    """
    在单个fold上训练和评估

    Returns:
        metrics: dict of evaluation metrics
    """
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练
    clf.fit(X_train_scaled, y_train)

    # 预测
    y_pred = clf.predict(X_test_scaled)

    # 概率预测（用于AUC）
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


def run_cross_validation(
        X, y,
        clf_name, clf,
        n_folds=5,
        n_repeats=10,
        logger=None
):
    """
    运行重复的分层交叉验证

    Args:
        X: 特征矩阵 [N, D]
        y: 标签 [N]
        clf_name: 分类器名称
        clf: 分类器对象
        n_folds: fold数量
        n_repeats: 重复次数
        logger: 日志记录器

    Returns:
        final_results: 最终统计结果
        all_results: 所有重复的详细结果
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Classifier: {clf_name}")
    logger.info(f"{'=' * 60}")

    all_results = {
        'accuracy': [],
        'balanced_accuracy': [],
        'f1': [],
        'auc': []
    }

    for repeat in range(n_repeats):
        logger.info(f"\nRepeat {repeat + 1}/{n_repeats}")

        # 创建分层K折交叉验证
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

            # 评估
            metrics = evaluate_single_fold(clf, X_train, y_train, X_test, y_test)

            # 记录结果
            for metric, value in metrics.items():
                repeat_results[metric].append(value)

            # 打印fold结果
            logger.info(
                f"  Fold {fold + 1}: "
                f"Acc={metrics['accuracy']:.4f}, "
                f"Bal_Acc={metrics['balanced_accuracy']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"AUC={metrics['auc']:.4f}"
            )

        # 记录本次重复的平均结果
        for metric in all_results.keys():
            mean_value = np.mean(repeat_results[metric])
            all_results[metric].append(mean_value)
            logger.info(f"  Repeat {repeat + 1} Mean {metric}: {mean_value:.4f}")

    # 计算最终统计
    final_results = {}
    logger.info(f"\n{'-' * 60}")
    logger.info(f"Final Results (Mean ± Std over {n_repeats} repeats):")
    logger.info(f"{'-' * 60}")

    for metric in all_results.keys():
        mean = np.mean(all_results[metric])
        std = np.std(all_results[metric])
        final_results[f'{metric}_mean'] = mean
        final_results[f'{metric}_std'] = std
        logger.info(f"  {metric.upper():20s}: {mean:.4f} ± {std:.4f}")

    return final_results, all_results


def save_results(results, dataset_name, meta, save_dir):
    """保存实验结果"""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存为CSV
    csv_file = os.path.join(save_dir, f'{dataset_name}_results_{timestamp}.csv')

    rows = []
    for clf_name, clf_results in results.items():
        row = {'Classifier': clf_name}
        row.update(clf_results['final'])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False, float_format='%.4f')

    logging.info(f"\n✅ Results saved to: {csv_file}")

    # 保存详细结果
    import pickle
    pickle_file = os.path.join(save_dir, f'{dataset_name}_detailed_{timestamp}.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump({
            'results': results,
            'meta': meta
        }, f)

    logging.info(f"✅ Detailed results saved to: {pickle_file}")

    return csv_file


def print_final_summary(results, logger):
    """打印最终结果摘要"""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"FINAL SUMMARY")
    logger.info(f"{'=' * 80}\n")

    # 创建汇总表
    summary_data = []

    for clf_name, clf_results in results.items():
        final = clf_results['final']
        summary_data.append({
            'Classifier': clf_name,
            'Accuracy': f"{final['accuracy_mean']:.4f} ± {final['accuracy_std']:.4f}",
            'Balanced Acc': f"{final['balanced_accuracy_mean']:.4f} ± {final['balanced_accuracy_std']:.4f}",
            'F1': f"{final['f1_mean']:.4f} ± {final['f1_std']:.4f}",
            'AUC': f"{final['auc_mean']:.4f} ± {final['auc_std']:.4f}"
        })

    df_summary = pd.DataFrame(summary_data)
    logger.info("\n" + df_summary.to_string(index=False))

    # 找出最佳模型
    best_clf = max(
        results.items(),
        key=lambda x: x[1]['final']['balanced_accuracy_mean']
    )

    logger.info(f"\n{'=' * 80}")
    logger.info(f"🏆 Best Classifier: {best_clf[0]}")
    logger.info(
        f"   Balanced Accuracy: "
        f"{best_clf[1]['final']['balanced_accuracy_mean']:.4f} ± "
        f"{best_clf[1]['final']['balanced_accuracy_std']:.4f}"
    )
    logger.info(f"{'=' * 80}\n")


def run_baseline_experiment(
        dataset_name,
        data_folder='./data',
        n_folds=5,
        n_repeats=10,
        save_dir='./results/baseline'
):
    """
    运行完整的baseline实验

    Args:
        dataset_name: 'ABIDE' or 'MDD'
        data_folder: 数据根目录
        n_folds: CV的fold数
        n_repeats: CV重复次数
        save_dir: 结果保存目录
    """
    # 设置日志
    logger = setup_logger(dataset_name, save_dir)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"BASELINE EXPERIMENT: FC + TRADITIONAL ML")
    logger.info(f"{'=' * 80}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Cross-validation: {n_folds}-fold, {n_repeats} repeats")
    logger.info(f"{'=' * 80}\n")

    # 1. 加载FC矩阵数据
    logger.info("1. Loading FC matrix data...")
    fc_matrices, labels, meta = load_fc_data(dataset_name, data_folder)

    logger.info(f"   Loaded {len(labels)} subjects")
    logger.info(f"   Atlas: {meta['atlas']}")
    logger.info(f"   Number of ROIs: {meta['n_rois']}")
    logger.info(f"   FC method: {meta['method']}")

    # 打印标签分布
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"\n   Label distribution:")
    for u, c in zip(unique, counts):
        logger.info(f"     Class {u}: {c} subjects ({c / len(labels) * 100:.1f}%)")

    # 2. Flatten FC矩阵为特征向量
    logger.info(f"\n2. Flattening FC matrices (upper triangle)...")
    X = flatten_fc_upper_triangle(fc_matrices)

    logger.info(f"   Feature matrix shape: {X.shape}")
    logger.info(f"   Expected features: {meta['n_rois'] * (meta['n_rois'] - 1) // 2}")
    logger.info(f"   Feature statistics:")
    logger.info(f"     Mean: {X.mean():.4f}")
    logger.info(f"     Std: {X.std():.4f}")
    logger.info(f"     Min: {X.min():.4f}")
    logger.info(f"     Max: {X.max():.4f}")

    # 3. 获取分类器
    classifiers = get_classifiers()

    # 4. 对每个分类器运行CV
    logger.info(f"\n3. Running cross-validation experiments...")

    all_classifier_results = {}

    for clf_name, clf in classifiers.items():
        final_results, detailed_results = run_cross_validation(
            X, labels,
            clf_name, clf,
            n_folds, n_repeats,
            logger
        )

        all_classifier_results[clf_name] = {
            'final': final_results,
            'detailed': detailed_results
        }

    # 5. 保存结果
    logger.info(f"\n4. Saving results...")
    save_results(all_classifier_results, dataset_name, meta, save_dir)

    # 6. 打印最终总结
    print_final_summary(all_classifier_results, logger)

    return all_classifier_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Baseline experiment: FC + Traditional ML'
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
        default='./results/baseline',
        help='Directory to save results'
    )

    args = parser.parse_args()

    # 运行实验
    results = run_baseline_experiment(
        dataset_name=args.dataset,
        data_folder=args.data_folder,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        save_dir=args.save_dir
    )

    print("\n✅ Baseline experiment completed successfully!")


if __name__ == '__main__':
    main()