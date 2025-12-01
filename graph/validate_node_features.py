"""
验证1：节点特征有效性验证 - Linear Probe
目的：检查预训练特征是否学到了有用信息
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from extract_node_features import extract_node_features_pretrained


def linear_probe_validation(encoder_path, timeseries_list, labels,
                            embedding_dim=64, device='cuda'):
    """
    Linear Probe验证特征质量

    期望：Acc > 55% 表示特征有效
    """
    print("\n" + "=" * 80)
    print("验证1：节点特征有效性验证 - Linear Probe")
    print("=" * 80)

    # 1. 提取特征
    features_list = extract_node_features_pretrained(
        timeseries_list,
        encoder_path,
        embedding_dim,
        device
    )

    # 2. 特征拉平 (Flatten)
    X = []
    for features in features_list:
        # features: (N_ROI, embedding_dim)
        X.append(features.flatten())

    X = np.array(X)  # (N_subjects, N_ROI * embedding_dim)
    y = np.array(labels)

    print(f"\n特征形状: {X.shape}")
    print(f"标签分布: {np.bincount(y)}")

    # 3. 5-Fold交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr_scores = []
    svm_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_scores.append(accuracy_score(y_test, lr_pred))

        # Linear SVM
        svm = LinearSVC(max_iter=5000, random_state=42)
        svm.fit(X_train, y_train)
        svm_pred = svm.predict(X_test)
        svm_scores.append(accuracy_score(y_test, svm_pred))

    # 4. 结果
    print(f"\n{'=' * 60}")
    print("Linear Probe结果:")
    print(f"{'=' * 60}")
    print(f"Logistic Regression: {np.mean(lr_scores):.4f} ± {np.std(lr_scores):.4f}")
    print(f"Linear SVM:          {np.mean(svm_scores):.4f} ± {np.std(svm_scores):.4f}")

    avg_acc = np.mean(lr_scores + svm_scores)

    print(f"\n判定:")
    if avg_acc < 0.52:
        print("  ❌ 特征无效 (< 52%)，检查预训练代码和Z-score标准化")
    elif avg_acc < 0.55:
        print("  ⚠️  特征质量一般 (52%-55%)，可能需要调整预训练参数")
    else:
        print("  ✅ 特征有效 (> 55%)，可以进入下一步")

    print(f"{'=' * 60}\n")

    return avg_acc


if __name__ == '__main__':
    from abide_data_baseline import ABIDEBaselineProcessor

    processor = ABIDEBaselineProcessor()
    timeseries_list, labels, _, _ = processor.download_and_extract(n_subjects=200)

    linear_probe_validation(
        encoder_path='./pretrained_models/node_encoder_best.pth',
        timeseries_list=timeseries_list,
        labels=labels,
        embedding_dim=64,
        device='cuda'
    )