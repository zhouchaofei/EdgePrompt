"""
验证2：功能连接图有效性验证 - FC Probe
目的：检查Ledoit-Wolf FC矩阵是否有效
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from fc_construction import FCConstructor


def fc_probe_validation(timeseries_list, labels, method='ledoit_wolf'):
    """
    FC Probe验证

    期望：Acc在62%-65%之间（这是baseline）
    """
    print("\n" + "=" * 80)
    print("验证2：功能连接图有效性验证 - FC Probe")
    print("=" * 80)

    # 1. 构建FC矩阵
    constructor = FCConstructor(method=method)

    X = []
    for ts in timeseries_list:
        fc = constructor.compute_fc_matrix(ts)
        # 取上三角（去除对角线）
        triu_idx = np.triu_indices_from(fc, k=1)
        fc_vector = fc[triu_idx]
        X.append(fc_vector)

    X = np.array(X)
    y = np.array(labels)

    print(f"\nFC特征形状: {X.shape}")
    print(f"标签分布: {np.bincount(y)}")

    # 2. 5-Fold交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # SVM
        svm = SVC(kernel='linear', C=1.0, random_state=42)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        scores.append(accuracy_score(y_test, pred))

    # 3. 结果
    avg_acc = np.mean(scores)

    print(f"\n{'=' * 60}")
    print(f"FC Probe结果 ({method}):")
    print(f"{'=' * 60}")
    print(f"准确率: {avg_acc:.4f} ± {np.std(scores):.4f}")

    print(f"\n判定:")
    if avg_acc < 0.60:
        print("  ❌ FC矩阵质量差 (< 60%)，检查FC构建方法")
    elif avg_acc < 0.62:
        print("  ⚠️  FC矩阵质量一般 (60%-62%)")
    else:
        print("  ✅ FC矩阵有效 (> 62%)，这是合理的baseline")

    print(f"{'=' * 60}\n")

    return avg_acc


if __name__ == '__main__':
    from abide_data_baseline import ABIDEBaselineProcessor

    processor = ABIDEBaselineProcessor()
    timeseries_list, labels, _, _ = processor.download_and_extract(n_subjects=400)

    fc_probe_validation(timeseries_list, labels, method='ledoit_wolf')