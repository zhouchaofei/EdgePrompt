# check_data_quality.py
import torch
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import os


def check_data():
    # 路径指向你 run_gnn_experiment 用的那个文件
    data_path = './data/REST-meta-MDD/gnn_datasets/MDD_pearson_hybrid.pkl'

    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        return

    print(f"Loading {data_path}...")
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
        # data_list, labels, metadata = pickle.load(f)

    data_list = data_dict['graph_list']
    labels = data_dict['labels']
    metadata = data_dict.get('metadata', {})

    print(f"Loaded {len(data_list)} graphs.")

    # 准备数据：模拟 Flatten 操作
    X = []
    y = []

    for i, data in enumerate(data_list):
        # Flatten: (116, 64) -> (7424,)
        # 确保转为numpy
        feat = data.x.numpy().flatten()
        X.append(feat)
        y.append(labels[i])

    X = np.array(X)
    y = np.array(y)

    print(f"Data shape: {X.shape}")
    print("Running Sklearn Logistic Regression on .pkl data...")

    # 使用和 validate_node_features.py 类似的逻辑
    clf = LogisticRegression(max_iter=1000, solver='liblinear')
    scores = cross_val_score(clf, X, y, cv=5)

    print(f"\n{'=' * 40}")
    print(f"Mean Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"{'=' * 40}")

    if scores.mean() < 0.55:
        print("\n❌ 结论: 数据文件里的特征是垃圾(随机噪音)。")
        print("原因: prepare_gnn_data.py 没有正确加载预训练权重。")
    else:
        print("\n✅ 结论: 数据文件正常。问题出在 PyTorch 训练参数上。")


if __name__ == '__main__':
    check_data()
