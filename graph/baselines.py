"""
基线方法实现
包括SVM、全局微调、各种Prompt方法等
"""
import os

import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from model import GIN
from prompt import EdgePrompt, NodePrompt


class SVMBaseline:
    """传统SVM基线"""

    def __init__(self):
        self.svm = SVC(kernel='rbf', C=1.0)
        self.scaler = StandardScaler()

    def extract_features(self, data_list):
        """提取图级特征"""
        features = []
        for data in data_list:
            # 简单的图统计特征
            node_feat = data.x.numpy()
            edge_num = data.edge_index.shape[1]

            feat = [
                node_feat.mean(),
                node_feat.std(),
                edge_num / len(node_feat),  # 密度
                node_feat.max(),
                node_feat.min()
            ]
            features.append(feat)

        return np.array(features)

    def train(self, train_data, train_labels):
        """训练SVM"""
        features = self.extract_features(train_data)
        features = self.scaler.fit_transform(features)
        self.svm.fit(features, train_labels)

    def evaluate(self, test_data, test_labels):
        """评估"""
        features = self.extract_features(test_data)
        features = self.scaler.transform(features)
        preds = self.svm.predict(features)
        return accuracy_score(test_labels, preds)


class FineTuningBaseline(nn.Module):
    """全局微调基线"""

    def __init__(self, num_layer=5, input_dim=12, hidden_dim=128,
                 num_classes=2, pretrained_path=None):
        super().__init__()

        self.gnn = GIN(num_layer, input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # 加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path)
            self.gnn.load_state_dict(checkpoint['model_state_dict'], strict=False)

    def forward(self, data):
        x = self.gnn(data, pooling='mean')
        return self.classifier(x)


class EdgePromptBaseline(nn.Module):
    """EdgePrompt基线"""

    def __init__(self, num_layer=5, input_dim=12, hidden_dim=128,
                 num_classes=2, num_prompts=5):
        super().__init__()

        self.gnn = GIN(num_layer, input_dim, hidden_dim)
        self.edge_prompt = EdgePrompt([input_dim] + [hidden_dim] * (num_layer - 1))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = self.gnn(data, prompt_type='EdgePrompt',
                     prompt=self.edge_prompt, pooling='mean')
        return self.classifier(x)


class NodePromptBaseline(nn.Module):
    """NodePrompt基线"""

    def __init__(self, num_layer=5, input_dim=12, hidden_dim=128,
                 num_classes=2):
        super().__init__()

        self.gnn = GIN(num_layer, input_dim, hidden_dim)
        self.node_prompts = nn.ModuleList([
            NodePrompt(dim) for dim in [input_dim] + [hidden_dim] * (num_layer - 1)
        ])
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = self.gnn(data, prompt_type='NodePrompt',
                     prompt=self.node_prompts, pooling='mean')
        return self.classifier(x)


def run_baseline_experiments(data_list, labels, shots=5):
    """运行所有基线实验"""
    results = {}

    # 1. SVM基线
    svm = SVMBaseline()
    train_data = data_list[:shots * 2]
    train_labels = labels[:shots * 2]
    test_data = data_list[shots * 2:]
    test_labels = labels[shots * 2:]

    svm.train(train_data, train_labels)
    results['SVM'] = svm.evaluate(test_data, test_labels)

    # 2. 其他基线类似处理...

    return results