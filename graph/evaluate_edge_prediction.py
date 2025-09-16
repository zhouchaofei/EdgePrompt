"""
评估Edge Prediction预训练模型在下游任务上的性能
"""
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from edge_prediction_pretrain import EdgePredictionModel
from abide_data import load_abide_data
from mdd_data import load_mdd_data
from load_data import GraphDownstream


def evaluate_edge_prediction_model(pretrained_path, target_dataset, args):
    """
    评估Edge Prediction预训练模型在目标数据集上的性能
    """
    print(f"\n评估Edge Prediction预训练模型: {pretrained_path}")
    print(f"目标数据集: {target_dataset}")

    # 加载目标数据集
    if target_dataset == 'ABIDE':
        graph_list, input_dim, output_dim = load_abide_data(
            data_folder=args.data_folder,
            graph_method=args.graph_method
        )
    elif target_dataset == 'MDD':
        graph_list, input_dim, output_dim = load_mdd_data(
            data_folder=args.data_folder,
            graph_method=args.graph_method
        )
    else:
        raise ValueError(f"未知数据集: {target_dataset}")

    # 准备few-shot数据
    train_data, test_data = GraphDownstream(
        graph_list, shots=args.shots, test_fraction=0.4
    )

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # 加载预训练模型
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    model = EdgePredictionModel(
        num_layer=args.num_layer,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 冻结编码器，只训练分类器
    for param in model.encoder.parameters():
        param.requires_grad = False

    # 添加分类头
    classifier = nn.Sequential(
        nn.Linear(args.hidden_dim, args.hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(args.hidden_dim // 2, output_dim)
    ).to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # 微调
    model.eval()  # 编码器保持eval模式
    best_acc = 0

    from torch_geometric.nn import global_mean_pool

    for epoch in range(100):
        # 训练
        classifier.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # 获取节点表示
            with torch.no_grad():
                node_rep = model.encode(batch)

            # 图级池化
            graph_rep = global_mean_pool(node_rep, batch.batch)

            # 分类
            logits = classifier(graph_rep)
            loss = nn.CrossEntropyLoss()(logits, batch.y.squeeze())

            loss.backward()
            optimizer.step()

        # 测试
        if (epoch + 1) % 10 == 0:
            classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)

                    # 获取图表示
                    node_rep = model.encode(batch)
                    graph_rep = global_mean_pool(node_rep, batch.batch)

                    logits = classifier(graph_rep)
                    pred = logits.argmax(dim=1)
                    correct += (pred == batch.y.squeeze()).sum().item()
                    total += batch.y.size(0)

            acc = correct / total
            if acc > best_acc:
                best_acc = acc

            print(f"Epoch {epoch + 1}: Accuracy = {acc:.4f}, Best = {best_acc:.4f}")

    return best_acc


def main():
    parser = argparse.ArgumentParser(description='评估Edge Prediction预训练模型')

    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--graph_method', type=str, default='correlation_matrix')
    parser.add_argument('--num_layer', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--shots', type=int, default=30)
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()

    # 评估ABIDE->MDD
    pretrained_path = f'./pretrained_models/edge_prediction/edge_prediction_ABIDE_for_MDD_{args.graph_method}.pth'
    acc = evaluate_edge_prediction_model(pretrained_path, 'MDD', args)
    print(f"\nEdge Prediction ABIDE->MDD: {acc:.4f}")

    # 评估MDD->ABIDE
    pretrained_path = f'./pretrained_models/edge_prediction/edge_prediction_MDD_for_ABIDE_{args.graph_method}.pth'
    acc = evaluate_edge_prediction_model(pretrained_path, 'ABIDE', args)
    print(f"\nEdge Prediction MDD->ABIDE: {acc:.4f}")


if __name__ == '__main__':
    main()