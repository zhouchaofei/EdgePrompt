"""
快速测试ABIDE数据处理
"""
from abide_data import ABIDEDataProcessor


def quick_test():
    print("快速测试ABIDE数据处理...")

    # 创建处理器
    processor = ABIDEDataProcessor(
        data_folder='./data',
        pipeline='cpac',
        atlas='ho',
        connectivity_kind='correlation',
        threshold=0.3
    )

    # 只测试1个被试，快速验证
    print("\n测试最少数据量（1个被试）...")
    graph_list = processor.process_and_save(n_subjects=1, graph_method='correlation_matrix')

    if graph_list:
        print("\n✅ 成功！")
        print(f"构建了 {len(graph_list)} 个图")
        sample = graph_list[0]
        print(f"节点特征形状: {sample.x.shape}")
        print(f"边数: {sample.edge_index.shape[1]}")
        print(f"标签: {sample.y.item()}")
    else:
        print("\n❌ 失败：未能构建任何图")

    return graph_list


if __name__ == '__main__':
    quick_test()