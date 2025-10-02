"""
简单的数据准备脚本
一键准备所有数据
"""
import os
import argparse


def prepare_abide():
    """准备ABIDE数据"""
    print("\n" + "=" * 60)
    print("准备ABIDE数据集")
    print("=" * 60)

    from abide_data import ABIDEDataProcessor

    processor = ABIDEDataProcessor(
        data_folder='./data',
        pipeline='cpac',
        atlas='ho',
        connectivity_kind='correlation',
        threshold=0.3
    )

    # 处理基础数据
    print("\n处理基础数据...")
    graph_list = processor.process_and_save(
        n_subjects=None,
        graph_method='correlation_matrix'
    )

    # 构建双流数据
    print("\n构建双流数据...")
    dual_stream = processor.process_dual_stream()

    print(f"\n✓ ABIDE处理完成: {len(graph_list)} 个基础样本, {len(dual_stream)} 个双流样本")

    return len(graph_list), len(dual_stream)


def prepare_mdd():
    """准备MDD数据"""
    print("\n" + "=" * 60)
    print("准备MDD数据集")
    print("=" * 60)

    from mdd_data import MDDDataProcessor

    processor = MDDDataProcessor(
        data_folder='./data',
        connectivity_kind='correlation',
        threshold=0.3
    )

    # 处理基础数据
    print("\n处理基础数据...")
    graph_list = processor.process_and_save(
        graph_method='correlation_matrix'
    )

    # 构建双流数据
    print("\n构建双流数据...")
    dual_stream = processor.process_dual_stream()

    print(f"\n✓ MDD处理完成: {len(graph_list)} 个基础样本, {len(dual_stream)} 个双流样本")

    return len(graph_list), len(dual_stream)


def main():
    parser = argparse.ArgumentParser(description='数据准备脚本')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'ABIDE', 'MDD'],
                        help='要准备的数据集')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("脑疾病数据准备工具")
    print("=" * 60)

    results = {}

    if args.dataset in ['all', 'ABIDE']:
        try:
            results['ABIDE'] = prepare_abide()
        except Exception as e:
            print(f"ABIDE处理失败: {e}")
            results['ABIDE'] = (0, 0)

    if args.dataset in ['all', 'MDD']:
        try:
            results['MDD'] = prepare_mdd()
        except Exception as e:
            print(f"MDD处理失败: {e}")
            results['MDD'] = (0, 0)

    # 打印总结
    print("\n" + "=" * 60)
    print("数据准备完成总结")
    print("=" * 60)

    for dataset, (basic, dual) in results.items():
        print(f"{dataset}: 基础样本={basic}, 双流样本={dual}")

    print("\n所有数据准备完成！")


if __name__ == "__main__":
    main()