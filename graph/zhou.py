#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABIDE数据集下载和探索脚本
使用nilearn.datasets.fetch_abide_pcp接口
"""

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_abide_pcp
import os


def download_and_explore_abide():
    """
    下载ABIDE数据集并探索其结构
    """
    print("开始下载ABIDE数据集...")
    print("注意：首次下载可能需要较长时间，数据集较大")

    # 下载ABIDE数据集
    # 这里只下载少量数据进行探索，可以根据需要调整参数
    abide_data = fetch_abide_pcp(
        data_dir='./data/zhou',  # 使用默认下载路径
        n_subjects=None,  # 只下载10个受试者的数据进行探索
        pipeline='cpac',  # 使用CPAC预处理流水线
        band_pass_filtering=True,
        global_signal_regression=False,
        derivatives=['rois_ho'],  # 下载功能预处理数据
        quality_checked=True,  # 只下载质量检查通过的数据
        verbose=1
    )

    print("\n" + "=" * 50)
    print("ABIDE数据集结构探索")
    print("=" * 50)

    # 查看数据集的基本信息
    print(f"\n1. 数据集类型: {type(abide_data)}")
    print(f"2. 数据集包含的字段: {list(abide_data.keys())}")

    # 查看功能数据
    if 'func_preproc' in abide_data:
        func_data = abide_data.func_preproc
        print(f"\n3. 功能数据:")
        print(f"   - 数据类型: {type(func_data)}")
        print(f"   - 文件数量: {len(func_data)}")
        print(f"   - 第一个文件路径: {func_data[0] if func_data else 'None'}")

        # 如果有文件，查看文件信息
        if func_data and os.path.exists(func_data[0]):
            import nibabel as nib
            try:
                img = nib.load(func_data[0])
                print(f"   - 第一个文件形状: {img.shape}")
                print(f"   - 数据类型: {img.get_data_dtype()}")
                print(f"   - 体素尺寸: {img.header.get_zooms()}")
            except Exception as e:
                print(f"   - 无法加载第一个文件: {e}")

    # 查看表型数据
    if 'phenotypic' in abide_data:
        phenotypic = abide_data.phenotypic
        print(f"\n4. 表型数据:")
        print(f"   - 数据类型: {type(phenotypic)}")
        if isinstance(phenotypic, pd.DataFrame):
            print(f"   - 形状: {phenotypic.shape}")
            print(f"   - 列名: {list(phenotypic.columns)}")
            print(f"\n   - 前5行数据:")
            print(phenotypic.head())

            # 查看诊断分布
            if 'DX_GROUP' in phenotypic.columns:
                print(f"\n   - 诊断组分布:")
                print(phenotypic['DX_GROUP'].value_counts())

            # 查看年龄分布
            if 'AGE_AT_SCAN' in phenotypic.columns:
                print(f"\n   - 年龄统计:")
                print(phenotypic['AGE_AT_SCAN'].describe())

    # 查看混淆变量
    if 'confounds' in abide_data:
        confounds = abide_data.confounds
        print(f"\n5. 混淆变量:")
        print(f"   - 数据类型: {type(confounds)}")
        print(f"   - 文件数量: {len(confounds) if confounds else 0}")
        if confounds:
            # 查看第一个混淆变量文件
            try:
                if isinstance(confounds[0], str) and os.path.exists(confounds[0]):
                    conf_df = pd.read_csv(confounds[0], sep='\t')
                    print(f"   - 第一个文件形状: {conf_df.shape}")
                    print(f"   - 列名: {list(conf_df.columns)}")
                elif isinstance(confounds[0], np.ndarray):
                    print(f"   - 第一个数组形状: {confounds[0].shape}")
                else:
                    print(f"   - 第一个元素类型: {type(confounds[0])}")
            except Exception as e:
                print(f"   - 无法读取混淆变量: {e}")

    # 查看其他可能的字段
    other_keys = [k for k in abide_data.keys()
                  if k not in ['func_preproc', 'phenotypic', 'confounds']]
    if other_keys:
        print(f"\n6. 其他字段: {other_keys}")
        for key in other_keys:
            data = abide_data[key]
            print(f"   - {key}: {type(data)}")
            if hasattr(data, '__len__'):
                try:
                    print(f"     长度: {len(data)}")
                except:
                    pass

    print(f"\n7. 数据下载位置:")
    # 尝试找到数据下载位置
    try:
        from nilearn.datasets.utils import _get_dataset_dir
        data_dir = _get_dataset_dir('abide_pcp', data_dir=None, verbose=1)
        print(f"   - {data_dir}")
    except:
        print("   - 无法确定具体下载位置，请检查nilearn默认数据目录")

    return abide_data


def explore_single_subject(abide_data, subject_index=0):
    """
    详细探索单个受试者的数据
    """
    if 'func_preproc' not in abide_data or not abide_data.func_preproc:
        print("没有功能数据可供探索")
        return

    print(f"\n" + "=" * 50)
    print(f"受试者 {subject_index} 详细信息")
    print("=" * 50)

    # 功能数据
    func_file = abide_data.func_preproc[subject_index]
    print(f"功能数据文件: {func_file}")

    if os.path.exists(func_file):
        try:
            import nibabel as nib
            from nilearn import image

            # 加载数据
            img = nib.load(func_file)
            print(f"图像形状: {img.shape}")
            print(f"仿射矩阵形状: {img.affine.shape}")
            print(f"体素尺寸: {img.header.get_zooms()}")

            # 使用nilearn获取更多信息
            print(f"时间点数: {image.get_data(img).shape[-1] if len(img.shape) == 4 else 'N/A'}")

        except Exception as e:
            print(f"加载功能数据时出错: {e}")

    # 表型数据
    if 'phenotypic' in abide_data:
        phenotypic = abide_data.phenotypic
        if isinstance(phenotypic, pd.DataFrame) and subject_index < len(phenotypic):
            print(f"\n表型信息:")
            subject_info = phenotypic.iloc[subject_index]
            for col in ['SUB_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX', 'SITE_ID']:
                if col in subject_info:
                    print(f"  {col}: {subject_info[col]}")

    # 混淆变量
    if 'confounds' in abide_data and abide_data.confounds:
        if subject_index < len(abide_data.confounds):
            conf_file = abide_data.confounds[subject_index]
            print(f"\n混淆变量文件: {conf_file}")

            if isinstance(conf_file, str) and os.path.exists(conf_file):
                try:
                    conf_df = pd.read_csv(conf_file, sep='\t')
                    print(f"混淆变量形状: {conf_df.shape}")
                    print(f"混淆变量列名: {list(conf_df.columns)}")
                except Exception as e:
                    print(f"读取混淆变量时出错: {e}")


if __name__ == "__main__":
    # 检查依赖
    try:
        import nilearn
        import nibabel

        print(f"nilearn版本: {nilearn.__version__}")
        print(f"nibabel版本: {nibabel.__version__}")
    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请安装: pip install nilearn nibabel pandas")
        exit(1)

    # 下载和探索数据集
    try:
        abide_data = download_and_explore_abide()

        # 探索第一个受试者的详细信息
        explore_single_subject(abide_data, 0)

        print(f"\n" + "=" * 50)
        print("探索完成！")
        print("=" * 50)

    except Exception as e:
        print(f"程序执行出错: {e}")
        print("可能的原因：")
        print("1. 网络连接问题")
        print("2. 磁盘空间不足")
        print("3. 权限问题")
        print("4. 依赖库版本不兼容")