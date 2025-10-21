"""
站点效应可视化分析
使用t-SNE和UMAP检查多站点fMRI数据的站点效应

功能：
1. 加载FC矩阵数据
2. 使用t-SNE/UMAP进行降维
3. 可视化站点和诊断标签的分布
4. 定量评估站点效应
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# 尝试导入UMAP（可选）
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    print("Warning: umap-learn not installed. Install with: pip install umap-learn")
    UMAP_AVAILABLE = False


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

    return X


def perform_site_effect_analysis(
        fc_matrices,
        labels,
        site_ids,
        method='tsne',
        n_components=2,
        save_dir='./results/site_effects',
        dataset_name='ABIDE'
):
    """
    执行站点效应分析

    Args:
        fc_matrices: FC矩阵 [N, N_ROI, N_ROI]
        labels: 诊断标签 [N]
        site_ids: 站点ID [N]
        method: 'tsne' or 'umap'
        n_components: 降维维度
        save_dir: 保存目录
        dataset_name: 数据集名称
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Site Effect Analysis - {dataset_name}")
    print(f"{'=' * 60}")

    # 1. Flatten FC矩阵
    print("\n1. Preparing features...")
    X = flatten_fc_upper_triangle(fc_matrices)
    print(f"   Feature shape: {X.shape}")

    # 2. 标准化
    print("\n2. Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 降维
    print(f"\n3. Performing dimensionality reduction ({method.upper()})...")

    if method == 'tsne':
        reducer = TSNE(
            n_components=n_components,
            random_state=42,
            perplexity=30,
            n_iter=1000,
            verbose=1
        )
        X_embedded = reducer.fit_transform(X_scaled)

    elif method == 'umap' and UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean'
        )
        X_embedded = reducer.fit_transform(X_scaled)

    else:
        raise ValueError(f"Method {method} not available")

    print(f"   Embedding shape: {X_embedded.shape}")

    # 4. 创建可视化
    print("\n4. Creating visualizations...")

    # 准备数据
    df = pd.DataFrame({
        'x': X_embedded[:, 0],
        'y': X_embedded[:, 1],
        'diagnosis': ['HC' if l == 0 else 'ASD' for l in labels],
        'site': site_ids
    })

    # 统计站点数量
    n_sites = len(df['site'].unique())
    print(f"   Number of sites: {n_sites}")

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 子图1：按诊断标签着色
    ax1 = axes[0]
    for diag in df['diagnosis'].unique():
        mask = df['diagnosis'] == diag
        ax1.scatter(
            df[mask]['x'],
            df[mask]['y'],
            label=diag,
            alpha=0.6,
            s=30
        )
    ax1.set_title(f'{method.upper()} - Colored by Diagnosis', fontsize=12)
    ax1.set_xlabel(f'{method.upper()} 1')
    ax1.set_ylabel(f'{method.upper()} 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：按站点着色
    ax2 = axes[1]

    # 如果站点太多，使用色板
    if n_sites > 20:
        # 使用tab20色板
        colors = plt.cm.tab20(np.linspace(0, 1, n_sites))
    else:
        colors = sns.color_palette("husl", n_sites)

    for idx, site in enumerate(df['site'].unique()):
        mask = df['site'] == site
        ax2.scatter(
            df[mask]['x'],
            df[mask]['y'],
            label=site if n_sites <= 15 else None,  # 站点太多时不显示图例
            alpha=0.6,
            s=30,
            color=colors[idx % len(colors)]
        )

    ax2.set_title(f'{method.upper()} - Colored by Site', fontsize=12)
    ax2.set_xlabel(f'{method.upper()} 1')
    ax2.set_ylabel(f'{method.upper()} 2')
    if n_sites <= 15:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 子图3：站点和诊断的交互
    ax3 = axes[2]

    # 创建组合标签
    df['combined'] = df['site'] + '_' + df['diagnosis']

    # 为每个站点分配基础颜色
    site_base_colors = {}
    for idx, site in enumerate(df['site'].unique()):
        site_base_colors[site] = colors[idx % len(colors)]

    # 绘制，HC用圆圈，ASD用三角形
    for site in df['site'].unique():
        # HC组
        mask_hc = (df['site'] == site) & (df['diagnosis'] == 'HC')
        if mask_hc.sum() > 0:
            ax3.scatter(
                df[mask_hc]['x'],
                df[mask_hc]['y'],
                marker='o',
                s=30,
                alpha=0.6,
                color=site_base_colors[site],
                label=f'{site}_HC' if n_sites <= 5 else None
            )

        # ASD组
        mask_asd = (df['site'] == site) & (df['diagnosis'] == 'ASD')
        if mask_asd.sum() > 0:
            ax3.scatter(
                df[mask_asd]['x'],
                df[mask_asd]['y'],
                marker='^',
                s=40,
                alpha=0.6,
                color=site_base_colors[site],
                label=f'{site}_ASD' if n_sites <= 5 else None
            )

    ax3.set_title(f'{method.upper()} - Site × Diagnosis\n(○=HC, △=ASD)', fontsize=12)
    ax3.set_xlabel(f'{method.upper()} 1')
    ax3.set_ylabel(f'{method.upper()} 2')
    if n_sites <= 5:
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图形
    save_path = os.path.join(save_dir, f'{dataset_name}_{method}_site_effects.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Saved visualization to: {save_path}")
    plt.show()

    # 5. 定量评估站点效应
    print("\n5. Quantitative evaluation...")

    # 计算Silhouette Score（站点聚类vs诊断聚类）
    # 值越高表示聚类越好

    # 站点聚类质量
    if n_sites > 1 and n_sites < len(X_embedded):
        site_silhouette = silhouette_score(X_embedded, site_ids)
        print(f"   Site clustering silhouette score: {site_silhouette:.4f}")
    else:
        site_silhouette = np.nan
        print(f"   Site clustering silhouette score: N/A (single site or too many sites)")

    # 诊断聚类质量
    diagnosis_silhouette = silhouette_score(X_embedded, labels)
    print(f"   Diagnosis clustering silhouette score: {diagnosis_silhouette:.4f}")

    # 判断站点效应
    print("\n6. Site effect assessment:")

    if not np.isnan(site_silhouette):
        if site_silhouette > diagnosis_silhouette:
            ratio = site_silhouette / diagnosis_silhouette
            print(f"   ⚠️  WARNING: Strong site effects detected!")
            print(f"   Site clustering is {ratio:.2f}x stronger than diagnosis clustering")
            print(f"   Consider:")
            print(f"     - Site harmonization (e.g., ComBat)")
            print(f"     - Including site as covariate")
            print(f"     - Site-stratified cross-validation")
        else:
            ratio = diagnosis_silhouette / site_silhouette
            print(f"   ✅ Good: Diagnosis clustering is stronger than site clustering")
            print(f"   Diagnosis clustering is {ratio:.2f}x stronger")
    else:
        print(f"   Cannot compare site vs diagnosis clustering (single site)")
        print(f"   Diagnosis silhouette score: {diagnosis_silhouette:.4f}")

    # 保存定量结果
    results = {
        'dataset': dataset_name,
        'method': method,
        'n_subjects': len(labels),
        'n_sites': n_sites,
        'site_silhouette': site_silhouette,
        'diagnosis_silhouette': diagnosis_silhouette,
        'site_effect_warning': site_silhouette > diagnosis_silhouette if not np.isnan(site_silhouette) else False
    }

    results_df = pd.DataFrame([results])
    results_path = os.path.join(save_dir, f'{dataset_name}_{method}_site_metrics.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n   Metrics saved to: {results_path}")

    return X_embedded, results


def analyze_site_effects_comprehensive(
        data_folder='./data',
        dataset_name='ABIDE',
        atlas='aal'
):
    """
    全面的站点效应分析（同时使用t-SNE和UMAP）

    Args:
        data_folder: 数据目录
        dataset_name: 数据集名称
        atlas: 脑图谱
    """
    print(f"\n{'=' * 60}")
    print(f"Comprehensive Site Effect Analysis")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    # 加载数据
    if dataset_name == 'ABIDE':
        from abide_data_baseline import load_abide_baseline
        fc_matrices, labels, subject_ids, site_ids, meta = load_abide_baseline(
            data_folder, atlas=atlas, normalized=True
        )

        if site_ids is None:
            print("❌ No site information available in the data!")
            print("Please re-run data preparation with the enhanced version")
            return

    elif dataset_name == 'MDD':
        from mdd_data_baseline import load_mdd_baseline
        # MDD数据集通常是单站点的
        fc_matrices, labels, subject_ids, site_ids, meta = load_mdd_baseline(
            data_folder, method='pearson'
        )
        # site_ids = np.array(['Site1'] * len(labels))  # 假设单站点
        if site_ids is None:
            print("❌ No site information available in the data!")
            print("Please re-run data preparation with the enhanced version")
            return

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\nData loaded:")
    print(f"  Subjects: {len(labels)}")
    print(f"  Sites: {len(np.unique(site_ids))}")
    print(f"  FC shape: {fc_matrices.shape}")

    save_dir = f'./results/site_effects/{dataset_name}'

    # t-SNE分析
    print("\n" + "=" * 40)
    print("t-SNE Analysis")
    print("=" * 40)
    tsne_embedding, tsne_results = perform_site_effect_analysis(
        fc_matrices, labels, site_ids,
        method='tsne',
        save_dir=save_dir,
        dataset_name=dataset_name
    )

    # UMAP分析（如果可用）
    if UMAP_AVAILABLE:
        print("\n" + "=" * 40)
        print("UMAP Analysis")
        print("=" * 40)
        umap_embedding, umap_results = perform_site_effect_analysis(
            fc_matrices, labels, site_ids,
            method='umap',
            save_dir=save_dir,
            dataset_name=dataset_name
        )

    # 创建总结报告
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)

    if tsne_results['site_effect_warning']:
        print("\n⚠️  CRITICAL: Significant site effects detected!")
        print("The model may learn site-specific patterns instead of disease patterns.")
        print("\nRecommended actions:")
        print("1. Apply harmonization methods (e.g., ComBat, neuroCombat)")
        print("2. Use site-stratified cross-validation")
        print("3. Include site as a covariate in the model")
        print("4. Consider multi-site training strategies")
    else:
        print("\n✅ Site effects appear to be minimal.")
        print("The model should be able to learn disease-specific patterns.")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return tsne_embedding, tsne_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze site effects in fMRI data')
    parser.add_argument('--dataset', type=str, default='ABIDE',
                        choices=['ABIDE', 'MDD'],
                        help='Dataset to analyze')
    parser.add_argument('--data_folder', type=str, default='./data',
                        help='Data folder path')
    parser.add_argument('--atlas', type=str, default='aal',
                        help='Brain atlas')

    args = parser.parse_args()

    # 运行综合分析
    analyze_site_effects_comprehensive(
        data_folder=args.data_folder,
        dataset_name=args.dataset,
        atlas=args.atlas
    )