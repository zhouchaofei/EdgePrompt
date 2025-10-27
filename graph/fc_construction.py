"""
功能连接矩阵构建模块
支持多种构图方式：
1. Pearson correlation (baseline)
2. Ledoit-Wolf shrinkage covariance
3. Ridge-shrinkage covariance
4. Graphical Lasso (sparse precision matrix)

以及多种阈值策略：
- none: 不做阈值处理
- top-k per node: 每个节点保留k个最强连接
"""

import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf, GraphicalLasso, empirical_covariance
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class FCConstructor:
    """功能连接矩阵构建器"""

    def __init__(self, method='pearson', **kwargs):
        """
        Args:
            method: 构图方法
                - 'pearson': Pearson correlation
                - 'ledoit_wolf': Ledoit-Wolf shrinkage
                - 'ridge': Ridge-shrinkage covariance
                - 'graphical_lasso': Graphical Lasso
            **kwargs: 方法特定参数
                - ridge_lambda: Ridge正则化参数
                - graphical_alpha: Graphical Lasso稀疏参数
        """
        self.method = method
        self.ridge_lambda = kwargs.get('ridge_lambda', 0.01)
        self.graphical_alpha = kwargs.get('graphical_alpha', 0.05)

    def compute_fc_matrix(self, timeseries):
        """
        从时间序列计算FC矩阵

        Args:
            timeseries: [T, N_ROI] 时间序列数据

        Returns:
            fc: [N_ROI, N_ROI] FC矩阵
        """
        if self.method == 'pearson':
            fc = self._compute_pearson(timeseries)
        elif self.method == 'ledoit_wolf':
            fc = self._compute_ledoit_wolf(timeseries)
        elif self.method == 'ridge':
            fc = self._compute_ridge(timeseries, self.ridge_lambda)
        elif self.method == 'graphical_lasso':
            fc = self._compute_graphical_lasso(timeseries, self.graphical_alpha)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # 清理NaN和Inf
        fc = np.nan_to_num(fc, nan=0.0, posinf=1.0, neginf=-1.0)

        # 对角线设为0（去掉自相关）
        np.fill_diagonal(fc, 0)

        return fc

    def _compute_pearson(self, timeseries):
        """标准Pearson相关"""
        return np.corrcoef(timeseries.T)

    def _compute_ledoit_wolf(self, timeseries):
        """
        Ledoit-Wolf shrinkage covariance转相关矩阵
        """
        # 标准化时间序列
        scaler = StandardScaler()
        ts_scaled = scaler.fit_transform(timeseries)

        # 计算Ledoit-Wolf协方差
        lw = LedoitWolf(assume_centered=False)
        lw.fit(ts_scaled)
        cov = lw.covariance_

        # 转换为相关矩阵
        fc = self._cov_to_corr(cov)
        return fc

    def _compute_ridge(self, timeseries, lambda_val):
        """
        Ridge-shrinkage covariance (regularized Pearson)
        """
        # 标准化
        scaler = StandardScaler()
        ts_scaled = scaler.fit_transform(timeseries)

        # 计算协方差
        cov = np.cov(ts_scaled.T)

        # Ridge正则化：cov + λI
        n_roi = cov.shape[0]
        cov_ridge = cov + lambda_val * np.eye(n_roi)

        # 转换为相关矩阵
        fc = self._cov_to_corr(cov_ridge)
        return fc

    def _compute_graphical_lasso(self, timeseries, alpha):
        """
        Graphical Lasso (稀疏逆协方差/精度矩阵)
        返回partial correlation
        增强版：包含错误处理和回退机制
        """
        # 标准化
        scaler = StandardScaler()
        ts_scaled = scaler.fit_transform(timeseries)

        # 检查数据条件
        n_samples, n_features = ts_scaled.shape

        # 如果样本数太少，增加正则化或使用回退方法
        if n_samples < n_features * 2:
            print(f"  Warning: Low sample/feature ratio ({n_samples}/{n_features}), "
                  f"increasing regularization")
            alpha = max(alpha * 2, 0.1)

        # 尝试多个alpha值，从原始值开始
        alpha_values = [alpha, alpha * 2, alpha * 5, 0.1, 0.2, 0.5]

        for try_alpha in alpha_values:
            try:
                # Graphical Lasso with error handling
                gl = GraphicalLasso(
                    alpha=try_alpha,
                    max_iter=100,
                    tol=1e-4,
                    mode='cd',  # coordinate descent mode
                    assume_centered=False
                )

                # 尝试拟合
                gl.fit(ts_scaled)

                # 获取precision matrix (逆协方差)
                precision = gl.precision_

                # 检查precision matrix是否有效
                if np.any(np.isnan(precision)) or np.any(np.isinf(precision)):
                    raise ValueError("Invalid precision matrix")

                # 转换为partial correlation
                # partial_corr_ij = -precision_ij / sqrt(precision_ii * precision_jj)
                diag = np.sqrt(np.abs(np.diag(precision)))  # 使用abs确保非负

                # 避免除零
                diag[diag < 1e-10] = 1e-10
                diag_mat = np.outer(diag, diag)

                # 计算partial correlation（注意负号）
                partial_corr = -precision / diag_mat

                # 对角线设为1（自相关）
                np.fill_diagonal(partial_corr, 1)

                # 确保对称性
                partial_corr = (partial_corr + partial_corr.T) / 2

                # 限制在[-1, 1]范围内
                partial_corr = np.clip(partial_corr, -1, 1)

                if try_alpha != alpha:
                    print(f"    Successfully computed with alpha={try_alpha:.3f}")

                return partial_corr

            except (FloatingPointError, ValueError, np.linalg.LinAlgError) as e:
                if try_alpha == alpha_values[-1]:
                    # 所有alpha都失败了，使用回退方法
                    print(f"    Graphical Lasso failed with all alpha values, "
                          f"falling back to Ledoit-Wolf")
                    return self._compute_ledoit_wolf_fallback(timeseries)
                else:
                    continue  # 尝试下一个alpha值

        # 不应该到达这里，但以防万一
        return self._compute_ledoit_wolf_fallback(timeseries)

    def _compute_ledoit_wolf_fallback(self, timeseries):
        """
        Ledoit-Wolf作为回退方法
        用于Graphical Lasso失败时
        """
        try:
            # 尝试Ledoit-Wolf
            return self._compute_ledoit_wolf(timeseries)
        except:
            # 如果还是失败，返回简单的Pearson相关
            print("    All methods failed, using Pearson correlation")
            return self._compute_pearson(timeseries)

    def _cov_to_corr(self, cov):
        """
        将协方差矩阵转换为相关矩阵
        增强版：处理数值问题
        """
        # 提取标准差
        std = np.sqrt(np.abs(np.diag(cov)))  # 使用abs处理数值误差

        # 避免除零
        std[std < 1e-10] = 1e-10
        std_mat = np.outer(std, std)

        # 计算相关矩阵
        corr = cov / std_mat

        # 确保对角线为1
        np.fill_diagonal(corr, 1)

        # 确保对称性
        corr = (corr + corr.T) / 2

        # 限制在[-1, 1]范围内
        corr = np.clip(corr, -1, 1)

        return corr


class ThresholdStrategy:
    """阈值策略类"""

    @staticmethod
    def apply_threshold(fc_matrix, strategy='none', k=8):
        """
        应用阈值策略

        Args:
            fc_matrix: [N_ROI, N_ROI] FC矩阵
            strategy: 阈值策略
                - 'none': 不做阈值处理
                - 'top_k': 每个节点保留k个最强连接
            k: top-k的k值

        Returns:
            fc_thresholded: 阈值化后的FC矩阵
            sparsity: 稀疏率
        """
        fc_thresh = fc_matrix.copy()

        if strategy == 'none':
            # 不做阈值处理
            pass

        elif strategy == 'top_k':
            # 每个节点保留前k个最强连接（基于绝对值）
            n_roi = fc_thresh.shape[0]

            for i in range(n_roi):
                # 获取第i个节点的所有连接（排除自身）
                row_values = fc_thresh[i, :].copy()
                row_values[i] = 0  # 排除自连接

                # 找到前k个最强连接的索引（基于绝对值）
                abs_values = np.abs(row_values)
                if k < n_roi - 1:
                    # 找到第k+1大的值作为阈值
                    threshold = np.partition(abs_values, -k - 1)[-k - 1] if k < len(abs_values) else 0
                    # 保留大于等于阈值的连接
                    mask = abs_values >= threshold
                    # 如果有并列值导致超过k个，只保留前k个
                    if mask.sum() > k:
                        sorted_indices = np.argsort(abs_values)[::-1]
                        mask[:] = False
                        mask[sorted_indices[:k]] = True
                    row_values[~mask] = 0

                fc_thresh[i, :] = row_values
                fc_thresh[:, i] = row_values  # 保持对称性

        else:
            raise ValueError(f"Unknown threshold strategy: {strategy}")

        # 计算稀疏率
        n_edges = fc_thresh.shape[0] * (fc_thresh.shape[0] - 1) / 2
        n_zeros = np.sum(np.triu(fc_thresh, k=1) == 0)
        sparsity = n_zeros / n_edges if n_edges > 0 else 0

        return fc_thresh, sparsity


def compute_fc_matrices_enhanced(
        timeseries_list,
        method='pearson',
        threshold_strategy='none',
        threshold_k=8,
        **method_kwargs
):
    """
    计算增强版FC矩阵（支持多种构图方式和阈值策略）
    包含错误处理和回退机制

    Args:
        timeseries_list: 时间序列列表
        method: 构图方法
        threshold_strategy: 阈值策略
        threshold_k: top-k的k值
        **method_kwargs: 方法特定参数

    Returns:
        fc_matrices: FC矩阵数组
        sparsity_rates: 稀疏率列表
    """
    print(f"\nComputing FC matrices:")
    print(f"  Method: {method}")
    print(f"  Threshold: {threshold_strategy}", end="")
    if threshold_strategy == 'top_k':
        print(f" (k={threshold_k})")
    else:
        print()

    # 打印方法特定参数
    if method == 'ridge' and 'ridge_lambda' in method_kwargs:
        print(f"  Ridge lambda: {method_kwargs['ridge_lambda']}")
    elif method == 'graphical_lasso' and 'graphical_alpha' in method_kwargs:
        print(f"  Graphical Lasso alpha: {method_kwargs['graphical_alpha']}")

    # 创建FC构建器
    fc_constructor = FCConstructor(method=method, **method_kwargs)

    fc_matrices = []
    sparsity_rates = []
    failed_subjects = []

    for i, ts in enumerate(timeseries_list):
        try:
            # 计算FC矩阵
            fc = fc_constructor.compute_fc_matrix(ts)

            # 检查FC矩阵有效性
            if np.any(np.isnan(fc)) or np.any(np.isinf(fc)):
                print(f"  Warning: Invalid FC matrix for subject {i}, using zero matrix")
                fc = np.zeros((ts.shape[1], ts.shape[1]))
                failed_subjects.append(i)

            # 应用阈值策略
            fc_thresh, sparsity = ThresholdStrategy.apply_threshold(
                fc, threshold_strategy, threshold_k
            )

            fc_matrices.append(fc_thresh)
            sparsity_rates.append(sparsity)

        except Exception as e:
            print(f"  Error processing subject {i}: {e}")
            # 使用零矩阵作为回退
            n_roi = ts.shape[1]
            fc_matrices.append(np.zeros((n_roi, n_roi)))
            sparsity_rates.append(1.0)
            failed_subjects.append(i)

        if (i + 1) % 100 == 0:
            print(f"  Processed: {i + 1}/{len(timeseries_list)}")

    fc_matrices = np.array(fc_matrices)
    sparsity_rates = np.array(sparsity_rates)

    # 打印统计信息
    print(f"\nFC matrices statistics:")
    print(f"  Shape: {fc_matrices.shape}")
    print(f"  Mean: {fc_matrices.mean():.4f}")
    print(f"  Std: {fc_matrices.std():.4f}")
    print(f"  Min: {fc_matrices.min():.4f}")
    print(f"  Max: {fc_matrices.max():.4f}")
    print(f"  Mean sparsity: {sparsity_rates.mean():.4f}")

    if failed_subjects:
        print(f"  Failed subjects: {len(failed_subjects)} / {len(timeseries_list)}")

    return fc_matrices, sparsity_rates


# 配置生成器保持不变
def generate_fc_configs():
    """
    生成所有FC构建配置的组合

    Returns:
        configs: 配置列表，每个配置是一个字典
    """
    configs = []

    # 1. Pearson (baseline)
    for threshold in ['none', 'top_k_8', 'top_k_12']:
        config = {
            'method': 'pearson',
            'threshold_strategy': 'none' if threshold == 'none' else 'top_k',
            'threshold_k': 8 if 'k_8' in threshold else 12 if 'k_12' in threshold else None,
            'name': f'pearson_{threshold}'
        }
        configs.append(config)

    # 2. Ledoit-Wolf
    for threshold in ['none', 'top_k_8', 'top_k_12']:
        config = {
            'method': 'ledoit_wolf',
            'threshold_strategy': 'none' if threshold == 'none' else 'top_k',
            'threshold_k': 8 if 'k_8' in threshold else 12 if 'k_12' in threshold else None,
            'name': f'ledoit_wolf_{threshold}'
        }
        configs.append(config)

    # 3. Ridge-shrinkage
    for lambda_val in [1e-3, 1e-2, 1e-1]:
        for threshold in ['none', 'top_k_8', 'top_k_12']:
            config = {
                'method': 'ridge',
                'ridge_lambda': lambda_val,
                'threshold_strategy': 'none' if threshold == 'none' else 'top_k',
                'threshold_k': 8 if 'k_8' in threshold else 12 if 'k_12' in threshold else None,
                'name': f'ridge_lambda{lambda_val}_{threshold}'
            }
            configs.append(config)

    # 4. Graphical Lasso - 调整alpha值使其更稳定
    for alpha in [0.05, 0.1, 0.2]:  # 增大alpha值以提高稳定性
        for threshold in ['none', 'top_k_8', 'top_k_12']:
            config = {
                'method': 'graphical_lasso',
                'graphical_alpha': alpha,
                'threshold_strategy': 'none' if threshold == 'none' else 'top_k',
                'threshold_k': 8 if 'k_8' in threshold else 12 if 'k_12' in threshold else None,
                'name': f'graphical_lasso_alpha{alpha}_{threshold}'
            }
            configs.append(config)

    return configs