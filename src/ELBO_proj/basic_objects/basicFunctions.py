import torch
import torch.nn as nn

# ============================================================
# 1. Basic utility functions
# ============================================================


# def reparameterize_full_cov(mu, cov, z_dim, T, eps=1e-6):

#     """

#     mu:  shape [z_dim * (T + 1)]

#     cov: shape [z_dim * (T + 1), z_dim * (T + 1)]

#     return:

#         Z: shape [T + 1, z_dim]

#     """

#     n = z_dim * (T + 1)
#     assert mu.shape == (n,)
#     assert cov.shape == (n, n)
#     # 为了数值稳定，给协方差矩阵加一个很小的对角项
#     cov_stable = cov + eps * torch.eye(n, device=cov.device, dtype=cov.dtype)
#     # Cholesky 分解：cov = L @ L.T
#     L = torch.linalg.cholesky(cov_stable)
#     # 从标准正态分布采样
#     epsilon = torch.randn(n, device=mu.device, dtype=mu.dtype)
#     # 重参数化采样
#     Z_vec = mu + L @ epsilon
#     # reshape 成时间序列形式
#     Z = Z_vec.reshape(T + 1, z_dim)
#     return Z

"""
    这里强制为协方差矩阵添加了一个噪声对角项以及正定化, 可以防止矩阵出现奇异性问题或者不严格镇定
"""
def reparameterize_full_cov(mu, cov, z_dim, T):
    """
    mu.shape  == [(T + 1) * z_dim]
    cov.shape == [(T + 1) * z_dim, (T + 1) * z_dim]
    """

    dim = mu.shape[0]

    # 先强制对称，避免数值误差导致 cov != cov.T
    cov = 0.5 * (cov + cov.T)

    eye = torch.eye(dim, device=cov.device, dtype=cov.dtype)

    # 自适应增加 jitter，直到 Cholesky 成功
    jitter = 1e-6
    max_tries = 4

    for _ in range(max_tries):
        try:
            cov_stable = cov + jitter * eye
            L = torch.linalg.cholesky(cov_stable)
            eps = torch.randn(dim, device=mu.device, dtype=mu.dtype)
            return mu + L @ eps
        except torch._C._LinAlgError:
            jitter *= 10.0

    # 如果仍然失败，使用特征值截断作为兜底方案
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = torch.clamp(eigvals, min=jitter)
    cov_stable = eigvecs @ torch.diag(eigvals) @ eigvecs.T

    L = torch.linalg.cholesky(cov_stable)
    eps = torch.randn(dim, device=mu.device, dtype=mu.dtype)

    return mu + L @ eps

def decoder(Z_j, nn_C, T, x_dim, z_dim):
    """
    Decode latent sequence Z_j into predicted state sequence X_hat_j.

    Input:
        Z_j: shape [(T + 1) * z_dim] or [T + 1, z_dim]
             corresponding to [z_0, z_1, ..., z_T]
        nn_C: linear decoder from z_dim to x_dim
        T: number of predicted state vectors, x_1 to x_T

    Output:
        X_hat_j: shape [T * x_dim]
                 corresponding to [x_1, x_2, ..., x_T], where x_i = C z_i
    """
    if Z_j.ndim == 1:
        Z_mat = Z_j.reshape(T + 1, z_dim)
    elif Z_j.ndim == 2:
        Z_mat = Z_j
    else:
        raise ValueError(f"Z_j must be a 1D or 2D tensor, but got shape {Z_j.shape}")

    if Z_mat.shape != (T + 1, z_dim):
        raise ValueError(
            f"Z_j should have shape [{T + 1}, {z_dim}] after reshape, "
            f"but got {Z_mat.shape}"
        )

    # Use z_1, z_2, ..., z_T to reconstruct x_1, x_2, ..., x_T.
    X_hat_mat = nn_C(Z_mat[1 : T + 1])

    if X_hat_mat.shape != (T, x_dim):
        raise ValueError(
            f"Decoded X_hat_mat should have shape [{T}, {x_dim}], "
            f"but got {X_hat_mat.shape}"
        )

    return X_hat_mat.reshape(T * x_dim)

def kl_divergence_gaussian(mu_1, cov_1, mu_2, cov_2, eps=1e-6):

    """

    Compute KL(N1 || N2) for two multivariate Gaussian distributions.

    N1 = N(mu_1, cov_1)

    N2 = N(mu_2, cov_2)

    mu_1:  shape [Z_dim]

    cov_1: shape [Z_dim, Z_dim]

    mu_2:  shape [Z_dim]

    cov_2: shape [Z_dim, Z_dim]

    return:

        scalar KL divergence

    """

    Z_dim = mu_1.shape[0]
    assert mu_1.shape == (Z_dim,)
    assert mu_2.shape == (Z_dim,)
    assert cov_1.shape == (Z_dim, Z_dim)
    assert cov_2.shape == (Z_dim, Z_dim)

    # 数值稳定：给协方差矩阵加一个很小的对角项

    eye = torch.eye(Z_dim, device=mu_1.device, dtype=mu_1.dtype)
    cov_1 = cov_1 + eps * eye
    cov_2 = cov_2 + eps * eye

    # 均值差

    diff = mu_2 - mu_1
    # log |cov_2| - log |cov_1|

    logdet_cov_1 = torch.logdet(cov_1)
    logdet_cov_2 = torch.logdet(cov_2)

    # cov_2^{-1} cov_1

    cov_2_inv_cov_1 = torch.linalg.solve(cov_2, cov_1)
    trace_term = torch.trace(cov_2_inv_cov_1)

    # (mu_2 - mu_1)^T cov_2^{-1} (mu_2 - mu_1)

    mahalanobis_term = diff @ torch.linalg.solve(cov_2, diff)
    kl = 0.5 * (

        logdet_cov_2

        - logdet_cov_1

        - Z_dim

        + trace_term

        + mahalanobis_term

    )

    return kl
