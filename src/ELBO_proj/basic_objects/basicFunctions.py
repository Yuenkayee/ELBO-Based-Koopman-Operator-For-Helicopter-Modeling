import torch
import torch.nn as nn

# ============================================================
# 1. Basic utility functions
# ============================================================


def reparameterize_full_cov(mu, cov, z_dim, T, eps=1e-6):

    """

    mu:  shape [z_dim * (T + 1)]

    cov: shape [z_dim * (T + 1), z_dim * (T + 1)]

    return:

        Z: shape [T + 1, z_dim]

    """

    n = z_dim * (T + 1)
    assert mu.shape == (n,)
    assert cov.shape == (n, n)
    # 为了数值稳定，给协方差矩阵加一个很小的对角项
    cov_stable = cov + eps * torch.eye(n, device=cov.device, dtype=cov.dtype)
    # Cholesky 分解：cov = L @ L.T
    L = torch.linalg.cholesky(cov_stable)
    # 从标准正态分布采样
    epsilon = torch.randn(n, device=mu.device, dtype=mu.dtype)
    # 重参数化采样
    Z_vec = mu + L @ epsilon
    # reshape 成时间序列形式
    Z = Z_vec.reshape(T + 1, z_dim)
    return Z

def decoder(Z_j, nn_C, T, x_dim, z_dim):
    X_hat_j = torch.zeros(x_dim * (T + 1))
    for i in range(T + 1):
        Z_ji = Z_j[(i - 1) * z_dim : i * z_dim - 1]
        X_hat_j[(i - 1) * x_dim : i * x_dim - 1] = nn_C(Z_ji)
    return X_hat_j

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
