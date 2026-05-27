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

def decoder(Z_j, nn_C, T, x_dim, z_dim):
    """
    Decode latent sequence Z_j into predicted state sequence X_hat_j.

    Accepted input shapes:
        Z_j.shape == [(T + 1) * z_dim]
        Z_j.shape == [T + 1, z_dim]
        Z_j.shape == [B, (T + 1) * z_dim]
        Z_j.shape == [B, T + 1, z_dim]

    Output shapes:
        If input is a single sample:
            X_hat_j.shape == [T * x_dim]
        If input is a batch:
            X_hat_j.shape == [B, T * x_dim]

    It uses z_1, z_2, ..., z_T to reconstruct x_1, x_2, ..., x_T,
    where x_i = C z_i.
    """
    single_input = False

    if Z_j.ndim == 1:
        Z_mat = Z_j.reshape(T + 1, z_dim)
        single_input = True
    elif Z_j.ndim == 2:
        if Z_j.shape == (T + 1, z_dim):
            Z_mat = Z_j
            single_input = True
        elif Z_j.shape[1] == (T + 1) * z_dim:
            Z_mat = Z_j.reshape(Z_j.shape[0], T + 1, z_dim)
        else:
            raise ValueError(
                f"2D Z_j should have shape [{T + 1}, {z_dim}] or "
                f"[B, {(T + 1) * z_dim}], but got {Z_j.shape}"
            )
    elif Z_j.ndim == 3:
        if Z_j.shape[1:] != (T + 1, z_dim):
            raise ValueError(
                f"3D Z_j should have shape [B, {T + 1}, {z_dim}], "
                f"but got {Z_j.shape}"
            )
        Z_mat = Z_j
    else:
        raise ValueError(f"Z_j must be a 1D, 2D, or 3D tensor, but got shape {Z_j.shape}")

    if single_input:
        X_hat_mat = nn_C(Z_mat[1 : T + 1])
        if X_hat_mat.shape != (T, x_dim):
            raise ValueError(
                f"Decoded X_hat_mat should have shape [{T}, {x_dim}], "
                f"but got {X_hat_mat.shape}"
            )
        return X_hat_mat.reshape(T * x_dim)

    X_hat_mat = nn_C(Z_mat[:, 1 : T + 1, :])
    if X_hat_mat.shape != (Z_mat.shape[0], T, x_dim):
        raise ValueError(
            f"Decoded X_hat_mat should have shape [{Z_mat.shape[0]}, {T}, {x_dim}], "
            f"but got {X_hat_mat.shape}"
        )

    return X_hat_mat.reshape(Z_mat.shape[0], T * x_dim)

# def kl_divergence_gaussian(mu_1, cov_1, mu_2, cov_2, eps=1e-6):

#     """

#     Compute KL(N1 || N2) for two multivariate Gaussian distributions.

#     N1 = N(mu_1, cov_1)

#     N2 = N(mu_2, cov_2)

#     mu_1:  shape [Z_dim]

#     cov_1: shape [Z_dim, Z_dim]

#     mu_2:  shape [Z_dim]

#     cov_2: shape [Z_dim, Z_dim]

#     return:

#         scalar KL divergence

#     """

#     Z_dim = mu_1.shape[0]
#     assert mu_1.shape == (Z_dim,)
#     assert mu_2.shape == (Z_dim,)
#     assert cov_1.shape == (Z_dim, Z_dim)
#     assert cov_2.shape == (Z_dim, Z_dim)

#     # 数值稳定：给协方差矩阵加一个很小的对角项

#     eye = torch.eye(Z_dim, device=mu_1.device, dtype=mu_1.dtype)
#     cov_1 = cov_1 + eps * eye
#     cov_2 = cov_2 + eps * eye

#     # 均值差

#     diff = mu_2 - mu_1
#     # log |cov_2| - log |cov_1|

#     logdet_cov_1 = torch.logdet(cov_1)
#     logdet_cov_2 = torch.logdet(cov_2)

#     # cov_2^{-1} cov_1

#     cov_2_inv_cov_1 = torch.linalg.solve(cov_2, cov_1)
#     trace_term = torch.trace(cov_2_inv_cov_1)

#     # (mu_2 - mu_1)^T cov_2^{-1} (mu_2 - mu_1)

#     mahalanobis_term = diff @ torch.linalg.solve(cov_2, diff)
#     kl = 0.5 * (

#         logdet_cov_2

#         - logdet_cov_1

#         - Z_dim

#         + trace_term

#         + mahalanobis_term

#     )

#     return kl


"""
    下面是用于 edge_block 方案的函数
"""

def _mu_to_mat(mu, T, z_dim):
    """
    Convert mu to time-sequence matrix form.

    Accepted input shapes:
        [(T + 1) * z_dim]
        [T + 1, z_dim]
        [B, (T + 1) * z_dim]
        [B, T + 1, z_dim]

    Return:
        mu_mat: shape [T + 1, z_dim] for a single sample, or
                shape [B, T + 1, z_dim] for a batch.
        single_input: bool
    """
    if mu.ndim == 1:
        return mu.reshape(T + 1, z_dim), True

    if mu.ndim == 2:
        if mu.shape == (T + 1, z_dim):
            return mu, True
        if mu.shape[1] == (T + 1) * z_dim:
            return mu.reshape(mu.shape[0], T + 1, z_dim), False

    if mu.ndim == 3 and mu.shape[1:] == (T + 1, z_dim):
        return mu, False

    raise ValueError(
        f"mu should have shape [{(T + 1) * z_dim}], [{T + 1}, {z_dim}], "
        f"[B, {(T + 1) * z_dim}], or [B, {T + 1}, {z_dim}], "
        f"but got {mu.shape}"
    )


def _cov_to_block(cov, T, z_dim):
    """
    Convert covariance to block-diagonal form.

    Accepted input shapes:
        [T + 1, z_dim, z_dim]
        [B, T + 1, z_dim, z_dim]
        [(T + 1) * z_dim, (T + 1) * z_dim]                 compatibility only
        [B, (T + 1) * z_dim, (T + 1) * z_dim]              compatibility only

    Return:
        cov_blocks: shape [T + 1, z_dim, z_dim] for a single sample, or
                    shape [B, T + 1, z_dim, z_dim] for a batch.
        single_input: bool
    """
    Z_dim = (T + 1) * z_dim

    if cov.ndim == 3 and cov.shape == (T + 1, z_dim, z_dim):
        return cov, True

    if cov.ndim == 4 and cov.shape[1:] == (T + 1, z_dim, z_dim):
        return cov, False

    if cov.ndim == 2 and cov.shape == (Z_dim, Z_dim):
        blocks = []
        for t in range(T + 1):
            row_start = t * z_dim
            row_end = (t + 1) * z_dim
            blocks.append(cov[row_start:row_end, row_start:row_end])
        return torch.stack(blocks, dim=0), True

    if cov.ndim == 3 and cov.shape[1:] == (Z_dim, Z_dim):
        blocks = []
        for t in range(T + 1):
            row_start = t * z_dim
            row_end = (t + 1) * z_dim
            blocks.append(cov[:, row_start:row_end, row_start:row_end])
        return torch.stack(blocks, dim=1), False

    raise ValueError(
        f"cov should have shape [{T + 1}, {z_dim}, {z_dim}], "
        f"[B, {T + 1}, {z_dim}, {z_dim}], [{Z_dim}, {Z_dim}], "
        f"or [B, {Z_dim}, {Z_dim}], but got {cov.shape}"
    )


def _make_block_spd(cov_blocks, eps=1e-4):
    """
    Stabilize block covariance matrices.

    Accepted input shapes:
        [T + 1, z_dim, z_dim]
        [B, T + 1, z_dim, z_dim]
    """
    if cov_blocks.ndim == 3:
        _, z_dim, _ = cov_blocks.shape
        eye = torch.eye(z_dim, device=cov_blocks.device, dtype=cov_blocks.dtype)
        cov_blocks = 0.5 * (cov_blocks + cov_blocks.transpose(-1, -2))
        return cov_blocks + eps * eye.unsqueeze(0)

    if cov_blocks.ndim == 4:
        _, _, z_dim, _ = cov_blocks.shape
        eye = torch.eye(z_dim, device=cov_blocks.device, dtype=cov_blocks.dtype)
        cov_blocks = 0.5 * (cov_blocks + cov_blocks.transpose(-1, -2))
        return cov_blocks + eps * eye.view(1, 1, z_dim, z_dim)

    raise ValueError(
        f"cov_blocks should have shape [T + 1, z_dim, z_dim] or "
        f"[B, T + 1, z_dim, z_dim], but got {cov_blocks.shape}"
    )


def reparameterize_block_diag(mu, cov_blocks, T, z_dim, eps=1e-4):
    """
    Reparameterization for block-diagonal Gaussian covariance.

    Accepted input shapes:
        mu.shape         == [(T + 1) * z_dim]
        mu.shape         == [T + 1, z_dim]
        mu.shape         == [B, (T + 1) * z_dim]
        mu.shape         == [B, T + 1, z_dim]
        cov_blocks.shape == [T + 1, z_dim, z_dim]
        cov_blocks.shape == [B, T + 1, z_dim, z_dim]

    Return:
        If input is a single sample:
            Z_vec.shape == [(T + 1) * z_dim]
        If input is a batch:
            Z_vec.shape == [B, (T + 1) * z_dim]
    """
    mu_mat, mu_single = _mu_to_mat(mu, T, z_dim)
    cov_blocks, cov_single = _cov_to_block(cov_blocks, T, z_dim)

    if mu_single != cov_single:
        raise ValueError(
            f"mu and cov_blocks should both be single-sample or both be batched, "
            f"but got mu_single={mu_single}, cov_single={cov_single}"
        )

    cov_blocks = _make_block_spd(cov_blocks, eps=eps)
    L = torch.linalg.cholesky(cov_blocks)

    if mu_single:
        eps_sample = torch.randn_like(mu_mat).unsqueeze(-1)
        Z_mat = mu_mat + torch.matmul(L, eps_sample).squeeze(-1)
        return Z_mat.reshape(-1)

    eps_sample = torch.randn_like(mu_mat).unsqueeze(-1)
    Z_mat = mu_mat + torch.matmul(L, eps_sample).squeeze(-1)
    return Z_mat.reshape(mu_mat.shape[0], (T + 1) * z_dim)


def kl_divergence_block_diag_gaussian(mu_q, cov_q, mu_p, cov_p, T, z_dim, eps=1e-4):
    """
    Compute KL(q || p) for block-diagonal multivariate Gaussian distributions.

    Accepted input shapes:
        mu_q, mu_p:
            [(T + 1) * z_dim], [T + 1, z_dim],
            [B, (T + 1) * z_dim], or [B, T + 1, z_dim]
        cov_q, cov_p:
            [T + 1, z_dim, z_dim] or [B, T + 1, z_dim, z_dim]

    Return:
        If input is a single sample:
            scalar KL divergence summed over time blocks.
        If input is a batch:
            scalar KL divergence averaged over batch samples.
    """
    mu_q, mu_q_single = _mu_to_mat(mu_q, T, z_dim)
    mu_p, mu_p_single = _mu_to_mat(mu_p, T, z_dim)
    cov_q, cov_q_single = _cov_to_block(cov_q, T, z_dim)
    cov_p, cov_p_single = _cov_to_block(cov_p, T, z_dim)

    single_flags = {mu_q_single, mu_p_single, cov_q_single, cov_p_single}
    if len(single_flags) != 1:
        raise ValueError(
            "mu_q, mu_p, cov_q, and cov_p should all be single-sample or all be batched."
        )
    single_input = mu_q_single

    cov_q = _make_block_spd(cov_q, eps=eps)
    cov_p = _make_block_spd(cov_p, eps=eps)

    L_q = torch.linalg.cholesky(cov_q)
    L_p = torch.linalg.cholesky(cov_p)

    logdet_q = 2.0 * torch.sum(
        torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1)),
        dim=-1,
    )
    logdet_p = 2.0 * torch.sum(
        torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1)),
        dim=-1,
    )

    cov_p_inv_cov_q = torch.cholesky_solve(cov_q, L_p)
    trace_term = torch.diagonal(cov_p_inv_cov_q, dim1=-2, dim2=-1).sum(dim=-1)

    diff = (mu_p - mu_q).unsqueeze(-1)
    cov_p_inv_diff = torch.cholesky_solve(diff, L_p)
    mahalanobis_term = torch.matmul(
        diff.transpose(-1, -2),
        cov_p_inv_diff,
    ).squeeze(-1).squeeze(-1)

    kl_per_block = 0.5 * (
        logdet_p
        - logdet_q
        - z_dim
        + trace_term
        + mahalanobis_term
    )

    if single_input:
        kl = kl_per_block.sum()
    else:
        kl = kl_per_block.sum(dim=1).mean()

    return torch.clamp(kl, min=0.0)