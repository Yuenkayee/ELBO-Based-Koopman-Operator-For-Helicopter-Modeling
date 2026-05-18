import torch
import torch.nn as nn
from basic_objects.convarianceNet import initCovarianceNet


class ConditionalPN(nn.Module):
    def __init__(self, z_dim, u_dim, x_dim, T):
        super().__init__()
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.T = T
        self.Z_dim = (T + 1) * z_dim
        self.U_dim = x_dim + T * u_dim
        # mu and cov are constructed in forward(), so that they are placed on
        # the same device and use the same dtype as the input data.
        self.net_Sigma_00 = initCovarianceNet(x_dim, z_dim)
    
    def forward(self, Z_j, U_j, nn_Wc, nn_A, nn_B):
        # U_j = [x_0, u_0, u_1, ..., u_{T-1}]
        x0 = U_j[0 : self.x_dim]

        Sigma_00 = self.net_Sigma_00(x0)
        cov = self.fullfill_cov_blocks(nn_A.weight, Sigma_00)
        mu = self.fullfill_mu(Z_j, U_j, nn_Wc, nn_A, nn_B)

        return mu, cov

    """_函数说明_
        这计算先验分布中, 隐变量 Z 的由矩阵块构成的协方差矩阵，考虑到 Z 的各子项z_t之间存在强相关性,
    因此协方差矩阵除了次对角线矩阵块不为零外, 主对角线的矩阵块还满足分布 Sigma_tt = A * Sigma_tt * A^T
    这里的矩阵 A 是一个待训练的参数，通过前面的网络获得, Sigma 是 z_t 的方差矩阵, 不含对数部分
    """
    def fullfill_cov_blocks(self, A, Sigma_00):
        cov = Sigma_00.new_zeros(self.Z_dim, self.Z_dim)

        Sigma_ii = Sigma_00

        for i in range(0, self.T + 1):
            if i == 0:
                Sigma_ii = Sigma_00
            else:
                Sigma_ii = A @ Sigma_ii @ A.T

            row_i_start = i * self.z_dim
            row_i_end = (i + 1) * self.z_dim
            cov[row_i_start:row_i_end, row_i_start:row_i_end] = Sigma_ii

            Sigma_ij = Sigma_ii
            for j in range(i + 1, self.T + 1):
                Sigma_ij = Sigma_ij @ A.T

                col_j_start = j * self.z_dim
                col_j_end = (j + 1) * self.z_dim

                # Sigma_ij = Cov(z_i, z_j)
                cov[row_i_start:row_i_end, col_j_start:col_j_end] = Sigma_ij
                # Sigma_ji = Cov(z_j, z_i) = Sigma_ij.T
                cov[col_j_start:col_j_end, row_i_start:row_i_end] = Sigma_ij.T

        return cov
                
    def fullfill_mu(self, Z_j, U_j, nn_Wc, nn_A, nn_B):
        mu = Z_j.new_zeros(self.Z_dim)

        if Z_j.ndim == 1:
            Z_mat = Z_j.reshape(self.T + 1, self.z_dim)
        elif Z_j.ndim == 2:
            Z_mat = Z_j
        else:
            raise ValueError(f"Z_j must be a 1D or 2D tensor, but got shape {Z_j.shape}")

        if Z_mat.shape != (self.T + 1, self.z_dim):
            raise ValueError(
                f"Z_j should have shape [{self.T + 1}, {self.z_dim}] after reshape, "
                f"but got {Z_mat.shape}"
            )

        x0 = U_j[0 : self.x_dim]
        U_flat = U_j[self.x_dim : self.x_dim + self.T * self.u_dim]
        U_mat = U_flat.reshape(self.T, self.u_dim)

        # Prior mean of z_0 is encoded from x_0.
        mu[0 : self.z_dim] = nn_Wc(x0)

        # Prior mean of z_t is propagated from z_{t-1} and u_{t-1}.
        for t in range(1, self.T + 1):
            z_prev = Z_mat[t - 1]
            u_prev = U_mat[t - 1]
            mu[t * self.z_dim : (t + 1) * self.z_dim] = nn_A(z_prev) + nn_B(u_prev)

        return mu