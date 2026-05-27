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
        self.process_noise_scale = 1e-4
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
        计算先验分布中隐变量 Z = [z_0, z_1, ..., z_T] 的块对角协方差。
        当前函数不再构造完整的 [(T + 1) * z_dim, (T + 1) * z_dim] 协方差矩阵，
        而是只返回每个时间步的主对角协方差块：
            cov_blocks.shape = [T + 1, z_dim, z_dim]

        隐空间协方差递推关系为：
            Sigma_{0,0} = Sigma_00
            Sigma_{t,t} = A Sigma_{t-1,t-1} A^T + Q,  t >= 1
        其中：
            Q = process_noise_scale * I

        该结构对应块对角协方差近似：
            Cov(z_i, z_j) = 0, i != j
            Cov(z_t, z_t) = Sigma_{t,t}
    """
    def fullfill_cov_blocks(self, A, Sigma_00):
        cov_blocks = []
        Q = self.process_noise_scale * torch.eye(
            self.z_dim,
            device=Sigma_00.device,
            dtype=Sigma_00.dtype,
        )

        Sigma_tt = 0.5 * (Sigma_00 + Sigma_00.T)
        cov_blocks.append(Sigma_tt)

        for _ in range(1, self.T + 1):
            Sigma_tt = A @ Sigma_tt @ A.T + Q
            Sigma_tt = 0.5 * (Sigma_tt + Sigma_tt.T)
            cov_blocks.append(Sigma_tt)

        return torch.stack(cov_blocks, dim=0)
                
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