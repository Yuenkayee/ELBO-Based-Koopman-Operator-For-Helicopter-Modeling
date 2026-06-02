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
        self.process_noise_scale = 1e-3
        # mu and cov are constructed in forward(), so that they are placed on
        # the same device and use the same dtype as the input data.
        self.net_Sigma_00 = initCovarianceNet(x_dim, z_dim)
    
    def forward(self, Z_j, U_j, nn_Wc, nn_A, nn_B):
        single_input = False
        if Z_j.ndim == 1:
            Z_j = Z_j.unsqueeze(0)
            U_j = U_j.unsqueeze(0)
            single_input = True
        elif Z_j.ndim == 2:
            if Z_j.shape == (self.T + 1, self.z_dim):
                Z_j = Z_j.reshape(1, self.Z_dim)
                U_j = U_j.unsqueeze(0)
                single_input = True
        else:
            raise ValueError(f"Z_j should be 1D or 2D, but got shape {Z_j.shape}")

        if Z_j.ndim != 2:
            raise ValueError(f"Z_j should be 2D after possible reshape, but got shape {Z_j.shape}")
        if U_j.ndim != 2:
            raise ValueError(f"U_j should be 2D after possible unsqueeze, but got shape {U_j.shape}")
        if Z_j.shape[0] != U_j.shape[0]:
            raise ValueError(
                f"Z_j and U_j should have the same batch size, "
                f"but got {Z_j.shape[0]} and {U_j.shape[0]}"
            )
        if Z_j.shape[1] != self.Z_dim:
            raise ValueError(
                f"Z_j should have shape [B, {self.Z_dim}], but got {Z_j.shape}"
            )

        # U_j = [x_0, u_0, u_1, ..., u_{T-1}]
        x0 = U_j[:, 0 : self.x_dim]

        Sigma_00 = self.net_Sigma_00(x0)
        cov = self.fullfill_cov_blocks(nn_A.weight, Sigma_00)
        mu = self.fullfill_mu(Z_j, U_j, nn_Wc, nn_A, nn_B)

        if single_input:
            return mu.squeeze(0), cov.squeeze(0)

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
        """
        Input:
            A.shape == [z_dim, z_dim]
            Sigma_00.shape == [z_dim, z_dim] or [B, z_dim, z_dim]

        Output:
            If Sigma_00 is single-sample:
                cov_blocks.shape == [T + 1, z_dim, z_dim]
            If Sigma_00 is batched:
                cov_blocks.shape == [B, T + 1, z_dim, z_dim]
        """
        single_input = False
        if Sigma_00.ndim == 2:
            Sigma_00 = Sigma_00.unsqueeze(0)
            single_input = True
        elif Sigma_00.ndim != 3:
            raise ValueError(
                f"Sigma_00 should have shape [z_dim, z_dim] or [B, z_dim, z_dim], "
                f"but got {Sigma_00.shape}"
            )

        B = Sigma_00.shape[0]
        cov_blocks = []
        Q = self.process_noise_scale * torch.eye(
            self.z_dim,
            device=Sigma_00.device,
            dtype=Sigma_00.dtype,
        ).view(1, self.z_dim, self.z_dim)

        Sigma_tt = 0.5 * (Sigma_00 + Sigma_00.transpose(-1, -2))
        cov_blocks.append(Sigma_tt)

        A_expand = A.unsqueeze(0).expand(B, -1, -1)
        A_T_expand = A.T.unsqueeze(0).expand(B, -1, -1)

        for _ in range(1, self.T + 1):
            Sigma_tt = A_expand @ Sigma_tt @ A_T_expand + Q
            Sigma_tt = 0.5 * (Sigma_tt + Sigma_tt.transpose(-1, -2))
            cov_blocks.append(Sigma_tt)

        cov_blocks = torch.stack(cov_blocks, dim=1)

        if single_input:
            return cov_blocks.squeeze(0)

        return cov_blocks
                
    def fullfill_mu(self, Z_j, U_j, nn_Wc, nn_A, nn_B):
        """
        Input:
            Z_j.shape == [B, (T + 1) * z_dim]
            U_j.shape == [B, x_dim + T * u_dim]

        Output:
            mu.shape == [B, (T + 1) * z_dim]
        """
        B = Z_j.shape[0]
        mu_mat = Z_j.new_zeros(B, self.T + 1, self.z_dim)

        Z_mat = Z_j.reshape(B, self.T + 1, self.z_dim)

        x0 = U_j[:, 0 : self.x_dim]
        U_flat = U_j[:, self.x_dim : self.x_dim + self.T * self.u_dim]
        U_mat = U_flat.reshape(B, self.T, self.u_dim)

        # Prior mean of z_0 is encoded from x_0.
        mu_mat[:, 0, :] = nn_Wc(x0)

        # Prior mean of z_t is propagated from z_{t-1} and u_{t-1}.
        for t in range(1, self.T + 1):
            z_prev = Z_mat[:, t - 1, :]
            u_prev = U_mat[:, t - 1, :]
            mu_mat[:, t, :] = nn_A(z_prev) + nn_B(u_prev)

        return mu_mat.reshape(B, self.Z_dim)