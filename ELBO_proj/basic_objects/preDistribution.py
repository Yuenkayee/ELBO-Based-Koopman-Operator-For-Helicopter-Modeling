import torch
import torch.nn as nn

class ConditionalPN(nn.Module):
    def __init__(self, z_dim, u_dim, x_dim, T):
        super().__init__()
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.T = T
        self.Z_dim = (T + 1) * z_dim
        self.U_dim = x_dim + T * u_dim
        self.mu = torch.zeros(self.Z_dim)
        self.cov = torch.zeros(self.Z_dim, self.Z_dim)
        self.net_logvar_z0 = nn.Sequential(
            nn.Linear(x_dim, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim)
            )
    
    def forward(self, Z_j, U_j, nn_Wc, nn_A, nn_B):
        ut_ = U_j[0 : self.x_dim - 1]
        logvar_z0 = self.net_logvar_z0(ut_)
        Sigma_z0 = torch.diag(torch.exp(logvar_z0))
        self.fullfill_diag_cov_blocks(nn_A.weight , Sigma_z0)
        self.fullfill_mu(self, Z_j, U_j, nn_Wc, nn_A, nn_B)
        return self.mu, self.cov

    """_函数说明_
        这计算先验分布中, 隐变量 Z 的分布协方差矩阵的主对角线上的矩阵块的函数，考虑到 Z 的各子项之间存在强相关性,
    因此协方差矩阵除了次对角线矩阵块不为零外, 主对角线的矩阵块还满足分布 Sigma_tt = A * Sigma_tt * A^T
    这里的矩阵 A 是一个待训练的参数，通过前面的网络获得, Sigma 是 z_t 的方差矩阵, 不含对数部分
    """
    def fullfill_diag_cov_blocks(self,A,Sigma_00):
        self.cov[0:self.z_dim - 1, 0:self.z_dim - 1] = Sigma_00
        Sigma_ii = Sigma_00
        for i in range(1, self.T):
            Sigma_ii = A @ Sigma_ii @ A.T
            self.cov[i * self.z_dim:(i + 1) * self.z_dim - 1,i * self.z_dim:(i + 1) * self.z_dim - 1] = Sigma_ii
            Sigma_ij = Sigma_ii
            for j in range(i + 1, self.T + 1):
                Sigma_ij = Sigma_ij @ A.T
                row_i_start = i * self.z_dim
                row_i_end = (i + 1) * self.z_dim
                col_j_start = j * self.z_dim
                col_j_end = (j + 1) * self.z_dim
                # Sigma_ij = Cov(z_i, z_j)
                self.cov[row_i_start:row_i_end, col_j_start:col_j_end] = Sigma_ij
                # Sigma_ji = Cov(z_j, z_i) = Sigma_ij.T
                self.cov[col_j_start:col_j_end, row_i_start:row_i_end] = Sigma_ij.T
                
    def fullfill_mu(self, Z_j, U_j, nn_Wc, nn_A, nn_B):
        for t in range(0, self.T + 1):
            if t == 0:
                ut_ = U_j[0 : self.x_dim - 1]
                self.mu[0 : self.z_dim - 1] = nn_Wc(ut_)
            else:
                ut_ = U_j[(t - 1) * self.u_dim + self.x_dim: t * self.u_dim + self.x_dim - 1]
                zt_ = Z_j[(t - 1) * self.z_dim : t * self.z_dim - 1]
                self.mu[t * self.z_dim :(t + 1) * self.z_dim - 1] = nn_A(zt_) + nn_B(ut_)