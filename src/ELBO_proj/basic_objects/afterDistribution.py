import torch
import torch.nn as nn
from basic_objects.convarianceNet import initCovarianceNet


class afterDistirbution(nn.Module):
    def __init__(self, z_dim, u_dim, x_dim, h_dim, embed_dim, T):
        super().__init__()
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.T = T
        self.Z_dim = (T + 1) * z_dim
        self.U_dim = x_dim + T * u_dim
        self.X_dim = T * x_dim
        self.process_noise_scale = 1e-4

        # mu, cov, and I_seq are constructed in forward(), so that they are
        # placed on the same device and use the same dtype as the input data.

        self.initCov = initCovarianceNet(embed_dim, z_dim)

        self.lstm = nn.LSTM(
            input_size=x_dim + u_dim,
            hidden_size=h_dim,
            batch_first=False,
            bidirectional=True,  # 表明这是双向 LSTM
        )

        self.proj = nn.Linear(2 * h_dim, embed_dim)
        self.mu_net = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, z_dim)
        )

    def forward(self, X_j, U_j, nn_A):
        I_seq = self.construct_I_seq(X_j, U_j)

        # nn.LSTM returns (output, (h_n, c_n)). Only output is needed here.
        # batch_first=False requires input shape [seq_len, batch_size, input_size].
        # I_seq has shape [T + 1, x_dim + u_dim], so we add the batch dimension at dim=1.
        lstm_out, _ = self.lstm(I_seq.unsqueeze(1))
        I_hat = torch.tanh(self.proj(lstm_out.squeeze(1)))

        # mu_t has shape [T + 1, z_dim], and mu has shape [(T + 1) * z_dim].
        mu_t = self.mu_net(I_hat)
        mu = mu_t.reshape(-1)

        # Use the embedding of the initial time step to construct Sigma_00.
        Sigma_00 = self.initCov(I_hat[0])
        cov = self.fullfill_cov_blocks(nn_A.weight, Sigma_00)

        return mu, cov

    def construct_I_seq(self, X_j, U_j):
        I_seq = X_j.new_zeros(self.T + 1, self.x_dim + self.u_dim)

        # U_j = [x_0, u_0, u_1, ..., u_{T-1}]
        x0 = U_j[0 : self.x_dim]
        U_flat = U_j[self.x_dim : self.x_dim + self.T * self.u_dim]
        U_mat = U_flat.reshape(self.T, self.u_dim)

        # X_j = [x_1, x_2, ..., x_T]
        X_mat = X_j.reshape(self.T, self.x_dim)

        I_seq[0, 0 : self.x_dim] = x0
        I_seq[0, self.x_dim : self.x_dim + self.u_dim] = U_mat[0]

        for t in range(1, self.T):
            I_seq[t, 0 : self.x_dim] = X_mat[t - 1]
            I_seq[t, self.x_dim : self.x_dim + self.u_dim] = U_mat[t]

        I_seq[self.T, 0 : self.x_dim] = X_mat[self.T - 1]
        # The last input is unknown because the data only contains u_0 to u_{T-1}.
        # Therefore the last u slot remains zero.

        return I_seq

    """_函数说明_
        计算后验分布中隐变量 Z = [z_0, z_1, ..., z_T] 的块协方差矩阵。
        当前假设隐空间动力学满足：
            z_{t+1} = A z_t + w_t,   w_t ~ N(0, Q)
        其中 Q = process_noise_scale * I。

        因此主对角块递推为：
            Sigma_{0,0} = Sigma_00
            Sigma_{t,t} = A Sigma_{t-1,t-1} A^T + Q,  t >= 1

        非对角块递推为：
            Sigma_{i,j} = Sigma_{i,j-1} A^T,          j > i
            Sigma_{j,i} = Sigma_{i,j}^T
        其中过程噪声只影响其注入时刻之后的主对角块，并通过后续 A 传播到更晚时刻的互协方差。
    """

    def fullfill_cov_blocks(self, A, Sigma_00):
        cov = Sigma_00.new_zeros(self.Z_dim, self.Z_dim)
        Q = self.process_noise_scale * torch.eye(
            self.z_dim,
            device=Sigma_00.device,
            dtype=Sigma_00.dtype,
        )

        diag_blocks = []
        Sigma_tt = Sigma_00
        diag_blocks.append(Sigma_tt)

        for t in range(1, self.T + 1):
            Sigma_tt = A @ Sigma_tt @ A.T + Q
            Sigma_tt = 0.5 * (Sigma_tt + Sigma_tt.T)
            diag_blocks.append(Sigma_tt)

        for i in range(0, self.T + 1):
            row_i_start = i * self.z_dim
            row_i_end = (i + 1) * self.z_dim

            Sigma_ii = diag_blocks[i]
            cov[row_i_start:row_i_end, row_i_start:row_i_end] = Sigma_ii

            Sigma_ij = Sigma_ii
            for j in range(i + 1, self.T + 1):
                Sigma_ij = Sigma_ij @ A.T

                col_j_start = j * self.z_dim
                col_j_end = (j + 1) * self.z_dim

                cov[row_i_start:row_i_end, col_j_start:col_j_end] = Sigma_ij
                cov[col_j_start:col_j_end, row_i_start:row_i_end] = Sigma_ij.T

        cov = 0.5 * (cov + cov.T)
        return cov
