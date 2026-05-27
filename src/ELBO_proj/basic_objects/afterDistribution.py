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
        计算后验分布中隐变量 Z = [z_0, z_1, ..., z_T] 的块对角协方差。
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
