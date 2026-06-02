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
        self.process_noise_scale = 1e-3

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
        single_input = False
        if X_j.ndim == 1:
            X_j = X_j.unsqueeze(0)
            U_j = U_j.unsqueeze(0)
            single_input = True
        elif X_j.ndim != 2:
            raise ValueError(f"X_j should be 1D or 2D, but got shape {X_j.shape}")

        if U_j.ndim != 2:
            raise ValueError(f"U_j should be 2D after possible unsqueeze, but got shape {U_j.shape}")
        if X_j.shape[0] != U_j.shape[0]:
            raise ValueError(
                f"X_j and U_j should have the same batch size, "
                f"but got {X_j.shape[0]} and {U_j.shape[0]}"
            )

        I_seq = self.construct_I_seq(X_j, U_j)

        # nn.LSTM returns (output, (h_n, c_n)). Only output is needed here.
        # batch_first=False requires input shape [seq_len, batch_size, input_size].
        # I_seq has shape [T + 1, B, x_dim + u_dim].
        lstm_out, _ = self.lstm(I_seq)
        I_hat = torch.tanh(self.proj(lstm_out))

        # mu_t has shape [T + 1, B, z_dim]. Convert it to [B, (T + 1) * z_dim].
        mu_t = self.mu_net(I_hat)
        mu = mu_t.permute(1, 0, 2).reshape(X_j.shape[0], self.Z_dim)

        # Use the embedding of the initial time step to construct Sigma_00.
        # I_hat[0].shape == [B, embed_dim], so Sigma_00.shape == [B, z_dim, z_dim].
        Sigma_00 = self.initCov(I_hat[0])
        cov = self.fullfill_cov_blocks(nn_A.weight, Sigma_00)

        if single_input:
            return mu.squeeze(0), cov.squeeze(0)

        return mu, cov

    def construct_I_seq(self, X_j, U_j):
        """
        Construct LSTM input sequence.

        Input:
            X_j.shape == [B, T * x_dim]
            U_j.shape == [B, x_dim + T * u_dim]

        Output:
            I_seq.shape == [T + 1, B, x_dim + u_dim]
        """
        B = X_j.shape[0]
        I_seq = X_j.new_zeros(self.T + 1, B, self.x_dim + self.u_dim)

        # U_j = [x_0, u_0, u_1, ..., u_{T-1}]
        x0 = U_j[:, 0 : self.x_dim]
        U_flat = U_j[:, self.x_dim : self.x_dim + self.T * self.u_dim]
        U_mat = U_flat.reshape(B, self.T, self.u_dim)

        # X_j = [x_1, x_2, ..., x_T]
        X_mat = X_j.reshape(B, self.T, self.x_dim)

        I_seq[0, :, 0 : self.x_dim] = x0
        I_seq[0, :, self.x_dim : self.x_dim + self.u_dim] = U_mat[:, 0, :]

        for t in range(1, self.T):
            I_seq[t, :, 0 : self.x_dim] = X_mat[:, t - 1, :]
            I_seq[t, :, self.x_dim : self.x_dim + self.u_dim] = U_mat[:, t, :]

        I_seq[self.T, :, 0 : self.x_dim] = X_mat[:, self.T - 1, :]
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
