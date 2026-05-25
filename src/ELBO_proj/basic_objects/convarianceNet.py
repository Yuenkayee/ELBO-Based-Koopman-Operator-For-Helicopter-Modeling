import torch
import torch.nn as nn
import torch.nn.functional as F

"""_功能介绍_
    这是一个通过指定输入 u0 或者 I_hat_0 来构造潜变量初始协方差矩阵 Sigma_00 的类。
    当前版本将 Sigma_00 构造为对角矩阵，其主对角线元素由 nn.Sequential 网络输出。
    为了保证协方差矩阵半正定，网络输出会经过 softplus 变换，使对角元素为正数。

    Inputs:
            u0.shape = [dim_u]
            或者
            u0.shape = [batch, dim_u]

    Returns:
            如果输入 u0.shape = [dim_u]：
                Sigma_00.shape = [dim_z, dim_z]
            如果输入 u0.shape = [batch, dim_u]：
                Sigma_00.shape = [batch, dim_z, dim_z]
"""


class initCovarianceNet(nn.Module):
    def __init__(self, dim_u, dim_z, hidden_dim=64):
        super().__init__()

        self.dim_z = dim_z

        # 输出 dim_z 个数，用来组成 Sigma_00 的主对角线元素。
        self.net = nn.Sequential(
            nn.Linear(dim_u, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim_z),
        )

    def forward(self, u0):
        """
        Construct Sigma_00 from a single input vector or a batch of input vectors.

        Accepted input shapes:
            u0.shape == [dim_u]
            u0.shape == [batch, dim_u]

        Return shapes:
            if input shape is [dim_u]:
                Sigma_00.shape == [dim_z, dim_z]
            if input shape is [batch, dim_u]:
                Sigma_00.shape == [batch, dim_z, dim_z]
        """
        single_input = False
        if u0.ndim == 1:
            u0 = u0.unsqueeze(0)
            single_input = True
        elif u0.ndim != 2:
            raise ValueError(f"u0 must be a 1D or 2D tensor, but got shape {u0.shape}")

        batch_size = u0.shape[0]

        raw_diag = self.net(u0)  # [batch, dim_z]

        # 保证协方差矩阵的主对角线元素为正数。
        diag_positive = F.softplus(raw_diag) + 1e-3

        # 构造对角协方差矩阵。
        Sigma_00 = torch.diag_embed(diag_positive)  # [batch, dim_z, dim_z]

        if single_input:
            Sigma_00 = Sigma_00.squeeze(0)

        return Sigma_00
