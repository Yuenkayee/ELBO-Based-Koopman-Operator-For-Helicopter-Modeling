import torch
import torch.nn as nn
import torch.nn.functional as F

"""_功能介绍_
    这是一个通过指定输入 u0 (或者 I_hat_0) 来构造潜变量协方差矩阵 Sigma_00 的类
    协方差矩阵 Sigma_00 通过对下三角矩阵 L 执行 L @ L.T 得到
    Inputs:
            u0.shape = [1, u_dim] (或者可以输入多个 batch 的 u_dim)
            ** 注意, 针对 u0.shape = u_dim 的情况，可以在外部使用
                u0 = u0.unsqueeze(0) 对 u_0 的维度进行扩展，然后再使用 CovarianceNet(u0)
    Returns:
            Sigma_00.shape = [batch, dim_z, dim_z]
            ** 注意, 如果只需要 shape = [dim_z, dim_z], 可以在外部使用
                Sigma_00 = Sigma_00[0]  # [dim_z, dim_z]
"""

class initCovarianceNet(nn.Module):
    def __init__(self, dim_u, dim_z, hidden_dim=64):
        super().__init__()

        self.dim_z = dim_z

        # 输出 dim_z * dim_z 个数，用来组成 raw_L
        self.net = nn.Sequential(
            nn.Linear(dim_u, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim_z * dim_z)
        )

    def forward(self, u0):
        """
        u0.shape = [batch, dim_u]

        return:
            Sigma_00.shape = [batch, dim_z, dim_z]
        """

        batch_size = u0.shape[0]

        raw_L = self.net(u0)  # [batch, dim_z * dim_z]
        raw_L = raw_L.view(batch_size, self.dim_z, self.dim_z)

        # 取下三角部分
        L = torch.tril(raw_L)

        # 保证对角线为正数
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        diag_positive = F.softplus(diag) + 1e-6

        L = L.clone()
        idx = torch.arange(self.dim_z, device=u0.device)
        L[:, idx, idx] = diag_positive

        # 构造协方差矩阵
        Sigma_00 = L @ L.transpose(-1, -2)

        return Sigma_00