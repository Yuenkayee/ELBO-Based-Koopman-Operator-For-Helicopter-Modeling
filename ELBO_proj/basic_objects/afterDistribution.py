import torch
import torch.nn as nn
from convarianceNet import initCovarianceNet

class afterDistirbution(nn.Module):
    def __init__(self, z_dim, u_dim, x_dim,h_dim,embed_dim, T):
        super().__init__()
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.T = T
        self.Z_dim = (T + 1) * z_dim
        self.U_dim = x_dim + T * u_dim
        self.X_dim = T * x_dim
        
        self.mu = torch.zeros(self.Z_dim)
        self.cov = torch.zeros(self.Z_dim, self.Z_dim)
        
        self.I_seq = torch.zeros(T + 1, x_dim + u_dim)
        self.initCov = initCovarianceNet(embed_dim, z_dim)
        
        self.lstm = nn.LSTM(
            input_size=x_dim + u_dim,
            hidden_size=h_dim,
            batch_first=True,
            bidirectional=True # 表明这是双向 LSTM
        )
        
        self.proj = nn.Linear(2 * h_dim, embed_dim)
        self.mu_net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim)
        )
    
    def forward(self, X_j, U_j,nn_A):
        self.construct_I_seq(X_j, U_j)
        lstm_out = self.lstm(self.I_seq)
        I_hat = torch.tanh(self.proj(lstm_out))
        self.mu_net(I_hat)
        Sigma_00 = self.initCov(I_hat)
        self.fullfill_cov_blocks(nn_A.weight , Sigma_00)
        return self.mu, self.cov
      
    def construct_I_seq (self, X_j, U_j):
        self.I_seq[0][0 : self.x_dim - 1] = U_j[0 : self.x_dim - 1]
        self.I_seq[0][self.x_dim : self.x_dim + self.u_dim - 1] = U_j[self.x_dim : self.x_dim + self.u_dim - 1]
        for t in range(1, self.T):
            self.I_seq[t][0 : self.x_dim - 1] = X_j[(t - 1) * self.x_dim : t * self.x_dim - 1]
            self.I_seq[t][self.x_dim : self.x_dim + self.u_dim - 1] = U_j[(t - 1) * self.u_dim + self.x_dim : t * self.u_dim + self.x_dim - 1]
        self.I_seq[self.T][0 : self.x_dim - 1] = X_j[(self.T - 1) * self.x_dim : self.T * self.x_dim - 1]
        self.I_seq[self.T][self.x_dim : self.x_dim + self.u_dim - 1] = torch.zeros(self.u_dim)
        
    """_函数说明_
        这计算先验分布中, 隐变量 Z 的由矩阵块构成的协方差矩阵，考虑到 Z 的各子项z_t之间存在强相关性,
    因此协方差矩阵除了次对角线矩阵块不为零外, 主对角线的矩阵块还满足分布 Sigma_tt = A * Sigma_tt * A^T
    这里的矩阵 A 是一个待训练的参数，通过前面的网络获得, Sigma 是 z_t 的方差矩阵, 不含对数部分
    """
    def fullfill_cov_blocks(self,A,Sigma_00):
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