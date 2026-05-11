import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_objects.data_reading import load_matlab_simulation_data

class ELBO(nn.Module):
    def __init__(self, x_dim, u_dim, z_dim, T):
        super().__init__()
        self.nn_Wc = nn.Linear(x_dim, z_dim, bias=False) # 矩阵 C 的逆矩阵
        self.nn_A = nn.Linear(z_dim, z_dim, bias=False)  # 矩阵 A
        self.nn_B = nn.Linear(u_dim, z_dim, bias=False)  # 矩阵 B