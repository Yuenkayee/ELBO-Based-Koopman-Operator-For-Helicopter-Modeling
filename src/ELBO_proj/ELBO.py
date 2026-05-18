import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from basic_objects.dataReading import load_matlab_simulation_data
from basic_objects.afterDistribution import afterDistirbution
from basic_objects.preDistribution import ConditionalPN
from basic_objects.basicFunctions import reparameterize_full_cov
from basic_objects.basicFunctions import decoder
from basic_objects.basicFunctions import kl_divergence_gaussian


class ELBO(nn.Module):
    def __init__(self, x_dim, u_dim, z_dim, h_dim, embed_dim, T, para_mu, para_lambda):
        super().__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.embed_dim = embed_dim
        self.T = T
        self.para_mu = para_mu
        self.para_lambda = para_lambda

        self.nn_A = nn.Linear(z_dim, z_dim, bias=False)  # 矩阵 A
        self.nn_B = nn.Linear(u_dim, z_dim, bias=False)  # 矩阵 B
        self.nn_C = nn.Linear(z_dim, x_dim, bias=False)  # 矩阵 C
        self.nn_Wc = nn.Linear(x_dim, z_dim, bias=False)  # 矩阵 C 的逆矩阵
        self.nn_pre = ConditionalPN(z_dim, u_dim, x_dim, T)  # 先验网络
        self.nn_after = afterDistirbution(
            z_dim, u_dim, x_dim, h_dim, embed_dim, T
        )  # 后验网络

    """ 
            训练用数据集 X_seq 和 U_seq 定义如下：
        X_seq: 
            X_seq = {X_0, X_1, ..., X_{S - 1}}
            其中, X_i = {x_1, ..., x_{T - 1}, x_T}
            因此, X_seq.shape = [S, T * x_dim] 
        U_seq:
            U_seq = {U_0, U_1, ..., U_{S - 1}}
            其中, U_i = {x_0, u_0, u_1, ..., u_{T - 1}}
            因此, U_seq.shape = [S, x_dim + T * u_dim]
    """

    def forward(self, X_seq, U_seq):
        S, _ = X_seq.shape
        loss = X_seq.new_tensor(0.0)

        for j in range(S):
            X_j = X_seq[j]
            U_j = U_seq[j]

            mu_after, cov_after = self.nn_after(X_j, U_j, self.nn_A)
            Z_j = reparameterize_full_cov(mu_after, cov_after, self.z_dim, self.T)
            mu_pre, cov_pre = self.nn_pre(Z_j, U_j, self.nn_Wc, self.nn_A, self.nn_B)
            X_hat_j = decoder(mu_pre, self.nn_C, self.T, self.x_dim, self.z_dim)

            eye_x = torch.eye(self.x_dim, device=X_seq.device, dtype=X_seq.dtype)
            reconstruction_loss = F.mse_loss(X_hat_j, X_j)
            inverse_loss = torch.mean((self.nn_C.weight @ self.nn_Wc.weight - eye_x) ** 2)
            kl_loss = kl_divergence_gaussian(mu_after, cov_after, mu_pre, cov_pre)

            loss = loss + reconstruction_loss + self.para_mu * inverse_loss + self.para_lambda * kl_loss

        loss = loss / S

        return {
            "A": self.nn_A.weight,
            "B": self.nn_B.weight,
            "C": self.nn_C.weight,
            "loss": loss,
        }


def train_elbo(
    model,
    trainData,
    num_epochs=1000,
    lr=5e-4,
    device="cpu",
):
    """_summary_

    Args:
        model (_type_): ELBO model
        trainData (_type_): X_seq and U_seq
        num_epochs (int, optional): Defaults to 1000.
        lr (_type_, optional): Defaults to 5e-4.
        device (str, optional): Defaults to None. If None, automatically selects cuda, mps, or cpu.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        X_seq = trainData.X_seq
        U_seq = trainData.U_seq
        X_seq = X_seq.to(device)
        U_seq = U_seq.to(device)

        optimizer.zero_grad()
        out = model(X_seq, U_seq)

        loss = out["loss"]
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1:04d}/{num_epochs:04d}] " f"Loss: {loss.item():.6f}")
    return model


def save_elbo_train_result(model, file_path=None):
    """
    Save the trained weight matrices A, B, and C of an ELBO model to a .mat file.

    The default output path is:
        ./data/trainResult.mat

    MATLAB can read the saved matrices using:
        load('data/trainResult.mat')

    Saved variables:
        A: shape [z_dim, z_dim]
        B: shape [z_dim, u_dim]
        C: shape [x_dim, z_dim]
    """
    if file_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "data", "trainResult.mat")

    output_dir = os.path.dirname(file_path)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    model_cpu = model.to("cpu")
    model_cpu.eval()

    train_result = {
        "A": model_cpu.nn_A.weight.detach().cpu().numpy(),
        "B": model_cpu.nn_B.weight.detach().cpu().numpy(),
        "C": model_cpu.nn_C.weight.detach().cpu().numpy(),
    }

    sio.savemat(file_path, train_result)

    return file_path
