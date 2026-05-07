import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from basicParts_dkin.basicFunctions import reparameterize, gaussian_kl
from basicParts_dkin.TemporalEncoder import TemporalEncoder
from basicParts_dkin.ObservationGenerator import ObservationGenerator
from basicParts_dkin.KoopmanLayer import KoopmanLayer
from basicParts_dkin.ConditionalPN import ConditionalPrior
from basicParts_dkin.Decoder import Decoder

# ============================================================
# 7. Full DKIN Model
# ============================================================

class DKIN(nn.Module):
    """
    Full DKIN model.

    Input:
        x_seq: [batch, T, x_dim]
        u_seq: [batch, T-1, u_dim]

    Output:
        Dictionary containing:
            mu_seq
            h_seq
            z_seq
            A
            B
            KL loss
    """

    def __init__(
        self,
        x_dim,
        u_dim,
        h_dim,
        temporal_hidden_dim=64,
        temporal_embed_dim=64,
        obs_gru_hidden_dim=64
    ):
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.h_dim = h_dim

        self.temporal_encoder = TemporalEncoder(
            x_dim=x_dim,
            u_dim=u_dim,
            hidden_dim=temporal_hidden_dim,
            embed_dim=temporal_embed_dim
        )

        self.observation_generator = ObservationGenerator(
            embed_dim=temporal_embed_dim,
            h_dim=h_dim,
            gru_hidden_dim=obs_gru_hidden_dim
        )

        self.conditional_prior = ConditionalPrior(
            h_dim=h_dim,
            u_dim=u_dim
        )

        self.koopman_layer = KoopmanLayer(
            h_dim=h_dim,
            u_dim=u_dim
        )

        self.decoder = Decoder(
            h_dim=h_dim,
            x_dim=x_dim
        )

    def forward(self, x_seq, u_seq):
        batch_size, T, _ = x_seq.shape

        # ------------------------------------------------------
        # Step 1: Temporal encoding
        # ------------------------------------------------------
        I_hat = self.temporal_encoder(x_seq, u_seq)

        # ------------------------------------------------------
        # Step 2: Initial observation inference
        # ------------------------------------------------------
        mu_h1, logvar_h1 = self.observation_generator.initial_distribution(I_hat)
        h_1 = reparameterize(mu_h1, logvar_h1)

        h_list = [h_1]

        # Prior for h_1 is standard Gaussian N(0, I)
        mu_prior_h1 = torch.zeros_like(mu_h1)
        logvar_prior_h1 = torch.zeros_like(logvar_h1)

        kl_loss = gaussian_kl(
            mu_q=mu_h1,
            logvar_q=logvar_h1,
            mu_p=mu_prior_h1,
            logvar_p=logvar_prior_h1
        )

        # ------------------------------------------------------
        # Step 3: Sequential observation generation
        # ------------------------------------------------------
        for t in range(1, T):
            # torch.stack 是 PyTorch 中用于把多个形状相同的张量沿一个新维度堆叠起来的函数。
            # torch.stack 的输出要比原来的张量多一个维度，这里使用该函数是为了对齐 recognition_distribution 的接口
            h_history = torch.stack(h_list, dim=1)

            I_hat_t = I_hat[:, t, :]

            mu_q, logvar_q = self.observation_generator.recognition_distribution(
                I_hat_t=I_hat_t,
                h_history=h_history
            )

            h_prev = h_list[-1]
            u_prev = u_seq[:, t - 1, :]

            mu_p, logvar_p = self.conditional_prior(
                h_prev=h_prev,
                u_prev=u_prev
            )

            h_t = reparameterize(mu_q, logvar_q)

            h_list.append(h_t) # 在 h_list 末尾添加一个新的元素

            kl_loss = kl_loss + gaussian_kl(
                mu_q=mu_q,
                logvar_q=logvar_q,
                mu_p=mu_p,
                logvar_p=logvar_p
            )

        h_seq = torch.stack(h_list, dim=1)

        # ------------------------------------------------------
        # Step 4: Koopman layer
        # ------------------------------------------------------
        A, B = self.koopman_layer(h_seq, u_seq)

        # ------------------------------------------------------
        # Step 5: Backward latent rollout
        # ------------------------------------------------------
        z_seq = self.koopman_layer.backward_rollout(
            h_seq=h_seq,
            u_seq=u_seq,
            A=A,
            B=B
        )

        # ------------------------------------------------------
        # Step 6: Decoding
        # ------------------------------------------------------
        mu_seq = self.decoder(z_seq)

        return {
            "mu_seq": mu_seq,
            "h_seq": h_seq,
            "z_seq": z_seq,
            "A": A,
            "B": B,
            "kl_loss": kl_loss
        }


# ============================================================
# 8. Dataset wrapper
# ============================================================

class SequenceDataset(Dataset):
    """
    Dataset for DKIN.

    x_data shape:
        [num_samples, T, x_dim]

    u_data shape:
        [num_samples, T-1, u_dim]
    """

    def __init__(self, x_data, u_data):
        super().__init__()

        assert x_data.ndim == 3
        assert u_data.ndim == 3
        assert x_data.shape[0] == u_data.shape[0]
        assert x_data.shape[1] == u_data.shape[1] + 1

        self.x_data = torch.as_tensor(x_data, dtype=torch.float32)
        self.u_data = torch.as_tensor(u_data, dtype=torch.float32)

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return self.x_data[idx], self.u_data[idx]


# ============================================================
# 9. Training function
# ============================================================

def train_dkin(
    model,
    dataloader,
    num_epochs=100,
    lr=5e-4,
    kappa_1=1.2,
    kappa_2=1.0,
    omega_T=5.0,
    device="cpu"
):
    """
    Train DKIN model.

    Loss:
        L_sum = kappa_1 * L_KL + kappa_2 * L_Pred

    where

        L_Pred = sum_{t=1}^{T-1} ||mu_t - x_t||^2
                 + omega_T ||mu_T - x_T||^2
    """

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()

        total_loss_value = 0.0
        total_pred_loss_value = 0.0
        total_kl_loss_value = 0.0

        for x_seq, u_seq in dataloader:
            x_seq = x_seq.to(device)
            u_seq = u_seq.to(device)

            out = model(x_seq, u_seq)

            mu_seq = out["mu_seq"]
            kl_loss = out["kl_loss"]

            # Prediction loss
            pred_loss_main = F.mse_loss(
                mu_seq[:, :-1, :],
                x_seq[:, :-1, :],
                reduction="mean"
            )

            pred_loss_terminal = F.mse_loss(
                mu_seq[:, -1, :],
                x_seq[:, -1, :],
                reduction="mean"
            )

            pred_loss = pred_loss_main + omega_T * pred_loss_terminal

            loss = kappa_1 * kl_loss + kappa_2 * pred_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_value += loss.item()
            total_pred_loss_value += pred_loss.item()
            total_kl_loss_value += kl_loss.item()

        num_batches = len(dataloader)

        print(
            f"Epoch [{epoch + 1:04d}/{num_epochs:04d}] "
            f"Loss: {total_loss_value / num_batches:.6f} | "
            f"Pred: {total_pred_loss_value / num_batches:.6f} | "
            f"KL: {total_kl_loss_value / num_batches:.6f}"
        )

    return model