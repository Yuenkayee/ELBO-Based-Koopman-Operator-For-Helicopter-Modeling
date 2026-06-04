import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Basic utility functions
# ============================================================

def reparameterize(mu, logvar, deterministic=False):
    """
    Reparameterization trick:
        h = mu + std * eps
    where eps ~ N(0, I).

    If deterministic=True, return mu directly. This is useful when
    exporting the final Koopman matrices after training.
    """
    if deterministic:
        return mu

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    """
    KL divergence between two diagonal Gaussian distributions:

        q = N(mu_q, diag(var_q))
        p = N(mu_p, diag(var_p))

    Return:
        KL(q || p), averaged over batch.
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    kl = 0.5 * (
        logvar_p - logvar_q
        + (var_q + (mu_q - mu_p) ** 2) / var_p
        - 1.0
    )

    return kl.sum(dim=-1).mean()


# ============================================================
# 2. Temporal Encoder: bidirectional LSTM
# ============================================================

class TemporalEncoder(nn.Module):
    """
    Temporal encoder:
        input:  I_t = [x_t; u_t]
        output: temporal embedding \hat{I}_t

    Shape:
        x_seq: [batch, T, x_dim]
        u_seq: [batch, T-1, u_dim]

    Since u has length T-1, we pad the last control input with zero.
    """

    def __init__(self, x_dim, u_dim, hidden_dim, embed_dim):
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim

        self.lstm = nn.LSTM(
            input_size=x_dim + u_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.proj = nn.Linear(2 * hidden_dim, embed_dim)

    def forward(self, x_seq, u_seq):
        batch_size, T, _ = x_seq.shape

        # Pad u_T as zero so that u_seq_pad has length T
        u_pad = torch.zeros(
            batch_size, 1, self.u_dim,
            device=x_seq.device,
            dtype=x_seq.dtype
        )

        u_seq_pad = torch.cat([u_seq, u_pad], dim=1)

        I_seq = torch.cat([x_seq, u_seq_pad], dim=-1)

        lstm_out, _ = self.lstm(I_seq)

        I_hat = torch.tanh(self.proj(lstm_out))

        return I_hat


# ============================================================
# 3. Observation Generation Block
# ============================================================

class ObservationGenerator(nn.Module):
    """
    Observation generation block.

    It includes:

    1. Initial recognition network:
        q_phi(h_1 | x_{1:T}, u_{1:T-1})

    2. Observation encoder:
        summarizes h_{1:t-1} using GRU

    3. Observation recognition network:
        q_psi(h_t | h_{1:t-1}, x_{1:t}, u_{1:t-1})
    """

    def __init__(self, embed_dim, h_dim, gru_hidden_dim):
        super().__init__()

        self.h_dim = h_dim
        self.gru_hidden_dim = gru_hidden_dim

        # Initial recognition network
        self.init_net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * h_dim)
        )

        # GRU for summarizing previous observations h_{1:t-1}
        self.obs_gru = nn.GRU(
            input_size=h_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True
        )

        # Recognition network for h_t
        self.recognition_net = nn.Sequential(
            nn.Linear(embed_dim + gru_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * h_dim)
        )

    def initial_distribution(self, I_hat):
        """
        Compute q_phi(h_1 | x_{1:T}, u_{1:T-1})

        Here we use the first temporal embedding I_hat[:, 0, :].
        One may also use mean pooling or the final hidden state.
        """
        init_feat = I_hat[:, 0, :]
        out = self.init_net(init_feat)

        mu, logvar = torch.chunk(out, chunks=2, dim=-1)
        return mu, logvar

    def recognition_distribution(self, I_hat_t, h_history):
        """
        Compute q_psi(h_t | h_{1:t-1}, x_{1:t}, u_{1:t-1})

        Parameters:
            I_hat_t:   [batch, embed_dim]
            h_history: [batch, t-1, h_dim]

        Return:
            mu_t, logvar_t
        """

        _, hidden = self.obs_gru(h_history)

        # hidden: [1, batch, gru_hidden_dim]
        hidden = hidden[-1]

        feat = torch.cat([I_hat_t, hidden], dim=-1)

        out = self.recognition_net(feat)

        mu, logvar = torch.chunk(out, chunks=2, dim=-1)

        return mu, logvar


# ============================================================
# 4. Conditional Prior Network
# ============================================================

class ConditionalPrior(nn.Module):
    """
    Conditional prior network:

        p_epsilon(h_t | h_{t-1}, u_{t-1})

    It outputs the mean and log-variance of a Gaussian prior.
    """

    def __init__(self, h_dim, u_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(h_dim + u_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * h_dim)
        )

    def forward(self, h_prev, u_prev):
        inp = torch.cat([h_prev, u_prev], dim=-1)

        out = self.net(inp)

        mu, logvar = torch.chunk(out, chunks=2, dim=-1)

        return mu, logvar


# ============================================================
# 5. Koopman Layer
# ============================================================

class KoopmanLayer(nn.Module):
    """
    Koopman layer.

    Given sampled observations h_{1:T} and controls u_{1:T-1},
    compute A and B from:

        [A B] = Y Z^\dagger

    where

        Y = [h_2, ..., h_T]
        Z = [h_1, ..., h_{T-1};
             u_1, ..., u_{T-1}]

    For each batch sample, A and B are computed separately.
    """

    def __init__(self, h_dim, u_dim, reg=1e-6):
        super().__init__()

        self.h_dim = h_dim
        self.u_dim = u_dim
        self.reg = reg

    def forward(self, h_seq, u_seq):
        """
        Compute one shared Koopman model A, B using all samples
        in the current mini-batch.

        Parameters:
            h_seq: [batch, T, h_dim]
            u_seq: [batch, T-1, u_dim]

        Returns:
            A: [h_dim, h_dim]
            B: [h_dim, u_dim]
        """
        A, B = self.solve_shared_koopman(h_seq, u_seq)
        return A, B

    def solve_shared_koopman(self, h_seq, u_seq):
        """
        Solve [A B] = Y Z^dagger by pooling all samples and all
        time steps in the provided mini-batch.

        Parameters:
            h_seq: [batch, T, h_dim]
            u_seq: [batch, T-1, u_dim]

        Returns:
            A: [h_dim, h_dim]
            B: [h_dim, u_dim]
        """
        batch_size, T, h_dim = h_seq.shape
        _, T_minus_1, u_dim = u_seq.shape

        assert T_minus_1 == T - 1

        # h_past: [batch, T-1, h_dim]
        # h_next: [batch, T-1, h_dim]
        h_past = h_seq[:, :-1, :]
        h_next = h_seq[:, 1:, :]

        # Pool batch and time dimensions.
        # h_past_flat: [batch * (T-1), h_dim]
        # h_next_flat: [batch * (T-1), h_dim]
        # u_flat:      [batch * (T-1), u_dim]
        h_past_flat = h_past.reshape(batch_size * T_minus_1, h_dim)
        h_next_flat = h_next.reshape(batch_size * T_minus_1, h_dim)
        u_flat = u_seq.reshape(batch_size * T_minus_1, u_dim)

        # H_past: [h_dim, batch * (T-1)]
        # H_next: [h_dim, batch * (T-1)]
        # U:      [u_dim, batch * (T-1)]
        H_past = h_past_flat.T
        H_next = h_next_flat.T
        U = u_flat.T

        # Z: [h_dim + u_dim, batch * (T-1)]
        # Y: [h_dim, batch * (T-1)]
        Z = torch.cat([H_past, U], dim=0)
        Y = H_next

        Z_pinv = torch.linalg.pinv(Z)
        K = Y @ Z_pinv

        A = K[:, :h_dim]
        B = K[:, h_dim:]

        A = A + self.reg * torch.eye(
            h_dim,
            device=h_seq.device,
            dtype=h_seq.dtype
        )

        return A, B

    def backward_rollout(self, h_seq, u_seq, A, B):
        """
        Backward-time latent rollout using one shared A, B:

            z_T = h_T
            z_t = A^{-1}(z_{t+1} - B u_t)

        Parameters:
            h_seq: [batch, T, h_dim]
            u_seq: [batch, T-1, u_dim]
            A:     [h_dim, h_dim]
            B:     [h_dim, u_dim]

        Return:
            z_seq: [batch, T, h_dim]
        """
        batch_size, T, h_dim = h_seq.shape

        z_list = [None for _ in range(T)]

        z_T = h_seq[:, -1, :]
        z_list[T - 1] = z_T

        for t in range(T - 2, -1, -1):
            z_next = z_list[t + 1]
            u_t = u_seq[:, t, :]

            # u_t: [batch, u_dim]
            # B.T: [u_dim, h_dim]
            # Bu: [batch, h_dim]
            Bu = u_t @ B.T

            rhs = z_next - Bu

            # Solve A z_t^T = rhs^T instead of explicitly computing A^{-1}.
            # rhs.T: [h_dim, batch]
            # z_t.T: [h_dim, batch]
            z_t = torch.linalg.solve(A, rhs.T).T

            z_list[t] = z_t

        z_seq = torch.stack(z_list, dim=1)

        return z_seq


# ============================================================
# 6. Decoder
# ============================================================

class Decoder(nn.Module):
    """
    Decoder:

        mu_t = C_mu z_t

    According to the paper, the decoder can be a linear map
    without activation function.
    """

    def __init__(self, h_dim, x_dim):
        super().__init__()

        self.linear = nn.Linear(h_dim, x_dim)

    def forward(self, z_seq):
        """
        z_seq: [batch, T, h_dim]

        return:
            mu_seq: [batch, T, x_dim]
        """
        return self.linear(z_seq)

    def get_C_matrix(self):
        """
        Return the linear decoder matrix C_mu used in
            mu_t = C_mu z_t + bias.

        Shape:
            C_mu: [x_dim, h_dim]
        """
        return self.linear.weight.detach().cpu()

    def get_decoder_bias(self):
        """
        Return the decoder bias term.

        Shape:
            bias: [x_dim]
        """
        return self.linear.bias.detach().cpu()


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

    def forward(self, x_seq, u_seq, deterministic=False):
        batch_size, T, _ = x_seq.shape

        # ------------------------------------------------------
        # Step 1: Temporal encoding
        # ------------------------------------------------------
        I_hat = self.temporal_encoder(x_seq, u_seq)

        # ------------------------------------------------------
        # Step 2: Initial observation inference
        # ------------------------------------------------------
        mu_h1, logvar_h1 = self.observation_generator.initial_distribution(I_hat)
        h_1 = reparameterize(mu_h1, logvar_h1, deterministic=deterministic)

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

            h_t = reparameterize(mu_q, logvar_q, deterministic=deterministic)

            h_list.append(h_t)

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

    def get_C_matrix(self):
        """
        Return the decoder matrix C_mu.

        Shape:
            C_mu: [x_dim, h_dim]
        """
        return self.decoder.get_C_matrix()

    def get_decoder_bias(self):
        """
        Return the decoder bias term.

        Shape:
            bias: [x_dim]
        """
        return self.decoder.get_decoder_bias()


    @torch.no_grad()
    def infer_h_sequence(self, x_seq, u_seq, deterministic=True):
        """
        Infer the Koopman observation sequence h_{1:T} without decoding.

        Parameters:
            x_seq: [batch, T, x_dim]
            u_seq: [batch, T-1, u_dim]
            deterministic: if True, use the posterior mean instead of sampling.

        Return:
            h_seq: [batch, T, h_dim]
        """
        batch_size, T, _ = x_seq.shape

        I_hat = self.temporal_encoder(x_seq, u_seq)

        mu_h1, logvar_h1 = self.observation_generator.initial_distribution(I_hat)
        h_1 = reparameterize(mu_h1, logvar_h1, deterministic=deterministic)

        h_list = [h_1]

        for t in range(1, T):
            h_history = torch.stack(h_list, dim=1)
            I_hat_t = I_hat[:, t, :]

            mu_q, logvar_q = self.observation_generator.recognition_distribution(
                I_hat_t=I_hat_t,
                h_history=h_history
            )

            h_t = reparameterize(mu_q, logvar_q, deterministic=deterministic)
            h_list.append(h_t)

        h_seq = torch.stack(h_list, dim=1)
        return h_seq

    @torch.no_grad()
    def fit_global_koopman_from_dataloader(self, dataloader, device="cpu", deterministic=True):
        """
        Fit one final global Koopman model A, B using all sequences in
        the given dataloader. This is the recommended export step after
        mini-batch training.

        Parameters:
            dataloader: yields x_seq [batch, T, x_dim], u_seq [batch, T-1, u_dim]
            device: torch device
            deterministic: if True, infer h_seq by posterior means.

        Returns:
            result: dict with A, B, C, decoder_bias
        """
        self.eval()

        Z_blocks = []
        Y_blocks = []

        for x_seq, u_seq in dataloader:
            x_seq = x_seq.to(device)
            u_seq = u_seq.to(device)

            h_seq = self.infer_h_sequence(
                x_seq=x_seq,
                u_seq=u_seq,
                deterministic=deterministic
            )

            batch_size, T, h_dim = h_seq.shape
            _, T_minus_1, u_dim = u_seq.shape
            assert T_minus_1 == T - 1

            h_past = h_seq[:, :-1, :]
            h_next = h_seq[:, 1:, :]

            h_past_flat = h_past.reshape(batch_size * T_minus_1, h_dim)
            h_next_flat = h_next.reshape(batch_size * T_minus_1, h_dim)
            u_flat = u_seq.reshape(batch_size * T_minus_1, u_dim)

            H_past = h_past_flat.T
            H_next = h_next_flat.T
            U = u_flat.T

            Z_block = torch.cat([H_past, U], dim=0)
            Y_block = H_next

            Z_blocks.append(Z_block.cpu())
            Y_blocks.append(Y_block.cpu())

        Z = torch.cat(Z_blocks, dim=1).to(device)
        Y = torch.cat(Y_blocks, dim=1).to(device)

        Z_pinv = torch.linalg.pinv(Z)
        K = Y @ Z_pinv

        A = K[:, :self.h_dim]
        B = K[:, self.h_dim:]

        A = A + self.koopman_layer.reg * torch.eye(
            self.h_dim,
            device=A.device,
            dtype=A.dtype
        )

        return {
            "A": A.detach().cpu(),
            "B": B.detach().cpu(),
            "C": self.get_C_matrix(),
            "decoder_bias": self.get_decoder_bias()
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
    device=None
):
    """
    Train DKIN model.

    Loss:
        L_sum = kappa_1 * L_KL + kappa_2 * L_Pred

    where

        L_Pred = sum_{t=1}^{T-1} ||mu_t - x_t||^2
                 + omega_T ||mu_T - x_T||^2
    """

    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Please run this training script on a machine "
                "with a CUDA-capable GPU and a CUDA-enabled PyTorch installation."
            )
        device = torch.device("cuda")
    else:
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "The requested device is CUDA, but CUDA is not available in this PyTorch environment."
            )

    print(f"Using training device: {device}")

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

            out = model(x_seq, u_seq, deterministic=False)

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