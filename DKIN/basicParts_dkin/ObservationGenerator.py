import torch
import torch.nn as nn
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
        # 将向量沿着指定维度切割成若干块
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
