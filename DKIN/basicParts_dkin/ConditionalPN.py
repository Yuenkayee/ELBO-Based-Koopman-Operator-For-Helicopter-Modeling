import torch
import torch.nn as nn
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

    def backward_rollout(self, h_seq, u_seq, A, B):
        """
        Backward-time latent rollout:

            z_T = h_T
            z_t = A^{-1}(z_{t+1} - B u_t)

        Parameters:
            h_seq: [batch, T, h_dim]
            u_seq: [batch, T-1, u_dim]
            A:     [batch, h_dim, h_dim]
            B:     [batch, h_dim, u_dim]

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

            Bu = torch.bmm(B, u_t.unsqueeze(-1)).squeeze(-1)

            rhs = z_next - Bu

            # Solve A z_t = rhs instead of explicitly computing A^{-1}
            z_t = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)

            z_list[t] = z_t

        z_seq = torch.stack(z_list, dim=1)

        return z_seq