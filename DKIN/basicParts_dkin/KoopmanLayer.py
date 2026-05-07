import torch
import torch.nn as nn
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
        Parameters:
            h_seq: [batch, T, h_dim]
            u_seq: [batch, T-1, u_dim]

        Returns:
            A: [batch, h_dim, h_dim]
            B: [batch, h_dim, u_dim]
        """

        batch_size, T, h_dim = h_seq.shape
        _, T_minus_1, u_dim = u_seq.shape

        assert T_minus_1 == T - 1

        A_list = []
        B_list = []

        for b in range(batch_size):
            # H_past: [h_dim, T-1]
            H_past = h_seq[b, :-1, :].T

            # H_next: [h_dim, T-1]
            H_next = h_seq[b, 1:, :].T

            # U: [u_dim, T-1]
            U = u_seq[b].T

            # Z: [h_dim + u_dim, T-1]
            Z = torch.cat([H_past, U], dim=0)

            # Y: [h_dim, T-1]
            Y = H_next

            # Moore-Penrose pseudoinverse
            Z_pinv = torch.linalg.pinv(Z)

            # K = [A B]: [h_dim, h_dim + u_dim]
            K = Y @ Z_pinv

            A = K[:, :h_dim]
            B = K[:, h_dim:]

            # Small regularization for numerical stability
            A = A + self.reg * torch.eye(
                h_dim,
                device=h_seq.device,
                dtype=h_seq.dtype
            )

            A_list.append(A)
            B_list.append(B)

        A = torch.stack(A_list, dim=0)
        B = torch.stack(B_list, dim=0)

        return A, B

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