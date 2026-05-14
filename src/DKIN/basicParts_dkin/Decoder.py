import torch
import torch.nn as nn

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
