import torch
import torch.nn as nn

# ============================================================
# 1. Basic utility functions
# ============================================================


def reparameterize(mu, logvar):
    """
    Reparameterization trick:
        h = mu + std * eps
    where eps ~ N(0, I)
    """
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

    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)

    return kl.sum(dim=-1).mean()
