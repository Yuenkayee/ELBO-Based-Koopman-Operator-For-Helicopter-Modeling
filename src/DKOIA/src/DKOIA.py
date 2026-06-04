

"""
Deep Koopman Operator with Input Augmentation, DKOIA.

The model follows the input-augmented Koopman structure

    z_k     = psi(x_k)
    ell_k   = [u_k, p_k, phi(x_k, u_k, p_k)]
    z_{k+1} = A z_k + B ell_k
    x_hat_k = C z_k

where psi and phi are feed-forward neural networks, and A, B, C are
trainable Koopman matrices. The training loss is a multi-step prediction
loss in the original state space.

Expected training data shapes:
    X_seq: [num_seq, T + 1, x_dim]
    U_seq: [num_seq, T,     u_dim]
    P_seq: [num_seq, T,     p_dim] or None

If the process has no known disturbance p_k, set p_dim=0 and pass P_seq=None.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from scipy.io import loadmat, savemat
except ImportError:  # pragma: no cover
    loadmat = None
    savemat = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int = 1) -> None:
    """Set random seeds for reproducible training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    """Simple fully-connected network used for psi and phi."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        output_activation: Optional[type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = in_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(activation())
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, out_dim))
        if output_activation is not None:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SequenceDataset(Dataset):
    """Dataset wrapper for sequence data."""

    def __init__(self, X_seq: Tensor, U_seq: Tensor, P_seq: Optional[Tensor] = None) -> None:
        if X_seq.ndim != 3:
            raise ValueError(f"X_seq must have shape [S, T+1, x_dim], got {tuple(X_seq.shape)}")
        if U_seq.ndim != 3:
            raise ValueError(f"U_seq must have shape [S, T, u_dim], got {tuple(U_seq.shape)}")
        if X_seq.shape[0] != U_seq.shape[0] or X_seq.shape[1] != U_seq.shape[1] + 1:
            raise ValueError("X_seq and U_seq must satisfy X_seq.shape=[S,T+1,x_dim], U_seq.shape=[S,T,u_dim]")

        if P_seq is None:
            P_seq = torch.zeros(X_seq.shape[0], U_seq.shape[1], 0, dtype=X_seq.dtype)
        if P_seq.ndim != 3:
            raise ValueError(f"P_seq must have shape [S, T, p_dim], got {tuple(P_seq.shape)}")
        if P_seq.shape[0] != X_seq.shape[0] or P_seq.shape[1] != U_seq.shape[1]:
            raise ValueError("P_seq and U_seq must have the same first two dimensions [S,T]")

        self.X_seq = X_seq.float()
        self.U_seq = U_seq.float()
        self.P_seq = P_seq.float()

    def __len__(self) -> int:
        return self.X_seq.shape[0]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.X_seq[index], self.U_seq[index], self.P_seq[index]


@dataclass
class TrainConfig:
    epochs: int = 1000
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    val_ratio: float = 0.1
    rollout_steps: Optional[int] = None
    grad_clip_norm: Optional[float] = 10.0
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: Optional[str] = None
    print_every: int = 10


# -----------------------------------------------------------------------------
# DKOIA model
# -----------------------------------------------------------------------------


class DKOIA(nn.Module):
    """Input-augmented deep Koopman model for general nonlinear processes."""

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        p_dim: int = 0,
        z_dim: int = 32,
        phi_dim: int = 32,
        psi_hidden_dims: Sequence[int] = (128, 128),
        phi_hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if x_dim <= 0 or u_dim <= 0 or p_dim < 0 or z_dim <= 0 or phi_dim < 0:
            raise ValueError("Invalid dimensions for DKOIA")

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.p_dim = p_dim
        self.z_dim = z_dim
        self.phi_dim = phi_dim
        self.ell_dim = u_dim + p_dim + phi_dim

        self.psi_net = MLP(x_dim, z_dim, psi_hidden_dims)
        self.phi_net = MLP(x_dim + u_dim + p_dim, phi_dim, phi_hidden_dims) if phi_dim > 0 else None

        # Linear Koopman matrices. Bias is not used so that these are exactly A, B, C.
        self.A_layer = nn.Linear(z_dim, z_dim, bias=False)
        self.B_layer = nn.Linear(self.ell_dim, z_dim, bias=False)
        self.C_layer = nn.Linear(z_dim, x_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize A close to identity and B, C with small random weights."""
        nn.init.eye_(self.A_layer.weight)
        self.A_layer.weight.data += 1e-3 * torch.randn_like(self.A_layer.weight)
        nn.init.xavier_uniform_(self.B_layer.weight, gain=0.1)
        nn.init.xavier_uniform_(self.C_layer.weight, gain=0.5)

    @property
    def A(self) -> Tensor:
        return self.A_layer.weight

    @property
    def B(self) -> Tensor:
        return self.B_layer.weight

    @property
    def C(self) -> Tensor:
        return self.C_layer.weight

    def psi(self, x: Tensor) -> Tensor:
        return self.psi_net(x)

    def phi(self, x: Tensor, u: Tensor, p: Optional[Tensor] = None) -> Tensor:
        if p is None:
            p = torch.zeros(*x.shape[:-1], 0, device=x.device, dtype=x.dtype)
        if self.phi_dim == 0:
            return torch.zeros(*x.shape[:-1], 0, device=x.device, dtype=x.dtype)
        return self.phi_net(torch.cat([x, u, p], dim=-1))

    def input_augmentation(self, x: Tensor, u: Tensor, p: Optional[Tensor] = None) -> Tensor:
        """Compute ell(x,u,p) = [u, p, phi(x,u,p)]."""
        if p is None:
            p = torch.zeros(*u.shape[:-1], 0, device=u.device, dtype=u.dtype)
        phi_val = self.phi(x, u, p)
        return torch.cat([u, p, phi_val], dim=-1)

    def decode(self, z: Tensor) -> Tensor:
        return self.C_layer(z)

    def koopman_step(self, z: Tensor, x_for_phi: Tensor, u: Tensor, p: Optional[Tensor] = None) -> Tensor:
        ell = self.input_augmentation(x_for_phi, u, p)
        return self.A_layer(z) + self.B_layer(ell)

    def rollout(
        self,
        x0: Tensor,
        U: Tensor,
        P: Optional[Tensor] = None,
        use_predicted_state_for_phi: bool = True,
    ) -> Tensor:
        """
        Multi-step prediction from x0.

        Args:
            x0: [batch, x_dim]
            U:  [batch, H, u_dim]
            P:  [batch, H, p_dim] or None
            use_predicted_state_for_phi:
                True  -> phi(x_hat_j|k, u_j, p_j), matching the paper's rollout constraint.
                False -> only useful for diagnostics if measured states are supplied elsewhere.

        Returns:
            X_hat: [batch, H + 1, x_dim]
        """
        if P is None:
            P = torch.zeros(U.shape[0], U.shape[1], 0, device=U.device, dtype=U.dtype)

        z = self.psi(x0)
        x_hat = self.decode(z)
        predictions = [x_hat]

        for t in range(U.shape[1]):
            x_for_phi = x_hat if use_predicted_state_for_phi else x0
            z = self.koopman_step(z, x_for_phi, U[:, t, :], P[:, t, :])
            x_hat = self.decode(z)
            predictions.append(x_hat)

        return torch.stack(predictions, dim=1)

    def forward(self, X: Tensor, U: Tensor, P: Optional[Tensor] = None) -> Tensor:
        return self.rollout(X[:, 0, :], U, P)

    def matrices_numpy(self) -> Dict[str, np.ndarray]:
        """Return trained A, B, C matrices as NumPy arrays."""
        return {
            "A": self.A.detach().cpu().numpy(),
            "B": self.B.detach().cpu().numpy(),
            "C": self.C.detach().cpu().numpy(),
        }


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


def weighted_multistep_loss(
    X_hat: Tensor,
    X_true: Tensor,
    step_weights: Optional[Tensor] = None,
) -> Tensor:
    """Weighted multi-step prediction MSE in the original state space."""
    err2 = (X_hat - X_true).pow(2).mean(dim=-1)  # [batch, H+1]
    if step_weights is None:
        return err2.mean()
    weights = step_weights.to(device=err2.device, dtype=err2.dtype)
    weights = weights / weights.sum().clamp_min(1e-12)
    return (err2 * weights[None, :]).sum(dim=1).mean()


def one_epoch(
    model: DKOIA,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    rollout_steps: Optional[int] = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0

    for X, U, P in loader:
        X = X.to(device)
        U = U.to(device)
        P = P.to(device)

        if rollout_steps is not None:
            H = min(rollout_steps, U.shape[1])
            X = X[:, : H + 1, :]
            U = U[:, :H, :]
            P = P[:, :H, :]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            X_hat = model(X, U, P)
            loss = weighted_multistep_loss(X_hat, X)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = X.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def train_dkoia(
    model: DKOIA,
    X_seq: Tensor | np.ndarray,
    U_seq: Tensor | np.ndarray,
    P_seq: Optional[Tensor | np.ndarray] = None,
    config: TrainConfig = TrainConfig(),
) -> Dict[str, list[float]]:
    """Train DKOIA using sequence data."""
    set_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)

    X_tensor = torch.as_tensor(X_seq, dtype=torch.float32)
    U_tensor = torch.as_tensor(U_seq, dtype=torch.float32)
    P_tensor = None if P_seq is None else torch.as_tensor(P_seq, dtype=torch.float32)
    dataset = SequenceDataset(X_tensor, U_tensor, P_tensor)

    val_size = int(len(dataset) * config.val_ratio)
    train_size = len(dataset) - val_size
    if val_size > 0:
        generator = torch.Generator().manual_seed(config.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = None if val_dataset is None else DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    best_val = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_loss = one_epoch(model, train_loader, optimizer, device, config.rollout_steps)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            with torch.no_grad():
                val_loss = one_epoch(model, val_loader, None, device, config.rollout_steps)
            history["val_loss"].append(val_loss)
        else:
            val_loss = train_loss

        if config.checkpoint_path is not None and val_loss <= best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, history, config.checkpoint_path)

        if epoch == 1 or epoch % config.print_every == 0 or epoch == config.epochs:
            print(f"Epoch [{epoch:04d}/{config.epochs:04d}] train_loss={train_loss:.6e} val_loss={val_loss:.6e}")

    return history


# -----------------------------------------------------------------------------
# MATLAB data IO helpers
# -----------------------------------------------------------------------------


def load_mat_sequences(
    mat_path: str | Path,
    x_prefix: str = "simu_result_",
    u_prefix: str = "simu_input_",
    p_prefix: Optional[str] = None,
    max_sequences: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load MATLAB data with variables such as simu_result_1 and simu_input_1.

    State arrays should be [T+1, x_dim]. Input arrays should be [T, u_dim].
    Disturbance arrays are optional and should be [T, p_dim].
    """
    if loadmat is None:
        raise ImportError("scipy is required to load .mat files: pip install scipy")

    data = loadmat(str(mat_path))
    indices: list[int] = []
    for key in data.keys():
        if key.startswith(x_prefix):
            suffix = key[len(x_prefix) :]
            if suffix.isdigit() and f"{u_prefix}{suffix}" in data:
                indices.append(int(suffix))
    indices = sorted(indices)
    if max_sequences is not None:
        indices = indices[:max_sequences]
    if not indices:
        raise ValueError(f"No sequence variables found in {mat_path}")

    X_list, U_list, P_list = [], [], []
    for idx in indices:
        X = np.asarray(data[f"{x_prefix}{idx}"], dtype=np.float32)
        U = np.asarray(data[f"{u_prefix}{idx}"], dtype=np.float32)
        if X.shape[0] != U.shape[0] + 1:
            raise ValueError(f"Sequence {idx}: expected X length T+1 and U length T, got {X.shape}, {U.shape}")
        X_list.append(X)
        U_list.append(U)

        if p_prefix is not None:
            p_key = f"{p_prefix}{idx}"
            if p_key not in data:
                raise ValueError(f"Missing disturbance variable {p_key}")
            P = np.asarray(data[p_key], dtype=np.float32)
            if P.shape[0] != U.shape[0]:
                raise ValueError(f"Sequence {idx}: expected P length T, got {P.shape}")
            P_list.append(P)

    X_seq = np.stack(X_list, axis=0)
    U_seq = np.stack(U_list, axis=0)
    P_seq = np.stack(P_list, axis=0) if p_prefix is not None else None
    return X_seq, U_seq, P_seq


def save_matrices_mat(model: DKOIA, mat_path: str | Path) -> None:
    """Save Koopman matrices A, B, C to a MATLAB .mat file."""
    if savemat is None:
        raise ImportError("scipy is required to save .mat files: pip install scipy")
    mat_path = Path(mat_path)
    mat_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(mat_path), model.matrices_numpy())


def save_checkpoint(
    model: DKOIA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Dict[str, list[float]],
    checkpoint_path: str | Path,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "dims": {
                "x_dim": model.x_dim,
                "u_dim": model.u_dim,
                "p_dim": model.p_dim,
                "z_dim": model.z_dim,
                "phi_dim": model.phi_dim,
            },
        },
        checkpoint_path,
    )


def load_checkpoint(
    model: DKOIA,
    checkpoint_path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


# -----------------------------------------------------------------------------
# Example entry point
# -----------------------------------------------------------------------------


def main() -> None:
    """
    Minimal example.

    Modify data_path, dimensions, and hyperparameters according to your process.
    If there is no disturbance p_k, keep p_dim=0 and p_prefix=None.
    """
    data_path = Path("data/trainData.mat")
    result_path = Path("data/trainResult.mat")
    checkpoint_path = Path("checkPoints/dkoia_best.pt")

    X_seq, U_seq, P_seq = load_mat_sequences(
        data_path,
        x_prefix="simu_result_",
        u_prefix="simu_input_",
        p_prefix=None,
    )

    x_dim = X_seq.shape[-1]
    u_dim = U_seq.shape[-1]
    p_dim = 0 if P_seq is None else P_seq.shape[-1]

    model = DKOIA(
        x_dim=x_dim,
        u_dim=u_dim,
        p_dim=p_dim,
        z_dim=32,
        phi_dim=32,
        psi_hidden_dims=(128, 128),
        phi_hidden_dims=(128, 128),
    )

    config = TrainConfig(
        epochs=1000,
        batch_size=128,
        learning_rate=1e-3,
        val_ratio=0.1,
        rollout_steps=None,
        checkpoint_path=str(checkpoint_path),
        print_every=10,
    )

    train_dkoia(model, X_seq, U_seq, P_seq, config)
    save_matrices_mat(model, result_path)
    print(f"Saved Koopman matrices to {result_path}")


if __name__ == "__main__":
    main()