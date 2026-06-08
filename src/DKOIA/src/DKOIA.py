

"""
Deep Koopman Operator with Input Augmentation, DKOIA.

The model follows the input-augmented Koopman structure

    z_k     = psi(x_k)
    ell_k   = [u_k, p_k, phi(x_k, u_k, p_k)]
    z_{k+1} = A z_k + B ell_k
    x_hat_k = C z_k

where psi is an LSTM network acting on a history window of states,
phi is a feed-forward neural network, and A, B, C are trainable Koopman
matrices. The training loss is a multi-step prediction loss in the original
state space.

Expected training data shapes:
    X_seq: [num_seq, T + 1, x_dim]
    U_seq: [num_seq, T,     u_dim]
    P_seq: [num_seq, T,     p_dim] or None

If the process has no known disturbance p_k, set p_dim=0 and pass P_seq=None.

For the LSTM-DKOIA version, psi is defined as
    z_k = psi_lstm(x_{k-L+1}, ..., x_k)
where L is psi_history_steps. During training, the first L states are used to
initialize z_{L-1}, and rollout starts from input u_{L-1}.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple
import time

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from scipy.io import loadmat, savemat
except ImportError:  # pragma: no cover
    loadmat = None
    savemat = None

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


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


class LSTMPsi(nn.Module):
    """LSTM-based observable map psi(x_{k-L+1:k}) -> z_k."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=x_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.proj = nn.Linear(hidden_dim, z_dim)

    def forward(self, x_hist: Tensor) -> Tensor:
        if x_hist.ndim != 3:
            raise ValueError(f"x_hist must have shape [batch, L, x_dim], got {tuple(x_hist.shape)}")
        _, (h_n, _) = self.lstm(x_hist)
        h_last = h_n[-1]
        return self.proj(h_last)


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
    resume_from_checkpoint: bool = True
    print_every: int = 10

def _to_numpy_from_mat_value(value) -> np.ndarray:
    """
    Convert a variable loaded from either scipy.io.loadmat or h5py to a NumPy array.
    MATLAB v7.3 files are HDF5 based. h5py reads MATLAB arrays with dimensions
    reversed compared with scipy.io.loadmat, so 2-D and higher arrays are
    transposed back to the usual MATLAB/Python orientation.
    """
    if h5py is not None and isinstance(value, h5py.Dataset):
        arr = np.array(value)
        if arr.ndim >= 2:
            arr = np.transpose(arr)
        return arr
    return np.asarray(value)

def _load_mat_file_auto(mat_path: str | Path) -> Tuple[Dict, bool]:
    """
    Load .mat data automatically.
    Returns:
        data: mapping from variable name to array-like object.
        is_hdf5: True for MATLAB v7.3/HDF5 files loaded by h5py.

    """
    mat_path = str(mat_path)
    if loadmat is None:
        raise ImportError("scipy is required to load non-v7.3 .mat files: pip install scipy")
    try:
        return loadmat(mat_path), False

    except NotImplementedError as exc:
        if "matlab v7.3" not in str(exc).lower() and "hdf" not in str(exc).lower():
            raise
        if h5py is None:
            raise ImportError(
                "This file is a MATLAB v7.3 HDF5 .mat file. Please install h5py: pip install h5py"
            ) from exc
        return h5py.File(mat_path, "r"), True

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
        psi_history_steps: int = 8,
        psi_lstm_hidden_dim: int = 128,
        psi_lstm_num_layers: int = 1,
        psi_lstm_dropout: float = 0.0,
        phi_hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if x_dim <= 0 or u_dim <= 0 or p_dim < 0 or z_dim <= 0 or phi_dim < 0:
            raise ValueError("Invalid dimensions for DKOIA")
        if psi_history_steps < 1:
            raise ValueError("psi_history_steps must be >= 1")
        if psi_lstm_hidden_dim <= 0 or psi_lstm_num_layers <= 0:
            raise ValueError("Invalid LSTM dimensions for psi_net")

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.p_dim = p_dim
        self.z_dim = z_dim
        self.phi_dim = phi_dim
        self.ell_dim = u_dim + p_dim + phi_dim

        self.psi_history_steps = psi_history_steps
        self.psi_lstm_hidden_dim = psi_lstm_hidden_dim
        self.psi_lstm_num_layers = psi_lstm_num_layers
        self.psi_lstm_dropout = psi_lstm_dropout

        self.psi_net = LSTMPsi(
            x_dim=x_dim,
            z_dim=z_dim,
            hidden_dim=psi_lstm_hidden_dim,
            num_layers=psi_lstm_num_layers,
            dropout=psi_lstm_dropout,
        )
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

    def psi(self, x_hist: Tensor) -> Tensor:
        return self.psi_net(x_hist)

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
        x_hist: Tensor,
        U: Tensor,
        P: Optional[Tensor] = None,
        use_predicted_state_for_phi: bool = True,
    ) -> Tensor:
        """
        Multi-step prediction initialized by a state history window.

        Args:
            x_hist: [batch, L, x_dim], where L = psi_history_steps. This window
                represents x_{k-L+1}, ..., x_k and is used to compute z_k.
            U: [batch, H, u_dim], control sequence starting from u_k.
            P: [batch, H, p_dim] or None, disturbance sequence starting from p_k.
            use_predicted_state_for_phi:
                True -> phi(x_hat_j|k, u_j, p_j), matching the paper's rollout constraint.

        Returns:
            X_hat: [batch, H + 1, x_dim], corresponding to x_hat_k, ..., x_hat_{k+H}.
        """
        if x_hist.ndim != 3:
            raise ValueError(f"x_hist must have shape [batch, L, x_dim], got {tuple(x_hist.shape)}")
        if x_hist.shape[1] != self.psi_history_steps:
            raise ValueError(
                f"x_hist length must equal psi_history_steps={self.psi_history_steps}, "
                f"got {x_hist.shape[1]}"
            )
        if P is None:
            P = torch.zeros(U.shape[0], U.shape[1], 0, device=U.device, dtype=U.dtype)

        z = self.psi(x_hist)
        x_hat = self.decode(z)
        predictions = [x_hat]

        for t in range(U.shape[1]):
            x_for_phi = x_hat if use_predicted_state_for_phi else x_hist[:, -1, :]
            z = self.koopman_step(z, x_for_phi, U[:, t, :], P[:, t, :])
            x_hat = self.decode(z)
            predictions.append(x_hat)

        return torch.stack(predictions, dim=1)

    def forward(self, X: Tensor, U: Tensor, P: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for a full training sequence.

        X: [batch, T+1, x_dim]
        U: [batch, T, u_dim]
        P: [batch, T, p_dim] or None

        The first psi_history_steps states initialize z at time k=L-1.
        Rollout then uses U[:, L-1:, :] and predicts X[:, L-1:, :].
        """
        L = self.psi_history_steps
        if X.shape[1] < L:
            raise ValueError(f"X sequence length must be at least psi_history_steps={L}")
        if U.shape[1] < L - 1:
            raise ValueError(f"U sequence length must be at least psi_history_steps-1={L - 1}")

        x_hist = X[:, :L, :]
        U_roll = U[:, L - 1 :, :]
        if P is None:
            P_roll = None
        else:
            P_roll = P[:, L - 1 :, :]
        return self.rollout(x_hist, U_roll, P_roll)

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
    grad_clip_norm: Optional[float] = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0

    for X, U, P in loader:
        X = X.to(device)
        U = U.to(device)
        P = P.to(device)

        L = model.psi_history_steps
        if rollout_steps is not None:
            H = min(rollout_steps, U.shape[1] - L + 1)
            if H < 1:
                raise ValueError(
                    f"rollout_steps is too short for psi_history_steps={L}; "
                    f"sequence U length is {U.shape[1]}"
                )
            X = X[:, : L + H, :]
            U = U[:, : L - 1 + H, :]
            P = P[:, : L - 1 + H, :]
        elif U.shape[1] < L - 1:
            raise ValueError(
                f"U sequence length {U.shape[1]} must be >= psi_history_steps - 1 = {L - 1}"
            )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            X_hat = model(X, U, P)
            X_target = X[:, model.psi_history_steps - 1 :, :]
            loss = weighted_multistep_loss(X_hat, X_target)
            if is_train:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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

    start_epoch = 1
    best_val = float("inf")
    if config.resume_from_checkpoint and config.checkpoint_path is not None and Path(config.checkpoint_path).is_file():
        checkpoint = load_checkpoint(model, config.checkpoint_path, optimizer=optimizer, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        history = checkpoint.get("history", history)
        if len(history.get("val_loss", [])) > 0:
            best_val = min(history["val_loss"])
        elif len(history.get("train_loss", [])) > 0:
            best_val = min(history["train_loss"])
        print(f"Resumed training from {config.checkpoint_path}, start_epoch={start_epoch}")

    if start_epoch > config.epochs:
        print(
            f"Checkpoint epoch is already {start_epoch - 1}, which is >= configured epochs={config.epochs}. "
            "No additional training will be performed. Increase config.epochs to continue training."
        )
        return history

    interval_start_time = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        train_loss = one_epoch(model, train_loader, optimizer, device, config.rollout_steps, config.grad_clip_norm)
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

        if epoch == start_epoch or epoch % config.print_every == 0 or epoch == config.epochs:
            elapsed = time.perf_counter() - interval_start_time
            num_epochs_in_interval = epoch - start_epoch + 1 if epoch == start_epoch else config.print_every
            if epoch == config.epochs and epoch % config.print_every != 0 and epoch != start_epoch:
                previous_print_epoch = epoch - ((epoch - start_epoch + 1) % config.print_every)
                if previous_print_epoch < start_epoch:
                    previous_print_epoch = start_epoch - 1
                num_epochs_in_interval = epoch - previous_print_epoch
            print(
                f"Epoch [{epoch:04d}/{config.epochs:04d}] "
                f"train_loss={train_loss:.6e} val_loss={val_loss:.6e} "
                f"time_for_last_{num_epochs_in_interval}_epochs={elapsed:.3f}s"
            )
            interval_start_time = time.perf_counter()

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
    data, is_hdf5 = _load_mat_file_auto(mat_path)

    try:
        indices: list[int] = []
        keys = list(data.keys())

        for key in keys:
            if key.startswith(x_prefix):
                suffix = key[len(x_prefix):]
                if suffix.isdigit() and f"{u_prefix}{suffix}" in data:
                    indices.append(int(suffix))

        indices = sorted(indices)

        if max_sequences is not None:
            indices = indices[:max_sequences]

        if not indices:
            raise ValueError(f"No sequence variables found in {mat_path}")

        X_list, U_list, P_list = [], [], []

        for idx in indices:
            X = _to_numpy_from_mat_value(data[f"{x_prefix}{idx}"]).astype(np.float32)
            U = _to_numpy_from_mat_value(data[f"{u_prefix}{idx}"]).astype(np.float32)

            X = np.squeeze(X)
            U = np.squeeze(U)

            if X.ndim == 1:
                X = X[:, None]
            if U.ndim == 1:
                U = U[:, None]

            if X.shape[0] != U.shape[0] + 1:
                raise ValueError(
                    f"Sequence {idx}: expected X length T+1 and U length T, "
                    f"got X.shape={X.shape}, U.shape={U.shape}. "
                    "If these dimensions look reversed, check the MATLAB save format."
                )

            X_list.append(X)
            U_list.append(U)

            if p_prefix is not None:
                p_key = f"{p_prefix}{idx}"

                if p_key not in data:
                    raise ValueError(f"Missing disturbance variable {p_key}")

                P = _to_numpy_from_mat_value(data[p_key]).astype(np.float32)
                P = np.squeeze(P)

                if P.ndim == 1:
                    P = P[:, None]

                if P.shape[0] != U.shape[0]:
                    raise ValueError(f"Sequence {idx}: expected P length T, got P.shape={P.shape}")

                P_list.append(P)

        X_seq = np.stack(X_list, axis=0)
        U_seq = np.stack(U_list, axis=0)
        P_seq = np.stack(P_list, axis=0) if p_prefix is not None else None

        return X_seq, U_seq, P_seq

    finally:
        if is_hdf5:
            data.close()


def save_matrices_mat(model: DKOIA, mat_path: str | Path) -> None:
    """Save Koopman matrices A, B, C to a MATLAB .mat file."""
    if savemat is None:
        raise ImportError("scipy is required to save .mat files: pip install scipy")
    mat_path = Path(mat_path)
    mat_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(mat_path), model.matrices_numpy())

def _linear_layer_to_mat(prefix: str, layer: nn.Linear) -> Dict[str, np.ndarray]:
    """Convert one nn.Linear layer to MATLAB-friendly arrays."""
    out: Dict[str, np.ndarray] = {f"{prefix}_weight": layer.weight.detach().cpu().numpy()}
    if layer.bias is not None:
        out[f"{prefix}_bias"] = layer.bias.detach().cpu().numpy()
    else:
        out[f"{prefix}_bias"] = np.zeros((layer.out_features,), dtype=np.float32)
    return out


def _mlp_to_mat(prefix: str, mlp: Optional[MLP]) -> Dict[str, np.ndarray]:
    """
    Export MLP Linear-layer weights and biases.

    The exported variables are named like:
        psi_layer_1_weight, psi_layer_1_bias, psi_layer_2_weight, ...
        phi_layer_1_weight, phi_layer_1_bias, phi_layer_2_weight, ...

    ReLU activation should be applied after every exported layer except the final layer.
    """
    out: Dict[str, np.ndarray] = {}
    if mlp is None:
        out[f"{prefix}_num_linear_layers"] = np.array([[0]], dtype=np.int64)
        return out

    linear_idx = 1
    for module in mlp.net:
        if isinstance(module, nn.Linear):
            out.update(_linear_layer_to_mat(f"{prefix}_layer_{linear_idx}", module))
            linear_idx += 1
    out[f"{prefix}_num_linear_layers"] = np.array([[linear_idx - 1]], dtype=np.int64)
    return out



def _lstm_psi_to_mat(psi_net: LSTMPsi) -> Dict[str, np.ndarray]:
    """Export LSTMPsi parameters to MATLAB-friendly arrays."""
    out: Dict[str, np.ndarray] = {
        "psi_type": np.array(["lstm"], dtype=object),
        "psi_num_lstm_layers": np.array([[psi_net.num_layers]], dtype=np.int64),
        "psi_lstm_hidden_dim": np.array([[psi_net.hidden_dim]], dtype=np.int64),
        "psi_lstm_input_dim": np.array([[psi_net.x_dim]], dtype=np.int64),
        "psi_lstm_output_dim": np.array([[psi_net.z_dim]], dtype=np.int64),
    }

    state = psi_net.lstm.state_dict()
    for key, value in state.items():
        out[f"psi_lstm_{key}"] = value.detach().cpu().numpy()

    out.update(_linear_layer_to_mat("psi_proj", psi_net.proj))
    return out


def export_model_for_matlab(model: DKOIA, mat_path: str | Path, pt_path: Optional[str | Path] = None) -> None:
    """
    Export A, B, C, LSTM psi_net and MLP phi_net for MATLAB/Simulink use.

    The .mat file contains Koopman matrices, the LSTM parameters of psi_net,
    the projection layer after the LSTM, and all Linear-layer weights/biases
    of phi_net. The LSTM gate order follows PyTorch convention:
        input gate, forget gate, cell gate, output gate.
    """
    if savemat is None:
        raise ImportError("scipy is required to save .mat files: pip install scipy")

    mat_path = Path(mat_path)
    mat_path.parent.mkdir(parents=True, exist_ok=True)

    export_dict: Dict[str, np.ndarray] = {
        "A": model.A.detach().cpu().numpy(),
        "B": model.B.detach().cpu().numpy(),
        "C": model.C.detach().cpu().numpy(),
        "x_dim": np.array([[model.x_dim]], dtype=np.int64),
        "u_dim": np.array([[model.u_dim]], dtype=np.int64),
        "p_dim": np.array([[model.p_dim]], dtype=np.int64),
        "z_dim": np.array([[model.z_dim]], dtype=np.int64),
        "phi_dim": np.array([[model.phi_dim]], dtype=np.int64),
        "ell_dim": np.array([[model.ell_dim]], dtype=np.int64),
        "psi_history_steps": np.array([[model.psi_history_steps]], dtype=np.int64),
        "psi_lstm_hidden_dim": np.array([[model.psi_lstm_hidden_dim]], dtype=np.int64),
        "psi_lstm_num_layers": np.array([[model.psi_lstm_num_layers]], dtype=np.int64),
        "psi_lstm_dropout": np.array([[model.psi_lstm_dropout]], dtype=np.float32),
    }
    export_dict.update(_lstm_psi_to_mat(model.psi_net))
    export_dict.update(_mlp_to_mat("phi", model.phi_net))

    savemat(str(mat_path), export_dict)

    if pt_path is not None:
        pt_path = Path(pt_path)
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "dims": {
                    "x_dim": model.x_dim,
                    "u_dim": model.u_dim,
                    "p_dim": model.p_dim,
                    "z_dim": model.z_dim,
                    "phi_dim": model.phi_dim,
                    "ell_dim": model.ell_dim,
                    "psi_history_steps": model.psi_history_steps,
                    "psi_lstm_hidden_dim": model.psi_lstm_hidden_dim,
                    "psi_lstm_num_layers": model.psi_lstm_num_layers,
                    "psi_lstm_dropout": model.psi_lstm_dropout,
                },
            },
            pt_path,
        )


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------

def save_checkpoint(
    model: DKOIA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Dict[str, list[float]],
    checkpoint_path: str | Path,
) -> None:
    """Save a training checkpoint for later continuation."""
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
                "ell_dim": model.ell_dim,
                "psi_history_steps": model.psi_history_steps,
                "psi_lstm_hidden_dim": model.psi_lstm_hidden_dim,
                "psi_lstm_num_layers": model.psi_lstm_num_layers,
                "psi_lstm_dropout": model.psi_lstm_dropout,
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
    data_path = Path("../data/trainData.mat")
    result_path = Path("../data/trainResult.mat")
    export_pt_path = Path("../data/trainResult_full_model.pt")
    checkpoint_path = Path("../checkPoints/dkoia_best.pt")

    X_seq, U_seq, P_seq = load_mat_sequences(
        data_path,
        x_prefix="simu_result_",
        u_prefix="simu_input_",
        p_prefix=None,
    )
    
    print(f"Loaded X_seq shape: {X_seq.shape}")
    print(f"Loaded U_seq shape: {U_seq.shape}")

    if P_seq is not None:
        print(f"Loaded P_seq shape: {P_seq.shape}")

    x_dim = X_seq.shape[-1]
    u_dim = U_seq.shape[-1]
    p_dim = 0 if P_seq is None else P_seq.shape[-1]

    model = DKOIA(
        x_dim=x_dim,
        u_dim=u_dim,
        p_dim=p_dim,
        z_dim=64,
        phi_dim=64,
        psi_history_steps=8,
        psi_lstm_hidden_dim=128,
        psi_lstm_num_layers=1,
        psi_lstm_dropout=0.0,
        phi_hidden_dims=(128, 128),
    )

    config = TrainConfig(
        epochs=1000,
        batch_size=128,
        learning_rate=1e-3,
        val_ratio=0.1,
        rollout_steps=None,
        checkpoint_path=str(checkpoint_path),
        resume_from_checkpoint=True,
        print_every=10,
    )

    train_dkoia(model, X_seq, U_seq, P_seq, config)
    export_model_for_matlab(model, result_path, export_pt_path)
    print(f"Saved Koopman matrices and neural-network parameters to {result_path}")
    print(f"Saved full PyTorch model parameters to {export_pt_path}")


if __name__ == "__main__":
    main()