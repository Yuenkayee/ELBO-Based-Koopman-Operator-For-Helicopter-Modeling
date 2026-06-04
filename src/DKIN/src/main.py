import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.io import loadmat, savemat

from dkin import DKIN, SequenceDataset


# ============================================================
# 1. Basic configuration
# ============================================================

CONFIG = {
    # Training hyperparameters
    "num_epochs": 600,
    "batch_size": 64,
    "lr": 5e-4,
    "kappa_1": 1.2,
    "kappa_2": 1.0,
    "omega_T": 5.0,

    # Model hyperparameters
    "h_dim": 48,
    "temporal_hidden_dim": 64,
    "temporal_embed_dim": 64,
    "obs_gru_hidden_dim": 64,

    # Checkpoint
    "resume": True,
}


# ============================================================
# 2. Path utilities
# ============================================================

def get_project_root():
    """
    Assume this file is located at:
        project_root/src/DKIN/main.py

    Then:
        project_root = parents[2]
    """
    return Path(__file__).resolve().parents[2]


def prepare_paths():
    project_root = get_project_root()

    data_dir = project_root / "DKIN/data"
    checkpoint_dir = project_root / "DKIN/checkPoints"

    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_data_path = data_dir / "trainData.mat"
    train_result_path = data_dir / "trainResult.mat"
    latest_checkpoint_path = checkpoint_dir / "latest.pt"

    return {
        "project_root": project_root,
        "data_dir": data_dir,
        "checkpoint_dir": checkpoint_dir,
        "train_data_path": train_data_path,
        "train_result_path": train_result_path,
        "latest_checkpoint_path": latest_checkpoint_path,
    }


# ============================================================
# 3. Load MATLAB training data
# ============================================================

def load_train_data_from_mat(train_data_path):
    """
    Load training data from trainData.mat.

    Expected variables:
        simu_input_1, simu_input_2, ..., simu_input_S
        simu_result_1, simu_result_2, ..., simu_result_S

    Each:
        simu_input_i:  [T, dim_u]
        simu_result_i: [T + 1, dim_x]

    Return:
        x_data: [S, T + 1, dim_x]
        u_data: [S, T, dim_u]
    """
    if not train_data_path.exists():
        raise FileNotFoundError(f"Cannot find training data file: {train_data_path}")

    try:
        mat_data = loadmat(train_data_path)
        is_hdf5_mat = False
    except NotImplementedError as exc:
        if "matlab v7.3" not in str(exc).lower():
            raise
        mat_data = h5py.File(train_data_path, "r")
        is_hdf5_mat = True

    input_keys = []
    result_keys = []

    for key in mat_data.keys():
        if key.startswith("simu_input_"):
            input_keys.append(key)
        elif key.startswith("simu_result_"):
            result_keys.append(key)

    def extract_index(name, prefix):
        return int(name.replace(prefix, ""))

    input_keys = sorted(input_keys, key=lambda name: extract_index(name, "simu_input_"))
    result_keys = sorted(result_keys, key=lambda name: extract_index(name, "simu_result_"))

    if len(input_keys) == 0:
        raise ValueError("No variables named simu_input_i were found in trainData.mat.")

    if len(result_keys) == 0:
        raise ValueError("No variables named simu_result_i were found in trainData.mat.")

    if len(input_keys) != len(result_keys):
        raise ValueError(
            f"The number of input sequences and result sequences is different: "
            f"{len(input_keys)} inputs vs {len(result_keys)} results."
        )

    x_list = []
    u_list = []

    for input_key, result_key in zip(input_keys, result_keys):
        input_idx = extract_index(input_key, "simu_input_")
        result_idx = extract_index(result_key, "simu_result_")

        if input_idx != result_idx:
            raise ValueError(
                f"Input/result index mismatch: {input_key} and {result_key}"
            )

        if is_hdf5_mat:
            # MATLAB v7.3 .mat files are HDF5-based. Arrays are often stored
            # with dimensions reversed compared with scipy.io.loadmat output.
            # Therefore, transpose 2-D matrices after reading.
            u_i = np.asarray(mat_data[input_key], dtype=np.float32)
            x_i = np.asarray(mat_data[result_key], dtype=np.float32)

            if u_i.ndim == 2:
                u_i = u_i.T
            if x_i.ndim == 2:
                x_i = x_i.T
        else:
            u_i = np.asarray(mat_data[input_key], dtype=np.float32)
            x_i = np.asarray(mat_data[result_key], dtype=np.float32)

        if u_i.ndim != 2:
            raise ValueError(f"{input_key} should be 2-D, but got shape {u_i.shape}")

        if x_i.ndim != 2:
            raise ValueError(f"{result_key} should be 2-D, but got shape {x_i.shape}")

        T_u = u_i.shape[0]
        T_x = x_i.shape[0]

        if T_x != T_u + 1:
            raise ValueError(
                f"Time length mismatch for index {input_idx}: "
                f"{input_key}.shape={u_i.shape}, {result_key}.shape={x_i.shape}. "
                f"Expected simu_result_i length = simu_input_i length + 1."
            )

        u_list.append(u_i)
        x_list.append(x_i)

    x_data = np.stack(x_list, axis=0)
    u_data = np.stack(u_list, axis=0)

    if is_hdf5_mat:
        mat_data.close()

    return x_data, u_data


# ============================================================
# 4. Checkpoint utilities
# ============================================================

def save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    epoch,
    config,
    x_dim,
    u_dim,
    loss_history,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "x_dim": x_dim,
        "u_dim": u_dim,
        "loss_history": loss_history,
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint_if_available(
    checkpoint_path,
    model,
    optimizer,
    device,
    resume=True,
):
    if not resume:
        return 0, []

    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}. Start training from epoch 0.")
        return 0, []

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    loss_history = checkpoint.get("loss_history", [])

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Resume training from epoch {start_epoch}")

    return start_epoch, loss_history


# ============================================================
# 5. Train one epoch
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    kappa_1,
    kappa_2,
    omega_T,
):
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

        pred_loss_main = F.mse_loss(
            mu_seq[:, :-1, :],
            x_seq[:, :-1, :],
            reduction="mean",
        )

        pred_loss_terminal = F.mse_loss(
            mu_seq[:, -1, :],
            x_seq[:, -1, :],
            reduction="mean",
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

    return {
        "loss": total_loss_value / num_batches,
        "pred_loss": total_pred_loss_value / num_batches,
        "kl_loss": total_kl_loss_value / num_batches,
    }


# ============================================================
# 6. Export Koopman matrices A, B, C
# ============================================================

@torch.no_grad()
def export_train_result(model, dataloader, device, train_result_path):
    """
    Export the final global Koopman matrices A, B and decoder parameters
    to trainResult.mat.

    This function is aligned with the mini-batch training scheme in dkin.py:
    during training, each mini-batch computes one temporary shared A, B;
    during export, all S training sequences are pooled to solve one final
    global A, B.

    Saved variables:
        A:              [h_dim, h_dim]
        B:              [h_dim, u_dim]
        C:              [x_dim, h_dim]
        decoder_bias:   [x_dim]
    """
    result = model.fit_global_koopman_from_dataloader(
        dataloader=dataloader,
        device=device,
        deterministic=True,
    )

    result_dict = {
        "A": result["A"].numpy(),
        "B": result["B"].numpy(),
        "C": result["C"].numpy(),
        "decoder_bias": result["decoder_bias"].numpy(),
    }

    savemat(train_result_path, result_dict)

    print(f"Saved Koopman matrices to {train_result_path}")
    print(f"A.shape = {result_dict['A'].shape}")
    print(f"B.shape = {result_dict['B'].shape}")
    print(f"C.shape = {result_dict['C'].shape}")
    print(f"decoder_bias.shape = {result_dict['decoder_bias'].shape}")


# ============================================================
# 7. Main function
# ============================================================

def main():
    paths = prepare_paths()

    train_data_path = paths["train_data_path"]
    train_result_path = paths["train_result_path"]
    latest_checkpoint_path = paths["latest_checkpoint_path"]
    checkpoint_dir = paths["checkpoint_dir"]

    print(f"Project root: {paths['project_root']}")
    print(f"Training data: {train_data_path}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    x_data, u_data = load_train_data_from_mat(train_data_path)

    num_samples, T_plus_1, x_dim = x_data.shape
    _, T, u_dim = u_data.shape

    print("Loaded training data:")
    print(f"x_data.shape = {x_data.shape}")
    print(f"u_data.shape = {u_data.shape}")
    print(f"num_samples = {num_samples}")
    print(f"T = {T}")
    print(f"x_dim = {x_dim}")
    print(f"u_dim = {u_dim}")

    dataset = SequenceDataset(x_data, u_data)

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    export_dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    model = DKIN(
        x_dim=x_dim,
        u_dim=u_dim,
        h_dim=CONFIG["h_dim"],
        temporal_hidden_dim=CONFIG["temporal_hidden_dim"],
        temporal_embed_dim=CONFIG["temporal_embed_dim"],
        obs_gru_hidden_dim=CONFIG["obs_gru_hidden_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    start_epoch, loss_history = load_checkpoint_if_available(
        checkpoint_path=latest_checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=device,
        resume=CONFIG["resume"],
    )

    num_epochs = CONFIG["num_epochs"]

    for epoch in range(start_epoch, num_epochs):
        metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            kappa_1=CONFIG["kappa_1"],
            kappa_2=CONFIG["kappa_2"],
            omega_T=CONFIG["omega_T"],
        )

        loss_history.append(
            {
                "epoch": epoch,
                "loss": metrics["loss"],
                "pred_loss": metrics["pred_loss"],
                "kl_loss": metrics["kl_loss"],
            }
        )

        print(
            f"Epoch [{epoch + 1:04d}/{num_epochs:04d}] "
            f"Loss: {metrics['loss']:.6f} | "
            f"Pred: {metrics['pred_loss']:.6f} | "
            f"KL: {metrics['kl_loss']:.6f}"
        )

    final_epoch = num_epochs - 1
    save_checkpoint(
        checkpoint_path=latest_checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=final_epoch,
        config=CONFIG,
        x_dim=x_dim,
        u_dim=u_dim,
        loss_history=loss_history,
    )
    print(f"Saved final checkpoint to {latest_checkpoint_path}")

    export_train_result(
        model=model,
        dataloader=export_dataloader,
        device=device,
        train_result_path=train_result_path,
    )


if __name__ == "__main__":
    main()