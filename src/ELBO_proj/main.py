import os

import torch

from ELBO import ELBO
from ELBO import save_elbo_train_result
from ELBO import train_elbo
from basic_objects.dataReading import load_matlab_simulation_data


def main():
    trainData = load_matlab_simulation_data()

    x_dim = 12
    u_dim = 4
    z_dim = 24
    h_dim = 24
    embed_dim = 36
    para_mu = 1.0
    para_lambda = 0.001
    para_z0 = 1.0
    para_dyn = 1.0
    para_rollout = 1.0
    batch_size = 32

    _, U_seq_dim = trainData.U_seq.shape
    if (U_seq_dim - x_dim) % u_dim != 0:
        raise ValueError(
            f"U_seq.shape[1] should satisfy x_dim + T * u_dim, "
            f"but got U_seq.shape[1]={U_seq_dim}, x_dim={x_dim}, u_dim={u_dim}"
        )

    T = (U_seq_dim - x_dim) // u_dim

    _, X_seq_dim = trainData.X_seq.shape
    if X_seq_dim != T * x_dim:
        raise ValueError(
            f"X_seq.shape[1] should be T * x_dim={T * x_dim}, "
            f"but got X_seq.shape[1]={X_seq_dim}"
        )

    model = ELBO(
        x_dim=x_dim,
        u_dim=u_dim,
        z_dim=z_dim,
        h_dim=h_dim,
        embed_dim=embed_dim,
        T=T,
        para_mu=para_mu,
        para_lambda=para_lambda,
        para_rollout=para_rollout,
        para_z0=para_z0,
        para_dyn=para_dyn,
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "data", "elbo_model_z0_dyn_rollout_checkpoint.pt")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model checkpoint from: {checkpoint_path}")
        except RuntimeError as err:
            print("Checkpoint is incompatible with the current model structure.")
            print(f"Skip loading checkpoint and train from scratch. Error: {err}")
    else:
        print("No checkpoint found. Train from scratch.")

    model = train_elbo(
        model=model,
        trainData=trainData,
        num_epochs=2000,
        lr=5e-4,
        device="cuda",
        batch_size=batch_size,
    )

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir != "":
        os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.to("cpu").state_dict(),
            "x_dim": x_dim,
            "u_dim": u_dim,
            "z_dim": z_dim,
            "h_dim": h_dim,
            "embed_dim": embed_dim,
            "T": T,
            "para_mu": para_mu,
            "para_lambda": para_lambda,
            "para_z0": para_z0,
            "para_dyn": para_dyn,
            "para_rollout": para_rollout,
            "batch_size": batch_size,
        },
        checkpoint_path,
    )
    print(f"Model checkpoint saved to: {checkpoint_path}")

    result_file = save_elbo_train_result(model)
    print(f"Training result saved to: {result_file}")


if __name__ == "__main__":
    main()