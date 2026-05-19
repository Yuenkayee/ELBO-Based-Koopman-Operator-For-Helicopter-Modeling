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
    h_dim = 16
    embed_dim = 16
    para_mu = 1000.0
    para_lambda = 1.2

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
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "data", "elbo_model_checkpoint.pt")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model checkpoint from: {checkpoint_path}")
    else:
        print("No checkpoint found. Train from scratch.")

    model = train_elbo(
        model=model,
        trainData=trainData,
        num_epochs=100,
        lr=5e-4,
        device="cuda",
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
        },
        checkpoint_path,
    )
    print(f"Model checkpoint saved to: {checkpoint_path}")

    result_file = save_elbo_train_result(model)
    print(f"Training result saved to: {result_file}")


if __name__ == "__main__":
    main()