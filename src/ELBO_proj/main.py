from ELBO import ELBO
from ELBO import save_elbo_train_result
from ELBO import train_elbo
from basic_objects.dataReading import load_matlab_simulation_data


def main():
    trainData = load_matlab_simulation_data()

    x_dim = 12
    u_dim = 4
    z_dim = 16
    h_dim = 16
    embed_dim = 32
    para_mu = 0.8
    para_lambda = 1.5

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

    model = train_elbo(model=model, trainData=trainData)
    result_file = save_elbo_train_result(model)
    print(f"Training result saved to: {result_file}")


if __name__ == "__main__":
    main()