import DKIN.dkin as dkin
import torch
from DKIN.dkin import SequenceDataset
from DKIN.dkin import DataLoader
from DKIN.dkin import DKIN
from DKIN.dkin import train_dkin

def main():
    num_samples = 1000
    batch = 100
    T = 30
    x_dim = 16
    u_dim = 4
    h_dim = 48

    x_data = torch.randn(num_samples, T, x_dim)
    u_data = torch.randn(num_samples, T - 1, u_dim)
    x_test = torch.randn(batch, T, x_dim)
    u_test = torch.randn(batch, T - 1, u_dim)
    dataset = SequenceDataset(x_data, u_data)

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True
    )

    model = DKIN(
        x_dim=x_dim,
        u_dim=u_dim,
        h_dim=h_dim,
        temporal_hidden_dim=64,
        temporal_embed_dim=64,
        obs_gru_hidden_dim=64
    )

    trained_model = train_dkin(
        model=model,
        dataloader=dataloader,
        num_epochs=100,
        lr=5e-4,
        kappa_1=1.2,
        kappa_2=1.0,
        omega_T=5.0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    trained_model.eval()

    with torch.no_grad():
        out = trained_model(x_test, u_test)
    A = out["A"]
    B = out["B"]
    z_seq = out["z_seq"]
    mu_seq = out["mu_seq"]
    print(A)
    print(B)
    print(z_seq)
    print(mu_seq)