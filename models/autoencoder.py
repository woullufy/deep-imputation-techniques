import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=10):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = torch.nan_to_num(x, nan=0.0)

        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, z


class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()

        hidden_dim = max(3, input_dim - 1)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),

            nn.BatchNorm1d(latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0)

        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, z
