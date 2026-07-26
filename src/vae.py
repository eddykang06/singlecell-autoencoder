"""VAE implementation"""
import torch
import torch.nn as nn
import torch.nn.functional as F


"""Condition embeddings"""
class LinearEmbed(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim 
        self.embed = nn.Linear(1, out_dim)

    def forward(self, x):
        out = self.embed(x)
        return out


class MLPEmbed(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.embed = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        out = self.embed(x)
        return out


class FiLM(nn.Module): 
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.get_gamma_beta = nn.Linear(embedding_dim, 2 * embedding_dim)

    def forward(self, x, condition):
        gamma, beta = self.get_gamma_beta(condition).chunk(2, dim = -1)
        out = gamma * x + beta
        return out


"""Adversarial regressor"""
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.model(x)
        return out


"""VAE architecture"""
class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Embdding modules
        self.time_embed = MLPEmbed(out_dim = latent_dim)
        self.dose_embed = MLPEmbed(out_dim = latent_dim)
        self.time_compose = FiLM(embedding_dim = latent_dim)
        self.dose_compose = FiLM(embedding_dim = latent_dim)

        # VAE architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_std = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        std = self.fc_std(h)
        return mu, std

    def reparam(self, mu, std):
        eps = torch.rand_like(std)
        new = mu + eps * std
        return new

    def compose(self, z, time_embedding, dose_embedding):
        out = self.compose(z, time_embedding)
        out = self.compose(out, dose_embedding)
    
    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x, t, d):
        time_embedding = self.time_embed(t)
        dose_embedding = self.dose_embed(d)
        mu, std = self.encode(x)
        z = self.reparam(mu, std)
        h = self.time_compose(z, time_embedding)
        h = self.dose_compose(z, dose_embedding)
        x_hat = self.decode(h)

        return x_hat