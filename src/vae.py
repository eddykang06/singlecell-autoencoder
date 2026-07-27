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


class LinearRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = self.model(x)
        return out


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None


def gradient_reverse(x, lambda_adv = 1.0):
    return GradientReversal.apply(x, lambda_adv)


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
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(min = -20, max = 20)
        return mu, logvar

    def reparam(self, mu, logvar, sample = True):
        if not sample:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        new = mu + eps * std
        return new

    def compose(self, z, time_embedding, dose_embedding):
        out = self.time_compose(z, time_embedding)
        out = self.dose_compose(out, dose_embedding)
        return out
    
    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x, d, t, sample = None):
        if sample is None:
            sample = self.training

        time_embedding = self.time_embed(t)
        dose_embedding = self.dose_embed(d)
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar, sample = sample)
        h = self.compose(z, time_embedding, dose_embedding)
        x_hat = self.decode(h)

        return x_hat, mu, logvar, z
