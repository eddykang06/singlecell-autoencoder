"""VAE implementation"""
import torch
import torch.nn as nn
import torch.nn.functional as F


"""Condition embeddings"""
class LinearEmbed(nn.Module):
    """
    Simple mapping from number to vector
    """
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim 
        self.embed = nn.Linear(1, out_dim)

    def forward(self, x):
        out = self.embed(x)
        return out


class MLPEmbed(nn.Module):
    """
    MLP mapping from number to vector
    """
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
    """
    Scale+shift composition to combine basal state with condition embedding
    """
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
    """
    Simple MLP regression module
    """
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
    """
    Simple linear regressor.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = self.model(x)
        return out


class GradientReversal(torch.autograd.Function):
    """
    Gradient reversal for usage on the adversarial loss function
    """
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
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # Log variance for stability
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus()
        )

    # Encoding x to latent mean and variance
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(min = -20, max = 20)
        return mu, logvar

    # Reparameterization trick to sample from latent distribution
    def reparam(self, mu, logvar, sample = True):
        if not sample:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        new = mu + eps * std
        return new

    # Composing latent sample with condition embeddings in the latent space
    def compose(self, z, time_embedding, dose_embedding):
        out = self.time_compose(z, time_embedding)
        out = self.dose_compose(out, dose_embedding)
        return out

    # Decoding to obtain a gene expression profile
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

    # Inference-time generation of new samples from trained VAE for specified does, time
    @torch.no_grad()
    def generate_samples(self, num_samples, d, t):
        """
        Note: simply provide a single dose d and single timepoint t
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        d = torch.as_tensor(d, dtype = dtype, device = device).expand(num_samples)
        t = torch.as_tensor(t, dtype = dtype, device = device).expand(num_samples)

        d = d.reshape(-1, 1)
        t = t.reshape(-1, 1)

        time_embedding = self.time_embed(t)
        dose_embedding = self.dose_embed(d)

        z = torch.randn(
            num_samples, 
            self.latent_dim,
            dtype = dtype,
            device = device
        )
        latent = self.compose(z, time_embedding, dose_embedding)
        x_hat = self.decode(latent)

        x_cpm = torch.expm1(x_hat)
        x_cpm = (
            1_000_000
            * x_cpm
            / x_cpm.sum(dim = 1, keepdim = True)
        )

        return x_cpm