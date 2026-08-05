import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from torch.optim import Adam
from src.vae import CVAE, MLPRegressor, gradient_reverse, LinearRegressor
from src.sc_data import df_to_tensors, get_data_loaders, get_standard_scaler


"""VAE training loop"""
def kl_loss_fn(
    mu: torch.Tensor,
    logvar: torch.Tensor
):
    """
    Formula for KL divergence between N(mu, variance) and N(0, 1)
    """
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim = 1
    ).mean()

    return kl_loss


def normalize_log_cpm(
    x_hat: torch.Tensor,
    library_size: float = 1_000_000.0
):
    """
    Convert log1p expression to CPM, enforce a fixed library size per
    sample, and return the normalized values to log1p space.
    """
    x_hat_cpm = torch.expm1(x_hat).clamp_min(0)
    row_totals = x_hat_cpm.sum(dim = 1, keepdim = True).clamp_min(1e-8)
    x_hat_cpm = library_size * x_hat_cpm / row_totals

    return torch.log1p(x_hat_cpm)


def train_cvae(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame,
    batch_size: int,
    epochs: int, 
    lr: float, 
    model_params: dict, 
    device: str,
    kl_weight: float, # Weight to apply to KL loss
    adv_weight: float, # Weight to apply to adversarial loss
    seed = 111    
):  
    """
    Training loop for CVAE
    """
    # Seed model initialization, latent sampling, and data shuffling
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)

    # Load data
    train_loader, val_loader = get_data_loaders(
        train_df = train_df,
        val_df = val_df,
        batch_size = batch_size,
        generator = generator
    )

    # Loss functions
    recon_loss_fn = F.mse_loss
    time_loss_fn = F.mse_loss
    dose_loss_fn = F.mse_loss

    # VAE + adversarial regressors
    model = CVAE(**model_params)
    model.to(device)
    model.train()

    dose_regressor = LinearRegressor(
        input_dim = model_params["latent_dim"]
    )
    dose_regressor.to(device)
    dose_regressor.train()
    time_regressor = LinearRegressor(
        input_dim = model_params["latent_dim"]
    )
    time_regressor.to(device)
    time_regressor.train()

    # Optimizer for all three models
    all_params = list(model.parameters()) + list(dose_regressor.parameters()) + list(time_regressor.parameters())
    optim = Adam(all_params, lr = lr)

    # Losses
    train_vae_losses = []
    train_adv_losses = []
    val_vae_losses = []
    val_adv_losses = []

    for epoch in range(epochs):
        train_vae_loss = 0.0
        train_adv_loss = 0.0
        val_vae_loss = 0.0
        val_adv_loss = 0.0

        for x, d, t in train_loader:

            x = torch.log1p(x.float().to(device))
            d = d.float().unsqueeze(-1).to(device)
            t = t.float().unsqueeze(-1).to(device)

            # Get reconstruction, latents, and adversarial preds
            x_hat, mu, logvar, z = model(x, d, t)
            x_hat = normalize_log_cpm(x_hat)
            d_hat = dose_regressor(gradient_reverse(z, lambda_adv = adv_weight)) 
            t_hat = time_regressor(gradient_reverse(z, lambda_adv = adv_weight))

            # VAE-specific loss
            recon_loss = recon_loss_fn(x_hat, x)
            reg_loss = kl_loss_fn(mu = mu, logvar = logvar)
            vae_loss = recon_loss + kl_weight * reg_loss

            # Adversarial loss
            adv_loss = time_loss_fn(t, t_hat) + dose_loss_fn(d, d_hat)

            # Total loss
            loss = recon_loss + kl_weight * reg_loss + adv_loss

            # Backprop
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Add loss
            train_vae_loss += vae_loss.item() * x.size(0)
            train_adv_loss += adv_loss.item() * x.size(0)

        # Store epoch losses
        train_vae_losses.append(train_vae_loss / len(train_loader.dataset))
        train_adv_losses.append(train_adv_loss / len(train_loader.dataset))

        model.eval()
        time_regressor.eval()
        dose_regressor.eval()

        with torch.no_grad():

            for x, d, t in val_loader:
                x = torch.log1p(x.float().to(device))
                d = d.float().unsqueeze(-1).to(device)
                t = t.float().unsqueeze(-1).to(device)

                # Get reconstruction, latents, and adversarial preds
                x_hat, mu, logvar, z = model(x, d, t)
                x_hat = normalize_log_cpm(x_hat)
                d_hat = dose_regressor(gradient_reverse(z, lambda_adv = adv_weight)) 
                t_hat = time_regressor(gradient_reverse(z, lambda_adv = adv_weight))

                # VAE-specific loss
                recon_loss = recon_loss_fn(x_hat, x)
                reg_loss = kl_loss_fn(mu = mu, logvar = logvar)
                vae_loss = recon_loss + kl_weight * reg_loss

                # Adversarial loss
                adv_loss = time_loss_fn(t, t_hat) + dose_loss_fn(d, d_hat)

                # Add loss
                val_vae_loss += vae_loss.item() * x.size(0)
                val_adv_loss += adv_loss.item() * x.size(0)

            # Store epoch losses
            val_vae_losses.append(val_vae_loss / len(val_loader.dataset))
            val_adv_losses.append(val_adv_loss / len(val_loader.dataset))

        model.train()
        time_regressor.train()
        dose_regressor.train()

        # Return loss every 10 
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1:3d} : Train VAE loss = {train_vae_losses[-1]:.4f}, Val VAE loss = {val_vae_losses[-1]:.4f}")
    
    return model, train_vae_losses, train_adv_losses, val_vae_losses, val_adv_losses
