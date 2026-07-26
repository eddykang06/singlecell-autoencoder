import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from torch.optim import Adam
from src.ae import SimpleAE, TorchStandardScaler
from src.vae import CVAE, MLPRegressor, gradient_reverse
from src.sc_data import df_to_tensors, get_data_loaders


"""Autoencoder training loop"""
def train_simple_ae(
    data: torch.Tensor, 
    batch_size: int,
    epochs: int, 
    lr: float, 
    model_params: dict, 
    device: str,
    train_size = 0.8, 
    seed = 111
):
    """
    Simple training loop for generic AE training
    """

    # Train val split
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(data)
    train_dataset, val_dataset = random_split(
        dataset = dataset, 
        lengths = [train_size, 1 - train_size], 
        generator = generator
    )
    train_loader = DataLoader(
        dataset = train_dataset, 
        batch_size = batch_size, 
        shuffle = True
    )
    val_loader = DataLoader(
        dataset = val_dataset, 
        batch_size = batch_size, 
        shuffle = True
    )

    model = SimpleAE(**model_params)
    model.to(device)
    model.train()

    optim = Adam(model.parameters(), lr = lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        for batch_tuple in train_loader:
            
            # Get batch
            (x,) = batch_tuple
            x = x.to(device).float()

            # Compute loss
            xhat = model(x)
            loss = F.mse_loss(xhat, x)

            # Backprop
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Total loss = avg loss * batchsize
            epoch_train_loss += loss.item() * x.size(0)
        
        # Compute average 
        train_losses.append(epoch_train_loss / len(train_dataset))

        model.eval()

        with torch.no_grad():
            

            for batch_tuple in val_loader:

                (x,) = batch_tuple
                x = x.to(device).float()

                # Compute loss
                xhat = model(x)
                loss = F.mse_loss(xhat, x)

                # Total loss = avg loss * batchsize
                epoch_val_loss += loss.item() * x.size(0)
            
            # Compute average 
            val_losses.append(epoch_val_loss / len(val_dataset))

            model.train()

        # Return loss every 10 
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1:3d} : Train MSE = {train_losses[-1]:.4f}, Val MSE = {val_losses[-1]:.4f}")
    
    return model, train_losses, val_losses


"""VAE training loop"""
def kl_loss_fn(
    mu: float, 
    std: float
):
    """
    Formula for KL divergence between N(0,1) and current mu, std
    """
    kl_loss = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=1).mean()

    return kl_loss


def train_cvae(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame,
    batch_size: int,
    epochs: int, 
    lr: float, 
    model_params: dict, 
    device: str,
    kl_weight: float,
    seed = 111    
):  
    # Load data
    generator = torch.Generator().manual_seed(seed)
    train_loader, val_loader = get_data_loaders(
        train_df = train_df,
        val_df = val_df
    )
    scaler = TorchStandardScaler()

    # Loss functions
    recon_loss_fn = F.mse_loss()
    time_loss_fn = F.mse_loss()
    dose_loss_fn = F.mse_loss()

    # VAE + adversarial regressors
    model = CVAE(**model_params)
    model.to(device)
    model.train()
    optim = Adam(model.parameters(), lr = lr)

    dose_regressor = MLPRegressor(
        input_dim = model_params["latent_dim"]
    )
    time_regressor = MLPRegressor(
        input_dim = model_params["latent_dim"]
    )

    # Losses
    train_vae_losses = []
    train_adv_losses = []
    val_vae_losses = []
    val_adv_losses = []

    for epoch in epochs:
        train_vae_loss = 0.0
        train_adv_loss = 0.0
        val_vae_loss = 0.0
        val_adv_loss = 0.0

        for x, d, t in train_loader:

            x = x.float().to(device)
            d = d.float().unsqueeze(-1).to(device)
            t = t.float().unsqueeze(-1).to(device)

            # Get reconstruction, latents, and adversarial preds
            x_hat, mu, std, z = model(x, d, t)
            d_hat = dose_regressor(gradient_reverse(z, lambda_adv = 0.1)) 
            t_hat = time_regressor(gradient_reverse(z, lambda_adv = 0.1))

            # VAE-specific loss
            recon_loss = recon_loss_fn(x_hat, x).item()
            reg_loss = kl_loss_fn(mu = mu, std = std).item()
            vae_loss = recon_loss + kl_weight * reg_loss

            # Adversarial loss
            adv_loss = time_loss_fn(t, t_hat).item() + dose_loss_fn(d, d_hat).item()

            # Total loss
            loss = recon_loss + kl_weight * reg_loss + adv_loss

            # Backprop
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Add loss
            train_vae_loss += vae_loss * x.size(0)
            train_adv_loss += adv_loss * x.size(0)

        # Store epoch losses
        train_vae_losses.append(train_vae_loss / len(train_loader.dataset))
        train_adv_losses.append(train_adv_loss / len(train_loader.dataset))

        model.eval()

        with torch.no_grad():

            for x, d, t in val_loader:
                x = x.float().to(device)
                d = d.float().unsqueeze(-1).to(device)
                t = t.float().unsqueeze(-1).to(device)

                # Get reconstruction, latents, and adversarial preds
                x_hat, mu, std, z = model(x, d, t)
                d_hat = dose_regressor(gradient_reverse(z, lambda_adv = 0.1)) 
                t_hat = time_regressor(gradient_reverse(z, lambda_adv = 0.1))

                # VAE-specific loss
                recon_loss = recon_loss_fn(x_hat, x).item()
                reg_loss = kl_loss_fn(mu = mu, std = std).item()
                vae_loss = recon_loss + kl_weight * reg_loss

                # Adversarial loss
                adv_loss = time_loss_fn(t, t_hat).item() + dose_loss_fn(d, d_hat).item()

                # Add loss
                val_vae_loss += vae_loss * x.size(0)
                val_adv_loss += adv_loss * x.size(0)

            # Store epoch losses
            val_vae_losses.append(val_vae_loss / len(val_loader.dataset))
            val_adv_losses.append(val_adv_loss / len(val_loader.dataset))

        model.train()

        # Return loss every 10 
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1:3d} : Train VAE loss = {train_vae_losses[-1]:.4f}, Val MSE = {val_vae_losses[-1]:.4f}")
    
    return model, train_vae_losses, train_adv_losses, val_vae_losses, val_adv_losses