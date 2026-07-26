import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from torch.optim import Adam
from src.ae import SimpleAE, TorchStandardScaler


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

    # Store losses
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
def kl_loss_fn(mu, std):
    """
    Formula for KL divergence between N(0,1) and current mu, std
    """
    kl_loss = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=1).mean()

    return kl_loss


def train_cvae(
    X_train: torch.Tensor, 
    X_val: torch.Tensor,
    batch_size: int,
    epochs: int, 
    lr: float, 
    model_params: dict, 
    device: str,
    kl_weight: float,
    seed = 111    
):  
    generator = torch.Generator().manual_seed(seed)

    # Get train and val data loaders
    train_dataset = TensorDataset(X_train)
    val_dataset = TensorDataset(X_val)
    train_loader = DataLoader(train_dataset)
    val_loader = DataLoader(val_dataset)

    scaler = TorchStandardScaler()

    # Define loss functions
    recon_loss_fn = F.mse_loss()
    kl_loss_fn = kl_loss_fn
    adv_loss_fn = F.mse_loss()

    for batch in train_loader:





    # Calculate MSE loss

    # Calculate KL loss
    kl_loss = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=1).mean()

    # Calculate adversarial losses


    # Sum loss

    
