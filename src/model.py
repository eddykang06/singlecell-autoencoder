"""Model architectures"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchStandardScaler:
    """
    Custom standard scaler class compatible with torch tensors
    """
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, tensor):
        """
        Note: Tensor must be sample x feature
        """
        self.means = tensor.mean(dim = 0)
        self.stds = tensor.std(dim = 0)

    def transform(self, tensor):
        if self.means is None or self.stds is None:
            raise ValueError(f"Standard scaler has not been fitted")
        if tensor.shape[1] != len(self.means):
            raise KeyError(f"Input tensor has incorrect number of features")
        
        means = self.means.to(device = tensor.device, dtype = tensor.dtype)
        stds = self.stds.to(device = tensor.device, dtype = tensor.dtype)
        scaled = (tensor - means) / stds

        return scaled

    def fit_transform(self, tensor):
        scaled = self.fit(tensor).transform(tensor)

        return scaled
    

class SimpleAE(nn.Module):

    def __init__(self, input_dim, h_dim, latent_dim):
        super().__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim

        self.enc_fc1 = nn.Linear(input_dim, h_dim)
        self.enc_fc2 = nn.Linear(h_dim, latent_dim)
        self.dec_fc1 = nn.Linear(latent_dim, h_dim)
        self.dec_fc2 = nn.Linear(h_dim, input_dim)    
    
    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        z = F.relu(self.enc_fc2(h))

        return z

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        x_hat = F.relu(self.dec_fc2(h))

        return x_hat
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)

        return x_hat



