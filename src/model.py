"""Model architectures"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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



