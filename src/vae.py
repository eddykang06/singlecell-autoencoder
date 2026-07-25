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


class SinusoidalEmbed(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim


class ConditionComposition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, time_embedding, )


    