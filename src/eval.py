import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA


def plot_latent_pca(
    data: pd.DataFrame,
    model: nn.Module,
    color_by: str
):
    """
    Plot PCA of latent means of all datapoints using trained CVAE encoder module

    Args:
        data     : Single-cell data with metadata
        model    : Trained CVAE model
        color_by : Metadata column to color PCA plot by ("dose" or "timepoint")
    """
    device = next(model.parameters()).device

    meta = data.iloc[:, ~data.columns.str.contains("SP")].reset_index()
    all_sc = data.iloc[:, data.columns.str.contains("SP")]
    all_sc = torch.tensor(all_sc.to_numpy()).float().to(device)

    latents = model.encode(all_sc)[0]

    latents = latents.detach().to("cpu").numpy()
    latent_pca = PCA(n_components = 2)
    proj = latent_pca.fit_transform(latents)
    var1, var2 = latent_pca.explained_variance_ratio_

    res = pd.DataFrame(proj)
    res = pd.concat((res, meta), axis = 1)
    res = res.rename(columns = {0: f"PC1", 1: f"PC2"})

    fig, ax = plt.subplots()
    sns.scatterplot(
        res, 
        x = "PC1",
        y = "PC2",
        hue = color_by,
        s = 20
    )
    ax.set_xlabel(f"PC1 ({var1 * 100:.2f} %)")
    ax.set_ylabel(f"PC2 ({var2 * 100:.2f} %)")
    ax.set_title("PCA on VAE latent means")


def get_composed_latents(
    model: nn.Module,
    d: float,
    t: int,
    num_samples: int
):
    """
    Get samples from latent space with dose and time information composed

    Note: later, add this as a class method in CVAE to make latent sampling easier

    Args:
        model       : Trained CVAE model
        d           : Dose information
        t           : Timepoint information
        num_samples : Number of latent samples
    """
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    d = torch.as_tensor(d, dtype = dtype, device = device).expand(num_samples)
    t = torch.as_tensor(t, dtype = dtype, device = device).expand(num_samples)  

    d = d.reshape(-1, 1)
    t = t.reshape(-1, 1)

    time_embedding = model.time_embed(t)
    dose_embedding = model.dose_embed(d)

    z = torch.randn(
        num_samples, 
        model.latent_dim,
        dtype = dtype,
        device = device
    )
    latent = model.compose(z, time_embedding, dose_embedding)

    return latent


def build_sampled_latent_df(
    model: nn.Module,
    timepoints: list,
    doses: list,
    num_samples: int
):
    """
    Build a dataframe of cells generated across set timepoints and doses.

    Args:
        timepoints  : List of timepoints to generate cells at
        doses       : List of doses to generate cells at
        num_samples : # of cells to generate for each condition

    Returns:
        df : Dataframe of generated cells and associated metadata
    """
    conditions = list(itertools.product(doses, timepoints))

    df = []

    for condition in conditions:
        d, t = condition
        latents = get_composed_latents(
            model = model,
            d = d,
            t = t ,
            num_samples = num_samples
        ).detach().to("cpu").numpy()
        dose_col = np.full((num_samples, 1), d)
        time_col = np.full((num_samples, 1), t)
        cells = np.concat((latents, dose_col, time_col), axis = 1)
        df.append(cells)

    df = pd.DataFrame(np.concat(df))
    df = df.rename(
        columns = {
            df.columns[-1]: "timepoint",
            df.columns[-2]: "dose"
        }
    )

    return df


def plot_composed_latent_pca(
    model: nn.Module,
    timepoints: list,
    doses: list,
    num_samples: int,
    color_by: str,
    title: str
):
    """
    Generated cells according to specified doses and timepoints, then plot PCA on composed latents

    Args:
        model       : Trained CVAE model
        timepoints  : List of timepoints to sample from
        doses       : List of doses to sample from
        num_samples : # of samples to draw from each condition
        color_by    : Metadata column to color plot by ("dose" or "timepoint")
        title       : PCA plot title

    """
    latent_df = build_sampled_latent_df(
        model = model,
        timepoints = timepoints,
        doses = doses,
        num_samples = num_samples
    )
    latent_mask = latent_df.columns.astype(str).str.isnumeric()

    meta = latent_df.loc[:, ~latent_mask].reset_index(drop = True)
    latents = latent_df.loc[:, latent_mask].to_numpy()

    latent_pca = PCA(n_components = 2)
    proj = latent_pca.fit_transform(latents)
    var1, var2 = latent_pca.explained_variance_ratio_

    res = pd.DataFrame(proj)
    res = pd.concat((res, meta), axis = 1)
    res = res.rename(columns = {0: f"PC1", 1: f"PC2"})

    fig, ax = plt.subplots()
    sns.scatterplot(
        res, 
        x = "PC1",
        y = "PC2",
        hue = color_by,
        s = 15,
        alpha = 0.7
    )
    ax.set_xlabel(f"PC1 ({var1 * 100:.2f} %)")
    ax.set_ylabel(f"PC2 ({var2 * 100:.2f} %)")
    ax.set_title(title)