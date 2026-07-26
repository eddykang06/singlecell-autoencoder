"""Data loading pipeline for scRNA-seq"""

import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from src.metadata import (
    condition_to_dose,
    condition_to_drug_id,
    condition_to_timepoint
)


def raw_to_cpm(
    df: pd.DataFrame
):
    """
    Convert raw single-cell counts to CPM values
    """
    df = df * 10**6 / df.sum(axis = 0)

    return df


def get_sc_data(
    path: str | Path
):
    """
    Function to load all single cell data from data directory
    """
    # Store processed dataframes
    dfs = []

    # Load each dataframe as CPMs
    files = os.listdir(path)
    for f in files:
        file_path = str(Path(path) / f)
        df = pd.read_table(file_path, sep = "\t", index_col = 0).T
        df = df.reset_index(drop = True)
        df = raw_to_cpm(df)

        # Metadata
        df["timepoint"] = condition_to_timepoint(f)
        df["dose"] = condition_to_dose(f)[0]
        df["drug"] = condition_to_drug_id(f)
        dfs.append(df)

    # Concatenate data and sort features
    cat = pd.concat(dfs, axis = 0, join = "outer")
    cat = cat.dropna(axis = 1, how = "all")
    cat = cat.sort_index(axis = 1)

    return cat


def df_to_tensors(
    df: pd.DataFrame
):
    """
    Convert a dataframe with metadata into a three tensors that can be fed into a torch TensorDataset
    """
    X = torch.tensor(df[df.columns.str.contains("SP")])
    d = torch.tensor(df["dose"])
    t = torch.tensor(df["timepoint"])

    return X, d, t


def get_data_loaders(
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        batch_size: int,
        shuffle = True
    ):
    """
    Get dataloaders from train and val dataframes
    """
    X_train, d_train, t_train = df_to_tensors(train_df)
    X_val, d_val, t_val = df_to_tensors(val_df)
    train_dataset = TensorDataset(X_train, d_train, t_train)
    val_dataset = TensorDataset(X_val, d_val, t_val)

    train_loader = DataLoader(
        train_dataset, 
        batch_size = batch_size, 
        shuffle = True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size = batch_size, 
        shuffle = True
    )

    return train_loader, val_loader