"""Data loading and preprocessing pipeline for scRNA-seq"""

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

"""Data loading"""
def raw_to_cpm(
    df: pd.DataFrame
):
    row_totals = df.sum(axis=1)
    row_totals = row_totals.replace(0, float("nan"))
    cpm = df.div(row_totals, axis = 0) * 1_000_000
    cpm = cpm.fillna(0)

    return cpm


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

    # Handle NA values
    cat = cat.fillna(0)

    return cat


"""Data processing"""
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
        return self

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

    def inverse_transform(self, tensor):
        if self.means is None or self.stds is None:
            raise ValueError(f"Standard scaler has not been fitted")
        if tensor.shape[1] != len(self.means):
            raise KeyError(f"Input tensor has incorrect number of features")

        means = self.means.to(device = tensor.device, dtype = tensor.dtype)
        stds = self.stds.to(device = tensor.device, dtype = tensor.dtype)
        unscaled = stds * tensor + means

        return unscaled


def df_to_tensors(
    df: pd.DataFrame
):
    """
    Convert a dataframe with metadata into a three tensors that can be fed into a torch TensorDataset
    """
    X = torch.tensor(df.iloc[:, df.columns.str.contains("SP")].to_numpy())
    d = torch.tensor(df["dose"].to_numpy())
    t = torch.tensor(df["timepoint"].to_numpy())

    return X, d, t

    
def get_data_loaders(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    batch_size: int,
    shuffle = True,
    generator = None
):
    """
    Get dataloaders from train and val dataframes
    """
    if not train_df.columns.equals(val_df.columns):
        raise ValueError("Train and validation dataframes must have identical columns in the same order")
    X_train, d_train, t_train = df_to_tensors(train_df)
    X_val, d_val, t_val = df_to_tensors(val_df)
    train_dataset = TensorDataset(X_train, d_train, t_train)
    val_dataset = TensorDataset(X_val, d_val, t_val)

    train_loader = DataLoader(
        train_dataset, 
        batch_size = batch_size, 
        shuffle = shuffle,
        generator = generator
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size = batch_size, 
        shuffle = False
    )

    return train_loader, val_loader


def get_standard_scaler(
    train_df: pd.DataFrame
):
    """
    Get means and stds for StandardScaler
    """
    scaler = TorchStandardScaler()
    X_train, _, _ = df_to_tensors(train_df)
    scaler.fit(X_train)

    return scaler