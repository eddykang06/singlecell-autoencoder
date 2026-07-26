"""Data loading pipeline for scRNA-seq"""

import pandas as pd
import os
from pathlib import Path
from src.metadata import (
    condition_to_dose,
    condition_to_drug_id,
    condition_to_timepoint
)


def raw_to_cpm(df):
    """
    Convert raw single-cell counts to CPM values
    """
    df = df * 10**6 / df.sum(axis = 0)

    return df


def get_sc_data(path):
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