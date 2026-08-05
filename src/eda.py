import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def plot_pseudo_v_bulk(
    sc_data: pd.DataFrame,
    bulk_data: pd.DataFrame,
    timepoint: int,
    dose: float
):
    """
    Plot psueodbulked profiles vs bulk profiles to measure correlation.

    Args:
        sc_data   : Single-cell CPM dataframe with metadata
        bulk_data : Bulk TPM dataframe with metadata
        timepoint : Data timepoint
        dose      : CEF dosage
    """
    # Filter and get pseudobulk
    sc_mask = (sc_data["timepoint"] == timepoint) & (sc_data["dose"] == dose)
    sc = sc_data[sc_mask]
    sc = sc.iloc[:, sc.columns.str.contains("SP")]
    sc = sc.mean(axis = 0)

    # Filter and get bulk
    bulk_mask = (bulk_data["timepoint"] == timepoint) & (bulk_data["drug1_dose"] == dose) & (bulk_data["drug1"] == "CEF") & (bulk_data["num_drugs"] == 1)
    bulk = bulk_data[bulk_mask]
    bulk = bulk.iloc[:, bulk.columns.str.contains("SP")]
    bulk = bulk.mean(axis = 0)

    # Bind and plot
    res = pd.concat((bulk, sc), join = "inner", axis = 1)
    res.columns = ["Bulk TPM", "Pseudobulked CPM"]
    fig, ax = plt.subplots()
    sns.scatterplot(
        res, 
        x = "Bulk TPM", 
        y = "Pseudobulked CPM", 
        alpha = 0.7,
        ax = ax
    )

    limits = [
        min(res["Bulk TPM"].min(), res["Pseudobulked CPM"].min()),
        max(res["Bulk TPM"].max(), res["Pseudobulked CPM"].max())
    ]

    pearson = pearsonr(res["Bulk TPM"], res["Pseudobulked CPM"])
    plt.plot(limits, limits, "k--", linewidth = 1)
    plt.title(f"CEF at {dose}x MIC, {timepoint} hour (Pearson r = {pearson.statistic:.3f})")

