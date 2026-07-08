"""All data extraction functions for models using TPM data to predict CFU"""
import pandas as pd
import numpy as np
import os
from src.metadata import attach_tpm_metadata


def read_fcnts_as_df(folder_path):
    """
    Extracts all feature count csvs from a specified folder and stores them as a list of dataframes

    Args:
        folder_path : File path containing Fcnts csv output from pipeline

    Returns: 
        fcnt_df_list : List of Fcnts dataframes
    """
    
    # Filter out the summary files and keep Fcnts from no-drug control (NDC) comparisons
    all_files = os.listdir(folder_path)
    files = [
        f for f in all_files
        if f.endswith(".csv")
        and ".summary" not in f
        and "NDC0hr" in f
            ]

    if len(files) == 0: 
        raise FileNotFoundError(f"No matching feature count files found in {folder_path}")

    # Convert each csv (which is stored as tab-delimited) to dataframe
    files = sorted(files)
    filenames = ["".join([folder_path, "/" , f]) for f in files]
    fcnt_df_list = [pd.read_table(f, sep = "\t", header = 0, skiprows = 1) for f in filenames]

    return fcnt_df_list


def sample_name_strip(name):
    """
    Convert a sample file name into an easy to read sample name
    Args:
        name : (ex: "/ExpOut/260107_AV242502_RNASeq_miniHT_SpnT4WT_CEF_CIP/Out/Rep/Bams/T4-wt12CEF12CIP1hr-a.bam")
        
    Returns:
        new_name : (ex: 12CEF12CIP1hr-a)
    """
    # Find index of the last / and remove entire prefix (OG file path)
    samplename_start_idx = name.strip().rfind("/") + 1
    new_name = name[samplename_start_idx:]

    # Find index of . (.bam is at end of sample name) and remove filetag
    filetag_start_idx = new_name.rfind(".")
    new_name = new_name[:filetag_start_idx]

    # Remove "T4-wt"
    new_name = new_name.replace("T4-wt", "")

    return new_name


def fcnts_to_tpms(fcnt_df_list):
    """
    Converts a list of RNA-seq feature count dataframes to a list of TPM dataframes

    Args:
        fcnt_df_list : List of feature count dataframes

    Returns:
        tpm_df_list : List of TPM dataframes

    """
    tpm_df_list = []

    for df in fcnt_df_list:

        # Move gene names to index and keep only length, Fcnts
        df = df.set_index("Geneid")
        df = df.loc[:, [col for col in list(df.columns) if col == "Length" or col.startswith("/")]]
        df = df.apply(lambda column: [int(entry) for entry in column])

        # Convert gene length from b -> kb
        df["Length"] = df["Length"].apply(lambda column: column / 1000)

        # Select Fcnts columns by excluding Length column
        fcnts_cols = [col for col in list(df.columns) if col != "Length"]
        
        # Fcnts / gene length = (Counts per kb)
        df[fcnts_cols] = df[fcnts_cols].apply(lambda column: column / df["Length"])

        # (Counts per kb) * 10^6 / (Total counts/kb) = TPM
        df[fcnts_cols] = df[fcnts_cols].apply(lambda column: column * 10**6 / sum(column))

        df.drop(columns = "Length", inplace = True)
        df = df.rename(columns = lambda column: sample_name_strip(column))
        tpm_df_list.append(df)        

    return tpm_df_list


def bind_tpm_data(tpm_df_list):
    """
    Function to take a list of TPM dataframes, then bind all into 1 dataframe
    Args: 
        tpm_df_list : list of TPM dataframes

    Returns:
        all_tpms [N,G] : dataframe with all TPM values (N samples on row, G genes on column)
    """
    tpm_df_list_uniq = []

    # Column names of 1st DF (all have last 3 cols)
    colnames = list(tpm_df_list[0].columns)

    # Select redundant NDC0hr columns and make new df with just those
    ndc0hr_cols = [col for col in colnames if "NDC0hr" in col]
    ndc0hr_df = tpm_df_list[0][ndc0hr_cols]
    tpm_df_list_uniq.append(ndc0hr_df)

    # Remove NDC0hr columns from all dfs
    for df in tpm_df_list:
        columns = list(df.columns)
        relevant_idx = [col for col in columns if "NDC0hr" not in col]
        stripped_df = df[relevant_idx]
        tpm_df_list_uniq.append(stripped_df)

    # Join and transpose to get genes on columns
    all_tpms = pd.concat(tpm_df_list_uniq, axis = 1, join = "outer").T

    return all_tpms


def get_all_tpm_data(fcnts_path):
    """
    Function to run entire data extraction pipeline

    Args:
        fcnts_path : Path to folder containing Fcnts output of lab pipeline
        cfu_path   : Path to folder containing CFU csv outputs

    Returns:
        all_data : [N, G+1] : Dataframe of all TPMs and CFUs as last column 
    """
    stored_fcnts = read_fcnts_as_df(fcnts_path)
    stored_tpms  = fcnts_to_tpms(stored_fcnts)
    tpm_df       = bind_tpm_data(stored_tpms)
    all_data     = attach_tpm_metadata(tpm_df)
    
    return all_data