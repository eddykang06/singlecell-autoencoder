"""All functions for metadata extraction from condition labels"""
import numpy as np 
import pandas as pd

def condition_to_replicate_idx(condition_list):
    """
    Function to convert a list of conditions into replicate labels

    Args:
        condition_list : List of condition labels, ex: ["12CIP1hr-a", "12CIP1hr-b",...]

    Returns:
        idx_list : List of condition labels converted into replicate indices, ex: [0, 0, 1,...]
    """
    idx_list = [0]

    for i in range(1, len(condition_list)):
        prev_condition = condition_list[i-1]   
        curr_condition = condition_list[i]

        # Exclude last character since "-a" "-b"
        if prev_condition[:-1] == curr_condition[:-1]:
            idx_list.append(idx_list[i-1])
        
        else:
            idx_list.append(idx_list[i-1] + 1)
    
    return idx_list
    

def find_first_alpha(str):
    """
    Function to find the index of the the first letter in a string
    (will be used for drug name parser function)

    Args:
        str : string

    Returns:
        first_alpha : idx of first letter
    """
    for i in range(len(str)):
        if str[i].isalpha():
            return i
            break 


def condition_to_drug_id(condition_label):
    """
    Function to convert a condition label to a drug ID

    Args:
        condition_label : Condition label, ex: "12CEF1hr", "12CIP13VNC2hr"

    Returns:
        drug_id : Drug Id as a single string, ex: "CEF", "CIP+VNC"
    """
    # Count # of alpha characters
    num_upper = np.sum([char.isupper() for char in condition_label])

    # Single drug case
    if num_upper == 3:

        # Find drug name using letter search
        first_alpha_idx = find_first_alpha(condition_label)
        drug_id = condition_label[first_alpha_idx:first_alpha_idx + 3]
        return drug_id
    
    # Multiple drug case
    elif num_upper == 6:

        # Find first letter and extract drug name
        first_idx = find_first_alpha(condition_label)
        drug1 = condition_label[first_idx:first_idx + 3]

        # Find first letter in remaining string and extract drug name
        substr     = condition_label[first_idx + 3:]
        second_idx = find_first_alpha(substr)
        drug2      = substr[second_idx:second_idx + 3]

        # Merge drug names
        drug_id = drug1 + "+" + drug2

        return drug_id

    else:
        raise KeyError("Condition label does not fit drug ID criteria")


def condition_to_drugs(condition_label):
    """
    Function to convert a condition label to a list of the two drugs represented

    Args:
        condition_label : Condition label, ex: "12CEF1hr", "12CIP13VNC2hr"

    Returns:
        drugs : List of drugs (length 2, "" if 2nd drug doesn't exist), ex: ["CEF", NA], ["CIP", NA]
    """
    # Count # of alpha characters
    num_upper = np.sum([char.isupper() for char in condition_label])

    # Single drug case
    if num_upper == 3:

        # Find drug name using letter search
        first_alpha_idx = find_first_alpha(condition_label)
        drug = condition_label[first_alpha_idx:first_alpha_idx + 3]
        drugs = [drug, np.nan]
        return drugs
    
    # Multiple drug case
    elif num_upper == 6:

        # Find first letter and extract drug name
        first_idx = find_first_alpha(condition_label)
        drug1 = condition_label[first_idx:first_idx + 3]

        # Find first letter in remaining string and extract drug name
        substr = condition_label[first_idx + 3:]
        second_idx = find_first_alpha(substr)
        drug2 = substr[second_idx:second_idx + 3]
        drugs = [drug1, drug2]

        return drugs

    else:
        raise KeyError("Condition label does not fit drug ID criteria")
    

def condition_to_dose(condition_label):
    """
    Function to convert a condition label to a list of the two drugs represented

    Args:
        condition_label : Condition label, ex: "12CEF1hr", "12CIP13VNC2hr"

    Returns:
        doses : List of doses (length 2, "" if 2nd drug doesn't exist), ex: [0.50, NA], [0.50, 0.33]
    """
    # Define dose mapping
    dose_map = {
        "14": 0.25,
        "13": 0.33,
        "12": 0.50,
        "34": 0.75,
        "1": 1
    }

    # Count # of alpha characters
    num_upper = np.sum([char.isupper() for char in condition_label])

    # Single drug case
    if num_upper == 3:
        
        if "NDC" in condition_label:
            return [np.nan, np.nan]
        
        # Find drug name using letter search
        first_alpha_idx = find_first_alpha(condition_label)
        dose = condition_label[:first_alpha_idx]
        dose = dose_map[dose]
        doses = [dose, np.nan]
        return doses
    
    # Multiple drug case
    elif num_upper == 6:

        # Find first letter and extract drug name
        first_idx = find_first_alpha(condition_label)
        dose1 = condition_label[:first_idx]
        dose1 = dose_map[dose1]

        # Find first letter in remaining string and extract drug name
        substr = condition_label[first_idx + 3:]
        second_idx = find_first_alpha(substr)
        dose2 = substr[:second_idx]
        dose2 = dose_map[dose2]
        doses = [dose1, dose2]

        return doses

    else:
        raise KeyError("Condition label does not fit drug ID criteria")


def condition_to_timepoint(condition_label):
    """
    Function to convert a condition label to a timepoint

    Args:
        condition_label : Condition label, ex: "12CEF1hr"

    Returns:
        timepoint : Time, ex: 1
    """
    time_map = {
        "0hr": 0,
        "1hr": 1,
        "2hr": 2,
        "4hr": 4
    }

    # Find the timepoint label
    time_idx = condition_label.find("hr")
    timepoint = condition_label[time_idx - 1:time_idx + 2]

    # Convert to int
    timepoint = time_map[timepoint]

    return timepoint


def condition_to_ndc_cfu(condition_label, ndc_cfus):
    """
    Function to convert a condition label to the corresponding timepoint CFU
    
    Args:
        condition_label : Condition label, ex: "12CEF1hr"
        ndc_cfus        : Array of 1hr, 2hr, 4hr NDC CFUs in order

    Returns:
        ndc_cfu : CFU value for NDC at corresponding timepoint
    """
    # Convert label to timepoint
    timepoint = condition_to_timepoint(condition_label)

    # Define map from timepoint to corresponding NDC CFU
    time_to_cfu = {
        1: ndc_cfus[0],
        2: ndc_cfus[1],
        4: ndc_cfus[2]
    }

    ndc_cfu = time_to_cfu[timepoint]

    return ndc_cfu


def attach_tpm_metadata(df):
    """
    Function to attach corresponding metadata to data df

    Args:
        df : Dataframe with gene expression values on columns
    
    Returns:
        df : Dataframe with metadata and time-matched NDC CFU data attached
    """
    # Get condition IDs
    labels = df.index

    # Extract metadata from condition IDs
    drug_id = [condition_to_drug_id(label) for label in labels]
    num_drugs = [2 if "+" in id else 1 for id in drug_id]
    drug = [condition_to_drugs(label) for label in labels]
    drug1 = [x[0] for x in drug]
    drug2 = [x[1] for x in drug]
    dose = [condition_to_dose(label) for label in labels]
    dose1 = [x[0] for x in dose]
    dose2 = [x[1] for x in dose]
    timepoint = [condition_to_timepoint(label) for label in labels]

    # Construct a new dataframe
    metadata_df = pd.DataFrame({
        "drug_id": drug_id,
        "num_drugs": num_drugs,
        "drug1": drug1,
        "drug2": drug2,
        "drug1_dose": dose1,
        "drug2_dose": dose2,
        "timepoint": timepoint,
    })
    metadata_df.index = labels

    # Join with original dataframe
    df = pd.merge(df, metadata_df, left_index = True, right_index = True, how = "left")

    return df