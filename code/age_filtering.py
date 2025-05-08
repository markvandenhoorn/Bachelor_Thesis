"""
Name: age_filtering.py
Author: Mark van den Hoorn
Desc: Creates datasets based on age. Creates a dataset for children's age of
0-14, 15-18, 19-22 etcetera.
"""
import pandas as pd
import os
from tqdm import tqdm

def filter_by_age_range(utterances_df, min_age, max_age):
    """
    Filter out utterances based on age range.
    """
    return utterances_df[(utterances_df['age_months'] > min_age) & (utterances_df['age_months'] <= max_age)]

def save_age_based_datasets(utterances_df, age_bins, filename_prefix):
    """
    Save a dataset with all incrementing age ranges, 0-14, 15-18, 19-22 etc.
    """
    output_dir = os.getcwd()

    for i in tqdm(range(len(age_bins) - 1), desc=f"Saving datasets by age range"):
        min_age = age_bins[i]
        max_age = age_bins[i + 1]
        filtered_data = filter_by_age_range(utterances_df, min_age, max_age)

        # save a file with the current data
        output_filename = f"{filename_prefix}_up_to_{max_age}_months.txt"
        filtered_data['utt'].to_csv(output_filename, index=False, header=False)
