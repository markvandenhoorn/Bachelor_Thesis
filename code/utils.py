"""
Name: utils.py
Author: Mark van den Hoorn
Desc: Utility file for file handling and importing and data handling.
"""
import os
import random
import pandas as pd
import spacy
from tqdm import tqdm
import re
import html

def set_wd():
    """
    Set working directory to data folder
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.chdir(data_dir)

def is_only_mask(text):
    """Util function to check if an utterance is only masks"""
    return text.strip() == "MASK" or text.strip() == "MASK MASK"

def clean_encoded_characters(df, column_name):
    """
    Clean HTML encoded characters like &8217 in the specified column,
    replacing them with regular apostrophes (').
    """
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'&\d{4}', "'", x))
    return df

def import_data(filename):
    """
    Reads in the data, drops NA, removes masks and replaces an with a.
    """
    data = pd.read_csv(filename, index_col=0)
    parent_data = data.dropna(subset=["p_utts", "age_months"])
    child_data = data.dropna(subset=["c_utts", "age_months"])

    # remove sentences with only MASK
    parent_utterances = parent_data[~parent_data["p_utts"].apply(is_only_mask)][["p_utts", "age_months"]]
    child_utterances = child_data[~child_data["c_utts"].apply(is_only_mask)][["c_utts", "age_months"]]

    # replace 'an' with 'a' for consistency
    parent_utterances['p_utts'] = parent_utterances['p_utts'].str.replace(r'\ban\b', 'a', regex=True)
    parent_utterances['p_utts'] = parent_utterances['p_utts'].str.replace(r'\bAn\b', 'A', regex=True)
    parent_utterances['p_utts'] = parent_utterances['p_utts'].str.replace(r'@', '', regex=True)
    child_utterances['c_utts'] = child_utterances['c_utts'].str.replace(r'\ban\b', 'a', regex=True)
    child_utterances['c_utts'] = child_utterances['c_utts'].str.replace(r'\bAn\b', 'A', regex=True)
    child_utterances['c_utts'] = child_utterances['c_utts'].str.replace(r'@', '', regex=True)

    # revert encoded characters back to original
    parent_utterances = clean_encoded_characters(parent_utterances, 'p_utts')
    child_utterances = clean_encoded_characters(child_utterances, 'c_utts')

    return parent_utterances, child_utterances
