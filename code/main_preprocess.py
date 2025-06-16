"""
Name: main_preprocess.py
Author: Mark van den Hoorn
Desc: Does all data preprocessing, creating datasets for model training etc. Uses
the other utility files to achieve this.
"""
import os
import pandas as pd
import pickle
import argparse
from utils import set_wd, import_data
from preprocessing import first_preprocess, filter_determiner_sentences
from pos_tagging import pos_tagging, assign_determiner_types, filter_det_noun_pairs, get_nouns, remove_empty_sentences, run_filter_iterations
from age_filtering import save_age_based_datasets
from collections import Counter, defaultdict

if __name__ == "__main__":
    # set the working directory and set up argparsing
    parser = argparse.ArgumentParser(description="Run full preprocessing pipeline.")
    parser.add_argument('--remove_conflicts', action='store_true', help="Remove conflicting det-noun pairs instead of adapting them")
    args = parser.parse_args()
    set_wd()

    # import data
    parent_utterances_raw, child_utterances_raw = import_data("ldp_data.csv")

    # import first names
    first_names = list(pd.read_csv("first_names.csv")['Name'])

    # preprocess the data (lemmatization, name replacement)
    parent_utterances_processed = first_preprocess(parent_utterances_raw, first_names)
    parent_utterances_processed = remove_empty_sentences(parent_utterances_processed)
    child_utterances_processed = first_preprocess(child_utterances_raw, first_names)

    # save current parent data as tokenizer train data
    parent_utterances_processed.to_csv("tokenizer_train_data.txt", index = False, header = False)

    # POS tagging and determiner filtering
    pos_tagged_child = pos_tagging(child_utterances_processed)
    pos_tagged_parent = pos_tagging(parent_utterances_processed)
    pos_tagged_parent.to_csv("pos_tagged_data.txt", index = False, header = False)
    pos_tagged_child.to_csv("pos_tagged_child.txt", index = False, header = False)

    # POS tag parent utts and filter out inconsistent determiner use
    # some nouns are allowed to be seen with both determiners, percentages reflect that
    pair_dict_regular = assign_determiner_types(pos_tagged_parent, both_chance = 1)
    pair_dict_0 = assign_determiner_types(pos_tagged_parent)
    pair_dict_25 = assign_determiner_types(pos_tagged_parent, both_chance = 0.25)
    pair_dict_50 = assign_determiner_types(pos_tagged_parent, both_chance = 0.50)
    pair_dict_75 = assign_determiner_types(pos_tagged_parent, both_chance = 0.75)

    # filter based on assigned determiners
    filtered_utterances_0 = run_filter_iterations(pos_tagged_parent, pair_dict_0,
        num_passes=2, remove_conflicts=args.remove_conflicts)
    filtered_utterances_25 = run_filter_iterations(pos_tagged_parent, pair_dict_25,
        num_passes=2, remove_conflicts=args.remove_conflicts)
    filtered_utterances_50 = run_filter_iterations(pos_tagged_parent, pair_dict_50,
        num_passes=2, remove_conflicts=args.remove_conflicts)
    filtered_utterances_75 = run_filter_iterations(pos_tagged_parent, pair_dict_75,
        num_passes=2, remove_conflicts=args.remove_conflicts)

    # save noun to determiner mapping so we can use it elsewhere
    with open("determiner_dicts.pkl", "wb") as f:
        pickle.dump({
            "regular": pair_dict_regular,
            "0": pair_dict_0,
            "25": pair_dict_25,
            "50": pair_dict_50,
            "75": pair_dict_75
        }, f)

    # save datasets based on age ranges
    age_ranges = [0, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58]
    save_age_based_datasets(filtered_utterances_0, age_ranges, "filtered_train_data_0")
    save_age_based_datasets(filtered_utterances_25, age_ranges, "filtered_train_data_25")
    save_age_based_datasets(filtered_utterances_50, age_ranges, "filtered_train_data_50")
    save_age_based_datasets(filtered_utterances_75, age_ranges, "filtered_train_data_75")
    save_age_based_datasets(parent_utterances_processed, age_ranges, "unfiltered_train_data")
