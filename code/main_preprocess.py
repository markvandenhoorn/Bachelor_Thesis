"""
Name: main.py
Author: Mark van den Hoorn
Desc: Does all data preprocessing, creating datasets for model training etc. Uses
the other utility files to achieve this.
"""
import os
from utils import set_wd, import_data
from preprocessing import first_preprocess, filter_determiner_sentences, filter_utterances_by_nouns
from pos_tagging import pos_tagging, assign_determiner_types, filter_det_noun_pairs, get_nouns
from age_filtering import save_age_based_datasets
import pandas as pd

### TODO: Might be that all sentences containing a noun that is not in training
### are removed, even if that noun is not part of a det+noun combo
### TODO: filter test sentences based on nouns that also appear in training
### TODO: Run code on all data (check import_data in utils.py)

if __name__ == "__main__":
    # set the working directory
    set_wd()

    # import data
    parent_utterances_raw, child_utterances_raw = import_data("ldp_data.csv")

    # import first names
    first_names = list(pd.read_csv("first_names.csv")['Name'])

    # preprocess the data (lemmatization, name replacement)
    parent_utterances_processed = first_preprocess(parent_utterances_raw, first_names)
    child_utterances_processed = first_preprocess(child_utterances_raw, first_names)

    # save current parent data as tokenizer train data
    parent_utterances_processed.to_csv("tokenizer_train_data.txt", index = False, header = False)

    # POS tagging and determiner filtering
    pos_tagged_child = pos_tagging(child_utterances_processed)
    child_test_regular, child_test_masked = filter_determiner_sentences(pos_tagged_child)

    pos_tagged_parent = pos_tagging(parent_utterances_processed)
    pair_dict = assign_determiner_types(pos_tagged_parent)
    filtered_utterances = filter_det_noun_pairs(pos_tagged_parent, pair_dict)

    # get nouns from the utterances
    filtered_nouns = get_nouns(filtered_utterances)
    regular_nouns = get_nouns(pos_tagged_parent)

    # make sure only det+noun are in test set where noun is also in training set
    child_test_regular_ = filter_utterances_by_nouns(child_test_regular, regular_nouns)
    child_test_masked_ = filter_utterances_by_nouns(child_test_masked, regular_nouns)
    child_test_regular_filtered = filter_utterances_by_nouns(child_test_regular, filtered_nouns)
    child_test_masked_filtered = filter_utterances_by_nouns(child_test_masked, filtered_nouns)

    # save test sets
    child_test_regular_.to_csv("test_data_regular.txt", index = False, header = False)
    child_test_masked_.to_csv("test_data_masked.txt", index = False, header = False)
    child_test_regular_filtered.to_csv("test_data_regular_filtered.txt", index = False, header = False)
    child_test_masked_filtered.to_csv("test_data_masked_filtered.txt", index = False, header = False)

    # save datasets based on age ranges
    age_ranges = [0, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58]
    save_age_based_datasets(filtered_utterances, age_ranges, "filtered_train_data")
    save_age_based_datasets(parent_utterances_processed, age_ranges, "unfiltered_train_data")
