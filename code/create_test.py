"""
Name: create_test.py
Author: Mark van den Hoorn
Desc: Uses the pos tagged children data to create test sets. These test sets
consist of sentences where nouns are used that are seen with only one type of
determiner throughout the data. It does this for every type of model, so for all
5 types, and for every age range as well.
"""
import pandas as pd
import ast
import pickle
from tqdm import tqdm
from utils import set_wd

def create_test_set(utterances_df, noun_to_types, age_ranges, lookahead=2):
    """
    Creates test sets per age range with utterances containing nouns seen with
    only one determiner type. Includes utterances where the noun is preceded by
    a determiner (optionally with one word in between). Creates a test set per
    age range.
    """
    # filter nouns seen with only one determiner type
    unambiguous_nouns = {noun for noun, types in noun_to_types.items() if len(types) == 1}
    results_by_age = {}

    # process for every age range
    for age_limit in tqdm(age_ranges, desc="Processing age ranges"):
        subset = utterances_df[utterances_df['age_months'] <= age_limit]
        filtered_rows = []

        # go through pos tagged sentences
        for _, row in subset.iterrows():
            tagged = row['tagged']
            if isinstance(tagged, str):
                try:
                    tagged = ast.literal_eval(tagged)
                except Exception:
                    continue

            valid_dets = {'a', 'the'}

            # find det + noun pairs and save sentence
            for idx, (word, pos) in enumerate(tagged):
                if pos == 'DET' and word.lower() in valid_dets:
                    for offset in range(1, lookahead + 1):
                        look_idx = idx + offset
                        if look_idx < len(tagged):
                            next_word, next_pos = tagged[look_idx]
                            if next_pos == 'NOUN' and next_word.lower() in unambiguous_nouns:
                                filtered_rows.append(row)
                                break
                    else:
                        continue
                    break

        results_by_age[age_limit] = pd.DataFrame(filtered_rows, columns=utterances_df.columns)

    return results_by_age

set_wd()
age_ranges = [14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58]

# read in pos tagged data
pos_tagged_child = pd.read_csv("pos_tagged_child.txt", header=None, names=["utt", "age_months", "tagged"])

# parse the tagged column (convert from string to list of tuples)
pos_tagged_child['tagged'] = pos_tagged_child['tagged'].apply(ast.literal_eval)

# create dictionaries with nouns and their determiner types
with open("determiner_dicts.pkl", "rb") as f:
    dicts = pickle.load(f)

dicts_with_labels = [
    (dicts["regular"], 100),
    (dicts["0"], 0),
    (dicts["25"], 25),
    (dicts["50"], 50),
    (dicts["75"], 75),
]

# test how many unambiguous nouns in normal data
regular_dict = dicts["regular"]
unambiguous_nouns = {noun for noun, det_set in regular_dict.items() if len(det_set) == 1}

print(f"Total nouns in regular dict: {len(regular_dict)}")
print(f"Unambiguous nouns in regular dict: {len(unambiguous_nouns)}")
print(f"Percentage unambiguous: {len(unambiguous_nouns) / len(regular_dict) * 100:.2f}%")

regular_dict = dicts["0"]
unambiguous_nouns = {noun for noun, det_set in regular_dict.items() if len(det_set) == 1}

print(f"Total nouns in 0 dict: {len(regular_dict)}")
print(f"Unambiguous nouns in 0 dict: {len(unambiguous_nouns)}")
print(f"Percentage unambiguous: {len(unambiguous_nouns) / len(regular_dict) * 100:.2f}%")

# get test data
for pair_dict, label in dicts_with_labels:
    test_set_complete = create_test_set(pos_tagged_child, pair_dict, age_ranges)

    # make test set for every age range
    for number in age_ranges:
        test_set = test_set_complete[number]
        test_set["utt"].to_csv(f"test_{number}_months_{label}.txt", header=False, index=False)
