"""
Name: create_test.py
Author: Mark van den Hoorn
Desc: Uses the pos tagged children data to create test sets. These test sets
consist of sentences where nouns are used that are seen with only one type of
determiner throughout the data, as well as the age in months and the pos tagged
utternace. It does this for every type of model, so for all 5 types, and for
every age range as well.
"""
import pandas as pd
import ast
import pickle
import argparse
from tqdm import tqdm
from utils import set_wd

def create_test_set(child_df, parent_df, noun_to_types, age_ranges, lookahead=2, unique_nouns=True):
    from collections import defaultdict

    valid_dets = {'a', 'the'}
    results_by_age = {}
    unambiguous_nouns_all = {noun for noun, types in noun_to_types.items() if len(types) == 1}

    for age_limit in tqdm(age_ranges, desc="Processing test sets by age"):
        parent_subset = parent_df[parent_df['age_months'] <= age_limit]
        training_nouns = set()

        for _, row in parent_subset.iterrows():
            try:
                tagged = ast.literal_eval(row['tagged']) if isinstance(row['tagged'], str) else row['tagged']
            except Exception:
                continue

            for idx, (word, pos) in enumerate(tagged):
                if pos == 'DET' and word.lower() in valid_dets:
                    for offset in range(1, lookahead + 1):
                        look_idx = idx + offset
                        if look_idx < len(tagged):
                            next_word, next_pos = tagged[look_idx]
                            if next_pos == 'NOUN':
                                training_nouns.add(next_word.lower())
                                break

        target_nouns = unambiguous_nouns_all & training_nouns
        used_nouns = set()
        selected_rows = []

        for _, row in child_df.iterrows():
            try:
                tagged = ast.literal_eval(row['tagged']) if isinstance(row['tagged'], str) else row['tagged']
            except Exception:
                continue

            for idx, (word, pos) in enumerate(tagged):
                if pos == 'DET' and word.lower() in valid_dets:
                    for offset in range(1, lookahead + 1):
                        look_idx = idx + offset
                        if look_idx < len(tagged):
                            next_word, next_pos = tagged[look_idx]
                            noun = next_word.lower()
                            if next_pos == 'NOUN' and noun in target_nouns:
                                if unique_nouns and noun in used_nouns:
                                    break
                                selected_rows.append(row)
                                used_nouns.add(noun)
                                break
                    else:
                        continue
                    break

        results_by_age[age_limit] = pd.DataFrame(selected_rows, columns=child_df.columns)

    return results_by_age

def main():
    parser = argparse.ArgumentParser(description="Create determiner test sets from child-parent data.")
    parser.add_argument("--unique_nouns", action="store_true", help="Ensure each noun appears only once per age bin")
    args = parser.parse_args()

    set_wd()
    age_ranges = [14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58]

    pos_tagged_child = pd.read_csv("pos_tagged_child.txt", header=None, names=["utt", "age_months", "tagged"])
    pos_tagged_parent = pd.read_csv("pos_tagged_data.txt", header=None, names=["utt", "age_months", "tagged"])
    pos_tagged_child['tagged'] = pos_tagged_child['tagged'].apply(ast.literal_eval)

    with open("determiner_dicts.pkl", "rb") as f:
        dicts = pickle.load(f)

    dicts_with_labels = [
        (dicts["regular"], 100),
        (dicts["0"], 0),
        (dicts["25"], 25),
        (dicts["50"], 50),
        (dicts["75"], 75),
    ]

    for pair_dict, label in dicts_with_labels:
        test_set_complete = create_test_set(
            pos_tagged_child, pos_tagged_parent,
            pair_dict, age_ranges,
            unique_nouns=args.unique_nouns
        )

        for number in age_ranges:
            test_set = test_set_complete[number]
            test_set.to_csv(f"test_{number}_months_{label}.txt", index=False)

if __name__ == "__main__":
    main()
