"""
Name: exploratory.py
Author: Mark van den Hoorn
Desc: Finds how many nouns are seen with the, a or both and prints it.
"""

import os
import ast
import pandas as pd
from collections import defaultdict

def count_nouns_with_determiners(tagged_utterances):
    """
    Counts how many nouns appear with 'a', 'the', and both determiners ('a' and 'the').
    """
    noun_determiner_counts = defaultdict(lambda: {'a': 0, 'the': 0})
    nouns_with_both = []

    determiners = {'a', 'the'}

    for sentence in tagged_utterances:
        for idx, (word, pos) in enumerate(sentence):
            if pos == 'DET' and word.lower() in determiners:
                det = word.lower()
                for j in range(idx + 1, len(sentence)):
                    next_word, next_pos = sentence[j]
                    if next_pos in {'NOUN', 'PROPN'}:
                        noun = next_word.lower()
                        noun_determiner_counts[noun][det] += 1
                        break

    both_count = 0
    only_a_count = 0
    only_the_count = 0
    nouns_with_both = []

    for noun, counts in noun_determiner_counts.items():
        if counts['a'] > 0 and counts['the'] > 0:
            both_count += 1
            nouns_with_both.append(noun)
        elif counts['a'] > 0:
            only_a_count += 1
        elif counts['the'] > 0:
            only_the_count += 1

    return both_count, only_a_count, only_the_count, nouns_with_both

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception as e:
        print(f"Failed to parse: {val}\nError: {e}")
        return []

def main():
    current_wd = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_wd, '../data/pos_tagged_data.txt')

    # read the tagged utterances column and convert string representation to actual list
    df = pd.read_csv(file_path, header=None, names=['utt', 'age', 'tagged'], keep_default_na=False)
    df['tagged'] = df['tagged'].apply(safe_literal_eval)

    both_count, only_a_count, only_the_count, nouns_with_both = count_nouns_with_determiners(df['tagged'])

    print(f"Nouns with both 'a' and 'the': {both_count}")
    print(f"Nouns with only 'a': {only_a_count}")
    print(f"Nouns with only 'the': {only_the_count}")

if __name__ == "__main__":
    main()
