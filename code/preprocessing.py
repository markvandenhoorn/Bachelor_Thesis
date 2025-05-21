"""
Name: preprocessing.py
Author: Mark van den Hoorn
Desc: File for lemmatizing, replacing MASK words etc.
"""

import spacy
import random
import pandas as pd
from utils import set_wd
from tqdm import tqdm
import re
import ast

# load spacy model for lemmatization
nlp = spacy.load('en_core_web_sm')

def first_preprocess(utterances, first_names):
    """
    Takes utterances. Lemmatizes, strips of punctuation and replaces MASK with
    a random first name.
    """
    utt_column = 'p_utts' if 'p_utts' in utterances.columns else 'c_utts'
    texts = utterances[utt_column].tolist()

    docs = nlp.pipe(texts, batch_size=1000, n_process=1)
    docs = tqdm(docs, total=len(texts), desc="Lemmatizing, removing punctuation and replacing MASK")

    # lemmatize, remove punctuation and clean spaces, also replace + with space
    lemmatized = []
    for doc in docs:
        lemmatized_tokens = [token.lemma_ for token in doc if not token.is_punct]

        # repeat words that have [x4] etc. and remove the [x4]
        expanded_tokens = []
        i = 0
        while i < len(lemmatized_tokens):
            token = lemmatized_tokens[i]
            if re.fullmatch(r'\[x\d+\]', token) and len(expanded_tokens) > 0:
                repeat_count = int(token[2:-1])
                expanded_tokens.extend([expanded_tokens[-1]] * (repeat_count - 1))
            else:
                expanded_tokens.append(token)
            i += 1

        # Remove '+' and join
        cleaned_text = " ".join(token.replace('+', '') for token in expanded_tokens)
        lemmatized.append(cleaned_text)

    # replace mask and 'name@x' with random name
    replaced = [
        re.sub(r"\b\w*@\w*\b", lambda _: random.choice(first_names),
        re.sub(r"\bMASK\b", lambda _: random.choice(first_names), text))
        for text in lemmatized
    ]

    return pd.DataFrame({'utt': replaced, 'age_months': utterances['age_months'].values})

def filter_determiner_sentences(utterances_df):
    """
    Filters utterances to keep only those where:
    - A determiner ('DET') is followed by a noun ('NOUN'), or
    - A determiner is followed by one other token, then a noun.

    Returns a DataFrame with columns: 'utt', 'age_months', 'tagged'
    """
    valid_determiners = {'a', 'the'}
    filtered_rows = []

    for _, row in tqdm(utterances_df.iterrows(), total=len(utterances_df), desc="Filtering tagged utterances"):
        tagged = row['tagged']

        for i, (word, tag) in enumerate(tagged):
            if tag == 'DET' and word.lower() in valid_determiners:
                # Case 1: DET followed by NOUN
                if i + 1 < len(tagged) and tagged[i + 1][1] == 'NOUN':
                    filtered_rows.append(row)
                    break
                # Case 2: DET followed by something else, then NOUN
                elif i + 2 < len(tagged) and tagged[i + 1][1] not in ['DET', 'NOUN'] and tagged[i + 2][1] == 'NOUN':
                    filtered_rows.append(row)
                    break

    return pd.DataFrame(filtered_rows)

# create an overall_performance test set if we directly run this file
if __name__ == "__main__":
    set_wd()
    df = pd.read_csv("pos_tagged_child.txt", header=None, names=["utt", "age_months", "tagged"])
    df["tagged"] = df["tagged"].apply(ast.literal_eval)
    filtered_df = filter_determiner_sentences(df)
    filtered_df.to_csv("overall_performance.txt", index=False)
