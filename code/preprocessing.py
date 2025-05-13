"""
Name: preprocessing.py
Author: Mark van den Hoorn
Desc: File for lemmatizing, replacing MASK words etc.
"""

import spacy
import random
import pandas as pd
from tqdm import tqdm
import re

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
    Keeps only sentences where:
    - A determiner ('DET') is followed by a noun ('NOUN'), or
    - A determiner is followed by exactly one token and then a noun.

    Sentences without any determiner ('DET') are excluded.
    Remaining sentences have the DET replaced by [MASK].
    Returns pd Series with masked sentences (test set) and non-masked sentences.
    """
    valid_determiners = ['a', 'the', 'A', 'The']
    masked_sentences = []
    regular_sentences = []
    age_months_masked = []
    age_months_regular = []

    # iterate through each sentence and its corresponding 'age_months'
    for i, tagged_sentence in tqdm(enumerate(utterances_df['tagged']), total=len(utterances_df), desc="Filtering out sentences without determiner + noun"):
        words = [word for word, tag in tagged_sentence]
        pos_tags = [tag for word, tag in tagged_sentence]

        # skip sentences without a determiner (DET)
        if 'DET' not in pos_tags:
            continue

        # process valid determiners ('a' or 'the')
        for i in range(len(pos_tags)):
            if pos_tags[i] == 'DET' and words[i].lower() in valid_determiners:
                # check if the DET is followed by NOUN
                if i + 1 < len(pos_tags) and pos_tags[i + 1] == 'NOUN':
                    # mask the determiner and add sentence to masked list
                    masked_words = words.copy()
                    masked_words[i] = '[MASK]'
                    masked_sentences.append(" ".join(w.strip() for w in masked_words))
                    regular_sentences.append(" ".join(w.strip() for w in words))
                    age_months_masked.append(utterances_df['age_months'].iloc[i])
                    age_months_regular.append(utterances_df['age_months'].iloc[i])
                    break

                # check if DET is followed by other token, then NOUN
                elif (
                    i + 2 < len(pos_tags)
                    and pos_tags[i + 1] not in ['DET', 'NOUN']
                    and pos_tags[i + 2] == 'NOUN'
                ):
                    # mask the determiner and add sentence to masked list
                    masked_words = words.copy()
                    masked_words[i] = '[MASK]'
                    masked_sentences.append(" ".join(masked_words))
                    regular_sentences.append(" ".join(words))
                    age_months_masked.append(utterances_df['age_months'].iloc[i])
                    age_months_regular.append(utterances_df['age_months'].iloc[i])
                    break

    # create DataFrames from the lists
    masked_df = pd.DataFrame({
        'utt': masked_sentences,
        'age_months': age_months_masked
    })
    regular_df = pd.DataFrame({
        'utt': regular_sentences,
        'age_months': age_months_regular
    })

    return regular_df, masked_df

def contains_train_noun(sentence, train_nouns):
    # check if a noun is in training nouns
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == 'NOUN' and token.text.lower() in train_nouns:
            return True
    return False

def filter_utterances_by_nouns(utterances, train_nouns):
    # filters out sentences where noun from det+noun is not in training data
    mask = utterances_df['utt'].apply(lambda s: contains_train_noun(s, train_nouns))
    return utterances_df[mask].reset_index(drop=True)

def extract_det_noun_pairs(pos_tagged_utterances):
    """
    Extracts all determiner + noun pairs from the child sentences.
    """
    det_noun_pairs = set()

    for row in pos_tagged_utterances['tagged']:
        # iterate over word and tag pairs
        for i in range(len(row) - 1):
            word1, tag1 = row[i]
            word2, tag2 = row[i + 1]

            # check if it is a det+noun pair
            if tag1 == 'DET' and tag2 == 'NOUN' and word1 in ['a', 'the']:
                # add the pair to the set
                pair = f"{word1.lower()} {word2.lower()}"
                det_noun_pairs.add(pair)

    return list(det_noun_pairs)
