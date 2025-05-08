"""
Name: pos_tagging.py
Author: Mark van den Hoorn
Desc: POS tag utterances for further processing
"""
import spacy
import random
import pandas as pd
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

def pos_tagging(utterances_df):
    """
    POS tags all utterances in given dataframe. Returns new dataframe.
    """
    tagged = []
    for sentence in tqdm(utterances_df['utt'], desc="POS tagging"):
        doc = nlp(sentence)
        tagged.append([(token.text, token.pos_) for token in doc])
    utterances_df = utterances_df.copy()
    utterances_df['tagged'] = tagged
    return utterances_df

def assign_determiner_types(utterances_df, lookahead=4):
    """
    Maps every noun to 1 determiner type. If a noun is seen with multiple types,
    randomize which one it gets assigned to.
    """
    valid_determiners = {'a': 'A', 'the': 'B'}
    noun_to_types = {}

    # give each noun a determiner type assigned
    for i, sentence in tqdm(enumerate(utterances_df['tagged']), total=len(utterances_df), desc="Assigning determiner types to nouns"):
        for idx in range(len(sentence)):
            word, pos = sentence[idx]

            # get the determiner type
            if pos == 'DET' and word.lower() in valid_determiners:
                det_type = valid_determiners[word.lower()]

                # check if followed by a noun
                for j in range(idx + 1, min(idx + 1 + lookahead, len(sentence))):
                    next_word, next_pos = sentence[j]
                    if next_pos == 'NOUN':
                        # add the determiner to this noun's determiner types
                        noun = next_word.lower()
                        noun_to_types.setdefault(noun, set()).add(det_type)
                        break

    # if a noun has multiple determiner types, choose one at random
    for noun, types in noun_to_types.items():
        if len(types) > 1:
            noun_to_types[noun] = {random.choice(list(types))}

    return noun_to_types

def filter_det_noun_pairs(utterances_df, noun_to_types):
    """
    Keeps only sentences where nouns are used with consistent determiners.
    """
    valid_determiners = {'a': 'A', 'the': 'B'}
    filtered_sentences = []
    age_months_filtered = []
    lookahead = 5

    for sentence_tuple, age_month in tqdm(zip(utterances_df['tagged'], utterances_df['age_months']), total=len(utterances_df), desc="Filtering consistent det-noun pairs"):
        original_words = [word for word, pos in sentence_tuple]
        sentence_is_consistent = True

        # check if the word is a determiner and save its value
        for idx, (word, pos) in enumerate(sentence_tuple):
            if pos == 'DET' and word.lower() in valid_determiners:
                current_det_word_lower = word.lower()
                current_det_type = valid_determiners[current_det_word_lower]

                # check if we find a nearby noun
                noun_found_in_window = False
                for j in range(idx + 1, min(idx + 1 + lookahead, len(sentence_tuple))):
                    next_word, next_pos = sentence_tuple[j]
                    if next_pos == 'NOUN':
                        noun = next_word.lower()
                        noun_found_in_window = True
                        if noun in noun_to_types:
                            # if noun is seen with wrong determiner, change determiner
                            assigned_noun_det_type = list(noun_to_types[noun])[0]
                            if current_det_type != assigned_noun_det_type:
                                for det, det_type in valid_determiners.items():
                                    if det_type == assigned_noun_det_type:
                                        original_words[idx] = det
                                        break
                        break

                # sentence_is_consistent belongs to previous code
                if not sentence_is_consistent:
                    # but I am too scared to remove it
                    break

        # rebuild sentence
        if sentence_is_consistent:
            filtered_sentences.append(" ".join(original_words))
            age_months_filtered.append(age_month)

    return pd.DataFrame({
        'utt': filtered_sentences,
        'age_months': age_months_filtered
    })

def get_nouns(utterances):
    """ get all unique nouns from the utterances"""
    train_nouns = set()
    for sentence in tqdm(utterances['utt'], desc="Extracting nouns from utterances"):
        doc = nlp(sentence)
        for token in doc:
            if token.pos_ == 'NOUN':
                train_nouns.add(token.text.lower())

    return train_nouns
