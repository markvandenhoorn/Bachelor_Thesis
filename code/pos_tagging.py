"""
Name: pos_tagging.py
Author: Mark van den Hoorn
Desc: POS tag utterances for further processing
"""
import spacy
import random
import pandas as pd
from tqdm import tqdm
import logging

nlp = spacy.load('en_core_web_sm')

# set up logging for checking which non-nouns have determiners changed
logging.basicConfig(
    filename="../data/determiner_rewrites.log",
    level=logging.INFO,
    filemode="a",
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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

def assign_determiner_types(utterances_df, lookahead=4, both_chance = 0):
    """
    Maps every noun to 1 determiner type. If a noun is seen with multiple types,
    randomize which one it gets assigned to, with a chance of both_chance to
    keep both types assigned.
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

    # seperate nouns by how many determiner types they've been seen with
    nouns_with_both = [noun for noun, types in noun_to_types.items() if len(types) > 1]

    # compute target number of nouns that should keep both determiner
    target_both_noun_count = int(len(nouns_with_both) * both_chance)

    # cap the target so we don't try to have too many nouns with both DETs
    target_both_noun_count = min(target_both_noun_count, len(nouns_with_both))

    # randomly select nouns to keep both determiners
    random.shuffle(nouns_with_both)
    nouns_to_keep_both = set(nouns_with_both[:target_both_noun_count])

    # for all other nouns with multiple determiners, choose one randomly
    for noun in nouns_with_both:
        if noun not in nouns_to_keep_both:
            noun_to_types[noun] = {random.choice(list(noun_to_types[noun]))}

    return noun_to_types

def filter_det_noun_pairs(utterances_df, noun_to_types):
    """
    Keeps only sentences where nouns are used with consistent determiners. If a
    word is used as some other pos part, but is in the list of nouns, still make
    sure its determiner use is consistent. Also outputs a logging file that notes
    the non-nouns that have their determiner changed.
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
                    noun = next_word.lower()
                    if noun in noun_to_types:
                        assigned_noun_det_type = noun_to_types[noun]
                        if current_det_type not in assigned_noun_det_type:
                            enforced_type = list(assigned_noun_det_type)[0]
                            for det, det_type in valid_determiners.items():
                                if det_type == enforced_type:
                                    original_words[idx] = det
                                    break

                            # log the sentence if change was made on non-noun
                            if next_pos != 'NOUN':
                                original_sentence = " ".join([word for word, _ in sentence_tuple])
                                modified_sentence = " ".join(original_words)
                                logging.info(f"Non-noun '{noun}' (POS: {next_pos}) triggered determiner change.")
                                logging.info(f"Original: {original_sentence}")
                                logging.info(f"Modified: {modified_sentence}")
                                logging.info("----")

                        break

                # sentence_is_consistent belongs to previous code
                if not sentence_is_consistent:
                    # but I am too scared to remove it
                    break

        # rebuild sentence
        if sentence_is_consistent:
            filtered_sentences.append(" ".join(original_words))
            age_months_filtered.append(age_month)

    logging.info("\n" + "="*60 + "\nNEW RUN\n" + "="*60)

    new_df = pd.DataFrame({
        'utt': filtered_sentences,
        'age_months': age_months_filtered
    })

    # remove empty lines
    new_df = remove_empty_sentences(new_df)

    return new_df

def get_nouns(utterances):
    """ get all unique nouns from the utterances"""
    train_nouns = set()
    for sentence in tqdm(utterances['utt'], desc="Extracting nouns from utterances"):
        doc = nlp(sentence)
        for token in doc:
            if token.pos_ == 'NOUN':
                train_nouns.add(token.text.lower())

    return train_nouns

def remove_empty_sentences(filtered_df):
    """
    Removes rows where the sentence is an empty string.
    """
    filtered_df = filtered_df[filtered_df['utt'].str.strip() != '']
    return filtered_df
