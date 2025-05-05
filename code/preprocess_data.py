"""
Name: preprocess_data.py
Author: Mark van den Hoorn
Desc: Outputs a txt file for tokenizer training, a txt file with
parent utterances (train data) per age range, a txt file with parent utterances
where each noun only occurs with 1 type of determiner per age range, and a txt
file with child utterances where they use determiner + noun combinations
(masked and unmasked) per age range.
"""
import os
import re
import string
import random
import spacy
import pandas as pd
from tqdm import tqdm

### TODO: filter test sentences based on nouns that also appear in training
### TODO: Run code on all data (check import_data)

def set_wd():
    # function to set current working directory to data folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')

    # Set it as the working directory
    os.chdir(data_dir)

def is_only_mask(text):
    # function to check if utterance has only MASK and nothing else
    return text.strip() == "MASK" or text.strip() == "MASK MASK"

def import_data(filename):
    # read data, remove NaN and divide into parent and child data
    data = pd.read_csv(filename, index_col = 0)
    data = data[0:10000]
    parent_data = data.dropna(subset=["p_utts", "age_months"])
    child_data = data.dropna(subset=["c_utts", "age_months"])

    # remove rows where utterance is only MASK and save only utterances
    parent_utterances = parent_data[~parent_data["p_utts"].apply(is_only_mask)][["p_utts", "age_months"]]
    child_utterances = child_data[~child_data["c_utts"].apply(is_only_mask)][["c_utts", "age_months"]]

    # replace every 'an' with 'a'
    parent_utterances['p_utts'] = parent_utterances['p_utts'].str.replace(r'\ban\b', 'a', regex=True)
    parent_utterances['p_utts'] = parent_utterances['p_utts'].str.replace(r'\bAn\b', 'A', regex=True)
    child_utterances['c_utts'] = child_utterances['c_utts'].str.replace(r'\ban\b', 'a', regex=True)
    child_utterances['c_utts'] = child_utterances['c_utts'].str.replace(r'\bAn\b', 'A', regex=True)

    return parent_utterances, child_utterances

def first_preprocess(utterances):
    """strip punctuation, lemmatize and replace MASK with random names"""
    utt_column = 'p_utts' if 'p_utts' in utterances.columns else 'c_utts'
    texts = utterances[utt_column].tolist()

    # batch process with spacy, include progress bar
    docs = nlp.pipe(texts, batch_size=1000, n_process=1)
    docs = tqdm(docs, total=len(texts), desc="Lemmatizing, removing punctuation and replacing MASK")

    # lemmatize, strip punctuation
    lemmatized = []
    for doc in docs:
        lemmatized_tokens = [token.lemma_ for token in doc if not token.is_punct]
        cleaned_text = " ".join(lemmatized_tokens)
        lemmatized.append(cleaned_text)

    # replace mask with random name
    replaced = [re.sub(r"MASK", lambda _: random.choice(first_names), text) for text in lemmatized]

    return pd.DataFrame({'utt': replaced, 'age_months': utterances['age_months'].values})

def pos_tagging(utterances_df):
    """
    Tags each word in the sentence as the part of speech that it is.
    Returns pd Series of tagged sentences.
    """
    tagged = []
    for sentence in tqdm(utterances_df['utt'], desc="POS tagging"):
        doc = nlp(sentence)
        tagged.append([(token.text, token.pos_) for token in doc])
    utterances_df = utterances_df.copy()
    utterances_df['tagged'] = tagged
    return utterances_df

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
                    masked_sentences.append(" ".join(masked_words))
                    regular_sentences.append(" ".join(words))
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

def assign_determiner_types(utterances_df, lookahead = 4):
    """
    Maps every noun to 1 determiner type. If a noun is seen with multiple types,
    randomize which one it gets assigned to.
    """
    valid_determiners = {'a': 'A', 'the': 'B'}
    noun_to_types = {}

    for i, sentence in tqdm(enumerate(utterances_df['tagged']), total=len(utterances_df), desc="Assigning determiner types to nouns"):
        # check for every word if it is a valid determiner and save the word
        for idx in range(len(sentence)):
            word, pos = sentence[idx]
            if pos == 'DET' and word.lower() in valid_determiners:
                det_type = valid_determiners[word.lower()]

                # check if there is a noun in the next 4 words
                for j in range(idx + 1, min(idx + 1 + lookahead, len(sentence))):
                    next_word, next_pos = sentence[j]
                    if next_pos == 'NOUN':
                        # save the noun and which determiner it is paired with
                        noun = next_word.lower()
                        noun_to_types.setdefault(noun, set()).add(det_type)
                        # stop if we find a noun
                        break

    # Now we need to ensure each noun gets assigned only 1 determiner
    for noun, types in noun_to_types.items():
        if len(types) > 1:
            # Assign one random determiner if multiple types are found
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

    for sentence_tuple, age_month in tqdm(zip(utterances_df['tagged'], utterances_df['age_months']),
                                           total=len(utterances_df),
                                           desc="Filtering consistent det-noun pairs"):
        original_words = [word for word, pos in sentence_tuple]
        sentence_is_consistent = True

        for idx, (word, pos) in enumerate(sentence_tuple):
            # check if the word is a key in valid determiners
            if pos == 'DET' and word.lower() in valid_determiners:
                current_det_word_lower = word.lower()
                # get the type of determiner it is paired with
                current_det_type = valid_determiners[current_det_word_lower]

                # check for a noun close to the determiner
                noun_found_in_window = False
                for j in range(idx + 1, min(idx + 1 + lookahead, len(sentence_tuple))):
                    next_word, next_pos = sentence_tuple[j]
                    if next_pos == 'NOUN':
                        noun = next_word.lower()
                        noun_found_in_window = True
                        if noun in noun_to_types:
                            assigned_noun_det_type = list(noun_to_types[noun])[0]
                            # compare types used
                            if current_det_type != assigned_noun_det_type:
                                sentence_is_consistent = False
                                break
                        break

                if not sentence_is_consistent:
                    break

        if sentence_is_consistent:
            filtered_sentences.append(" ".join(original_words))
            age_months_filtered.append(age_month)

    return pd.DataFrame({
        'utt': filtered_sentences,
        'age_months': age_months_filtered
    })

def filter_by_age_range(utterances_df, max_age):
    """
    Filters the utterances such that the 'age_months' is less than or equal to max_age.
    """
    return utterances_df[utterances_df['age_months'] <= max_age]

def save_age_based_datasets(utterances_df, age_ranges):
    """
    Saves filtered datasets based on different age ranges to separate text files,
    including only the sentence column (no age info).
    """
    for max_age in tqdm(age_ranges, desc="Saving train sets by age range"):
        filtered_data = filter_by_age_range(utterances_df, max_age)
        filtered_data['utt'].to_csv(
            f"train_data_up_to_{max_age}_months.txt",
            index=False,
            header=False
        )

if __name__ == "__main__":
    # set working directory and load data
    set_wd()
    parent_utterances, child_utterances = import_data("ldp_data.csv")

    # load pos tagger from spacy
    nlp = spacy.load('en_core_web_sm')

    # import the first name data into a list
    first_names = list(pd.read_csv("first_names.csv")['Name'])

    # create lemmatized datasets, replace MASK with random names
    parent_utterances = first_preprocess(parent_utterances)
    child_utterances = first_preprocess(child_utterances)

    # save current parent data as unedited data, also serves as tokenizer train data
    parent_utterances.to_csv("train_data_all_unedited.txt", index = False, header = False)

    # pos tag child data, then mask determiners and save test sets
    pos_tagged_child = pos_tagging(child_utterances)
    child_test_regular, child_test_masked = filter_determiner_sentences(pos_tagged_child)
    child_test_regular.to_csv("test_data_regular.txt", index = False, header = False)
    child_test_masked.to_csv("test_data_masked.txt", index = False, header = False)

    # create parent utt dataset where each noun only appears with 1 determiner
    pos_tagged_parent = pos_tagging(parent_utterances)
    pair_dict = assign_determiner_types(pos_tagged_parent)
    filtered_utterances = filter_det_noun_pairs(pos_tagged_parent, pair_dict)

    # set age ranges for the children
    age_ranges = [14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54]

    # create training data sets per age cap
    save_age_based_datasets(filtered_utterances, age_ranges)
