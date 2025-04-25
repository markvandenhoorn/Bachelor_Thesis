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
# python -m spacy download en_core_web_sm to download

### TODO: import_data remove first 1000 thingies and do everything

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
    parent_data = data.dropna(subset=["p_utts"])
    child_data = data.dropna(subset=["c_utts"])

    # remove rows where utterance is only MASK and save only utterances
    parent_utterances = parent_data[~parent_data["p_utts"].apply(is_only_mask)]['p_utts']
    child_utterances = child_data[~child_data["c_utts"].apply(is_only_mask)]['c_utts']

    # replace every 'an' with 'a'
    parent_utterances = parent_utterances.str.replace(r'\ban\b', 'a', regex=True)
    parent_utterances = parent_utterances.str.replace(r'\bAn\b', 'A', regex=True)
    child_utterances = child_utterances.str.replace(r'\ban\b', 'a', regex=True)
    child_utterances = child_utterances.str.replace(r'\bAn\b', 'A', regex=True)

    return parent_utterances, child_utterances

def strip_punctuation_and_lemmatize(text):
    """
    Takes a string, removes extraneous punctuation, and lemmatizes the words.
    """
    # let spacy process text
    doc = nlp(text)

    lemmatized_tokens = []

    # go through each 'token'
    for token in doc:
        # if it is a normal word, lemmatize it and add to list
        if not token.is_punct:
            lemmatized_tokens.append(token.lemma_)

    # re-create the sentence from the list
    cleaned_text = " ".join(lemmatized_tokens)

    return cleaned_text

def replace_mask_with_random(text):
    # replace MASK with a random first name
    return re.sub(r"MASK", lambda _: random.choice(first_names), text)

def replace_mask_data(utterances):
    # strip punctuation, lemmatize and replace MASK with random names
    utterances = utterances.apply(strip_punctuation_and_lemmatize)
    return utterances.apply(replace_mask_with_random)

def pos_tagging(utterances):
    """
    Tags each word in the sentence as the part of speech that it is.
    Returns pd Series of tagged sentences.
    """
    tagged_utterances = []
    for sentence in utterances:
        # pos tag the sentence
        doc = nlp(sentence)

        # create tuple of word + POS tag for every word
        tagged_utterances.append([(token.text, token.pos_) for token in doc])

    return pd.Series(tagged_utterances)

def filter_determiner_sentences(tagged_sentences):
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
    sentences = []

    # extract words and pos tags from sentences
    for tagged_sentence in tagged_sentences:
        words = [word for word, tag in tagged_sentence]
        pos_tags = [tag for word, tag in tagged_sentence]

        # check if the sentence contains any determiner (DET), skip if no
        if 'DET' not in pos_tags:
            continue

        # keep only sentences with 'a' or 'the' as the determiner
        for i in range(len(pos_tags)):
            if pos_tags[i] == 'DET' and words[i].lower() in valid_determiners:
                # check if the DET is followed by NOUN
                if i + 1 < len(pos_tags) and pos_tags[i + 1] == 'NOUN':
                    # mask the determiner and add sentence to the lists
                    masked_words = words.copy()
                    masked_words[i] = '[MASK]'
                    masked_sentences.append(" ".join(masked_words))
                    sentences.append(" ".join(words))
                    break

                # check if DET is followed by other token, then NOUN
                elif (
                    i + 2 < len(pos_tags)
                    and pos_tags[i + 1] not in ['DET', 'NOUN']
                    and pos_tags[i + 2] == 'NOUN'
                ):
                    # mask the determiner and add sentence to the list
                    masked_words = words.copy()
                    masked_words[i] = '[MASK]'
                    masked_sentences.append(" ".join(masked_words))
                    sentences.append(" ".join(words))
                    break

    # Return the reconstructed sentences as a pandas Series
    return pd.Series(sentences), pd.Series(masked_sentences)

def assign_determiner_types(tagged_sentences, lookahead = 4):
    """
    Maps every noun to 1 determiner type. If a noun is seen with multiple types,
    randomize which one it gets assigned to.
    """
    valid_determiners = {'a': 'A', 'the': 'B'}
    noun_to_types = {}
    sentence_records = []

    for sentence in tagged_sentences:
        noun_pairs = set()

        # check for every word if it is a valid determiner and save the word
        for i in range(len(sentence)):
            word, pos = sentence[i]
            if pos == 'DET' and word.lower() in valid_determiners:
                det_type = valid_determiners[word.lower()]

                # check if there is a noun in the next 4 words
                for j in range(i+1, min(i + 1 + lookahead, len(sentence))):
                    next_word, next_pos = sentence[j]
                    if next_pos == 'NOUN':
                        # save the noun and which determiner it is paired with
                        noun = next_word.lower()
                        noun_to_types.setdefault(noun, set()).add(det_type)
                        noun_pairs.add((noun, det_type))
                        # stop if we find a noun
                        break

        # keep a record of sentence + which pairs were present in that sentence
        sentence_records.append((sentence, noun_pairs))

    return noun_to_types, sentence_records

def filter_det_noun_pairs(noun_to_types, sentence_records):
    """
    Keeps only sentences where nouns are used with consistent determiners.
    """
    filtered_sentences = []

    # assign a determiner type to the noun
    final_determiner_type = {
        noun: list(types)[0]
        for noun, types in noun_to_types.items()
        if len(types) == 1
    }

    # check for each sentence if it is consistent, otherwise remove it
    for sentence, noun_pairs in sentence_records:
        if all(
            noun in final_determiner_type and used_type == final_determiner_type[noun]
            for noun, used_type in noun_pairs
        ):
            # remove extra spaces around punctuation etc.
            plain_sentence = " ".join(word.strip() for word, _ in sentence)
            filtered_sentences.append(plain_sentence)

    return pd.Series(filtered_sentences)

if __name__ == "__main__":
    # set working directory and load data
    set_wd()
    parent_utterances, child_utterances = import_data("temp_data.csv")

    # load pos tagger from spacy
    nlp = spacy.load('en_core_web_sm')

    # import the first name data into a list
    first_names = list(pd.read_csv("first_names.csv")['Name'])

    # create lemmatized datasets, replace MASK with random names
    parent_utterances = replace_mask_data(parent_utterances)
    child_utterances = replace_mask_data(child_utterances)

    # save current parent data as unedited data, also serves as tokenizer train data
    parent_utterances.to_csv("train_data_all_unedited.txt", index = False, header = False)

    # pos tag child data, then mask determiners and save test sets
    pos_tagged_child = pos_tagging(child_utterances)
    child_test_regular, child_test_masked = filter_determiner_sentences(pos_tagged_child)
    child_test_regular.to_csv("test_data_regular.txt", index = False, header = False)
    child_test_masked.to_csv("test_data_masked.txt", index = False, header = False)

    # create parent utt dataset where each noun only appears with 1 determiner
    pos_tagged_parent = pos_tagging(parent_utterances)
    pair_dict, sentence_records = assign_determiner_types(pos_tagged_parent)
    filtered_utterances = filter_det_noun_pairs(pair_dict, sentence_records)
    filtered_utterances.to_csv('testdit.txt', index = False, header = False)
