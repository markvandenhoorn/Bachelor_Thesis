import os
import re
import string
import random
import spacy
import pandas as pd
# pip install spacy
# python -m spacy download en_core_web_sm to download

### TODO: pos tag all child data, not just first x amount

def set_wd():
    # function to set current working directory to data folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')

    # Set it as the working directory
    os.chdir(data_dir)

def is_only_mask(text):
    # function to check if utterance has only MASK and nothing else
    return text.strip() == "MASK"

def import_data(filename):
    # read data, remove NaN and divide into parent and child data
    data = pd.read_csv(filename, index_col = 0)
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

def remove_mask(utterances):
    # function to remove MASK and make sure no extra spaces are left
    return utterances.apply(lambda text: re.sub(r'\s*MASK\s*', ' ', text).strip())

def random_word(length=8):
    # function to create a word of 8 random letters
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def replace_mask_with_random(text):
    # replace MASKed personal info by random word
    return re.sub(r"MASK", lambda _: random_word(), text)

def replace_mask_data(utterances):
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
    Returns pd Series with masked sentences (test set).
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

def filter_parent_utterances(tagged_sentences):
    """
    Takes pd Series of POS tagged sentences.
    Creates a new Series of regular sentences. New Series has only sentences
    where each noun appears with only 1 type of determiner: 'a' or 'the'.
    """
    valid_determiners = {'a': 'A', 'the': 'B'}
    noun_to_types = {}
    sentence_records = []

    # first collect noun-determiner relationships
    for sentence in tagged_sentences:
        noun_pairs = set()

        # get all word and POS tag combinations
        for i in range(len(sentence)):
            word, pos = sentence[i]

            # if we find a valid determiner, save the actual determiner word
            if pos == 'DET' and word.lower() in valid_determiners:
                det_type = valid_determiners[word.lower()]

                # check if there is a noun immediately after the determiner
                if i + 1 < len(sentence) and sentence[i + 1][1] == 'NOUN':
                    noun = sentence[i + 1][0].lower()

                    # if new noun, add it to the dictionary with an empty set
                    if noun not in noun_to_types:
                        noun_to_types[noun] = set()

                    # add the determiner type we encountered it with
                    noun_to_types[noun].add(det_type)
                    noun_pairs.add((noun, det_type))

                # otherwise check if there is a noun after another token
                elif (
                    i + 2 < len(sentence)
                    and sentence[i + 1][1] not in ['DET', 'NOUN']
                    and sentence[i + 2][1] == 'NOUN'
                ):
                    noun = sentence[i + 2][0].lower()
                    if noun not in noun_to_types:
                        noun_to_types[noun] = set()
                    noun_to_types[noun].add(det_type)
                    noun_pairs.add((noun, det_type))

        # store the sentence with the noun-determiner pairs that occur there
        sentence_records.append((sentence, noun_pairs))

    # now decide which type the noun is going to be associated with
    final_determiner_type = {}
    for noun, types in noun_to_types.items():
        # if there is only 1 determiner found, choose that one
        if len(types) == 1:
            final_determiner_type[noun] = list(types)[0]
        else:
            # if both types are found, randomly assign a determiner to the noun
            final_determiner_type[noun] = random.choice(['A', 'B'])

    # now filter the sentences based on the determiner the noun is paired with
    filtered_sentences = []
    for sentence, noun_pairs in sentence_records:
        # ensure each noun in the sentence has the correct determiner
        if all(used_type == final_determiner_type[noun] for noun, used_type in noun_pairs):
            # make the sentence without the pos tags (string)
            plain_sentence = " ".join(word for word, _ in sentence)
            filtered_sentences.append(plain_sentence)

    return pd.Series(filtered_sentences)

if __name__ == "__main__":
    # set working directory and load data
    set_wd()
    parent_utterances, child_utterances = import_data("temp_data.csv")

    # create and save dataset without MASK, to train tokenizer on
    tokenizer_data = remove_mask(parent_utterances)
    tokenizer_data.to_csv("tokenizer_data.csv")

    # create parent and child datasets, replace MASK with random words
    parent_utterances = replace_mask_data(parent_utterances)
    child_utterances = replace_mask_data(child_utterances)

    # save current parent data as unedited data
    parent_utterances.to_csv("train_data_all_unedited.csv")

    # load pos tagger from spacy
    nlp = spacy.load('en_core_web_sm')

    # pos tag child data, then mask determiners and save test sets
    # pos_tagged_child = pos_tagging(child_utterances[0:5000])
    # child_test_regular, child_test_masked = filter_determiner_sentences(pos_tagged_child)
    # child_test_regular.to_csv("test_data_regular.csv")
    # child_test_masked.to_csv("test_data_masked.csv")

    # create parent utt dataset where each noun only appears with 1 determiner
    pos_tagged_parent = pos_tagging(parent_utterances[0:1000])
    filtered_utterances = filter_parent_utterances(pos_tagged_parent)
    filtered_utterances.to_csv('testdit.csv')
