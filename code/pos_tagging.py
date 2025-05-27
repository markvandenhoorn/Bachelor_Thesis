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

def get_types_to_nouns_map(noun_to_types_map):
    types_to_nouns = {'A': [], 'B': []}
    # Using sets to ensure noun uniqueness initially
    temp_a_nouns_exclusive = set()
    temp_b_nouns_exclusive = set()

    for noun_key, det_types_set in noun_to_types_map.items():
        # Check for exclusive assignment
        if det_types_set == {'A'}: # Exclusively type 'A'
            temp_a_nouns_exclusive.add(noun_key)
        elif det_types_set == {'B'}: # Exclusively type 'B'
            temp_b_nouns_exclusive.add(noun_key)
        # Nouns assigned to {'A', 'B'} or other combinations are NOT added
        # to this specific map, as they are not suitable for unambiguous adaptation.

    types_to_nouns['A'] = list(temp_a_nouns_exclusive)
    types_to_nouns['B'] = list(temp_b_nouns_exclusive)
    return types_to_nouns

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

def filter_det_noun_pairs(utterances_df, noun_to_types, remove_conflicts = False):
    """
    Modifies sentences based on determiner-noun consistency rules.
    Additionally tracks how often two typed words had conflicting determiner types.
    """
    valid_determiners = {'a': 'A', 'the': 'B'}
    processed_sentences_texts = []
    age_months_processed = []
    lookahead = 3
    conflict_count = 0

    types_to_nouns_map_for_adaptation = get_types_to_nouns_map(noun_to_types)

    for sentence_tuple, age_month in tqdm(zip(utterances_df['tagged'], utterances_df['age_months']), total=len(utterances_df), desc="Filtering consistent det-noun pairs"):
        current_sentence_words = [word for word, pos in sentence_tuple]
        skip_sentence = False

        for idx, (original_det_text_in_tuple, original_det_pos_in_tuple) in enumerate(sentence_tuple):
            if original_det_pos_in_tuple == 'DET' and original_det_text_in_tuple.lower() in valid_determiners:
                current_det_word_in_sentence = current_sentence_words[idx].lower()
                if current_det_word_in_sentence not in valid_determiners:
                    continue
                current_det_type = valid_determiners[current_det_word_in_sentence]

                typed_words_in_window = []
                for j in range(idx + 1, min(idx + 1 + lookahead, len(sentence_tuple))):
                    next_word_text_original, next_word_pos_original = sentence_tuple[j]
                    noun_key = next_word_text_original.lower()
                    if noun_key in noun_to_types:
                        typed_words_in_window.append(
                            (next_word_text_original, next_word_pos_original, j, noun_to_types[noun_key])
                        )

                num_typed_words = len(typed_words_in_window)

                if num_typed_words == 0:
                    continue

                elif num_typed_words == 1:
                    text1, pos1, original_idx1_sentence, types1 = typed_words_in_window[0]
                    if current_det_type not in types1 and len(types1) == 1:
                        enforced_type1 = list(types1)[0]
                        new_det_text = next(
                            (det_val for det_val, det_t_val in valid_determiners.items() if det_t_val == enforced_type1),
                            None
                        )
                        if new_det_text:
                            current_sentence_words[idx] = new_det_text

                elif num_typed_words >= 2:
                    text1, pos1, original_idx1_sentence, types1 = typed_words_in_window[0]
                    text2, pos2, original_idx2_sentence, types2 = typed_words_in_window[1]

                    final_det_type_for_phrase = current_det_type
                    adapt_word2 = False
                    word2_target_det_type = None

                    if types1 == {'A', 'B'} and types2 == {'A', 'B'}:
                        pass
                    elif types1 == {'A', 'B'} and len(types2) == 1:
                        final_det_type_for_phrase = list(types2)[0]
                    elif types2 == {'A', 'B'} and len(types1) == 1:
                        final_det_type_for_phrase = list(types1)[0]
                    elif len(types1) == 1 and len(types2) == 1:
                        req_type1 = list(types1)[0]
                        req_type2 = list(types2)[0]

                        if req_type1 == req_type2:
                            final_det_type_for_phrase = req_type1
                        else:
                            conflict_count += 1
                            if remove_conflicts:
                                skip_sentence = True
                                break
                            final_det_type_for_phrase = req_type1
                            adapt_word2 = True
                            word2_target_det_type = req_type1

                    if current_det_type != final_det_type_for_phrase:
                        new_det_text = next(
                            (det_val for det_val, det_t_val in valid_determiners.items() if det_t_val == final_det_type_for_phrase),
                            None
                        )
                        if new_det_text:
                            current_sentence_words[idx] = new_det_text

                    if adapt_word2:
                        candidate_new_word2s = [
                            n_word for n_word in types_to_nouns_map_for_adaptation.get(word2_target_det_type, [])
                            if n_word != text1.lower()
                        ]
                        if candidate_new_word2s:
                            new_word2_selected_text = random.choice(candidate_new_word2s)
                            current_sentence_words[original_idx2_sentence] = new_word2_selected_text

        if remove_conflicts and skip_sentence:
            continue

        processed_sentences_texts.append(" ".join(current_sentence_words))
        age_months_processed.append(age_month)

    logging.info(f"Total typed-word pair conflicts found: {conflict_count}")
    logging.info("="*40 + " NEW RUN " + "="*40)

    new_df = pd.DataFrame({
        'utt': processed_sentences_texts,
        'age_months': age_months_processed
    })

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

def run_filter_iterations(initial_utterances_df, noun_to_types, num_passes=2,
        remove_conflicts=False):
    """
    Runs the filter_det_noun_pairs function multiple times.
    After each pass (except the last), sentences are re-POS-tagged.

    Args:
        initial_utterances_df (pd.DataFrame): DataFrame with 'utt', 'age_months', and 'tagged' columns.
        noun_to_types (dict): The map of nouns to their determiner types.
        num_passes (int): Number of times to run the filtering process.

    Returns:
        pd.DataFrame: DataFrame after the specified number of filtering passes.
    """
    if 'tagged' not in initial_utterances_df.columns:
        # If the very first input doesn't have 'tagged', POS tag it.
        logging.info("Initial input missing 'tagged' column. Performing initial POS tagging.")
        current_df = pos_tagging(initial_utterances_df)
    else:
        current_df = initial_utterances_df.copy()

    num_passes = num_passes
    if remove_conflicts:
        num_passes = 1

    for i in range(num_passes):
        logging.info(f"\n>>>> Starting Filtering Pass {i + 1} of {num_passes} <<<<\n")

        # Run the main filtering function
        # It's assumed that filter_det_noun_pairs returns a df with 'utt' and 'age_months'
        df_after_pass = filter_det_noun_pairs(current_df, noun_to_types, remove_conflicts)

        if i < num_passes - 1:
            # If it's not the last pass, re-tag the sentences for the next pass
            logging.info(f"Re-POS-tagging sentences after Pass {i + 1}...")
            # df_after_pass currently has 'utt' and 'age_months'.
            # pos_tagging will add/update the 'tagged' column based on the modified 'utt'.
            current_df = pos_tagging(df_after_pass)
            # Ensure age_months is carried over if pos_tagging doesn't preserve all columns
            # (though the provided pos_tagging seems to copy and add)
            if 'age_months' not in current_df.columns and 'age_months' in df_after_pass.columns:
                 current_df['age_months'] = df_after_pass['age_months']
        else:
            # This is the last pass
            current_df = df_after_pass

    logging.info(f"\n>>>> Completed {num_passes} Filtering Passes <<<<\n")
    return current_df
