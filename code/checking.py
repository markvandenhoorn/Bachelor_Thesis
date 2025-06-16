import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pos_tagging import pos_tagging
import spacy
import argparse

nlp = spacy.load("en_core_web_sm")

def analyze_determiner_noun_usage_spacy(df, lookahead=4):
    """
    Find out which determiners every noun is seen with after they were already
    filtered.
    """
    valid_determiner_words = {"a", "the"}
    noun_to_determiners_seen = defaultdict(set)

    print("Running spaCy POS tagging...")
    df_tagged = pos_tagging(df)

    print("Analyzing determiner-noun pairs...")
    for sentence_tags in tqdm(df_tagged["tagged"], desc="Analyzing pairs"):
        for idx, (word, pos) in enumerate(sentence_tags):
            if word.lower() in valid_determiner_words:
                current_det_word = word.lower()

                # look ahead up to N words to find a noun (not just anything after DT)
                for j in range(idx + 1, min(idx + 1 + lookahead, len(sentence_tags))):
                    next_word, next_pos = sentence_tags[j]

                    # ensure next word is really used as a noun
                    if next_pos.startswith("NOUN"):
                        noun = next_word.lower()
                        noun_to_determiners_seen[noun].add(current_det_word)
                        break
                    # optionally skip if next word is an adjective or verb
                    elif next_pos.startswith(("ADJ", "VERB")):
                        continue
                    else:
                        # stop if it's something else like a preposition or punctuation
                        break

    return noun_to_determiners_seen

def print_determiner_usage_summary(noun_to_determiners_seen):
    """
    Print how many nouns are seen with both a and the per dataset.
    """
    nouns_with_a_only = 0
    nouns_with_the_only = 0
    nouns_with_both = 0

    list_a_only = []
    list_the_only = []
    list_both = []

    for noun, dets_seen in noun_to_determiners_seen.items():
        has_a = 'a' in dets_seen
        has_the = 'the' in dets_seen

        if has_a and has_the:
            nouns_with_both += 1
            list_both.append(noun)
        elif has_a:
            nouns_with_a_only += 1
            list_a_only.append(noun)
        elif has_the:
            nouns_with_the_only += 1
            list_the_only.append(noun)

    total_nouns_tracked = len(noun_to_determiners_seen)

    # Print summary
    print("\n--- Determiner Usage Summary ---")
    print(f"Total distinct nouns found with 'a' or 'the': {total_nouns_tracked}")
    print(f"Nouns appearing with ONLY 'a': {nouns_with_a_only}")
    if list_a_only:
        print(f"  Examples: {', '.join(list_a_only[:10])}{'...' if len(list_a_only) > 10 else ''}")

    print(f"Nouns appearing with ONLY 'the': {nouns_with_the_only}")
    if list_the_only:
        print(f"  Examples: {', '.join(list_the_only[:10])}{'...' if len(list_the_only) > 10 else ''}")

    print(f"Nouns appearing with BOTH 'a' and 'the': {nouns_with_both}")
    if list_both:
        print(f"  Examples: {', '.join(list_both[:10])}{'...' if len(list_both) > 10 else ''}")

def main():
    parser = argparse.ArgumentParser(description="Run determiner-noun analysis on training data.")
    parser.add_argument("filter_level", choices=["0", "25", "50", "75", "unfiltered"], help="Filtering level or 'unfiltered'")
    args = parser.parse_args()

    filter_level = args.filter_level

    # Run the analysis
    for number in ["14", "18", "22", "26", "30", "34", "38", "42", "46", "50", "54", "58"]:
        if filter_level == "unfiltered":
            file_path = f"../data/unfiltered_train_data_up_to_{number}_months.txt"
        else:
            file_path = f"../data/filtered_train_data_{filter_level}_up_to_{number}_months.txt"

        training_df = pd.read_csv(file_path, header = None, names = ["utt"])
        analysis_results = analyze_determiner_noun_usage_spacy(training_df)
        print_determiner_usage_summary(analysis_results)

if __name__ == "__main__":
    main()
