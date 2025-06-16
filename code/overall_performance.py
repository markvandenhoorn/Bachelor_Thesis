"""
Name: overall_performance.py
Author: Mark van den Hoorn
Description:
Evaluates the performance of a BERT-based model trained on DET-NOUN prediction
tasks. Outputs flexible use plots and a confusion matrix comparing model
predictions with child utterances.
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import re
import ast
import random
import argparse
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from statistics import median
from transformers import BertForMaskedLM, BertTokenizerFast
from pos_tagging import pos_tagging
from utils import set_wd, is_only_mask, clean_encoded_characters

def import_data(filename):
    """
    Reads in the data, drops NA, removes masks and replaces an with a.
    """
    data = pd.read_csv(filename, index_col=0)
    child_data = data.dropna(subset=["c_utts", "age_months"])

    # remove sentences with only MASK
    child_utterances = child_data[~child_data["c_utts"].apply(is_only_mask)][["subject","c_utts", "age_months"]]

    # replace 'an' with 'a' for consistency
    child_utterances['c_utts'] = child_utterances['c_utts'].str.replace(r'\ban\b', 'a', regex=True)
    child_utterances['c_utts'] = child_utterances['c_utts'].str.replace(r'\bAn\b', 'A', regex=True)
    child_utterances['c_utts'] = child_utterances['c_utts'].str.replace(r'@', '', regex=True)

    # revert encoded characters back to original
    child_utterances = clean_encoded_characters(child_utterances, 'c_utts')

    return child_utterances

def first_preprocess(utterances, first_names, nlp):
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

    return pd.DataFrame({
        'utt': replaced,
        'age_months': utterances['age_months'].values,
        'subject': utterances['subject'].values
    })

def preprocess_and_filter_ldp(df):
    """Load and filter the LDP corpus for DET+(X)+NOUN utterances."""
    df = df.rename(columns={"c_utts": "utt"})
    df = df[["subject", "age_months", "utt"]].dropna(subset=["utt"])
    df = pos_tagging(df)
    df = df[df["tagged"].apply(contains_det_noun_or_det_x_noun)]
    return df

def contains_det_noun_or_det_x_noun(tagged):
    """Return True if tagged utterance contains 'DET NOUN' or 'DET X NOUN' pattern."""
    for i in range(len(tagged)):
        if i + 1 < len(tagged) and tagged[i][1] == 'DET' and tagged[i + 1][1] == 'NOUN':
            return True
        if i + 2 < len(tagged) and tagged[i][1] == 'DET' and tagged[i + 2][1] == 'NOUN':
            return True
    return False

def load_model_and_tokenizer(model_name):
    """Load fine-tuned model and tokenizer."""
    current_wd = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(current_wd, '..', 'custom_tokenizer')
    model_path = os.path.join(current_wd, '..', 'models', model_name)

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(model_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer, current_wd

def process_data(df):
    """Extract gold determiners, create masked inputs, and track target nouns."""
    gold_dets, masked_sentences, target_nouns = [], [], []

    for tagged in df['tagged']:
        det_indices = [i for i, (word, _) in enumerate(tagged) if word.lower() in ["a", "the"]]
        if len(det_indices) != 1:
            continue
        idx = det_indices[0]
        gold_dets.append(tagged[idx][0].lower())

        masked_tokens = ["[MASK]" if i == idx else word for i, (word, _) in enumerate(tagged)]
        masked_sentences.append(" ".join(masked_tokens))

        noun = tagged[idx + 1][0].lower() if idx + 1 < len(tagged) and tagged[idx + 1][1] == 'NOUN' else None
        target_nouns.append(noun)

    return gold_dets, masked_sentences, target_nouns

def replace_det_in_tagged(tagged, new_det):
    """Replace the first DET token in a tagged list with the given determiner."""
    new_tagged = []
    replaced = False
    for word, pos in tagged:
        if not replaced and word.lower() in ["a", "the"]:
            new_tagged.append((new_det, pos))
            replaced = True
        else:
            new_tagged.append((word, pos))
    return new_tagged

def get_predictions(model, tokenizer, masked_sentences):
    """Run masked language model predictions on a list of sentences."""
    predicted_dets = []

    for sentence in masked_sentences:
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        mask_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        if len(mask_indices) != 1:
            predicted_dets.append("other")
            continue

        logits = outputs.logits[0, mask_indices[0], :]
        predicted_token_id = logits.argmax().item()
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)

        predicted_dets.append(predicted_token if predicted_token in ["a", "the"] else "other")

    return predicted_dets

def compute_flexible_use(df, group_keys=["age_months", "subject"]):
    grouped = df.groupby(group_keys)
    noun_counts_per_group = defaultdict(list)

    for keys, group in grouped:
        noun_dets = defaultdict(set)
        for tagged in group["tagged"]:
            for i in range(len(tagged)):
                if i + 1 < len(tagged) and tagged[i][1] == "DET" and tagged[i + 1][1] == "NOUN":
                    noun_dets[tagged[i + 1][0].lower()].add(tagged[i][0].lower())
                if i + 2 < len(tagged) and tagged[i][1] == "DET" and tagged[i + 2][1] == "NOUN":
                    noun_dets[tagged[i + 2][0].lower()].add(tagged[i][0].lower())

        count = sum(1 for dets in noun_dets.values() if {"a", "the"}.issubset(dets))
        noun_counts_per_group[keys[0]].append(count)

    # For each age group, calculate median and IQR (Q1 and Q3)
    result = []
    for age, counts in sorted(noun_counts_per_group.items()):
        counts_sorted = sorted(counts)
        med = median(counts_sorted)
        q1 = counts_sorted[len(counts_sorted)//4]
        q3 = counts_sorted[(3*len(counts_sorted))//4]
        result.append((age, med, q1, q3))

    return result

def plot_flexible_use(child_data, model_data, model_name, current_wd):
    """Plot flexible determiner use by age for children and model, with IQR error bars."""
    age_c, med_c, q1_c, q3_c = zip(*child_data)
    age_m, med_m, q1_m, q3_m = zip(*model_data)

    # Calculate error bar sizes
    err_lower_c = [med - q1 for med, q1 in zip(med_c, q1_c)]
    err_upper_c = [q3 - med for med, q3 in zip(med_c, q3_c)]

    err_lower_m = [med - q1 for med, q1 in zip(med_m, q1_m)]
    err_upper_m = [q3 - med for med, q3 in zip(med_m, q3_m)]

    plt.figure(figsize=(12, 5))

    plt.errorbar(age_c, med_c, yerr=[err_lower_c, err_upper_c], fmt='-o', label='Children', color='orange', capsize=5)
    plt.errorbar(age_m, med_m, yerr=[err_lower_m, err_upper_m], fmt='-d', label='Model', color='deepskyblue', capsize=5)

    plt.axhline(2, linestyle='--', color='black', linewidth=0.5)
    plt.xlabel("Age (months old)")
    plt.ylabel("Median noun types with both 'a' and 'the'")
    plt.title("Median Use of Both Determiners for a Noun")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(current_wd, '..', 'output', f"both_det_use_{model_name}.png")
    plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(gold_dets, predicted_dets, model_name, current_wd):
    """Plot and save normalized confusion matrix comparing model predictions."""
    labels = ["a", "other", "the"]
    cm = confusion_matrix(gold_dets, predicted_dets, labels=labels)
    acc = accuracy_score(gold_dets, predicted_dets)
    coverage = sum(1 for p in predicted_dets if p in ["a", "the"]) / len(predicted_dets)

    cm_normalized = cm / cm.sum()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=labels)
    disp.plot(cmap="Greys", values_format=".2%")
    plt.title(f"LDP Corpus\nOverall accuracy: {acc:.2%}\nCoverage: {coverage:.2%}")
    plt.tight_layout()

    save_path = os.path.join(current_wd, '..', 'output', f"overall_performance_{model_name}.png")
    plt.savefig(save_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate age-specific model performance")
    parser.add_argument('--model_type', help="Prefix for model directory (e.g., 'filtered_0_model')", required=False)
    args = parser.parse_args()

    set_wd()
    nlp = spacy.load('en_core_web_sm')

    if not args.model_type:
        # Preprocessing step to create 'overall_performance.txt'
        ldp_csv_path = 'ldp_data.csv'
        first_names = list(pd.read_csv("first_names.csv")['Name'])

        df = import_data(ldp_csv_path)
        df = first_preprocess(df, first_names, nlp)
        df = preprocess_and_filter_ldp(df)
        df.to_csv("overall_performance.txt", index=False)

    else:
        model_prefix = args.model_type
        df = pd.read_csv("overall_performance.txt")
        df["tagged"] = df["tagged"].apply(ast.literal_eval)

        # Compute baseline (children)
        child_medians = compute_flexible_use(df)

        # Get current working directory for plotting/saving
        current_wd = os.path.dirname(os.path.abspath(__file__))

        # Collect predictions across age-specific models
        all_preds, all_gold, all_nouns, all_tagged, all_ages, all_subjects = [], [], [], [], [], []

        for age, group in df.groupby("age_months"):
            model_name = f"{model_prefix}_up_to_{int(age)}_months"
            try:
                model, tokenizer, _ = load_model_and_tokenizer(model_name)
            except Exception as e:
                print(f"[Skipping age {age}] Could not load model '{model_name}': {e}")
                continue

            gold_dets, masked_sentences, target_nouns = process_data(group)
            if not masked_sentences:
                continue

            predicted_dets = get_predictions(model, tokenizer, masked_sentences)

            all_preds.extend(predicted_dets)
            all_gold.extend(gold_dets)
            all_nouns.extend(target_nouns)
            all_tagged.extend(group["tagged"].iloc[:len(predicted_dets)])
            all_ages.extend([age] * len(predicted_dets))
            all_subjects.extend(group["subject"].iloc[:len(predicted_dets)])

        pred_df = pd.DataFrame({
            "age_months": all_ages,
            "subject": all_subjects,
            "tagged": all_tagged,
            "pred_det": all_preds,
            "noun": all_nouns
        })
        pred_df = pred_df[pred_df["pred_det"].isin(["a", "the"])].copy()
        pred_df["tagged"] = [replace_det_in_tagged(t, d) for t, d in zip(pred_df["tagged"], pred_df["pred_det"])]

        model_medians = compute_flexible_use(pred_df, group_keys=["age_months", "subject"])

        plot_flexible_use(child_medians, model_medians, model_prefix + "_age_aligned", current_wd)

        max_age = 58
        model_name = f"{model_prefix}_up_to_{max_age}_months"
        group = df[df["age_months"] <= max_age]

        try:
            model, tokenizer, _ = load_model_and_tokenizer(model_name)
        except Exception as e:
            print(f"[Skipping confusion matrix] Could not load model '{model_name}': {e}")
            return

        gold_dets, masked_sentences, _ = process_data(group)
        if not masked_sentences:
            print("[No valid masked sentences for confusion matrix]")
            return

        predicted_dets = get_predictions(model, tokenizer, masked_sentences)

        filtered = [
            (g, p) for g, p in zip(gold_dets, predicted_dets)
            if g in ["a", "the"] and p in ["a", "the", "other"]
        ]
        if filtered:
            gold_dets_filtered, predicted_dets_filtered = zip(*filtered)
        else:
            gold_dets_filtered, predicted_dets_filtered = [], []

        plot_confusion_matrix(gold_dets_filtered, predicted_dets_filtered, model_name + "_age_aligned", current_wd)


if __name__ == "__main__":
    main()
