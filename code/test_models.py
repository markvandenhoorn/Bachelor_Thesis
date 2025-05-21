import os
import re
import torch
import pandas as pd
import ast
import pickle
import matplotlib.pyplot as plt
from utils import set_wd
from transformers import BertForMaskedLM, BertTokenizer
from collections import defaultdict

# set wd and set device to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_wd()

# load dictionary of noun to assigned determiner
with open("determiner_dicts.pkl", "rb") as f:
    dicts = pickle.load(f)

# map model_type to the appropriate dictionary
determiner_dicts_by_model_type = {
    100: dicts["regular"],
    0: dicts["0"],
    25: dicts["25"],
    50: dicts["50"],
    75: dicts["75"],
}

# set paths to directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
TOKEN_DIR = os.path.join(BASE_DIR, '..', 'custom_tokenizer')

# get the pattern for the test files
test_file_pattern = re.compile(r"test_(\d+)_months_(\d+).txt")

# find all test files using pattern
test_files = [f for f in os.listdir(DATA_DIR) if test_file_pattern.match(f)]
results = []

for test_file in test_files:
    match = test_file_pattern.match(test_file)
    if not match:
        continue

    age = int(match.group(1))
    model_type = int(match.group(2))

    # set name of the folder and load tokenizer and model
    model_folder_name = f"filtered_{model_type}_model_up_to_{age}_months"
    if "filtered_100_model" in model_folder_name:
        model_folder_name = model_folder_name.replace("filtered_100_model", "unfiltered_model")
    model_path = os.path.normpath(os.path.join(MODEL_DIR, model_folder_name))
    tokenizer = BertTokenizer.from_pretrained(TOKEN_DIR)
    model = BertForMaskedLM.from_pretrained(model_path).to(device).eval()

    # load in test data
    file_path = os.path.join(DATA_DIR, test_file)
    df = pd.read_csv(file_path)
    df["tagged"] = df["tagged"].apply(ast.literal_eval)

    sentences = df["utt"].tolist()
    pos_tags_list = df["tagged"].tolist()

    a_to_the = 0
    the_to_a = 0
    pred_a_or_the = 0
    correct_overall = 0
    total_sentences = 0

    # iterate over the sentences in test data and their corresponding POS tags
    for sentence, pos_tags in zip(sentences, pos_tags_list):
        total_sentences += 1

        # find index of the determiner in the sentence
        det_indices = [
            i for i, (word, tag) in enumerate(pos_tags)
            if word.lower() in ["a", "the"]
        ]

        # find noun after determiner
        candidate_indices = []
        for i in det_indices:
            if i + 1 < len(pos_tags) and pos_tags[i + 1][1].startswith("NOUN"):
                candidate_indices.append(i)
            elif (
                i + 2 < len(pos_tags)
                and pos_tags[i + 1][1] not in ["NOUN"]
                and pos_tags[i + 2][1].startswith("NOUN")
            ):
                candidate_indices.append(i)

        # we only want sentences with 1 det + noun
        if len(candidate_indices) != 1:
            continue

        det_index = candidate_indices[0]

        # check if noun is in dictionary with determiner types
        if det_index + 1 < len(pos_tags) and pos_tags[det_index + 1][1].startswith("NOUN"):
            noun = pos_tags[det_index + 1][0].lower()
        elif (
            det_index + 2 < len(pos_tags)
            and not pos_tags[det_index + 1][1].startswith("NOUN")
            and pos_tags[det_index + 2][1].startswith("NOUN")
        ):
            noun = pos_tags[det_index + 2][0].lower()
        else:
            print("no noun found")
            continue

        det_dict = determiner_dicts_by_model_type.get(model_type, {})
        det_type_set = det_dict.get(noun)

        # check if we have a determiner in the dictionary for this noun
        if not det_type_set or not det_type_set.intersection({"A", "B"}):
            # skip it if not
            continue

        # choose determiner based on what's in the set
        if "A" in det_type_set:
            gold_det = "a"
        elif "B" in det_type_set:
            gold_det = "the"
        else:
            continue

        # also save the actual determiner in sentence for accuracy
        actual_det = pos_tags[det_index][0].lower()

        # mask the determiner
        masked_tokens = [tok for tok, _ in pos_tags]
        masked_tokens[det_index] = "[MASK]"
        masked_sentence = " ".join(masked_tokens)

        # run model on test sentence
        inputs = tokenizer(masked_sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        mask_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        if len(mask_index) != 1:
            continue

        # get predicted token
        logits = outputs.logits[0, mask_index[0]]
        predicted_token_id = logits.argmax().item()
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)

        # don't need to check what prediction is if it is not a determiner
        if predicted_token not in ["a", "the"]:
            continue

        # check if prediction is the right one for this sentence
        if predicted_token == actual_det:
            correct_overall += 1

        # check if prediction is other than the assigned det type
        pred_a_or_the += 1
        if gold_det == "a" and predicted_token == "the":
            a_to_the += 1
        elif gold_det == "the" and predicted_token == "a":
            the_to_a += 1

    # calculate misclassification ratio and accuracies
    a_to_the_ratio = a_to_the / pred_a_or_the if pred_a_or_the > 0 else 0.0
    the_to_a_ratio = the_to_a / pred_a_or_the if pred_a_or_the > 0 else 0.0
    overall_accuracy = correct_overall / total_sentences if total_sentences > 0 else 0.0
    determiner_accuracy = correct_overall / pred_a_or_the if pred_a_or_the > 0 else 0.0

    results.append({
        "model_type": model_type,
        "age": age,
        "a_to_the": a_to_the,
        "the_to_a": the_to_a,
        "pred_a_or_the": pred_a_or_the,
        "a_to_the_ratio": a_to_the_ratio,
        "the_to_a_ratio": the_to_a_ratio,
        "total_misclassified": (a_to_the + the_to_a) / pred_a_or_the if pred_a_or_the > 0 else 0.0,
        "overall_accuracy": overall_accuracy,
        "determiner_accuracy": determiner_accuracy,
        "total_predictions": total_sentences
    })

# save to CSV
df = pd.DataFrame(results)
df.sort_values(["model_type", "age"], inplace=True)
csv_path = os.path.join(BASE_DIR, "misclassification_ratios.csv")
df.to_csv(csv_path, index=False)
print(f"Saved summary to {csv_path}")

# plot per model type
output_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(output_dir, exist_ok=True)

# plot misclassification ratios
for model_type in df["model_type"].unique():
    subset = df[df["model_type"] == model_type]

    plt.figure()
    plt.plot(subset["age"], subset["a_to_the_ratio"], label="'a' → 'the'", marker='o')
    plt.plot(subset["age"], subset["the_to_a_ratio"], label="'the' → 'a'", marker='o')
    plt.plot(subset["age"], subset["total_misclassified"], label="Total misclassifications", marker='x', linestyle='--', color='gray')
    plt.title(f"Model Type {model_type}: Misclassification Ratios")
    plt.xlabel("Age (months)")
    plt.ylabel("Ratio")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"model_type_{model_type}.png"))
    plt.close()

# plot accuracies
for model_type in df["model_type"].unique():
    subset = df[df["model_type"] == model_type]

    plt.figure()
    plt.plot(subset["age"], subset["overall_accuracy"], label="Overall Accuracy", marker='o')
    plt.plot(subset["age"], subset["determiner_accuracy"], label="Accuracy (Given Determiner Prediction)", marker='s')
    plt.title(f"Model Type {model_type}: Accuracy Over Time")
    plt.xlabel("Age (months)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"accuracy_model_type_{model_type}.png"))
    plt.close()
