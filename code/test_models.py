import os
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertForMaskedLM, BertTokenizer
from collections import defaultdict
from pos_tagging import pos_tagging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust these paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
TOKEN_DIR = os.path.join(BASE_DIR, '..', 'custom_tokenizer')

# Match test file pattern: test_14_months_0.txt
test_file_pattern = re.compile(r"test_(\d+)_months_(\d+).txt")

# Collect all matching test files
test_files = [f for f in os.listdir(DATA_DIR) if test_file_pattern.match(f)]

results = []

for test_file in test_files:
    match = test_file_pattern.match(test_file)
    if not match:
        continue

    age = int(match.group(1))
    model_type = int(match.group(2))

    model_folder_name = f"filtered_{model_type}_model_up_to_{age}_months"
    #
    #
    #
    #
    #
    # remove this when 75 model has been trained
    if "filtered_75_model" in model_folder_name:
        print(f"Skipping model {model_folder_name}")
        continue
    if "filtered_100_model" in model_folder_name:
        model_folder_name = model_folder_name.replace("filtered_100_model", "unfiltered_model")
    model_path = os.path.normpath(os.path.join(MODEL_DIR, model_folder_name))
    tokenizer = BertTokenizer.from_pretrained(TOKEN_DIR)
    model = BertForMaskedLM.from_pretrained(model_path).to(device).eval()

    file_path = os.path.join(DATA_DIR, test_file)
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    a_to_the = 0
    the_to_a = 0
    pred_a_or_the = 0

    # convert sentences to a DataFrame for POS tagging
    df_sentences = pd.DataFrame({"utt": sentences})

    # perform POS tagging for the entire dataframe
    pos_tags_df = pos_tagging(df_sentences)
    pos_tags_list = pos_tags_df['tagged'].tolist()

    # Iterate over the sentences and their corresponding POS tags
    for sentence, pos_tags in zip(sentences, pos_tags_list):
        det_indices = [
            i for i, (word, tag) in enumerate(pos_tags)
            if word.lower() in ["a", "the"]
        ]

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

        if len(candidate_indices) != 1:
            continue

        det_index = candidate_indices[0]
        gold_det = pos_tags[det_index][0].lower()

        # Mask the determiner
        masked_tokens = [tok for tok, _ in pos_tags]
        masked_tokens[det_index] = "[MASK]"
        masked_sentence = " ".join(masked_tokens)

        # Run model
        inputs = tokenizer(masked_sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        mask_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        if len(mask_index) != 1:
            continue

        logits = outputs.logits[0, mask_index[0]]
        predicted_token_id = logits.argmax().item()
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)

        if predicted_token not in ["a", "the"]:
            continue

        pred_a_or_the += 1
        if gold_det == "a" and predicted_token == "the":
            a_to_the += 1
        elif gold_det == "the" and predicted_token == "a":
            the_to_a += 1

    a_to_the_ratio = a_to_the / pred_a_or_the if pred_a_or_the > 0 else 0.0
    the_to_a_ratio = the_to_a / pred_a_or_the if pred_a_or_the > 0 else 0.0

    results.append({
        "model_type": model_type,
        "age": age,
        "a_to_the": a_to_the,
        "the_to_a": the_to_a,
        "pred_a_or_the": pred_a_or_the,
        "a_to_the_ratio": a_to_the_ratio,
        "the_to_a_ratio": the_to_a_ratio,
    })

# Save to CSV
df = pd.DataFrame(results)
df.sort_values(["model_type", "age"], inplace=True)
csv_path = os.path.join(BASE_DIR, "misclassification_ratios.csv")
df.to_csv(csv_path, index=False)
print(f"Saved summary to {csv_path}")

# Plot per model type
output_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(output_dir, exist_ok=True)

for model_type in df["model_type"].unique():
    subset = df[df["model_type"] == model_type]

    plt.figure()
    plt.plot(subset["age"], subset["a_to_the_ratio"], label="'a' → 'the'", marker='o')
    plt.plot(subset["age"], subset["the_to_a_ratio"], label="'the' → 'a'", marker='o')
    plt.title(f"Model Type {model_type}: Misclassification Ratios")
    plt.xlabel("Age (months)")
    plt.ylabel("Ratio")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"model_type_{model_type}.png"))
    plt.close()
