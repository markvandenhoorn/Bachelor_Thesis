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
from scipy.stats import ttest_ind

# set wd and set device to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_wd()
output_dir = os.path.join("..", "output")
os.makedirs(output_dir, exist_ok=True)

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
    df = pd.read_csv(file_path, decimal=',')
    df["tagged"] = df["tagged"].apply(ast.literal_eval)

    sentences = df["utt"].tolist()
    pos_tags_list = df["tagged"].tolist()

    a_to_the = 0
    the_to_a = 0
    pred_a_or_the = 0
    a_predictions = 0
    the_predictions = 0
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

        # count a and the predictions
        if predicted_token == "a":
            a_predictions += 1
        elif predicted_token == "the":
            the_predictions += 1

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
        "a_predictions": a_predictions,
        "the_predictions": the_predictions,
        "total_predictions": total_sentences,
        "a_to_the": a_to_the,
        "the_to_a": the_to_a,
        "pred_a_or_the": pred_a_or_the,
        "a_to_the_ratio": a_to_the_ratio,
        "the_to_a_ratio": the_to_a_ratio,
        "total_misclassified": (a_to_the + the_to_a) / pred_a_or_the if pred_a_or_the > 0 else 0.0,
        "overall_accuracy": overall_accuracy,
        "determiner_accuracy": determiner_accuracy
    })

# save to CSV
df = pd.DataFrame(results)
df.sort_values(["model_type", "age"], inplace=True)

# Compute summary statistics per model type
summary_stats = df.groupby("model_type").agg({
    "total_misclassified": ["mean", "std"],
    "determiner_accuracy": ["mean", "std"],
    "overall_accuracy": ["mean", "std"],
    "pred_a_or_the": "sum",
    "total_predictions": "sum"
}).reset_index()

# Flatten multi-level column names
summary_stats.columns = ['model_type', 'misclass_mean', 'misclass_std',
                         'det_acc_mean', 'det_acc_std',
                         'overall_acc_mean', 'overall_acc_std',
                         'total_determiner_preds', 'total_preds']

# Save to CSV
summary_csv_path = os.path.join(output_dir, "summary_statistics_by_model_type.csv")
summary_stats.to_csv(summary_csv_path, index=False)
print(f"Saved summary statistics to {summary_csv_path}")

# Compare misclassification ratio across model types pairwise
model_types = df["model_type"].unique()
ttest_results = []

for i, m1 in enumerate(model_types):
    for m2 in model_types[i+1:]:
        group1 = pd.to_numeric(df[df["model_type"] == m1]["total_misclassified"], errors='coerce')
        group2 = pd.to_numeric(df[df["model_type"] == m2]["total_misclassified"], errors='coerce')
        stat, pval = ttest_ind(group1.dropna(), group2.dropna(), equal_var=False)
        ttest_results.append({
            "model_type_1": m1,
            "model_type_2": m2,
            "t_stat": stat,
            "p_value": pval
        })

# Save t-test results
ttest_df = pd.DataFrame(ttest_results)
ttest_csv_path = os.path.join(output_dir, "ttest_generalization_ratios.csv")
ttest_df.to_csv(ttest_csv_path, index=False)
print(f"Saved t-test results to {ttest_csv_path}")

csv_path = os.path.join(output_dir, "misclassification_ratios.csv")
df.to_csv(csv_path, index=False)
print(f"Saved summary to {csv_path}")

# plot colors
custom_colors = {
    0: "#cadcff",
    25: "#c5ffed",
    50: "#9dbaf4",
    75: "#70b4e5",
    100: "#2dad9d"
}

# plot misclassification ratios
for model_type in df["model_type"].unique():
    subset = df[df["model_type"] == model_type]

    plt.figure()
    plt.plot(subset["age"], subset["a_to_the_ratio"], label="'a' → 'the'", marker='o')
    plt.plot(subset["age"], subset["the_to_a_ratio"], label="'the' → 'a'", marker='o')
    plt.plot(subset["age"], subset["total_misclassified"], label="Total generalized predictions", marker='x', linestyle='--', color='gray')
    plt.title(f"Model Type {model_type}: Generalized predictions")
    plt.xlabel("Age (months)")
    plt.ylabel("Ratio")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Generalization_ratios_{model_type}.png"))
    plt.close()

# plot misclassification numbers
for model_type in df["model_type"].unique():
    subset = df[df["model_type"] == model_type]

    plt.figure()
    plt.plot(subset["age"], subset["a_to_the"], label="'a' → 'the'", marker='o')
    plt.plot(subset["age"], subset["the_to_a"], label="'the' → 'a'", marker='o')
    plt.plot(subset["age"], (subset["a_to_the"]+subset["the_to_a"]), label="Total generalized predictions", marker='x', linestyle='--', color='gray')
    plt.title(f"Model Type {model_type}: Generalization in predictions")
    plt.xlabel("Age (months)")
    plt.ylabel("amount")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Generalization_numbers_{model_type}.png"))
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
    plt.savefig(os.path.join(output_dir, f"accuracy_{model_type}.png"))
    plt.close()

# plot prediction ratios ('a' and 'the') per model type
for model_type in df["model_type"].unique():
    subset = df[df["model_type"] == model_type]

    plt.figure()
    plt.plot(subset["age"], subset["a_predictions"] / subset["total_predictions"], label="Ratio of 'a' predictions", marker='o')
    plt.plot(subset["age"], subset["the_predictions"] / subset["total_predictions"], label="Ratio of 'the' predictions", marker='s')
    plt.plot(subset["age"], (subset["the_predictions"]+subset["a_predictions"]) / subset["total_predictions"], label="Ratio of any determiner predictions", marker='s')
    plt.title(f"Model Type {model_type}: Prediction Ratios Over Time")
    plt.xlabel("Age (months)")
    plt.ylabel("Prediction Ratio")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"prediction_ratios_{model_type}.png"))
    plt.close()

# Plot: Total Generalization Ratio Across Model Types
plt.figure()
for model_type in df["model_type"].unique():
    subset = df[df["model_type"] == model_type]
    color = custom_colors.get(model_type, None)
    plt.plot(subset["age"], subset["total_misclassified"], label=f"Model {model_type}", marker='o', color = color)
plt.title("Generalization Ratio")
plt.xlabel("Age (months)")
plt.ylabel("Generalization Ratio")
plt.ylim(0, 1)
plt.grid(True)
plt.legend(title="Model Type")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "total_generalization_ratio_all_models.png"))
plt.close()

# Plot: Determiner Prediction Accuracy Across Model Types
plt.figure()
for model_type in df["model_type"].unique():
    subset = df[df["model_type"] == model_type]
    color = custom_colors.get(model_type, None)
    plt.plot(subset["age"], subset["determiner_accuracy"], label=f"Model {model_type}", marker='s', color = color)
plt.title("Determiner Accuracy")
plt.xlabel("Age (months)")
plt.ylabel("Accuracy (Given Determiner Prediction)")
plt.ylim(0, 1)
plt.grid(True)
plt.legend(title="Model Type")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "determiner_accuracy_all_models.png"))
plt.close()

# plot total generalization amount grouped
plt.figure()
for model_type in df["model_type"].unique():
    subset = df[df["model_type"] == model_type]
    total_generalizations = subset["a_to_the"] + subset["the_to_a"]
    color = custom_colors.get(model_type, None)
    plt.plot(subset["age"], total_generalizations, label=f"Model {model_type}", marker='o', color=color)
plt.title("Total Generalization Count")
plt.xlabel("Age (months)")
plt.ylabel("Generalized Predictions (Count)")
plt.grid(True)
plt.legend(title="Model Type")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "total_generalization_count_all_models.png"))
plt.close()

# Plot: Any Determiner Prediction Ratio Across Model Types
plt.figure()
for model_type in df["model_type"].unique():
    subset = df[df["model_type"] == model_type]
    color = custom_colors.get(model_type, None)
    det_ratio = (subset["a_predictions"] + subset["the_predictions"]) / subset["total_predictions"]
    plt.plot(subset["age"], det_ratio, label=f"Model {model_type}", marker='o', color=color)
plt.title("Ratio of Any Determiner Prediction Across Model Types")
plt.xlabel("Age (months)")
plt.ylabel("Determiner Prediction Ratio")
plt.ylim(0, 1)
plt.grid(True)
plt.legend(title="Model Type")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "any_determiner_prediction_ratio_all_models.png"))
plt.close()

# Error bar plot: Mean and std of total misclassified per model
plt.figure()
plt.bar(summary_stats["model_type"].astype(str),
        summary_stats["misclass_mean"],
        yerr=summary_stats["misclass_std"],
        capsize=5,
        color=[custom_colors.get(mt, "#cccccc") for mt in summary_stats["model_type"]])
plt.title("Avg. Total Generalized Ratio by Model Type")
plt.xlabel("Model Type")
plt.ylabel("Mean Generalization Ratio")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mean_misclassification_ratio_per_model.png"))
plt.close()
