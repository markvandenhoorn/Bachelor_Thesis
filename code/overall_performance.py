"""
Name: overall_performance.py
Author: Mark van den Hoorn
Desc: For checking the performance of the full, 58 month unfiltered model. Used
for making sure that the model has been trained correctly etcetera.
Outputs a confusion matrix.
"""
from transformers import BertForMaskedLM, BertTokenizerFast
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd

# load tokenizer and model
current_wd = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(current_wd, '..', 'custom_tokenizer')
model_path = os.path.join(current_wd, '..', 'models', 'unfiltered_model_up_to_58_months')
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
model = BertForMaskedLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

file_path = os.path.join(current_wd, '..', 'data', 'test_54_months_100.txt')

# get the det+nouns from the file
with open(file_path, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# have sentences with determiner and also one with mask + noun
gold_dets = []
masked_sentences = []

for line in lines:
    words = line.split()
    masked_sentence = []
    temp_gold_dets = []
    num_dets = 0

    for word in words:
        if word in ["a", "the"]:
            temp_gold_dets.append(word)
            masked_sentence.append("[MASK]")
            num_dets += 1
        else:
            masked_sentence.append(word)

    # keep only sentences with a single determiner
    if num_dets == 1:
        masked_sentences.append(" ".join(masked_sentence))
        gold_dets.extend(temp_gold_dets)

predicted_dets = []
all_dets = ["a", "the"]

# get model outputs for masked sentences
for sentence in masked_sentences:
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    # get predicted word from tokens
    mask_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    if len(mask_indices) != 1:
        # skip if somehow multiple masks got through
        continue

    logits = outputs.logits[0, mask_indices[0], :]
    predicted_token_id = logits.argmax(dim=-1).item()
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)

    if predicted_token in all_dets:
        predicted_dets.append(predicted_token)
    else:
        predicted_dets.append("other")

# make confusion matrix, calculate accuracy and some metrics
labels = ["a", "other", "the"]
cm = confusion_matrix(gold_dets, predicted_dets, labels=labels)
acc = accuracy_score(gold_dets, predicted_dets)
predicted_any_det = sum([1 for pred in predicted_dets if pred in ["a", "the"]])
coverage = predicted_any_det / len(predicted_dets)

# normalize
cm_normalized = cm / cm.sum()
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=labels)
disp.plot(cmap="Greys", values_format=".2%")

gold_counts = Counter(gold_dets)
print(f"Total gold test samples: {len(gold_dets)}")
print(f"Occurrences of 'a': {gold_counts['a']}")
print(f"Occurrences of 'the': {gold_counts['the']}")

# display
plt.title(f"LDP Corpus\nOverall accuracy: {acc:.2%}\nCoverage: {coverage:.2%}")
plt.tight_layout()
plt.show()
