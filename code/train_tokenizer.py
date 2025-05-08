"""
Name: train_tokenizer.py
Author: Mark van den Hoorn
Desc: Trains a BertWordPieceTokenizer on all parent utterances from LDP corpus.
"""
from transformers import BertTokenizer, BertForMaskedLM, BertTokenizerFast, BertConfig
from tokenizers.implementations import ByteLevelBPETokenizer, BertWordPieceTokenizer
from tokenizers.pre_tokenizers import Whitespace, BertPreTokenizer
from preprocess_data import set_wd
import os

def create_tokenizer(files):
    """Creates a tokenizer by training it from scratch"""
    # initialize tokenizer
    tokenizer = BertWordPieceTokenizer()

    # split words based on whitespace
    tokenizer.pre_tokenizer = BertPreTokenizer()

    # include all numbers in the special tokens
    tokenizer.train(files, vocab_size=30_000, min_frequency=2, special_tokens=[
        "<s>",
        "[UNK]",
        "[PAD]",
        "[CLS]",
        "</s>",
        "[MASK]",
        "[SEP]",
    ])

    return tokenizer

# set working directory to data folder
set_wd()

# create and train the tokenizer
tokenizer = create_tokenizer(["tokenizer_train_data.txt"])

# save tokenizer
save_dir = os.path.join(os.getcwd(), "..", "custom_tokenizer")
os.makedirs(save_dir, exist_ok=True)
tokenizer.save_model(save_dir)
