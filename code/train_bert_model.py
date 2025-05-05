"""
Name: train_bert_model.py
Author: Mark van den Hoorn
Desc: Trains a bert model on child-directed utterances. The age of the children
increases incrementally, models get made with every increment.
Needs 1 argument to be run: filtered/unfiltered. Example of how to run:
python train_bert_model.py unfiltered
"""
import torch
import os
import argparse
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizerFast
from transformers import BertTokenizer, BertForMaskedLM, BertTokenizerFast, BertConfig
from datasets import load_dataset

def create_model(config):
    model = BertForMaskedLM(config=config)
    model = model.to(device)
    print("Number of parameters: ", model.num_parameters())
    return model

def prepare_data(filepath, tokenizer):
    """
    Prepares data using the Hugging Face `datasets` library for MLM.
    """
    dataset = load_dataset(
        "text",
        data_files={"train": filepath},
        split="train"
    )

    # tokenize the dataset
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Remove text column since it's now tokenized
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    return tokenized_dataset, data_collator

def train_single_model(model, tokenizer, datafile, name, modelconfig,
    trainingconfig, outputpath, tokenizer_path):
    """
    Trains a single BERT model on specified data. Model is saved in given output
    folder.
    """
    # tokenize data and create collator
    dataset, data_collator = prepare_data(datafile, tokenizer)

    # instantiate a trainer
    trainer = Trainer(model=model,
                      args=trainingconfig,
                      data_collator=data_collator,
                      train_dataset=dataset
                     )

    # start training
    train_output = trainer.train()

    # save model in output folder
    trainer.save_model(os.path.join(outputpath, name))

def train_bert_incremental(modelconfig, trainingconfig, path_to_data, output_path,
    tokenizer_path, age_ranges, data_type):
    """
    Trains a BERT model incrementally over multiple sessions/phases, by
    age ranges of LDP corpus data.
    """
    # load  our pretrained tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        max_len=128
    )

    # update model config with vocab size and create the model
    modelconfig.vocab_size = tokenizer.vocab_size
    model = create_model(modelconfig)

    # set input and output names based on argparsed choice
    input_fnames = f"{data_type}_train_data_up_to_{{}}_months.txt"
    output_fnames = f"{data_type}_model_up_to_{{}}_months"

    for i, age_limit in enumerate(age_ranges):
        session = i + 1
        print(f"\n--- Training for Session {session} (Age <= {age_limit} months) ---")

        # create file name
        datafile = os.path.join(path_to_data, input_fnames.format(age_limit))
        # output model with correct name
        output_name = output_fnames.format(age_limit)

        # train model
        train_single_model(model, tokenizer, datafile, output_name, modelconfig,
                           trainingconfig, output_path, tokenizer_path)

if __name__ == "__main__":
    # argparser to specify whether you want to use filtered or unfiltered data
    parser = argparse.ArgumentParser(description="Use Filtered or Unfiltered data for model training.")
    parser.add_argument(
        "data_type",
        type=str,
        choices=["filtered", "unfiltered"],
        help="Specify whether to use 'filtered' or 'unfiltered' training data."
    )
    args = parser.parse_args()

    # BERT configuration
    modelconfig = BertConfig(
        seed=29032001,
        num_attention_heads=2,
        num_hidden_layers=1,
        type_vocab_size=1
    )

    # trainer configuration
    trainingconfig = TrainingArguments(
        overwrite_output_dir=True,
        per_device_train_batch_size = 64,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        resume_from_checkpoint=None
    )

    # set age ranges for file naming
    age_ranges = [14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54]

    # set paths for input, output and tokenizer
    current_wd = os.path.dirname(os.path.abspath(__file__))
    path_to_data = os.path.join(current_wd, '..', 'data')
    output_path = os.path.join(current_wd, '..', 'models')
    tokenizer_path =os.path.join(current_wd, '..', 'custom_tokenizer')

    # check for GPU and set device
    print("Cuda available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train incremental models
    train_bert_incremental(modelconfig, trainingconfig, path_to_data, output_path,
        tokenizer_path, age_ranges, args.data_type)
