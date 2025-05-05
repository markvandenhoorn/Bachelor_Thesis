"""
Name: train_bert_model.py
Author: Mark van den Hoorn
Desc: Trains a bert model on child-directed utterances. The age of the children
increases incrementally, models get made with every increment.
"""
import torch
import os
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
    tokenizer_path, max_phase, input_fnames="", output_fnames=""):
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

    # loop through each training phase/session
    for session in range(1, max_phase + 1):
        print(f"\n Training for session {session}:")

        datafile = os.path.join(path_to_data, input_fnames.format(session))
        output_name = output_fnames.format(session)

        train_single_model(model, tokenizer, datafile, output_name, modelconfig,
            trainingconfig, output_path, tokenizer_path)






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

# set paths for input, output and tokenizer
current_wd = os.path.dirname(os.path.abspath(__file__))
path_to_data = os.path.join(current_wd, '..', 'data')
output_path = os.path.join(current_wd, '..', 'models')
tokenizer_path =os.path.join(current_wd, '..', 'custom_tokenizer')

# set name formats for input and output
input_fnames = "train_data_up_to_{}_months.txt"
output_fnames = "model_output_session_{}"

# check for GPU and set device
print("Cuda available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train incremental models
train_bert_incremental(modelconfig, trainingconfig, path_to_data, output_path,
    tokenizer_path, 2, input_fnames, output_fnames)
