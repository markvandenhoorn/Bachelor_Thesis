# Bachelor_Thesis
This project studies the role of the type of child-directed input in computationally modelling determiner use.

## Getting started
### Preparing the data
First, make sure you have the ldp data downloaded and stored in the 'data' folder. Make sure to name the file 'ldp_data.csv'. You will also need the first names dataset, so download that (see link at the bottom of this README) and store it as 'first_names.csv' in the data folder as well. 
Then, navigate to the 'code' folder in your terminal. From there, run: 
```
python main_preprocessing.py
```
This will create all the necessary data, for each type of model (restrained, unrestrained and the 3 types in between) as well as the tokenizer dataset. These files are stored in the data folder.

### Training the tokenizer
After creating the data, train a tokenizer using:
```
python train_tokenizer.py
```
This tokenizer will be saved in the custom_tokenizer folder.

### Training incremental models
Now you can start training incremental models. You need to specify what type of models you want to train when running the code. The 5 options are:
```
python train_bert_model.py filtered
```
```
python train_bert_model.py unfiltered
```
The resulting models are stored in the models folder.
