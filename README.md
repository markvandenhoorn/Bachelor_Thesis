# Bachelor_Thesis
This project studies the role of the type of child-directed input in computationally modelling determiner use.

## Getting started
### Installing dependencies
This codebase was entire written in Python 3.10.8. Requirements.txt contains all the necessary packages to run the code succesfully. These can be easily installed using pip by using this instruction:
```
pip install -r requirements.txt
```

Or using conda:

```
conda install --file requirements.txt
```
### Preparing the data
First, make sure you have the ldp data downloaded and stored in the 'data' folder (you might need to create this folder in the root directory). Make sure to name the file 'ldp_data.csv'. You will also need the first names dataset, so download that (see link at the bottom of this README) and store it as 'first_names.csv' in the data folder as well. 
Then, navigate to the 'code' folder in your terminal. From there, run: 
```
python main_preprocessing.py
```
This will create all the necessary data, for each type of model (restrained, unrestrained and the 3 types in between) as well as the tokenizer dataset. These files are stored in the data folder. Optionally, you can add an argument to filter out sentences where two words want conflicting determiners. E.g. "the blue one", where blue is assigned 'the' and one is assigned 'a'. By default the second word gets changed into a new noun that has the same determiner type as the first. By running this:
```
python main_preprocessing.py --remove_conflicts
```
These conflicting sentences get filtered out of the training data. This is removal of conflicts is what was done in the study this code was created for. 

### Training the tokenizer
After creating the data, train a tokenizer using:
```
python train_tokenizer.py
```
This tokenizer will be saved in the custom_tokenizer folder.

### Training incremental models
Now you can start training incremental models. You need to specify what type of models you want to train when running the code. The 5 options are:
```
python train_bert_model.py 0
```
```
python train_bert_model.py 25
```
```
python train_bert_model.py 50
```
```
python train_bert_model.py 75
```
```
python train_bert_model.py unfiltered
```
The resulting models are stored in the models folder.

### Exploratory functions
The files exploratory.py, checking.py and overall_performance.py are for exploratory data analysis. Exploratory.py checks how many unique nouns are seen with a determiner in the data, and divides this into groups of 'seen with a', 'seen with the' and 'seen with both'. Checking.py is for finding how many nouns in the training data appear with 'a', 'the' or both. The results from these files are only printed to the console. 

Overall_performance tests the fully trained unrestricted model on sentences with DET + NOUN or DET + x + NOUN where x is any word that is not a NOUN. It outputs a confusion matrix and also a graph of productive determiner use. The first time you run this, run it without arguments to create a sub-testset. Then you run it by providing a prefix of the name of the model you want to test, like these examples:
```
python overall_performance --model_type filtered_0_model
```
```
python overall_performance --model_type unfiltered_model
```
The results are stored in the output folder.

### Creating test sets
To create test sets, you run create_test.py. By default it creates a test set containing all relevant sentences. If you want test sets where each noun is seen only once in each test set (like in the study), you run it like this:
```
python create_test.py --unique_nouns
```
The sets are stored in the data folder.

### Testing the models
With the test sets created, you can run:
```
python test_models.py
```
This outputs plots with misclassification scores and accuracy scores for all model types, per age range.
The results are stored in the output folder.
