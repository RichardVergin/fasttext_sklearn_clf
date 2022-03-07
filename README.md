# Wrap fasttext model to use as a sklearn classifier
Use fasttext as a sklearn estimator. This will e.g. allow using sklearn randomized / grid search or adding
fasttext clf to sklearn pipeline.

# Content
## src
- train_from_df.py: illustrates how to use wrapper to train and save clf. Includes hyperparameter tuning with random
search.
- train_from_txt.py: illustrates how to train model from txt file.
- predict_drom_df.py: illustrates how to load and wrap trained model to predict from dataframe and use class methods.
- predict_from_txt.py: illustrated how to predict from txt file.

## src/utils
- fasttext_sklearn.py: contains FastTextSklearnEstimator that does the magic.
- load_and_clean_exampleset.py: load sklearn dataset "fetch_20newsgroups" and do minimum preparation. Main function
also stores prepared trainings- and testset into data/...
- prepare.py: functions to prepare label for fasttext clf (format __label__n for n in num_labels) and to store txt.file
in format to train and predict with fasttext.

# How to install and run project
- packages & versions listed in requirements.txt
- to simply run code and classifiy sklearn dataset:
    - run src/utils/load_and_clean_exampleset.py
    - run train_from_df.py
    - run predict_from_df.py
- to use your own data: adapt paths in train and predict files


