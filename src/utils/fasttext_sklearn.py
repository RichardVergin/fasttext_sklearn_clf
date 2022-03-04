import fasttext
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import tempfile


class FastTextSklearnEstimator(BaseEstimator):
    def __init__(
        self,
        model=None,
        k=3,
        lr=.01,
        lrUpdateRate=100,
        dim=100,
        wordNgrams=1,
        minCount=1,
        minCountLabel=0,
        ws=5,
        epoch=5,
        neg=5,
        loss='softmax',
        thread=16,
        verbose=0
    ):
        super().__init__()
        self.k = k
        self.lr = lr
        self.lrUpdateRate = lrUpdateRate
        self.dim = dim
        self.wordNgrams = wordNgrams
        self.minCount = minCount
        self.minCountLabel = minCountLabel
        self.ws = ws
        self.epoch = epoch
        self.neg = neg
        self.loss = loss
        self.thread = thread
        self.verbose = verbose
        self.model = model

    def get_params(self, deep=True):
        """
        Get Hyper parameters of FastText Model
        """
        return {
            'lr': self.lr,
            'lrUpdateRate': self.lrUpdateRate,
            'dim': self.dim,
            'wordNgrams': self.wordNgrams,
            'minCount': self.minCount,
            'minCountLabel': self.minCountLabel,
            'ws': self.ws,
            'epoch': self.epoch,
            'neg': self.neg,
            'loss': self.loss,
            'thread': self.thread
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def set_input_stream(self, X):
        # Create temp file
        temp_train_text = tempfile.NamedTemporaryFile(delete=False)

        # Write into temp file
        with open(temp_train_text.name, 'w') as train:
            # for _, line in X.items():
            for line in X:
                if not line.endswith('\n'):
                    line = line + '\n'
                train.write(line)

        # Assign input stream
        self.input_stream = temp_train_text

    def fit(self, X, y=None, txtflag=False):
        try:
            if txtflag is False:
                # create tempfile from input dataset
                self.set_input_stream(X)

                # fit model
                self.model = fasttext.train_supervised(
                    input=self.input_stream.name,
                    **self.get_params(deep=True)
                )

                # Close temp file
                self.input_stream.close()

            elif txtflag:
                # fit model on file
                self.model = fasttext.train_supervised(
                    X,
                    **self.get_params(deep=True)
                )
        except TypeError:
            print("Error in input dataset.. please see if the file/list of sentences is of correct format")
        return self

    def convert_fasttext_tuples(self, prediction_fasttext):
        # assign probabilities correctly in ascending order (label 0 to n)
        probabilities = np.array([])

        # extract for each class (0 to n)
        for k in range(self.k):
            for label in list(prediction_fasttext[0]):
                if str(k) in label:
                    # get index of tuples containing the labels
                    index_k = list(prediction_fasttext[0]).index(label)
                    prob_k = list(prediction_fasttext[1])[index_k]
                    probabilities = np.append(probabilities, np.array(prob_k))

        return probabilities

    def predict_proba(self, X):
        if isinstance(X, str):
            # predict direclty
            prediction_fasttext = self.model.predict(X, k=self.k)

            # convert fasttext tuples to probabilites for sklearn in according order
            probabilities = self.convert_fasttext_tuples(prediction_fasttext)

            return probabilities

        elif isinstance(X, pd.Series):
            # iterate over each element, compute probabilities and add to all probabilites
            probabilities_series = np.empty([0, self.k])

            for i, x in X.iteritems():
                # predict
                prediction_fasttext = self.model.predict(x, k=self.k)

                # convert fasttext tuples to probabilites for sklearn in according order
                probabilites = self.convert_fasttext_tuples(prediction_fasttext)

                # add to all probabilites
                probabilities_series = np.vstack([probabilities_series, probabilites])

            return probabilities_series

        else:
            raise TypeError('feed model with string or pd.Series')

    def predict(self, X):
        if isinstance(X, str):
            # predict direclty
            # get probabilites
            probabilites = self.predict_proba(X)

            # return class of highest probability based on index
            prediction = np.argmax(probabilites, axis=0)

            return prediction

        elif isinstance(X, pd.Series):
            # iterate over each element, compute prediction and add to predictions
            predictions = np.array([])

            # get probabilites
            probabilities_series = self.predict_proba(X)

            # return class of highest probability based on index
            for probabilites in probabilities_series:
                prediction = np.argmax(probabilites, axis=0)
                # add to all predictions
                predictions = np.append(predictions, prediction)

            return predictions

        else:
            raise TypeError('feed model with string or pd.Series')

    def predict_from_file(self, X):
        try:
            # open file and predict
            texts = open(X, "r").readlines()
            sentences = [x[:-1] for x in texts]
            return self.model.predict(sentences, k=self.k)
        except TypeError:
            print("Error in input dataset.. please see if the file/list of sentences is of correct format")

    def labels(self):
        return self.model.labels

    def save_model(self, filename):
        return self.model.save_model(filename)
