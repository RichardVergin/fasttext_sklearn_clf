import fasttext
import pandas as pd
from sklearn.metrics import accuracy_score
from utils.fasttext_sklearn import FastTextSklearnEstimator


def main(filepath):
    # load y_true
    y_test = pd.read_csv('../data/y_test.csv')

    # load fasttext model and initialize wrapper
    fasttext_model = fasttext.load_model('../models/fasttext_from_txt.bin')
    model = FastTextSklearnEstimator(
        model=fasttext_model,
        k=3
    )

    # predict
    y_hat = model.predict_from_file(
        X=filepath
    )
    return y_hat


if __name__ == '__main__':
    main(
        filepath='../data/test.txt'
    )
