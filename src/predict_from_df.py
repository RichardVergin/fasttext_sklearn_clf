import fasttext
import pandas as pd
from sklearn.metrics import accuracy_score
from utils.fasttext_sklearn import FastTextSklearnEstimator


def main():
    # load data
    X_test = pd.read_csv('../data/X_test.csv')
    y_test = pd.read_csv('../data/y_test.csv')

    # load fasttext model and initialize wrapper
    fasttext_model = fasttext.load_model('../models/fasttext_from_df.bin')
    model = FastTextSklearnEstimator(
        model=fasttext_model,
        k=3
    )

    # predict
    y_hat = model.predict(X_test['text'])
    acc = accuracy_score(
        y_true=y_test,
        y_pred=y_hat
    )
    print(f'achieved an awesome acc of: {acc}')

    probas = model.predict_proba(X_test['text'])

    return y_hat, probas


if __name__ == '__main__':
    main()
