import csv

import pandas as pd


def prepare_trainingsset(X, y, col_text):
    # add label to text and keep prepared text only
    X['label'] = y

    X[col_text + '_prepared'] = ''
    for i in range(X['label'].nunique()):
        X.loc[
            X.label == i, col_text + '_prepared'
        ] = '__label__' + str(i) + ' ' + X[col_text]

    return X[col_text + '_prepared']


def store_txt_for_fasttext(series, output_path):
    series.to_csv(
        output_path,
        index=False,
        sep=' ',
        header=None,
        quoting=csv.QUOTE_NONE,
        quotechar='',
        escapechar=' '
    )


def main():
    # load data
    X_train = pd.read_csv('../../data/X_train.csv')
    y_train = pd.read_csv('../../data/y_train.csv')
    X_test = pd.read_csv('../../data/X_test.csv')

    X_train = prepare_trainingsset(
        X=X_train,
        y=y_train,
        col_text='text'
    )

    store_txt_for_fasttext(
        series=X_train,
        output_path='../../data/train.txt'
    )
    store_txt_for_fasttext(
        series=X_test,
        output_path='../../data/test.txt'
    )

    return X_train


if __name__ == '__main__':
    main()
