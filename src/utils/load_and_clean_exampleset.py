import pandas as pd
import string
from sklearn.datasets import fetch_20newsgroups


def cut_to_article(df):
    # keep everything after writes
    df['text'] = df['text'].str.split('writes:').str[-1]

    # keep everything before --
    df['text'] = df['text'].str.split('--').str[0]

    return df


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


def quick_cleaning(df):
    df['text'] = df['text'].replace('\n', '', regex=True)
    df['text'] = df['text'].apply(remove_punctuations)
    return df


def main():
    # load example dataset from sklearn
    categories = [
        'rec.sport.baseball',
        'sci.space',
        'talk.politics.misc'
    ]

    # load and clean trainingsset
    twenty_train = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=42
    )
    X_train = pd.DataFrame(
        data=twenty_train.data,
        columns=['text']
    )
    X_train = cut_to_article(X_train)
    X_train = quick_cleaning(X_train)

    y_train = pd.Series(
        data=twenty_train.target,
        index=X_train.index
    )

    # load and clean testset
    twenty_test = fetch_20newsgroups(
        subset='test',
        categories=categories,
        shuffle=True,
        random_state=42
    )
    X_test = pd.DataFrame(
        data=twenty_test.data,
        columns=['text']
    )
    X_test = cut_to_article(X_test)
    X_test = quick_cleaning(X_test)

    y_test = pd.Series(
        data=twenty_test.target,
        index=X_test.index
    )

    # save
    X_train.to_csv('../data/X_train.csv', index=False)
    X_test.to_csv('../data/X_test.csv', index=False)
    y_train.to_csv('../data/y_train.csv', index=False)
    y_test.to_csv('../data/y_test.csv', index=False)


if __name__ == '__main__':
    main()
