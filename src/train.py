import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from utils.fasttext_sklearn import FastTextSklearnEstimator
from utils.prepare import prepare_trainingsset


def main(distributions, n_iter, scoring, cv):
    # load data
    X_train = pd.read_csv('../../data/X_train.csv')
    y_train = pd.read_csv('../../data/y_train.csv')

    X_test = pd.read_csv('../../data/X_test.csv')
    y_test = pd.read_csv('../../data/y_test.csv')

    # prepare trainingsset
    X_train = prepare_trainingsset(
        X=X_train,
        y=y_train,
        col_text='text'
    )

    # initialize model and perform random search
    model = FastTextSklearnEstimator()

    clf = RandomizedSearchCV(
        estimator=model,
        param_distributions=distributions,
        n_iter=n_iter,
        scoring=scoring,
        random_state=None,
        n_jobs=-1,
        refit=False,
        cv=cv
    )
    
    search = clf.fit(
        X=X_train,
        y=y_train
    )

if __name__ == '__main__':
    main()
