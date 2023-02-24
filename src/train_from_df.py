import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from utils.fasttext_sklearn import FastTextSklearnEstimator
from utils.prepare import prepare_trainingsset


distributions = {
    'epoch': [5, 10],
    'lr': [0.01, 0.1],
    'lrUpdateRate': [50, 100],
    'wordNgrams': [1, 3, 5, 7],
    'dim': [50, 100],
    'ws': [3, 5, 7],
    'loss': ['ns', 'hs', 'softmax']
}
n_iter = 5
scorer = make_scorer(f1_score, average='micro')
cv = 10


def main(distributions, n_iter, scorer, cv):
    # load data
    X_train = pd.read_csv('../data/X_train.csv')
    y_train = pd.read_csv('../data/y_train.csv')

    # prepare trainingsset
    X_train = prepare_trainingsset(
        X=X_train,
        y=y_train,
        col_text='text'
    )

    # initialize model and perform random search
    model = FastTextSklearnEstimator(
        k=3
    )

    clf = RandomizedSearchCV(
        estimator=model,
        param_distributions=distributions,
        n_iter=n_iter,
        scoring=scorer,
        random_state=None,
        n_jobs=-1,
        refit=False,
        cv=cv
    )
    
    search = clf.fit(
        X=X_train,
        y=y_train
    )

    # refit (must be done manually to insert params into fasttext model within wrapper since random search does not
    # trigger self.model)
    model = FastTextSklearnEstimator(
        k=3
    )
    model.set_params(**search.best_params_)
    model.fit(
        X=X_train,
        txtflag=False
    )

    # save
    model.save_model(
        filename='../models/fasttext_from_df.bin'
    )


if __name__ == '__main__':
    main(
        distributions=distributions,
        n_iter=n_iter,
        scorer=scorer,
        cv=cv
    )
