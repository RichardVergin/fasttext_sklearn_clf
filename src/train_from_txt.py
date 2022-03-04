from utils.fasttext_sklearn import FastTextSklearnEstimator


def main(filepath):
    # initialize model
    model = FastTextSklearnEstimator(
        k=3
    )
    
    # fit on txt
    model.fit(
        X=filepath,
        txtflag=True
    )

    # save
    model.save_model(
        filename='../models/fasttext_from_txt.bin'
    )


if __name__ == '__main__':
    main(
        filepath='../data/train.txt'
    )
