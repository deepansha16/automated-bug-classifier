import logging
import sys

import numpy
import pandas as pd
from sklearn import tree, naive_bayes, svm, neural_network, ensemble, preprocessing
from sklearn.base import ClassifierMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, train_test_split
# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingGridSearchCV

scores = ["f1"]


def hyperparameter_grid_search_halving(model: ClassifierMixin,
                                       X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                                       tuned_parameters,
                                       scores: list,
                                       verbose: bool = False):
    # Split the dataset in two equal parts
    # X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.5, random_state=0)

    string = ""

    def join_newline(to_join):
        nonlocal string
        string = "\n".join([string, to_join])

    for score in scores:
        clf = HalvingGridSearchCV(estimator=model, param_grid=tuned_parameters, scoring="%s_macro" % score, n_jobs=1)

        y_true, y_pred = y_test, clf.predict(X_test)

        join_newline(f"Best parameters set found on development set:\n")
        # Tuning hyper-parameters for {score}
        join_newline(f"{clf.best_params_}\n")
        join_newline("Grid scores on development set:\n")

        if verbose:
            means = clf.cv_results_["mean_test_score"]
            stds = clf.cv_results_["std_test_score"]
            for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
                join_newline("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        join_newline("Detailed classification report:\n")
        join_newline("The model is trained on the full development set.")
        join_newline("The scores are computed on the full evaluation set.\n")
        join_newline(f"{classification_report(y_true, y_pred, labels=numpy.unique(y_true))}")

    return string


def train_naive_bayes(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    param_grid = [
        {"var_smoothing": [1e-9, 1e-5, 1e-2, 1e-1]}
    ]

    return hyperparameter_grid_search_halving(naive_bayes.GaussianNB(), X.sparse.to_dense(), y, X_test.sparse.to_dense(), y_test, param_grid, scores)


def train_multilayer_perceptron(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    # https://scikit-learn.org/stable/modules/preprocessing.html
    X = preprocessing.StandardScaler(with_mean=False).fit_transform(X)
    X_test = preprocessing.StandardScaler(with_mean=False).fit_transform(X_test)

    param_grid = [
        {"solver": ["lbfgs", "adam"], "activation": ["identity", "logistic", "tanh", "relu"], "max_iter": [10000]},
        {"solver": ["sgd"], "learning_rate": ["constant", "invscaling", "adaptive"], "max_iter": [10000]}
    ]
    return hyperparameter_grid_search_halving(neural_network.MLPClassifier(),
                                              X, y, X_test, y_test, param_grid, scores)


def train_random_forest(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    param_grid = {"criterion": ["gini", "entropy"],
                  "min_samples_split": [5, 10, 15, 20, 30], "min_samples_leaf": [2, 4, 8, 16, 32],
                  "max_depth": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

    return hyperparameter_grid_search_halving(ensemble.RandomForestClassifier(),
                                              X, y, X_test, y_test, param_grid, scores)


def start_tuning(training_source: str, feature_source: str):
    # Get issues for training from CSV
    df_train = pd.read_csv(training_source)
    feature_vectors = pd.read_pickle(feature_source, compression="gzip")

    X = feature_vectors.iloc[
        : len(df_train[df_train["id"] >= 130000])
        ]

    y = X["assignee"]

    X_test = feature_vectors.drop("assignee", axis=1).iloc[len(df_train):]

    y_test = feature_vectors["assignee"].iloc[len(df_train):]

    # jobs = [train_naive_bayes, train_multilayer_perceptron, train_svm, train_decision_tree, train_random_forest]
    jobs = [train_naive_bayes]

    for job in jobs:
        logging.info("-------------------------------------")
        logging.info(f"starting {job.__name__}")
        logging.info(job(X.drop("assignee", axis=1), y, X_test, y_test))


def main(args):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%d/%m/%Y %H:%M:%S %z",
        filename="hyperparameter_tuning.log",
        encoding="utf-8",
        filemode="w",
        level=logging.DEBUG,
    )
    start_tuning(args.training_source, args.feature_source)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--training-source",
        dest="training_source",
        help="Path of the training set.",
        default="data/preprocessed_issues_for_training.csv",
    )
    parser.add_argument(
        "-f",
        "--feature-source",
        dest="feature_source",
        help="Path of the feature vectors.",
        default="data/feature_vectors.gzip",
    )

    main(parser.parse_args())
