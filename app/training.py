import logging

import numpy as np
import pandas as pd
from os import path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn import preprocessing
from joblib import dump


class ModelTrainer:
    BIGRAM_THRESHOLD = 0.1

    def __init__(
        self,
        training_source: str,
        testing_source: str,
        feature_source: str,
        recent: bool,
        output_feature: bool,
        output_vectorizer: bool,
        output_model: bool,
        no_feature_source: bool,
        bigram: bool,
    ):
        self.df_train = pd.read_csv(training_source)
        self.df_test = pd.read_csv(testing_source)
        self.recent = recent
        self.output_feature = output_feature
        self.output_vectorizer = output_vectorizer
        self.output_model = output_model
        self.no_feature_source = no_feature_source
        self.bigram = bigram
        if not self.no_feature_source and path.exists(feature_source):
            self.feature_vectors = pd.read_pickle(feature_source, compression="gzip")
        else:
            self.feature_vectors = None

    def run(self):
        if self.feature_vectors is None:
            self._get_feature_vectors()
        if self.output_feature:
            self.feature_vectors.to_pickle(
                "data/feature_vectors.gzip",
                compression="gzip",
            )
        self._train_and_test()

    def _train_and_test(self):
        # encode assignees
        le = preprocessing.LabelEncoder()
        le.fit(self.feature_vectors["assignee"].unique())

        if self.recent:
            X = self.feature_vectors.iloc[
                : len(self.df_train[self.df_train["id"] >= 130000])
            ]
        else:
            X = self.feature_vectors.iloc[: len(self.df_train)]

        y = le.transform(X["assignee"])
        X = X.drop(["assignee", "id"], axis=1)

        classifier = RandomForestClassifier(random_state=0)
        classifier.fit(X, y)
        test_vectors = self.feature_vectors.iloc[len(self.df_train) :]
        X_test = test_vectors.drop(["assignee", "id"], axis=1)
        y_true = le.transform(test_vectors["assignee"].to_numpy())
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)

        # compute and store metrics and scores
        logging.info(
            "\n"
            + classification_report(
                y_true,
                y_pred,
                target_names=list(le.inverse_transform(np.unique(y_true))),
                labels=np.unique(y_true),
                zero_division=0,
            )
        )
        accuracy = round(100 * accuracy_score(y_true, y_pred), 2)
        logging.info(f"\nAccuracy: {accuracy}%")

        # ignore predictions of assignees that appear in test set but not in training set
        # because y_true contains labels not in y_pred_proba
        indices_to_delete = []
        for label in y_true:
            if label not in classifier.classes_:
                indices_to_delete += list(np.where(y_true == label)[0])
        auc_score = roc_auc_score(
            np.delete(y_true, indices_to_delete),
            np.delete(y_pred_proba, indices_to_delete, axis=0),
            multi_class="ovo",
            labels=classifier.classes_,
        )
        logging.info(f"\nROC AUC: {auc_score}")

        if self.output_model:
            if self.recent:
                dump(classifier, "data/vscode_recent_issues_rfc.joblib")
            else:
                dump(classifier, "data/vscode_issues_rfc.joblib")

    def _get_feature_vectors(self):
        df = pd.concat([self.df_train, self.df_test])

        # remove na first
        df["body"] = df["body"].fillna("").astype(str)
        df["title"] = df["title"].fillna("").astype(str)
        bodies = df["body"] + " " + df["title"]

        if self.bigram:
            # build vectorizer with bigrams
            vectorizer = TfidfVectorizer(ngram_range=(2, 2), use_idf=True)
        else:
            vectorizer = TfidfVectorizer(use_idf=True)
        vectors = vectorizer.fit_transform(bodies)
        self.feature_vectors = pd.DataFrame.sparse.from_spmatrix(
            vectors, columns=vectorizer.get_feature_names_out()
        )

        if self.output_vectorizer:
            dump(vectorizer, "data/vectorizer.joblib")

        if self.bigram:
            # filter out bad bigrams
            original_len_columns = len(self.feature_vectors.columns)
            self._filter_out_uncommon_bigrams(self.BIGRAM_THRESHOLD)
            filtered_len_columns = len(self.feature_vectors.columns)
            reduced_percent = (
                100
                * (original_len_columns - filtered_len_columns)
                / original_len_columns
            )
            logging.info("Columns are reduced by {:.2f}%".format(reduced_percent))
            logging.info(f"Now {filtered_len_columns} columns")

        # add assignee column and id column
        self.feature_vectors = self.feature_vectors.assign(
            assignee=df["assignee"].to_list(), id=df["id"].to_list()
        )

    def _filter_out_uncommon_bigrams(self, drop_threshold):
        # Filter out uncommon bigrams by counting the number of times they appear in the dataset (tf-idf is different than
        # zero), and then dropping the ones that appear in less than 10% of the issues
        to_drop = self.feature_vectors.apply(lambda x: x > 0).sum()
        logging.info("quantile threshold: " + str(to_drop.quantile(drop_threshold)))
        to_drop = to_drop[to_drop <= to_drop.quantile(drop_threshold)]
        logging.info("dropped # bigram columns: " + str(len(to_drop)))
        self.feature_vectors = self.feature_vectors.drop(to_drop.index, axis=1)


def main(args):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%d/%m/%Y %H:%M:%S %z",
        filename="training.log",
        encoding="utf-8",
        filemode="w",
        level=logging.DEBUG,
    )
    model_trainer = ModelTrainer(
        args.training_source,
        args.testing_source,
        args.feature_source,
        args.recent,
        args.output_feature,
        args.output_vectorizer,
        args.output_model,
        args.no_feature_source,
        args.bigram,
    )
    model_trainer.run()


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
        "-e",
        "--testing-source",
        dest="testing_source",
        help="Path of the test set.",
        default="data/preprocessed_issues_for_testing.csv",
    )
    parser.add_argument(
        "-f",
        "--feature-source",
        dest="feature_source",
        help="Path of the feature vectors.",
        default="data/feature_vectors.gzip",
    )
    parser.add_argument(
        "-r", "--recent", action="store_true", dest="recent", default=False
    )
    parser.add_argument(
        "-of",
        "--output-feature",
        action="store_true",
        dest="output_feature",
        default=False,
    )
    parser.add_argument(
        "-ov",
        "--output-vectorizer",
        action="store_true",
        dest="output_vectorizer",
        default=False,
    )
    parser.add_argument(
        "-om", "--output-model", action="store_true", dest="output_model", default=False
    )
    parser.add_argument(
        "-nf",
        "--no-feature-source",
        action="store_true",
        dest="no_feature_source",
        default=False,
    )
    parser.add_argument(
        "-bg",
        "--bigram",
        action="store_true",
        dest="bigram",
        default=False,
    )
    # Remove -r to train on all data
    main(parser.parse_args(["-r", "-nf", "-ov", "-of", "-om"]))
