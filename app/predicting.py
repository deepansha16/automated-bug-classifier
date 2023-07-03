from joblib import load
from sklearn import preprocessing
import pandas as pd
from app.issue_mining import IssueMiner
from nltk.corpus import stopwords
import nltk
from app.preprocessing import preprocess
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



class Predictor:
    def __init__(
        self, id: str, vectorizer_source: str, feature_source: str, recent: bool
    ):
        self.id = id
        self.vectorizer = load(vectorizer_source)
        self.feature_vectors = pd.read_pickle(feature_source, compression="gzip")
        if recent:
            self.classifier = load("model/vscode_recent_issues_rfc.joblib")
        else:
            self.classifier = load("model/vscode_issues_rfc.joblib")
        self.contributors_info = pd.read_json(
            "data/vscode_contributors_response.json", lines=True
        )[["login", "name", "contributions"]]
        self.contributors_info["name"] = self.contributors_info["name"].apply(
            lambda x: "".join(str(x).lower().split() if x else x)
        )

    def _get_issue_vector(self):
        nltk.download("stopwords", quiet=True)
        issue_miner = IssueMiner()
        issue_dict = issue_miner.get_an_issue_by_id(self.id)
        stop = stopwords.words("english")
        title = preprocess(issue_dict["title"], stop, True)
        body = preprocess(issue_dict["body"], stop, True)
        issue_content = title + " " + body
        vector = self.vectorizer.transform([issue_content])
        feature_vector = pd.DataFrame.sparse.from_spmatrix(
            vector, columns=self.vectorizer.get_feature_names_out()
        )
        feature_vector = feature_vector.drop(["id"], axis=1)
        if issue_dict["assignees"] is not None and len(issue_dict["assignees"]) > 0:
            assignee = [assignee["login"] for assignee in issue_dict["assignees"]]
        elif issue_dict["assignee"] is not None:
            assignee = issue_dict["assignee"]["login"]
        else:
            assignee = None
        return (feature_vector, assignee)

    def predict(self):
        le = preprocessing.LabelEncoder()
        le.fit(self.feature_vectors["assignee"].unique())
        X, y_true = self._get_issue_vector()
        y_pred_proba = self.classifier.predict_proba(X)
        # store results in a dataframe
        data = list(
            zip(y_pred_proba[0], le.inverse_transform(self.classifier.classes_))
        )
        df = pd.DataFrame(data, columns=["Probability", "Assignee"]).sort_values(
            by="Probability", ascending=False
        )
        df["Probability"] = df["Probability"].apply(lambda x: "{:.2f}%".format(100 * x))
        df["Authored Commits"] = df.apply(
            lambda x: Predictor.get_num_of_authored_commits(x, self.contributors_info),
            axis=1,
        ).astype("int32")
        print(f"Ground truth (if exists): {y_true}")
        print(f"Predict results:")
        print(df.to_string(index=False))

    @staticmethod
    def get_num_of_authored_commits(row, contributors_info):
        name = "".join(row["Assignee"].lower().split())
        df_login = contributors_info[contributors_info["login"] == row["Assignee"]]
        df_name = contributors_info[contributors_info["name"] == name]
        if df_login.empty and df_name.empty:
            return 0
        elif df_login.empty:
            return df_name.iloc[0]["contributions"]
        else:
            return df_login.iloc[0]["contributions"]


def main(args):
    predictor = Predictor(
        args.id, args.vectorizer_source, args.feature_source, args.recent
    )
    predictor.predict()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--recent", action="store_true", dest="recent", default=False
    )
    parser.add_argument(
        "-v",
        "--vectorizer-source",
        dest="vectorizer_source",
        help="Path of the vectorizer.",
        default="data/vectorizer.joblib",
    )
    parser.add_argument(
        "-f",
        "--feature-source",
        dest="feature_source",
        help="Path of the feature vectors.",
        default="data/feature_vectors.gzip",
    )
    parser.add_argument("-id", "--commit-id", dest="id", default=None)
    main(parser.parse_args(["-r", "-id", "164915"]))
