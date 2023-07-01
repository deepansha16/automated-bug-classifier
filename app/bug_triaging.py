import sys
import argparse


def list_get(li, index, fallback=None):
    try:
        return li[index]
    except IndexError:
        return fallback


def preprocessing(args):
    from app import preprocessing

    preprocessing.main(args)


def issue_mining(args):
    from app import issue_mining

    issue_mining.main()


def contributor_mining(args):
    from app import contributor_mining

    contributor_mining.main()


def issue_filtering(args):
    from app import issue_filtering

    issue_filtering.main()


def training(args):
    from app import training

    training.main(args)


def predicting(args):
    from app import predicting

    predicting.main(args)


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()


# Subparser: preprocessing
p_preprocessing = subparsers.add_parser("preprocessing")
p_preprocessing.add_argument("-s", "--source", dest="source", default=None)
p_preprocessing.add_argument(
    "-L", "--with-lemmatizer", action="store_true", dest="do_lemma", default=False
)
p_preprocessing.add_argument(
    "-T", "--training", action="store_true", dest="is_training", default=False
)
p_preprocessing.set_defaults(func=preprocessing)

# Subparser: issue filtering
p_issue_filtering = subparsers.add_parser("issue_filtering")
p_issue_filtering.set_defaults(func=issue_filtering)

# Subparser: issue_mining
p_issue_mining = subparsers.add_parser("issue_mining")
p_issue_mining.set_defaults(func=issue_mining)

# Subparser: contributor_mining
p_contributor_mining = subparsers.add_parser("contributor_mining")
p_contributor_mining.set_defaults(func=contributor_mining)

# Subparser: training
p_training = subparsers.add_parser("training")
p_training.add_argument(
    "-a",
    "--training-source",
    dest="training_source",
    help="Path of the training set.",
    default="data/preprocessed_issues_for_training.csv",
)
p_training.add_argument(
    "-e",
    "--testing-source",
    dest="testing_source",
    help="Path of the test set.",
    default="data/preprocessed_issues_for_testing.csv",
)
p_training.add_argument(
    "-f",
    "--feature-source",
    dest="feature_source",
    help="Path of the feature vectors.",
    default="data/feature_vectors.gzip",
)
p_training.add_argument(
    "-r", "--recent", action="store_true", dest="recent", default=False
)
p_training.add_argument(
    "-of",
    "--output-feature",
    action="store_true",
    dest="output_feature",
    default=False,
)
p_training.add_argument(
    "-ov",
    "--output-vectorizer",
    action="store_true",
    dest="output_vectorizer",
    default=False,
)
p_training.add_argument(
    "-om", "--output-model", action="store_true", dest="output_model", default=False
)
p_training.add_argument(
    "-nf",
    "--no-feature-source",
    action="store_true",
    dest="no_feature_source",
    default=False,
)
p_training.add_argument(
    "-bg",
    "--bigram",
    action="store_true",
    dest="bigram",
    default=False,
)
p_training.set_defaults(func=training)

# Subparser: predicting
p_predicting = subparsers.add_parser("predicting")
p_predicting.add_argument(
    "-r", "--recent", action="store_true", dest="recent", default=False
)
p_predicting.add_argument(
    "-v",
    "--vectorizer-source",
    dest="vectorizer_source",
    help="Path of the vectorizer.",
    default="data/vectorizer.joblib",
)
p_predicting.add_argument(
    "-f",
    "--feature-source",
    dest="feature_source",
    help="Path of the feature vectors.",
    default="data/feature_vectors.gzip",
)
p_predicting.add_argument("-id", "--commit-id", dest="id", default=None)
p_predicting.set_defaults(func=predicting)


def main(argv):
    import app.utils.doc_utils as docs

    helpstrings = {"-h", "--help"}
    command = list_get(argv, 0, "").lower()

    # No command has been entered by the user or unrecognizable command.
    if command not in docs.MODULE_DOCSTRINGS:
        print(docs.DOCSTRING)
        if command == "":
            print("You are viewing the help text as no command has been entered.")
        elif command not in helpstrings:
            print(
                'You are viewing the help text because "%s" was not recognized'
                % command
            )
        return 1

    # The user entered a command, but no further arguments, or just help.
    argument = list_get(argv, 1, "").lower()
    if (
        argument in helpstrings
        and command not in "issue_mining"
        and command not in "contributor_mining"
        and command not in "issue_filtering"
    ):
        print(docs.MODULE_DOCSTRINGS[command])
        return 1

    args = parser.parse_args(argv)
    args.func(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
