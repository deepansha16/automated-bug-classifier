DOCSTRING = """
Project: Automated Bug Triaging

Below you will find the list of commands useful for the operation of the tool:
{predicting}
{training}
{preprocessing}
{issue_mining}
{contributor_mining}
{issue_filtering}


For details on every command, use the following:
> python3 -m app.bug_triaging <command> -h
"""

MODULE_DOCSTRINGS = {
    "issue_mining": """
issue_mining:
    Mines all the issues from the VS Code Github repository using Github REST API.
    Usage:
        $ python3 -m app.bug_triaging issue_mining
""",
    "contributor_mining": """
contributor_mining:
    Mines the list of contributors from the VS Code repository and sorts them
     by the number of commits per contributor in descending order.
    Usage:
        $ python3 -m app.bug_triaging contributor_mining
""",
    "issue_filtering": """
issue_filtering:
    Filters out unwanted records based on issue id, number of assignees, language and labels and
        get contributors who have been assigned more than once
    Usage:
        $ python3 -m app.bug_triaging issue_filtering
""",
    "preprocessing": """
preprocessing:
    Preprocess the training files and iterate through all the methods to get the final json.
    Usage:
        $ python3 -m app.bug_triaging preprocessing -s <path-to-issues-for-training>
    Flags:
    -s <path-to-issues-for-training> | --source <path-to-issues-for-training>:
        The path to the files
    -L | --with-lemmatizer:
        Flag to enable lemmatization
    -T | --training:
        Flag to enable filtering with thresholds for the training set
""",
    "training": """
training:
    Extract Github issue feature vectors and train Random Forest model.
    Usage:
        $ python3 -m app.bug_triaging training
        $ python3 -m app.bug_triaging training -a <path-to-training-set> -e <path-to-test-set> -f <path-to-feature-vectors> -r -of -om
    Flags:
    -a <path-to-training-set> | --training-source <path-to-training-set>:
        The path to the training set file, default is data/preprocessed_issues_for_training.csv.
    -e <path-to-test-set> | --testing-source <path-to-test-set>:
        The path to the test set file, default is data/preprocessed_issues_for_testing.csv.
    -f <path-to-feature-vectors> | --source <path-to-feature-vectors>:
        The path to the feature vector file, default is data/feature_vectors.csv.
    -r | --recent:
        Flag to enable training with recent issues (130000 <= id <= 150000).
    -of | --output-feature:
        Flag to enable output of feature vectors.
    -ov | --output-vectorizer:
        Flag to enable output of the vectorizer.
    -om | --output-model:
        Flag to enable output of Random Forest model.
    -nf | --no-feature-source:
        Flag to disable using of the feature vector file.
    -bg | --bigram:
        Flag to enable bigram.
""",
    "predicting": """
predicting:
    Take as input the id of an open vscode's issue on GitHub and provide as output a 
    ranked list of candidate assignees.
    Usage:
        $ python3 -m app.bug_triaging predicting -r -id 186955
    Flags:
    -r | --recent:
        Flag to enable predicting with the model trained with recent issues (130000 <= id <= 150000).
    -id | --commit-id:
        The id of an open vscode's issue on GitHub.
    -v | --vectorizer-source:
        The path to the vectorizer file, default is data/vectorizer.joblib.
    -f <path-to-feature-vectors> | --source <path-to-feature-vectors>:
        The path to the feature vector file, default is data/feature_vectors.csv.
""",
}


def docstring_preview(text):
    return text.split("\n\n")[0]


def indent(text, spaces=4):
    spaces = " " * spaces
    return "\n".join(
        spaces + line if line.strip() != "" else line for line in text.split("\n")
    )


docstring_headers = {
    key: indent(docstring_preview(value)) for (key, value) in MODULE_DOCSTRINGS.items()
}

DOCSTRING = DOCSTRING.format(**docstring_headers)
