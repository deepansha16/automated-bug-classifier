import string
import sys
import pandas as pd
import regex as re
from nltk.corpus import stopwords
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import os
import matplotlib.pyplot as plt

# Variables to determine the minimum and maximum amount of times that words can appear across the dataset
MINIMUM_APPEARANCE_THRESHOLD = 80
MAXIMUM_APPEARANCE_THRESHOLD = 40000

# Valid options are `"n"` for nouns,`"v"` for verbs, `"a"` for adjectives,
# `"r"` for adverbs and `"s"`for satellite adjectives.
# Lookup table to translate part-of-speech tags into valid arguments for the lemmatizer
# Tags that are not explicitly declared by the 5 available arguments will default to 'n'
# as it is the default argument of the lemmatizer with no tags.
PART_OF_SPEECH_TABLE = {
    "CC": "n",
    "CD": "n",
    "DT": "n",
    "EX": "n",
    "FW": "n",
    "IN": "n",
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
    "LS": "n",
    "MD": "n",
    "NN": "n",
    "NNS": "n",
    "NNP": "n",
    "NNPS": "n",
    "PDT": "n",
    "POS": "n",
    "PRP": "n",
    "PRP$": "n",
    "RB": "r",
    "RBR": "r",
    "RBS": "r",
    "RP": "n",
    "SYM": "n",
    "TO": "n",
    "UH": "n",
    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v",
    "WDT": "n",
    "WP": "n",
    "WP$": "n",
    "WRB": "r",
    "$": "n",
    "''": "n",
}


# Assuming the input is a string (as title and body are) we remove punctuation, stopwords and we stem the terms
def preprocess(str_row, stop, do_lemma):
    if str_row is None or str_row == "":
        return ""

    # Replace all URLs with empty strings
    to_return = re.sub(r"http\S+", "", str_row)

    # Remove words with numbers in them
    to_return = re.sub("\S*\d\S*", "", to_return).strip()

    # Only keep characters
    to_return = re.sub(r"[^a-zA-Z ]+", " ", to_return)

    # Remove punctuation and turn into lowercase
    to_return = to_return.translate(str.maketrans("", "", string.punctuation)).lower()

    # Stopwords removal
    to_return = " ".join([word for word in to_return.split() if word not in stop])

    # Lemmatization with part-of-speech tagging stemming
    stemmer = SnowballStemmer("english")

    if do_lemma:
        lemmatizer = WordNetLemmatizer()
        tagged = nltk.pos_tag(to_return.split())

        to_return = " ".join(
            [
                stemmer.stem(lemmatizer.lemmatize(word, PART_OF_SPEECH_TABLE[tag]))
                for word, tag in tagged
            ]
        )
    else:
        to_return = " ".join([stemmer.stem(word) for word in to_return.split()])

    if to_return is None or to_return == "":
        return ""

    return to_return


def count_words(s, wc):
    """
    Given a string, count the words of that string and
    store them into the passed dictionary with their count.
    """
    if s is None:
        return

    for word in s.split():
        if word not in wc:
            wc[word] = 1
        else:
            wc[word] += 1


def delete_words(s, wc):
    """
    Given a string and a dictionary that maps words to their count,
    remove words that have less than MINIMUM_APPEARANCE_THRESHOLD or more than
    MAXIMUM_APPEARANCE_THRESHOLD occurrences.
    """
    if s is None:
        return ""

    words = s.split()
    return " ".join(
        [
            word
            for word in words
            if not (
                wc[word] < MINIMUM_APPEARANCE_THRESHOLD
                or wc[word] > MAXIMUM_APPEARANCE_THRESHOLD
            )
        ]
    )


def reduce_dimensions(data):
    """
    Given a dataframe with columns 'title', 'body' and 'assignee', it counts the occurrences
    of each word within each title and body and removes the words that come up
    less than MINIMUM_APPEARANCE_THRESHOLD times and more than MAXIMUM_APPEARANCE_THRESHOLD times
    meaning we remove features that are too specific or too broad for training.
    """
    to_return = data

    # Count words
    wc = {}

    to_return["title"].apply(count_words, args=(wc,))
    to_return["body"].apply(count_words, args=(wc,))

    # Plot initial count
    df = pd.DataFrame(list(wc.items()), columns=["word", "count"])
    fig = plt.figure(figsize=(10, 15))
    df.boxplot(column="count")
    fig.savefig("data/wordcount.png", format="png")

    total_count = df["count"].sum()

    # Delete words that exceed thresholds
    to_return["title"] = to_return["title"].apply(delete_words, args=(wc,))
    to_return["body"] = to_return["body"].apply(delete_words, args=(wc,))

    # Count clean words
    clean_wc = {}

    to_return["title"].apply(count_words, args=(clean_wc,))
    to_return["body"].apply(count_words, args=(clean_wc,))

    # Plot count after cleaning
    clean_df = pd.DataFrame(list(clean_wc.items()), columns=["word", "count"])
    fig = plt.figure(figsize=(10, 15))
    clean_df.boxplot(column="count")
    fig.savefig("data/clean_wordcount.png", format="png")

    clean_count = clean_df["count"].sum()

    print(
        "The total count is now "
        + str(clean_count)
        + " from the original "
        + str(total_count)
        + "."
    )

    return to_return


def main(args):
    # Get filename
    if args.source is None or not os.path.exists(args.source):
        print("Enter valid path for the source code..")
        sys.exit(0)

    filename = args.source
    formatted_filename = os.path.basename(filename).replace(".json", "")

    # Download stopwords
    nltk.download("stopwords")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

    stop = stopwords.words("english")

    data = pd.read_json(filename, lines=True)

    # Labels not considered so far, may want to include them later on.
    # Copy title and body into the processed data as we do not need the rest.
    proc_data = pd.DataFrame(columns=["id", "title", "body", "assignee"])
    assignees = []
    for row in data.iterrows():
        if row[1]["assignee"] is None:
            assignees.append(row[1]["assignees"][0]["login"])
        else:
            assignees.append(row[1]["assignee"]["login"])

    proc_data["id"] = data["number"]
    proc_data["assignee"] = pd.Series(assignees)
    proc_data["title"] = data["title"]
    proc_data["body"] = data["body"]

    proc_data["title"] = proc_data["title"].apply(
        preprocess,
        args=(
            stop,
            args.do_lemma,
        ),
    )
    proc_data["body"] = proc_data["body"].apply(
        preprocess,
        args=(
            stop,
            args.do_lemma,
        ),
    )

    if args.is_training:
        proc_data = reduce_dimensions(proc_data)

    # Output to CSV
    proc_data.to_csv("data/preprocessed_" + formatted_filename + ".csv", index=False)
