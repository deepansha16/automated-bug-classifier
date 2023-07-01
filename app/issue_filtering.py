import pandas as pd
from app.contributor_counting import ContributorCounter


class IssueFilter:
    # found these set of labels vague and not useful for the analysis
    INVALID_LABELS = [
        "*english-please",
        "duplicate",
        "invalid-testplan-item",
        "label-to-subscribe-to",
        "translation-required-chinese simplified",
        "translation-required-chinese traditional",
        "translation-required-arabic",
        "translation-required-german",
        "translation-required-spanish",
        "translation-required-french",
        "translation-required-italian",
        "translation-required-japanese",
        "translation-required-korean",
        "translation-required-portuguese",
        "translation-required-russian",
        "translation-required-turkish",
        "translation-required-polish",
        "translation-required-romanian",
        "translation-required-czech",
        "translation-required-persian",
        "translation-required-portuguese-brazil",
        "translation-required-russian",
        "translation-required-thai",
        "translation-required-undefined",
    ]

    def __init__(self, contributor_counter: ContributorCounter):
        # store issues in a dataframe, each row represents an issue dict
        # e.g. {'url': 'http...', 'id': 1392647560, 'title': 'Open multiple terminals', ...}
        self.data = (
            pd.read_json(
                "./data/complete_vscode_issues_response.json",
            )
            .stack()
            .reset_index(drop=True)
            .to_frame("issue")
        )
        self.contributor_counter = contributor_counter

        # deprecated
        # preparation for _is_not_english
        # nltk.download("crubadan")
        # nltk.download("punkt")
        # self.text_cat = nltk.classify.TextCat()

    def filter(self):
        """
        Filter out unwanted records based on issue id, number of assignees, language and labels and
        get contributors who have been assigned more than once
        """
        self.data = self.data.assign(Valid=True)
        # remove pull requests and issues without exactly one assignee
        for index in self.data.index:
            # log running
            # issue_id = self.data.loc[index, "issue"]["number"]
            # if index % 100 == 0:
            #     logging.info(
            #         f"Index {index}/{self.data.index.stop}: issue id {issue_id}"
            #     )
            if (
                self._is_pull_request(index)
                or self._has_more_than_one_assignees(index)
                or self._has_invalid_id(index)
                or self._has_invalid_label(index)
            ):
                self.data.loc[index, "Valid"] = False
            else:
                self._handle_labels(index)
                self.contributor_counter.count(self.data.loc[index, "issue"])

        self.data = self.data.loc[self.data.Valid]
        for index in self.data.index:
            if self.contributor_counter.is_invalid_assignee(
                self.data.loc[index, "issue"]
            ):
                self.data.loc[index, "Valid"] = False
        self.data = self.data.loc[self.data.Valid]
        self.contributor_counter.filter().save()

    def _is_pull_request(self, index) -> bool:
        return "pull_request" in self.data.loc[index, "issue"]

    def _has_more_than_one_assignees(self, index) -> bool:
        issue = self.data.loc[index, "issue"]
        if "assignees" in issue and issue["assignees"]:
            return len(issue["assignees"]) != 1
        elif "assignee" in issue and issue["assignee"]:
            return False
        else:
            return True

    def _has_invalid_id(self, index) -> bool:
        return self.data.loc[index, "issue"]["number"] > 160000

    def _is_not_english(self, index) -> bool:
        # consider texts from issue title and body separately
        # because sometimes the language of title and language of body are different, e.g. index=2478
        # we check title first because it's shorter
        title_language = self.text_cat.guess_language(
            self.data.loc[index, "issue"]["title"]
        )
        if title_language != "eng":
            return True
        if self.data.loc[index, "issue"]["body"] is not None:
            body_language = self.text_cat.guess_language(
                self.data.loc[index, "issue"]["body"]
            )
            return body_language != "eng"

    def _has_invalid_label(self, index) -> bool:
        for label in self.data.loc[index, "issue"]["labels"]:
            if label["name"] in self.INVALID_LABELS:
                return True
        return False

    def _handle_labels(self, index):
        new_label_structure = []
        for label in self.data.loc[index, "issue"]["labels"]:
            new_label_structure.append(label["name"])
        self.data.loc[index, "issue"]["labels"] = new_label_structure

    def split(self):
        """
        Split issues into training and test sets based on issue id.
        id ≤ 150000 as training set
        id from 150001 to 160000 as test set
        """
        self.data = self.data.assign(ForTraining=False, Recent=False)

        # only consider issues with issue id ≤ 150000 as training set
        for index in self.data.index:
            if self.data.loc[index, "issue"]["number"] <= 150000:
                self.data.loc[index, "ForTraining"] = True
        self.data.loc[self.data.ForTraining == True]["issue"].to_json(
            "./data/issues_for_training.json", orient="records", lines=True
        )
        self.data.loc[self.data.ForTraining == False]["issue"].to_json(
            "./data/issues_for_testing.json", orient="records", lines=True
        )


def main():
    # logging.basicConfig(
    #     format="%(asctime)s %(levelname)s:%(message)s",
    #     datefmt="%d/%m/%Y %H:%M:%S %z",
    #     filename="issue_filtering.log",
    #     encoding="utf-8",
    #     level=logging.DEBUG,
    # )
    contributor_counter = ContributorCounter()
    issue_filter = IssueFilter(contributor_counter)
    issue_filter.filter()
    issue_filter.split()


if __name__ == "__main__":
    main()
