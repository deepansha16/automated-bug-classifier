import pandas as pd


class ContributorCounter:
    def __init__(self):
        self.assignee_count = dict()

    def count(self, issue: dict):
        assignee = self._get_assignee(issue)
        if assignee is None:
            return
        if assignee in self.assignee_count:
            self.assignee_count[assignee] += 1
        else:
            self.assignee_count[assignee] = 1

    def is_invalid_assignee(self, issue: dict) -> bool:
        """Filter out issues assigned to developers who have been assignees less than five times"""
        assignee = self._get_assignee(issue)
        return assignee is None or self.assignee_count[assignee] < 5

    def _get_assignee(self, issue: dict):
        if "assignee" in issue and issue["assignee"] is not None:
            return issue["assignee"]["login"]
        elif "assignees" in issue and issue["assignees"][0] is not None:
            return issue["assignees"][0]["login"]
        else:
            return

    def filter(self):
        self.assignee_count = {k: v for (k, v) in self.assignee_count.items() if v > 4}
        return self


    def save(self):
        assignees = (
            pd.Series(self.assignee_count)
            .to_frame("count")
            .reset_index(names="login")
            .sort_values(by="count")
        )
        assignees.to_csv("./data/vscode_contributors_count.csv", index=False)
