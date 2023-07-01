from typing import Any
import requests, json, logging
from app.mining import Miner


class IssueMiner(Miner):
    VSCODE_ISSUES_URL = "https://api.github.com/repos/microsoft/vscode/issues"

    def get_an_issue_by_id(self, id: str) -> dict:
        headers = {"Accept": self.MEDIA_TYPE}
        r = requests.get(self.VSCODE_ISSUES_URL + id, headers=headers)
        self._handle_response(r)
        return r.json()

    def get_issues(self):
        """
        Get all closed issues from VS Code GitHub repository and
        write them to the file vscode_issues_response.json.
        """
        self._initialize_response_file()
        headers = {
            "Accept": self.MEDIA_TYPE,
            "Authorization": "token " + self.token,
        }
        payload = {"state": "closed", "per_page": self.MAX_NUM_OF_RESULTS_PER_PAGE}

        r = requests.get(self.VSCODE_ISSUES_URL, headers=headers, params=payload)
        self._handle_response(r)
        self._store_response(r.json(), False)

        while "next" in r.links:
            next_url = r.links["next"]["url"]
            r = requests.get(next_url, headers=headers)
            self._handle_response(r)
            self._store_response(r.json(), True)

        self._finalize_response_file()

    def _store_response(self, response: Any, with_comma: bool):
        """
        Append obtained issues to the file vscode_issues_response.json.

        Parameters:
            response (Any): Any objects that can be serialized in json, e.g. str, list, dict, .etc.
        """
        json_object = json.dumps(response, indent=4)
        with open("vscode_issues_response.json", "a", encoding="utf-8") as f:
            if with_comma:
                f.write(",\n")
            f.write(json_object)

    def _initialize_response_file(self):
        with open("vscode_issues_response.json", "a", encoding="utf-8") as f:
            f.write("[")

    def _finalize_response_file(self):
        with open("vscode_issues_response.json", "a", encoding="utf-8") as f:
            f.write("]")


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%d/%m/%Y %H:%M:%S %z",
        filename="issue_mining.log",
        encoding="utf-8",
        level=logging.DEBUG,
    )
    issue_miner = IssueMiner()
    issue_miner.read_token()
    issue_miner.get_issues()


if __name__ == "__main__":
    main()
