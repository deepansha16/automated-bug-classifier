from app.mining import Miner
from typing import Any
import requests, logging, os
import pandas as pd


class ContributorMiner(Miner):
    VSCODE_CONTRIBUTORS_URL = (
        "https://api.github.com/repos/microsoft/vscode/contributors"
    )

    def get_contributors(self):
        """
        Lists contributors to the specified repository and sorts them by
        the number of commits per contributor in descending order.
        """
        headers = {
            "Accept": self.MEDIA_TYPE,
            "Authorization": "token " + self.token,
        }
        payload = {"anon": "true", "per_page": self.MAX_NUM_OF_RESULTS_PER_PAGE}
        r = requests.get(self.VSCODE_CONTRIBUTORS_URL, headers=headers, params=payload)
        self._handle_response(r)
        page = 1
        self._store_response(r.json(), page)
        while "next" in r.links:
            next_url = r.links["next"]["url"]
            r = requests.get(next_url, headers=headers)
            self._handle_response(r)
            page += 1
            self._store_response(r.json(), page)

        self._merge_contributors_response()

    def _store_response(self, response: Any, page: int):
        """
        Store obtained contributors to the file vscode_contributors_response_page<i>.json.
        <i> is the page number

        Parameters:
            response (Any): Any objects that can be serialized in json, e.g. str, list, dict, .etc.
            page (int): Page number.
        """
        contributors = pd.Series(response)
        contributors.to_json(
            f"vscode_contributors_response_page{page}.json",
            orient="records",
            lines=True,
        )

    def _merge_contributors_response(self):
        """
        Merge all vscode_contributors_response_page<i>.json files into vscode_contributors_response.json.
        """
        for filename in os.listdir(os.path.curdir):
            if os.path.isfile(filename) and filename.startswith(
                "vscode_contributors_response_page"
            ):
                with open(os.path.join(os.path.curdir, filename)) as source_file:
                    data = source_file.read()
                with open("vscode_contributors_response.json", "a") as target_file:
                    target_file.write(data)
                os.remove(filename)


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%d/%m/%Y %H:%M:%S %z",
        filename="contributor_mining.log",
        encoding="utf-8",
        level=logging.DEBUG,
    )
    contributor_miner = ContributorMiner()
    contributor_miner.read_token()
    contributor_miner.get_contributors()


if __name__ == "__main__":
    main()
