import requests, datetime, time, logging


class Miner:
    MEDIA_TYPE = "application/vnd.github+json"
    MAX_NUM_OF_RESULTS_PER_PAGE = 100

    def __init__(self, token: str = ""):
        """
        Constructs all the necessary attributes for the Miner object.
        Parameters:
            token (str): A GitHub access token
        """
        self.token = token

    def _handle_response(self, r: requests.Response):
        logging.info(f"Response of GET request: {r.status_code}")
        r.raise_for_status()

        remaining_limit = int(r.headers["X-RateLimit-Remaining"])
        reset_timestamp = float(r.headers["X-RateLimit-Reset"])
        logging.info(
            f"Remaining limit: {remaining_limit} Reset timestamp: {reset_timestamp}"
        )
        if remaining_limit == 0:
            self._wait_until_reset_remaining_limit(reset_timestamp)

    def _wait_until_reset_remaining_limit(self, reset_timestamp: float):
        """
        Wait until GitHub reset remaining limit to 5000.
        To avoid being blocked by GitHub, add 60 seconds more to wait.

        Parameters:
            reset_timestamp (float): A Unix timestamp returned by GitHub that states when to reset remaining limit
        """
        present = datetime.datetime.now()
        present_timestamp = datetime.datetime.timestamp(present)
        interval = int(reset_timestamp - present_timestamp) + 60
        time.sleep(interval)

    def read_token(self):
        # for temporary usage, probably replace it with a command line argument later
        print("Please enter your GitHub access token:")
        self.token = input()
