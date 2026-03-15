import os

import requests
from dotenv import load_dotenv

from src.utils import NPZ_FILE


def submit():
    load_dotenv()
    ENDPOINT = "task4"
    API_TOKEN = os.getenv("TEAM_TOKEN")
    SERVER_URL = os.getenv("SERVER_URL")

    if not API_TOKEN:
        raise ValueError(
            "TEAM_TOKEN not provided. Define TEAM_TOKEN in .env"
        )

    if not SERVER_URL:
        raise ValueError(
            "SERVER_URL not defined. Define SERVER_URL in .env"
        )

    headers = {
        "X-API-Token": API_TOKEN
    }

    # Important, the name of key in files - "npz_file" must be exact
    response = requests.post(
        f"{SERVER_URL}/{ENDPOINT}",
        files={"npz_file": open(NPZ_FILE, "rb")},
        headers=headers
    )

    try:
        data = response.json()
    except Exception:
        data = response.text

    print("response:", response.status_code, data)


if __name__ == "__main__":
    submit()