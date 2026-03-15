import argparse
import os

import requests
from dotenv import load_dotenv


# Load .env file if present
load_dotenv()

ENDPOINT = "task-status"


def parse_args():
    parser = argparse.ArgumentParser(description="Query task status from server")

    parser.add_argument(
        "request_id",
        help="Request ID to query"
    )
    parser.add_argument(
        "--team-token",
        help="Team API token (overrides TEAM_TOKEN from .env)"
    )

    parser.add_argument(
        "--server-url",
        help="Server URL (overrides SERVER_URL from .env)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    team_token = args.team_token or os.getenv("TEAM_TOKEN")
    server_url = args.server_url or os.getenv("SERVER_URL")

    if not team_token:
        raise ValueError(
            "TEAM_TOKEN not provided. Provide it via --team-token or define TEAM_TOKEN in .env"
        )

    if not server_url:
        raise ValueError(
            "SERVER_URL not defined. Provide it via --server-url or define SERVER_URL in .env"
        )

    payload = {
        "request_id": args.request_id
    }

    headers = {
        "X-API-Token": team_token
    }

    response = requests.post(
        f"{server_url}/{ENDPOINT}",
        json=payload,
        headers=headers
    )

    try:
        data = response.json()
    except Exception:
        data = response.text

    sh = data['results'][0]['sh']
    ti = data['results'][0]['ti']
    amp = data['results'][0]['amp']

    print("response:", response.status_code, data)

    print(f'shape: {sh}')
    print(f'amplitude: {amp}')
    print(f'time: {ti}')

    # total = 0.6 * sh + 0.2 * amp + 0.2 * ti
    total = sh + amp + ti
    print(f'total: {total}')


if __name__ == "__main__":
    main()