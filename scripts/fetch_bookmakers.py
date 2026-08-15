import json
import os
import sys
from pathlib import Path

import requests


def load_dotenv(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def main():
    root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv(root_env)
    token = os.getenv("SPORTMONKS_API_TOKEN", "").strip()
    if not token:
        print("Missing SPORTMONKS_API_TOKEN.")
        sys.exit(1)

    url = "https://api.sportmonks.com/v3/odds/bookmakers"
    params = {"api_token": token}
    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        print(f"Bookmakers error: {response.status_code} {response.text}")
        sys.exit(1)

    payload = response.json()
    data = payload.get("data") or []
    out_path = Path(__file__).resolve().parent.parent / "data" / "bookmakers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
