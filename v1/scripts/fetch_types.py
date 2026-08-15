import json
import os
import sys
import time

import requests


def load_dotenv(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def fetch_types(token):
    types = []
    page = 1
    url = "https://api.sportmonks.com/v3/core/types"
    while True:
        params = {"api_token": token, "page": page}
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Types error: {response.status_code} {response.text}")
        payload = response.json()
        data = payload.get("data") or []
        if not data:
            break
        types.extend(data)
        pagination = payload.get("pagination") or payload.get("meta", {}).get("pagination")
        if isinstance(pagination, dict):
            has_more = pagination.get("has_more")
            current = pagination.get("current_page")
            total = pagination.get("total_pages") or pagination.get("last_page")
            if has_more is False:
                break
            if current and total and current >= total:
                break
        page += 1
        time.sleep(0.2)
    return types


def main():
    root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv(root_env)

    token = os.getenv("SPORTMONKS_API_TOKEN", "").strip()
    if not token:
        print("Missing SPORTMONKS_API_TOKEN.")
        sys.exit(1)

    types = fetch_types(token)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "types.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(types, handle, indent=2)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
