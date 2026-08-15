import json
import os
import sys
import time
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


def build_url(fixture_id, feed):
    if feed == "premium":
        return f"https://api.sportmonks.com/v3/football/odds/premium/fixtures/{fixture_id}"
    return f"https://api.sportmonks.com/v3/football/odds/pre-match/fixtures/{fixture_id}"


def main():
    root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv(root_env)
    token = os.getenv("SPORTMONKS_API_TOKEN", "").strip()
    if not token:
        print("Missing SPORTMONKS_API_TOKEN.")
        sys.exit(1)

    feed = os.getenv("SPORTMONKS_ODDS_FEED", "pre-match").strip().lower()
    if feed not in {"pre-match", "premium"}:
        print("SPORTMONKS_ODDS_FEED must be 'pre-match' or 'premium'.")
        sys.exit(1)

    fixtures_dir = Path(__file__).resolve().parent.parent / "data" / "fixtures"
    if not fixtures_dir.exists():
        print(f"Missing fixtures directory: {fixtures_dir}")
        sys.exit(1)

    fixture_ids = sorted(
        int(path.stem.split("_")[-1]) for path in fixtures_dir.glob("fixture_*.json")
    )
    output_dir = Path(__file__).resolve().parent.parent / "data" / "odds"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, fixture_id in enumerate(fixture_ids, start=1):
        url = build_url(fixture_id, feed)
        params = {"api_token": token}
        out_path = output_dir / f"fixture_{fixture_id}.json"
        if out_path.exists():
            print(f"[{index}/{len(fixture_ids)}] {fixture_id} -> cached")
            continue
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            print(f"[{index}/{len(fixture_ids)}] {fixture_id} -> {response.status_code}")
            continue
        payload = response.json()
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[{index}/{len(fixture_ids)}] wrote {out_path}")
        time.sleep(0.2)


if __name__ == "__main__":
    main()
