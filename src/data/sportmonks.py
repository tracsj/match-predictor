"""SportMonks v3 ingest for the player-level tier.

    uv run python -m src.data.sportmonks

This is the depth half of the two-tier design: Danish Superliga and Scottish
Premiership only, but with full starting XIs and 36-41 statistics per player.
Both leagues also appear in football-data.co.uk with Pinnacle closing odds, so
the same network can be trained on identical fixtures with and without a
player encoder and the difference measured.

Facts this is built around, all probed 2026-08-15 (see
docs/research/00-measured-facts.md):

  - The key is on the Football Free Plan: leagues 271 and 501 only, 3,000
    requests per hour, counted PER ENTITY TYPE (fixtures and odds have
    separate budgets).
  - 22 seasons are entitled, but rich per-player statistics only begin at
    2019/20. 2018/19 has odds and almost no player detail (6 stat types, 52
    rows, against 35-41 types and ~400-500 rows from 2019/20).
  - Odds must be market-filtered. `/odds/pre-match/fixtures/{id}/markets/1`
    returns 50 KB where the unfiltered endpoint returns 2.0 MB of ~30 markets
    of which only Fulltime Result is ever read. v1 skipped this and spent
    1.35 GB, 96% of its repo, on data nothing consumed.

Everything is cached per fixture and the run is resumable: an interrupted
fetch picks up where it stopped rather than starting over.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

BASE = "https://api.sportmonks.com/v3/football"
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "sportmonks"

# Verified entitled season ids, 2019/20 -> 2025/26. Hardcoded because they were
# probed and confirmed; `discover_seasons` re-derives them if the plan changes.
SEASONS: dict[int, dict[str, int]] = {
    271: {  # Danish Superliga
        "2019-20": 16020, "2020-21": 17328, "2021-22": 18334, "2022-23": 19686,
        "2023-24": 21644, "2024-25": 23584, "2025-26": 25536,
    },
    501: {  # Scottish Premiership
        "2019-20": 16222, "2020-21": 17141, "2021-22": 18369, "2022-23": 19735,
        "2023-24": 21787, "2024-25": 23690, "2025-26": 25598,
    },
}
LEAGUE_NAMES = {271: "Danish Superliga", 501: "Scottish Premiership"}

FIXTURE_INCLUDE = "participants;scores;lineups.details;events;statistics"
MARKET_FULLTIME_RESULT = 1

# Leave headroom rather than riding the limit to zero -- a 429 costs more than
# a pause, and the budget resets hourly.
RATE_FLOOR = 40


def _token() -> str:
    tok = os.environ.get("SPORTMONKS_API_TOKEN")
    if not tok:
        env = REPO_ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("SPORTMONKS_API_TOKEN="):
                    tok = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not tok:
        raise RuntimeError("SPORTMONKS_API_TOKEN not found in env or .env")
    return tok


@dataclass
class FetchStats:
    fixtures_cached: int = 0
    fixtures_fetched: int = 0
    odds_cached: int = 0
    odds_fetched: int = 0
    empty_odds: int = 0
    errors: list[str] = field(default_factory=list)
    waited_seconds: float = 0.0

    def line(self) -> str:
        return (f"fixtures {self.fixtures_fetched} new / {self.fixtures_cached} cached | "
                f"odds {self.odds_fetched} new / {self.odds_cached} cached "
                f"({self.empty_odds} empty) | waited {self.waited_seconds / 60:.1f} min | "
                f"errors {len(self.errors)}")


class Client:
    """Thin wrapper that respects the per-entity hourly budget.

    SportMonks returns a `rate_limit` block on every response carrying
    `remaining` and `resets_in_seconds` for the entity just requested. Reading
    it is far better than guessing an interval: v1 used a flat 0.2s sleep with
    no retry and no 429 handling at all.
    """

    def __init__(self, token: str | None = None, verbose: bool = True):
        self.token = token or _token()
        self.session = requests.Session()
        self.verbose = verbose
        self.stats = FetchStats()

    def get(self, path: str, **params) -> dict | None:
        params["api_token"] = self.token
        for attempt in range(4):
            try:
                r = self.session.get(f"{BASE}/{path}", params=params, timeout=60)
            except requests.RequestException as exc:
                self.stats.errors.append(f"{path}: {exc}")
                time.sleep(3 * (attempt + 1))
                continue

            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                if self.verbose:
                    print(f"    429, sleeping {wait}s", flush=True)
                time.sleep(wait)
                self.stats.waited_seconds += wait
                continue
            if r.status_code != 200:
                self.stats.errors.append(f"{path}: HTTP {r.status_code}")
                return None

            body = r.json()
            self._respect_budget(body)
            return body

        self.stats.errors.append(f"{path}: gave up after retries")
        return None

    def _respect_budget(self, body: dict) -> None:
        rl = body.get("rate_limit") or {}
        remaining = rl.get("remaining")
        if remaining is None or remaining > RATE_FLOOR:
            return
        wait = float(rl.get("resets_in_seconds") or 60) + 5
        if self.verbose:
            print(f"    budget for {rl.get('requested_entity')} down to "
                  f"{remaining}; sleeping {wait:.0f}s", flush=True)
        time.sleep(wait)
        self.stats.waited_seconds += wait


def discover_seasons(client: Client) -> dict[int, dict[str, int]]:
    """Re-derive entitled seasons. Paging matters: entitlement filtering
    happens after pagination, so page 1 can return fewer than per_page."""
    found: dict[int, dict[str, int]] = {}
    page = 1
    while page <= 6:
        body = client.get("seasons", per_page=50, page=page)
        if not body or not body.get("data"):
            break
        for s in body["data"]:
            lg = s.get("league_id")
            if lg in SEASONS:
                name = str(s.get("name", "")).replace("/", "-")
                if len(name) == 9:                     # "2019-2020" -> "2019-20"
                    name = name[:5] + name[-2:]
                found.setdefault(lg, {})[name] = s["id"]
        if not body.get("pagination", {}).get("has_more"):
            break
        page += 1
    return found


def season_fixture_ids(client: Client, season_id: int) -> list[int]:
    """Fixture ids for a season.

    Note the endpoint: `/fixtures/seasons/{id}` does NOT exist and returns
    "The requested endpoint does not exist". The season object with an
    `include=fixtures` is the working route.
    """
    body = client.get(f"seasons/{season_id}", include="fixtures")
    if not body or "data" not in body:
        return []
    return [f["id"] for f in (body["data"].get("fixtures") or [])]


def fetch_fixture(client: Client, fixture_id: int) -> bool:
    dest = RAW_DIR / "fixtures" / f"{fixture_id}.json"
    if dest.exists():
        client.stats.fixtures_cached += 1
        return True
    body = client.get(f"fixtures/{fixture_id}", include=FIXTURE_INCLUDE)
    if not body or "data" not in body:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(body["data"]))
    client.stats.fixtures_fetched += 1
    return True


def fetch_odds(client: Client, fixture_id: int) -> bool:
    """Fulltime Result only. See the module docstring for why."""
    dest = RAW_DIR / "odds" / f"{fixture_id}.json"
    if dest.exists():
        client.stats.odds_cached += 1
        return True
    body = client.get(
        f"odds/pre-match/fixtures/{fixture_id}/markets/{MARKET_FULLTIME_RESULT}")
    if body is None or "data" not in body:
        return False
    data = body["data"] or []
    if not data:
        client.stats.empty_odds += 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data))
    client.stats.odds_fetched += 1
    return True


def run(first_season: str = "2019-20", verbose: bool = True) -> FetchStats:
    client = Client(verbose=verbose)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[int, str, int]] = []
    for league, seasons in SEASONS.items():
        for name, sid in sorted(seasons.items()):
            if name >= first_season:
                plan.append((league, name, sid))

    for league, name, sid in plan:
        ids_path = RAW_DIR / "season_fixtures" / f"{sid}.json"
        if ids_path.exists():
            ids = json.loads(ids_path.read_text())
        else:
            ids = season_fixture_ids(client, sid)
            ids_path.parent.mkdir(parents=True, exist_ok=True)
            ids_path.write_text(json.dumps(ids))
        if verbose:
            print(f"[{LEAGUE_NAMES[league]} {name}] {len(ids)} fixtures", flush=True)

        for i, fid in enumerate(ids, 1):
            fetch_fixture(client, fid)
            fetch_odds(client, fid)
            if verbose and i % 50 == 0:
                print(f"    {i}/{len(ids)}  {client.stats.line()}", flush=True)

    if verbose:
        print(f"DONE  {client.stats.line()}", flush=True)
        for e in client.stats.errors[:10]:
            print(f"  ERR {e}")
    return client.stats


if __name__ == "__main__":
    run()
