"""Ingest upcoming fixtures — the matches that have not been played yet.

    uv run python -m src.data.fixtures

Deliberately a separate module from `footballdata.py`. That parser ends with

    df = df[df["fthg"].notna() & df["ftag"].notna()].copy()

which is correct for a results corpus and fatal for a fixture list, and
`tests/test_footballdata.py` asserts the resulting invariant. Rather than
weaken it, upcoming fixtures get their own table and are joined in only where
a caller explicitly asks for a forward horizon.

Measured 2026-08-17, with the commands recorded in
docs/research/00-measured-facts.md:

  - `fixtures.csv` is a ROLLING ~4-DAY WINDOW, not a season fixture list. The
    snapshot that day held 127 rows across 14 of the 22 main divisions. So the
    schedule that reads it must fire at least every four days or fixtures are
    silently never predicted.
  - Its prices are a genuine PRE-CLOSE snapshot, collected Friday <=17:00 BST
    for the weekend and Tuesday <=13:00 for midweek. Every closing (`*C*`)
    column is present in the header and empty, as it must be before kickoff.
  - There is NO PINNACLE. `PSH/PSD/PSA` are absent from the schema entirely,
    as they now are from the 2026/27 results files. The sharpest price
    available on both legs is the Betfair Exchange (`BFEH/BFED/BFEA` here,
    `BFEC*` in the results file).
  - It retains already-played fixtures, so filtering on kickoff is mandatory
    rather than defensive.
  - It carries a UTF-8 BOM on the first column, which `_read_csv_bytes`
    already handles.
  - Bare `curl` works; no browser user-agent is needed.

Extra-country leagues have their own feed, `new_league_fixtures.csv`, with a
different schema. It is not read here — see docs/PROGRAMME.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.data.footballdata import (
    BASE, MAIN_CORE, MAIN_ODDS, MAIN_DIVISIONS, OUT_DIR, RAW_DIR,
    _country_of, _parse_dates, _read_csv_bytes, _select, normalize_team,
)
from src.features.horizon import UNPLAYED_COL

__all__ = ["FIXTURES_URL", "download_fixtures", "load_fixtures", "build_fixtures",
           "season_of", "uk_now_naive"]

FIXTURES_URL = f"{BASE}/fixtures.csv"
FIXTURES_CSV = RAW_DIR / "fixtures.csv"
FIXTURES_PARQUET = OUT_DIR / "fixtures.parquet"

# Kickoff times in this feed are UK local, and so are the corpus kickoffs it
# has to sort against. The comparison therefore happens in UK local on both
# sides rather than being converted to UTC.
UK_TZ = "Europe/London"


def uk_now_naive() -> pd.Timestamp:
    """Current UK wall-clock time, tz dropped, for comparison against a naive
    UK-local kickoff.

    The runner is UTC and the feed is UK local, so a bare `Timestamp.now()`
    is an hour out under BST. The scheduled run times leave enough margin that
    it would not currently bite, which is exactly why this cannot be left to
    the cron schedule: the schedule is the thing most likely to be changed
    casually later.
    """
    return pd.Timestamp.now(tz=UK_TZ).tz_localize(None)


def season_of(ts: pd.Timestamp) -> str:
    """football-data's season label for a kickoff, e.g. '2026-27'.

    A season starting in year Y is published as Y/Y+1. July is inside the new
    season, not the old one: the cached 2026/27 Scottish Premiership file opens
    on 31/07/2026.
    """
    ts = pd.Timestamp(ts)
    start = ts.year if ts.month >= 7 else ts.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def download_fixtures(session: requests.Session | None = None) -> Path:
    """Fetch the fixtures feed and cache it. Returns the cached path."""
    sess = session or requests.Session()
    r = sess.get(FIXTURES_URL, timeout=60)
    r.raise_for_status()
    if not r.content.strip():
        raise RuntimeError(f"{FIXTURES_URL} returned an empty body")
    FIXTURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    FIXTURES_CSV.write_bytes(r.content)
    return FIXTURES_CSV


def load_fixtures(path: Path | None = None, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Parse the cached fixtures CSV into corpus-shaped rows.

    Output columns match `matches.parquet` where they overlap, so the result can
    be concatenated onto the corpus and run through the same feature pass. The
    differences are all deliberate: `fthg`/`ftag`/`result` are NA, `unplayed` is
    True, and every closing-odds column is empty because closing prices do not
    exist before kickoff.
    """
    path = path or FIXTURES_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} not cached. Run download_fixtures() first.")

    raw = _read_csv_bytes(path)
    if "HomeTeam" not in raw.columns or "Date" not in raw.columns:
        raise RuntimeError(f"{path} does not look like the fixtures feed: {list(raw.columns)[:8]}")

    df = _select(raw, {**MAIN_CORE, **MAIN_ODDS})
    df = df[df["home_raw"].notna() & df["date"].notna()].copy()

    df["date"] = _parse_dates(df["date"])
    df = df[df["date"].notna()].copy()

    # Same kickoff construction as build_matches, so the two sort together.
    t = df["time"].astype("string").str.strip() if "time" in df.columns else None
    if t is not None:
        td = pd.to_timedelta(t + ":00", errors="coerce")
        df["kickoff"] = df["date"] + td.fillna(pd.Timedelta(0))
        df["has_kickoff_time"] = td.notna()
    else:
        df["kickoff"] = df["date"]
        df["has_kickoff_time"] = False

    for col in df.columns:
        if col.startswith(("b365", "ps", "max", "avg", "bfe", "ah")):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Trust the Div column, never a filename or row order.
    df = df[df["div"].isin(MAIN_DIVISIONS)].copy()

    df["season"] = df["kickoff"].map(season_of)
    df["source"] = "fixtures"
    df["league"] = df["div"].map(MAIN_DIVISIONS).fillna(df["div"])
    df["home_key"] = df["home_raw"].map(normalize_team)
    df["away_key"] = df["away_raw"].map(normalize_team)
    df["country"] = df["div"].map(_country_of)

    # No result, and said explicitly rather than left to inference.
    df["fthg"] = pd.Series(pd.NA, index=df.index, dtype="Int16")
    df["ftag"] = pd.Series(pd.NA, index=df.index, dtype="Int16")
    df["result"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df[UNPLAYED_COL] = True

    # The join key, byte-identical to footballdata.build_matches. A mismatch
    # here orphans every prediction and nothing downstream would complain, so
    # tests/test_forward.py asserts it against a real corpus row.
    df["match_id"] = (
        df["div"].astype(str) + "|" + df["date"].dt.strftime("%Y%m%d")
        + "|" + df["home_key"] + "|" + df["away_key"]
    )

    now = uk_now_naive() if now is None else pd.Timestamp(now)
    df = df[df["kickoff"] > now].copy()

    df = df.sort_values(["kickoff", "div", "home_key"]).reset_index(drop=True)
    df = df[~df["match_id"].duplicated()].copy()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("string")
    return df.reset_index(drop=True)


def build_fixtures(refresh: bool = True, write: bool = True,
                   now: pd.Timestamp | None = None) -> pd.DataFrame:
    if refresh:
        download_fixtures()
    df = load_fixtures(now=now)
    if write:
        FIXTURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(FIXTURES_PARQUET, index=False)
    return df


def _describe(fx: pd.DataFrame, label: str) -> None:
    print(f"{label}: {len(fx)} fixtures across {fx['div'].nunique()} divisions")
    if not len(fx):
        return
    print(f"  kickoffs {fx['kickoff'].min()} -> {fx['kickoff'].max()}")
    print(f"  divisions {', '.join(sorted(fx['div'].unique()))}")
    for ps in (("bfeh", "bfed", "bfea"), ("b365h", "b365d", "b365a"),
               ("maxh", "maxd", "maxa"), ("avgh", "avgd", "avga")):
        have = fx[list(ps)].notna().all(axis=1).sum() if all(c in fx for c in ps) else 0
        print(f"  {ps[0][:-1]:6s} priced on {have}/{len(fx)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-refresh", action="store_true", help="parse the cached copy")
    ap.add_argument("--all", action="store_true",
                    help="also show the whole window, including fixtures already "
                         "played -- the feed retains those, so this is how to tell "
                         "'nothing scheduled' apart from 'parsed nothing'")
    args = ap.parse_args()

    if not args.no_refresh:
        download_fixtures()
    print(f"UK now: {uk_now_naive():%Y-%m-%d %H:%M}")
    if args.all:
        _describe(load_fixtures(now=pd.Timestamp("1900-01-01")), "whole window")
        print()
    _describe(build_fixtures(refresh=False), "upcoming")
