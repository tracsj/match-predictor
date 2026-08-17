"""Parse cached SportMonks fixtures into match and player-match tables.

    uv run python -m src.data.sportmonks_parse

Two outputs, both parquet:

  sm_matches.parquet        one row per fixture, keyed so it joins to the
                            football-data corpus on (country, home, away, date)
  sm_players.parquet        one row per player per fixture, carrying minutes,
                            position and 37 raw statistics

The stat type-id table is inherited from v1. Those 37 ids were derived by
dumping SportMonks' type dictionary and counting frequencies against a real
season -- genuinely hard-won and not worth re-deriving. What has changed is
everything around it: the join to football-data goes through the reviewed
alias map, and nothing here computes a rolling feature, because that belongs
in one chronological pass alongside the rest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.footballdata import OUT_DIR
from src.data.sportmonks import LEAGUE_NAMES, RAW_DIR
from src.data.team_aliases import resolve_team

__all__ = ["STAT_DEFS", "parse_all", "MATCHES_PARQUET", "PLAYERS_PARQUET"]

MATCHES_PARQUET = OUT_DIR / "sm_matches.parquet"
PLAYERS_PARQUET = OUT_DIR / "sm_players.parquet"

# SportMonks stat type ids, inherited from v1's build_player_match_dataset.
# "count" statistics are converted to per-90 downstream; "rate" statistics are
# already normalised and are averaged instead.
STAT_DEFS = [
    ("goals", 52, "count"), ("assists", 79, "count"),
    ("shots_total", 42, "count"), ("shots_on_target", 86, "count"),
    ("shots_off_target", 41, "count"), ("key_passes", 117, "count"),
    ("passes", 80, "count"), ("accurate_passes", 116, "count"),
    ("accurate_passes_pct", 1584, "rate"), ("touches", 120, "count"),
    ("total_crosses", 98, "count"), ("long_balls", 122, "count"),
    ("long_balls_won", 123, "count"), ("long_balls_won_pct", 27270, "rate"),
    ("total_duels", 105, "count"), ("duels_won", 106, "count"),
    ("duels_lost", 1491, "count"), ("duels_won_pct", 27276, "rate"),
    ("aerials", 27274, "count"), ("aerials_won", 107, "count"),
    ("aerials_lost", 27266, "count"), ("aerials_won_pct", 27275, "rate"),
    ("tackles", 78, "count"), ("tackles_won", 27267, "count"),
    ("tackles_won_pct", 27268, "rate"), ("interceptions", 100, "count"),
    ("clearances", 101, "count"), ("fouls", 56, "count"),
    ("fouls_drawn", 96, "count"), ("dribble_attempts", 108, "count"),
    ("successful_dribbles", 109, "count"), ("dribbled_past", 110, "count"),
    ("dispossessed", 94, "count"), ("possession_lost", 27273, "count"),
    ("ball_recovery", 27271, "count"), ("rating", 118, "rate"),
    ("goals_conceded", 88, "count"),
]
STAT_BY_ID = {tid: (name, kind) for name, tid, kind in STAT_DEFS}
COUNT_STATS = [n for n, _, k in STAT_DEFS if k == "count"]
RATE_STATS = [n for n, _, k in STAT_DEFS if k == "rate"]

MINUTES_TYPE_ID = 119
COUNTRY_OF_LEAGUE = {271: "DNK", 501: "SC0"}

# SportMonks lineup type ids: 11 = starting XI, 12 = bench.
STARTING_XI, BENCH = 11, 12


def _stat_value(detail: dict) -> float:
    """SportMonks nests the number under `data.value`, sometimes as a string."""
    v = (detail.get("data") or {}).get("value")
    if v is None:
        v = detail.get("value")
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _parse_fixture(raw: dict) -> tuple[dict, list[dict]] | None:
    parts = raw.get("participants") or []
    if len(parts) != 2:
        return None

    home = away = None
    for p in parts:
        loc = ((p.get("meta") or {}).get("location") or "").lower()
        if loc == "home":
            home = p
        elif loc == "away":
            away = p
    if home is None or away is None:
        # Never fall back to list order. In v1, five scripts assumed the first
        # listed team was home; measured, it was home only 63.6% of the time,
        # and roughly a third of simulated bets were graded against the wrong
        # side's price. Skipping is honest; guessing is not.
        return None

    # Full-time score. `scores` carries several periods; take CURRENT/2ND_HALF.
    gh = ga = None
    for sc in (raw.get("scores") or []):
        if sc.get("description") in ("CURRENT", "2ND_HALF"):
            s = sc.get("score") or {}
            if s.get("participant") == "home":
                gh = s.get("goals")
            elif s.get("participant") == "away":
                ga = s.get("goals")
    if gh is None or ga is None:
        return None

    league = raw.get("league_id")
    match = {
        "sm_fixture_id": raw.get("id"),
        "league_id": league,
        "league": LEAGUE_NAMES.get(league, str(league)),
        "div": COUNTRY_OF_LEAGUE.get(league),
        "season_id": raw.get("season_id"),
        "kickoff": pd.to_datetime(raw.get("starting_at"), errors="coerce"),
        "sm_home_id": home.get("id"), "sm_away_id": away.get("id"),
        "home_raw": home.get("name"), "away_raw": away.get("name"),
        "home_key": resolve_team(home.get("name")),
        "away_key": resolve_team(away.get("name")),
        "fthg": int(gh), "ftag": int(ga),
    }
    match["result"] = "H" if gh > ga else ("D" if gh == ga else "A")

    home_id = home.get("id")
    players = []
    for entry in (raw.get("lineups") or []):
        details = {}
        for d in (entry.get("details") or []):
            tid = d.get("type_id")
            if tid == MINUTES_TYPE_ID:
                details["minutes"] = _stat_value(d)
            elif tid in STAT_BY_ID:
                details[STAT_BY_ID[tid][0]] = _stat_value(d)

        team_id = entry.get("team_id")
        ff = entry.get("formation_field")
        players.append({
            "sm_fixture_id": raw.get("id"),
            "kickoff": match["kickoff"],
            "player_id": entry.get("player_id"),
            "player_name": entry.get("player_name"),
            "team_id": team_id,
            "is_home": int(team_id == home_id),
            "lineup_type": entry.get("type_id"),
            "is_starter": int(entry.get("type_id") == STARTING_XI),
            "formation_field": ff,
            # The leading digit of formation_field encodes the line the player
            # occupies: 1 keeper, 2 defence, 3 midfield, 4 attack.
            "position_group": int(str(ff)[0]) if ff not in (None, "") and str(ff)[0].isdigit() else 0,
            "jersey": entry.get("jersey_number"),
            **details,
        })

    return match, players


def parse_all(write: bool = True, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted((RAW_DIR / "fixtures").glob("*.json"))
    matches, players, skipped = [], [], 0

    for f in files:
        try:
            raw = json.loads(f.read_text())
        except json.JSONDecodeError:
            skipped += 1
            continue
        got = _parse_fixture(raw)
        if got is None:
            skipped += 1
            continue
        m, ps = got
        matches.append(m)
        players.extend(ps)

    mdf = pd.DataFrame(matches).sort_values("kickoff").reset_index(drop=True)
    pdf = pd.DataFrame(players).sort_values("kickoff").reset_index(drop=True)

    # A missing statistic means "not recorded", which for a count is zero
    # appearances of that event -- but only for players who actually played.
    # Leaving NaN for unplayed entries keeps "did not appear" distinguishable
    # from "appeared and did nothing".
    if not pdf.empty:
        played = pdf["minutes"].fillna(0) > 0
        for col in COUNT_STATS:
            if col in pdf.columns:
                pdf.loc[played, col] = pdf.loc[played, col].fillna(0.0)

    if verbose:
        print(f"parsed {len(mdf):,} matches, {len(pdf):,} player-match rows "
              f"({skipped} fixtures skipped)")
        if not mdf.empty:
            print(f"  {mdf['kickoff'].min().date()} .. {mdf['kickoff'].max().date()}")
            print(mdf.groupby(["league", "season_id"]).size().to_string())

    if write and not mdf.empty:
        MATCHES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        mdf.to_parquet(MATCHES_PARQUET, index=False)
        pdf.to_parquet(PLAYERS_PARQUET, index=False)
    return mdf, pdf


if __name__ == "__main__":
    parse_all()
