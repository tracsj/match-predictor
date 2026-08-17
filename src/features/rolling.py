"""Rolling form, computed strictly from matches before kickoff.

Yeung et al. (2024) use last-5 recency features decomposed into attacking
strength, defensive strength, opposition strength and home advantage -- the
decomposition matters more than the window. Raw rolling goals conflate "this
team scores a lot" with "this team has played weak defences"; carrying the
opponent's strength alongside lets the model separate them instead of being
handed a pre-baked adjustment that may be the wrong one.

Windows of 5 and 10 are both emitted. The Kaggle Football Match Probability
Prediction competition framed the task as each team's previous 10 matches, and
its leaderboard is the best public evidence that a sequence of that length
carries signal.

**Same-day ordering.** football-data has no kickoff time before 2019/20. This
does not affect a team's own rolling window -- no team plays twice in a day,
so its history is unambiguous whatever the within-day order. It WOULD affect
any league-wide running aggregate, so `league_goals_avg` is lagged by a full
day rather than computed up to the instant of kickoff.

**Box-score availability.** Shots, shots on target and corners exist from
2000/01 in the main files and not at all in the extra-country files. Missing
values stay NaN rather than being filled: a zero would read as "took no
shots", which is a different and false claim.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.horizon import unplayed_flags

__all__ = ["RollingParams", "add_rolling", "rolling_features"]

WINDOWS = (5, 10)


@dataclass(frozen=True)
class RollingParams:
    windows: tuple[int, ...] = WINDOWS
    max_rest_days: float = 30.0     # cap so a summer break does not dominate


# Per-match record kept for each team, always from that team's point of view.
_FIELDS = ("gf", "ga", "pts", "sf", "sa", "sotf", "sota", "cf", "ca",
           "opp_elo", "was_home")


def _blank_row(windows) -> dict[str, float]:
    out: dict[str, float] = {}
    for side in ("h", "a"):
        out[f"{side}_played"] = 0.0
        out[f"{side}_rest_days"] = np.nan
        for w in windows:
            for f in ("pts", "gf", "ga", "gd", "sot_f", "sot_a", "corners_f",
                      "opp_elo", "home_share"):
                out[f"{side}_{f}_{w}"] = np.nan
    out["league_goals_avg"] = np.nan
    out["h_days_since_season_start"] = np.nan
    return out


def rolling_features(df: pd.DataFrame, params: RollingParams = RollingParams()) -> pd.DataFrame:
    """One row per match with each side's pre-match form.

    `df` must be sorted by kickoff and carry the rating columns (elo_home /
    elo_away), because the opponent-strength component is the opponent's Elo
    at the time those matches were played.
    """
    if not df["kickoff"].is_monotonic_increasing:
        raise ValueError("df must be sorted by kickoff before building rolling features")
    for need in ("elo_home", "elo_away"):
        if need not in df.columns:
            raise KeyError(f"rolling features need {need!r}; call add_ratings first")

    W = params.windows
    maxw = max(W)
    hist: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=maxw))
    last_played: dict[tuple, pd.Timestamp] = {}
    season_start: dict[tuple, pd.Timestamp] = {}

    # League-wide running goal average, lagged a day so same-day fixtures
    # cannot contribute to each other.
    league_goals: dict[tuple, list[float]] = defaultdict(list)
    league_pending: dict[tuple, list[tuple]] = defaultdict(list)

    cols = df.columns
    def col(name):
        return df[name].to_numpy() if name in cols else np.full(len(df), np.nan)

    country, div, season = col("country"), col("div"), col("season")
    hk, ak = col("home_key"), col("away_key")
    gh, ga_ = col("fthg"), col("ftag")
    hs, as_ = col("hs"), col("as_")
    hst, ast = col("hst"), col("ast")
    hc, ac = col("hc"), col("ac")
    eh, ea = col("elo_home"), col("elo_away")
    kick = pd.to_datetime(df["kickoff"]).to_numpy()
    unplayed = unplayed_flags(df)

    rows = []
    for i in range(len(df)):
        H, A = (country[i], hk[i]), (country[i], ak[i])
        lg = (country[i], div[i], season[i])
        now = pd.Timestamp(kick[i])

        # Release yesterday's league goals into the running average.
        keep = []
        for when, total in league_pending[lg]:
            (league_goals[lg].append(total) if when.date() < now.date() else keep.append((when, total)))
        league_pending[lg] = keep

        row = _blank_row(W)
        row["league_goals_avg"] = (float(np.mean(league_goals[lg]))
                                   if league_goals[lg] else np.nan)

        for side, key in (("h", H), ("a", A)):
            past = hist[key]
            row[f"{side}_played"] = float(len(past))
            if key in last_played:
                row[f"{side}_rest_days"] = min(
                    (now - last_played[key]).total_seconds() / 86400.0,
                    params.max_rest_days)
            for w in W:
                recent = list(past)[-w:]
                if not recent:
                    continue
                arr = {f: np.array([r[f] for r in recent], dtype=float) for f in _FIELDS}
                row[f"{side}_pts_{w}"] = float(np.nanmean(arr["pts"]))
                row[f"{side}_gf_{w}"] = float(np.nanmean(arr["gf"]))
                row[f"{side}_ga_{w}"] = float(np.nanmean(arr["ga"]))
                row[f"{side}_gd_{w}"] = float(np.nanmean(arr["gf"] - arr["ga"]))
                row[f"{side}_sot_f_{w}"] = float(np.nanmean(arr["sotf"])) if not np.isnan(arr["sotf"]).all() else np.nan
                row[f"{side}_sot_a_{w}"] = float(np.nanmean(arr["sota"])) if not np.isnan(arr["sota"]).all() else np.nan
                row[f"{side}_corners_f_{w}"] = float(np.nanmean(arr["cf"])) if not np.isnan(arr["cf"]).all() else np.nan
                # The opposition-strength component: without it, "scored a lot"
                # and "played weak defences" are the same number.
                row[f"{side}_opp_elo_{w}"] = float(np.nanmean(arr["opp_elo"]))
                row[f"{side}_home_share_{w}"] = float(np.nanmean(arr["was_home"]))

        if lg not in season_start or season_start[lg] > now:
            season_start.setdefault(lg, now)
        row["h_days_since_season_start"] = (now - season_start[lg]).total_seconds() / 86400.0
        rows.append(row)

        # --- absorb this match into history, AFTER recording the features ---
        # Nothing to absorb from a fixture that has not been played. The naive
        # version of this loop records it as a 0-0 defeat for both sides,
        # because `3.0 if gd > 0 else (1.0 if gd == 0 else 0.0)` takes the last
        # branch when gd is NaN, and stamps `last_played` so the next fixture's
        # rest_days is wrong too. See src/features/horizon.py.
        if unplayed[i]:
            continue

        gd = gh[i] - ga_[i]
        hist[H].append({"gf": gh[i], "ga": ga_[i],
                        "pts": 3.0 if gd > 0 else (1.0 if gd == 0 else 0.0),
                        "sf": hs[i], "sa": as_[i], "sotf": hst[i], "sota": ast[i],
                        "cf": hc[i], "ca": ac[i], "opp_elo": ea[i], "was_home": 1.0})
        hist[A].append({"gf": ga_[i], "ga": gh[i],
                        "pts": 3.0 if gd < 0 else (1.0 if gd == 0 else 0.0),
                        "sf": as_[i], "sa": hs[i], "sotf": ast[i], "sota": hst[i],
                        "cf": ac[i], "ca": hc[i], "opp_elo": eh[i], "was_home": 0.0})
        last_played[H] = last_played[A] = now
        league_pending[lg].append((now, float(gh[i] + ga_[i])))

    return pd.DataFrame(rows, index=df.index)


def add_rolling(df: pd.DataFrame, params: RollingParams = RollingParams()) -> pd.DataFrame:
    return pd.concat([df, rolling_features(df, params)], axis=1)


def rolling_feature_names(params: RollingParams = RollingParams()) -> list[str]:
    return [c for c in _blank_row(params.windows)]
