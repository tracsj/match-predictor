"""Squad tensors: each starting XI as a set of player vectors.

This is the tier-2 experiment. The literature sweep found **no published
pre-match football model that encodes a starting XI as a permutation-invariant
set of player vectors and predicts 1X2** -- the nearest precedent is an NBA
paper (Hubacek et al. 2019) that ran a convolution over player statistics to
consume a variable roster. So there is no recipe to follow here, which is the
point of running it as a measured experiment rather than an assumption.

Output shape `(n_matches, 2, squad_size, n_features)` with side 0 = home, plus
a `(n_matches, 2, squad_size)` mask. Each player vector holds that player's
rolling per-90 form from matches strictly BEFORE this one, plus position and
experience. A player with no history is masked out rather than zero-filled,
because "debutant" and "does nothing" are different claims.

**Order within a squad is not meaningful.** Players are emitted in formation
order for reproducibility, but the encoder that consumes this must be
permutation-invariant (mean/max pooling, or attention), or it will learn
something about the arbitrary ordering instead of about the squad.

Built in one forward chronological pass, like every other feature table here.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.sportmonks_parse import COUNT_STATS, RATE_STATS

__all__ = ["SquadParams", "build_squads", "player_feature_names"]

# Statistics SportMonks actually MEASURES across the whole window.
#
# The criterion is coverage -- the fraction of player-matches where the value
# is present -- not how often it is zero. That distinction was a real error
# worth recording, because the first version of this list got it backwards.
#
# SportMonks OMITS a detail row entirely when a statistic is not collected; it
# does not return zero. The parser originally filled every absent count with
# zero for anyone who played, which turned "not measured" into "did it zero
# times". `touches` then read as 10.9 per 90 against 38.8 passes -- physically
# impossible, since a player touches the ball more often than they pass it --
# and I blamed the vendor's data rather than my own fill. With the fill
# corrected (coverage is now decided per fixture from the set of type ids the
# feed carries for that match), touches measures 54.1 when present, comfortably
# above passes at 36.7, exactly as the ordinary definition predicts.
#
# What survives is a genuine and much narrower coverage gap: 10 statistics are
# collected in only 4 of 14 seasons (touches, aerials, aerials_lost,
# ball_recovery, possession_lost, tackles_won and the percentage variants) and
# 3 more in 9 of 14 (long_balls, long_balls_won, shots_off_target). Those are
# excluded because a feature present for a quarter of the corpus cannot carry
# a rolling window across it.
#
# The 24 kept below are present in all 14 seasons at 84-98% coverage.
# `test_players.py` re-derives this from the data so a future fetch cannot
# leave the list stale.
MIN_COVERAGE = 0.80

CORE_COUNTS = [
    "goals", "assists", "shots_total", "shots_on_target", "key_passes",
    "passes", "accurate_passes", "total_crosses", "total_duels", "duels_won",
    "duels_lost", "aerials_won", "tackles", "interceptions", "clearances",
    "fouls", "fouls_drawn", "dribble_attempts", "successful_dribbles",
    "dribbled_past", "dispossessed", "goals_conceded",
]
CORE_RATES = ["rating", "accurate_passes_pct"]


@dataclass(frozen=True)
class SquadParams:
    squad_size: int = 11          # starters only
    window: int = 10              # appearances of rolling history per player
    min_minutes: int = 1          # what counts as an appearance


def player_feature_names(params: SquadParams = SquadParams()) -> list[str]:
    return (
        [f"p90_{c}" for c in CORE_COUNTS]
        + [f"avg_{r}" for r in CORE_RATES]
        + ["hist_matches", "hist_avg_minutes", "is_gk", "is_def", "is_mid", "is_att"]
    )


def build_squads(matches: pd.DataFrame, players: pd.DataFrame,
                 params: SquadParams = SquadParams()
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Return (squads, mask), row-aligned with `matches`.

    `matches` must be sorted by kickoff. `players` is the player-match table
    from src.data.sportmonks_parse.
    """
    if not matches["kickoff"].is_monotonic_increasing:
        raise ValueError("matches must be sorted by kickoff")

    names = player_feature_names(params)
    F, S = len(names), params.squad_size
    n = len(matches)
    squads = np.zeros((n, 2, S, F), dtype=np.float32)
    mask = np.zeros((n, 2, S), dtype=bool)

    # Rolling history per player: one entry per appearance, most recent last.
    hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=params.window))

    by_fixture = {fid: g for fid, g in players.groupby("sm_fixture_id", sort=False)}
    fixture_ids = matches["sm_fixture_id"].to_numpy()

    for i, fid in enumerate(fixture_ids):
        g = by_fixture.get(fid)
        if g is None:
            continue
        starters = g[g["is_starter"] == 1]

        for side in (1, 0):        # 1 = home flag, side index 0
            idx = 0 if side == 1 else 1
            squad = starters[starters["is_home"] == side]
            squad = squad.sort_values("formation_field", na_position="last")
            for slot, (_, row) in enumerate(squad.iterrows()):
                if slot >= S:
                    break
                past = hist.get(row["player_id"])
                if not past:
                    # Debutant, or first appearance in this corpus. Masked
                    # rather than zero-filled: an all-zero vector would read
                    # as a player who does nothing, which is a strong and
                    # false claim about a player we simply have no data on.
                    continue

                mins = np.array([r["minutes"] for r in past], dtype=float)
                total = np.nansum(mins)
                vec = np.empty(F, dtype=np.float32)
                k = 0
                for c in CORE_COUNTS:
                    vals = np.array([r.get(c, np.nan) for r in past], dtype=float)
                    # per-90 over the whole window, not a mean of per-90s: a
                    # 5-minute cameo should not carry the same weight as a
                    # full match.
                    vec[k] = (np.nansum(vals) / total * 90.0) if total > 0 else 0.0
                    k += 1
                for r_ in CORE_RATES:
                    vals = np.array([r.get(r_, np.nan) for r in past], dtype=float)
                    vec[k] = np.nanmean(vals) if not np.isnan(vals).all() else 0.0
                    k += 1
                vec[k] = len(past); k += 1
                vec[k] = float(np.nanmean(mins)) if len(mins) else 0.0; k += 1
                pg = int(row.get("position_group") or 0)
                vec[k:k + 4] = [pg == 1, pg == 2, pg == 3, pg == 4]

                squads[i, idx, slot] = vec
                mask[i, idx, slot] = True

        # --- absorb this match into player history, AFTER reading it ---
        for _, row in g.iterrows():
            if (row.get("minutes") or 0) >= params.min_minutes:
                hist[row["player_id"]].append(row.to_dict())

    np.nan_to_num(squads, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return squads, mask
