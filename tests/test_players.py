"""Tests for the squad tensors feeding the player-set encoder.

Two things carry the weight here. Leak-freeness, as everywhere else. And the
statistic-availability audit: 16 of SportMonks' 37 per-player statistics are
collected in some seasons and written as literal ZERO in others, and a zero
looks exactly like data.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.sportmonks_parse import (
    COUNT_STATS, MATCHES_PARQUET, PLAYERS_PARQUET, RATE_STATS,
)
from src.features.players import (
    CORE_COUNTS, CORE_RATES, MAX_SEASON_ZERO_SPREAD, SquadParams, build_squads,
    player_feature_names,
)

HAS_DATA = MATCHES_PARQUET.exists() and PLAYERS_PARQUET.exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="SportMonks not parsed yet")
NAMES = player_feature_names()


def toy(n_matches=6, n_players=11):
    """Two teams of fixed players, playing each other repeatedly."""
    matches, players = [], []
    for i in range(n_matches):
        fid = 1000 + i
        matches.append({
            "sm_fixture_id": fid,
            "kickoff": pd.Timestamp("2021-01-01") + pd.Timedelta(days=7 * i),
            "home_key": "alpha", "away_key": "beta", "div": "SC0",
            "fthg": 1, "ftag": 0, "result": "H", "season_id": 1,
        })
        for team, is_home, base in ((100, 1, 0), (200, 0, 100)):
            for j in range(n_players):
                players.append({
                    "sm_fixture_id": fid, "player_id": base + j,
                    "kickoff": matches[-1]["kickoff"],
                    "team_id": team, "is_home": is_home, "is_starter": 1,
                    "formation_field": f"{1 + j % 4}:{j}",
                    "position_group": 1 + j % 4,
                    "minutes": 90.0,
                    **{c: float(j) for c in COUNT_STATS},
                    **{r: 7.0 for r in RATE_STATS},
                })
    return (pd.DataFrame(matches).sort_values("kickoff").reset_index(drop=True),
            pd.DataFrame(players))


# --------------------------------------------------------------------------
# Leak-freeness
# --------------------------------------------------------------------------

def test_first_appearance_is_masked_out_not_zero_filled():
    """A debutant and a player who does nothing are different claims. An
    all-zero vector asserts the second when we only know the first."""
    m, p = toy(n_matches=1)
    sq, mask = build_squads(m, p)
    assert not mask.any()
    assert (sq == 0).all()


def test_a_players_own_match_never_enters_their_vector():
    m, p = toy(n_matches=2)
    sq, mask = build_squads(m, p)
    assert not mask[0].any(), "nobody has history at the first fixture"
    assert mask[1].all(), "everyone has exactly one prior appearance by the second"
    assert sq[1, 0, 0, NAMES.index("hist_matches")] == 1.0


def test_appending_later_matches_cannot_change_an_earlier_squad():
    m3, p3 = toy(n_matches=3)
    m5, p5 = toy(n_matches=5)
    s3, k3 = build_squads(m3, p3)
    s5, k5 = build_squads(m5, p5)
    assert np.array_equal(s3, s5[:3])
    assert np.array_equal(k3, k5[:3])


def test_history_accumulates_up_to_the_window_then_stops():
    m, p = toy(n_matches=14)
    sq, mask = build_squads(m, p, SquadParams(window=10))
    hi = NAMES.index("hist_matches")
    assert sq[5, 0, 0, hi] == 5.0
    assert sq[13, 0, 0, hi] == 10.0, "window must cap the history"


# --------------------------------------------------------------------------
# Shape and encoding
# --------------------------------------------------------------------------

def test_shapes_and_sides():
    m, p = toy(n_matches=4)
    sq, mask = build_squads(m, p)
    assert sq.shape == (4, 2, 11, len(NAMES))
    assert mask.shape == (4, 2, 11)
    assert sq.dtype == np.float32


def test_side_zero_is_home():
    m, p = toy(n_matches=3)
    # The toy fixture already gives every player goals = their index (0..10),
    # so the marker has to sit clear of that range to be unambiguous.
    p.loc[p.is_home == 0, "goals"] = 50.0
    p.loc[p.is_home == 1, "goals"] = 1.0
    sq, _ = build_squads(m, p)
    gi = NAMES.index("p90_goals")
    assert sq[2, 0, :, gi].max() == pytest.approx(1.0), "side 0 must be home"
    assert sq[2, 1, :, gi].max() == pytest.approx(50.0), "side 1 must be away"


def test_per_90_is_computed_over_total_minutes_not_as_a_mean_of_ratios():
    """A five-minute cameo must not carry the same weight as a full match."""
    m, p = toy(n_matches=3, n_players=1)
    p.loc[(p.sm_fixture_id == 1000) & (p.player_id == 0), ["minutes", "goals"]] = [5.0, 1.0]
    p.loc[(p.sm_fixture_id == 1001) & (p.player_id == 0), ["minutes", "goals"]] = [90.0, 0.0]
    sq, mask = build_squads(m, p)
    gi = NAMES.index("p90_goals")
    # 1 goal in 95 total minutes = 0.947 per 90. A mean of per-90s would give
    # (18.0 + 0.0) / 2 = 9.0, which is nonsense.
    assert sq[2, 0, 0, gi] == pytest.approx(1 / 95 * 90, abs=1e-4)


def test_position_group_is_one_hot():
    m, p = toy(n_matches=3)
    sq, mask = build_squads(m, p)
    cols = [NAMES.index(c) for c in ("is_gk", "is_def", "is_mid", "is_att")]
    filled = sq[2, 0][mask[2, 0]]
    assert set(np.unique(filled[:, cols])) <= {0.0, 1.0}
    assert (filled[:, cols].sum(axis=1) == 1).all()


def test_unsorted_matches_are_rejected():
    m, p = toy(n_matches=4)
    with pytest.raises(ValueError, match="sorted by kickoff"):
        build_squads(m.iloc[::-1], p)


def test_no_nan_or_inf_reaches_the_tensor():
    m, p = toy(n_matches=4)
    p.loc[p.index[:20], "rating"] = np.nan
    sq, _ = build_squads(m, p)
    assert np.isfinite(sq).all()


# --------------------------------------------------------------------------
# The statistic-availability audit
# --------------------------------------------------------------------------

def test_core_stats_are_a_subset_of_what_the_parser_produces():
    assert set(CORE_COUNTS) <= set(COUNT_STATS)
    assert set(CORE_RATES) <= set(RATE_STATS)


def test_the_known_collection_gaps_are_excluded():
    """Regression test for a real trap.

    `touches` is 100% zero in 9 of 11 seasons and averages 56.7 per full match
    in the other two -- a collection gap written as data. Including it would
    teach the model that players before 2024 never touched the ball.
    """
    for stat in ("touches", "possession_lost", "ball_recovery", "long_balls",
                 "tackles_won", "aerials_won", "clearances", "duels_won"):
        assert stat not in CORE_COUNTS, f"{stat} is a known collection gap"
    for stat in ("aerials_won_pct", "tackles_won_pct", "long_balls_won_pct"):
        assert stat not in CORE_RATES, f"{stat} is a known collection gap"


@needs_data
def test_the_safe_list_is_still_correct_against_the_real_data():
    """Re-derives the audit rather than trusting the hardcoded list.

    If SportMonks starts collecting a statistic it previously skipped, or a new
    fetch changes the season mix, this fails and the list needs revisiting --
    which is better than the list silently going stale.

    The discriminator is the SPREAD of the per-season zero-rate, not its level.
    A genuinely rare event is rare consistently (goals: 0.067); a collection
    gap swings from near-zero to total (touches: 1.000).
    """
    players = pd.read_parquet(PLAYERS_PARQUET)
    matches = pd.read_parquet(MATCHES_PARQUET)[["sm_fixture_id", "season_id"]]
    pl = players[players["minutes"].fillna(0) >= 60].merge(matches, on="sm_fixture_id")

    for stat in CORE_COUNTS + CORE_RATES:
        per = pl.groupby("season_id")[stat].apply(lambda s: (s.fillna(0) == 0).mean())
        spread = float(per.max() - per.min())
        assert spread <= MAX_SEASON_ZERO_SPREAD, (
            f"{stat} zero-rate spans {spread:.3f} across seasons -- it looks "
            "like a collection gap and should leave CORE_*")


@needs_data
def test_real_squads_carry_football_like_numbers():
    m = pd.read_parquet(MATCHES_PARQUET).sort_values("kickoff").reset_index(drop=True)
    p = pd.read_parquet(PLAYERS_PARQUET)
    sq, mask = build_squads(m, p)
    flat = sq.reshape(-1, sq.shape[-1])[mask.reshape(-1)]

    def mean_of(name):
        return float(flat[:, NAMES.index(name)].mean())

    assert 0.05 < mean_of("p90_goals") < 0.30, "goals per 90 out of range"
    assert 20 < mean_of("p90_passes") < 70, "passes per 90 out of range"
    assert 6.0 < mean_of("avg_rating") < 8.0, "SportMonks rating out of range"
    assert mask.mean() > 0.90, "too many players lack usable history"


@needs_data
def test_most_matches_have_a_full_starting_eleven():
    m = pd.read_parquet(MATCHES_PARQUET).sort_values("kickoff").reset_index(drop=True)
    p = pd.read_parquet(PLAYERS_PARQUET)
    _, mask = build_squads(m, p)
    full_home = (mask[:, 0].sum(axis=1) == 11).mean()
    assert full_home > 0.80, f"only {full_home:.1%} of matches have a full home XI"
