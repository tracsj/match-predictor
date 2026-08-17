"""Tests for the per-team match sequences feeding the recurrent branch.

The one that matters is leak-freeness. A per-team groupby-then-tail would look
perfectly reasonable and silently include matches after the target kickoff, so
the ordering guarantee is asserted directly rather than inferred from a metric.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.sequences import SEQ_FEATURES, SeqParams, build_sequences

IDX = {f: i for i, f in enumerate(SEQ_FEATURES)}


def frame(rows) -> pd.DataFrame:
    """rows = [(home, away, hg, ag)]"""
    df = pd.DataFrame(rows, columns=["home_key", "away_key", "fthg", "ftag"])
    df["country"] = "X"
    df["div"] = "E0"
    df["season"] = "2020-21"
    df["kickoff"] = pd.date_range("2020-08-01", periods=len(df), freq="7D")
    df["elo_home"] = 1500.0
    df["elo_away"] = 1500.0
    df["hst"] = 4.0
    df["ast"] = 3.0
    return df


# --------------------------------------------------------------------------
# Leak-freeness
# --------------------------------------------------------------------------

def test_first_match_has_an_entirely_empty_sequence():
    seq, mask = build_sequences(frame([("a", "b", 3, 0)]))
    assert not mask.any(), "a team with no history must have an empty sequence"
    assert (seq == 0).all()


def test_a_match_never_contains_itself():
    seq, mask = build_sequences(frame([("a", "b", 7, 0)]))
    assert not mask[0].any()
    # And with one prior match, the window holds exactly that one.
    seq, mask = build_sequences(frame([("a", "b", 7, 0), ("a", "c", 0, 0)]))
    assert mask[1, 0].sum() == 1
    assert seq[1, 0, -1, IDX["gf"]] == 7.0


def test_appending_later_matches_cannot_change_an_earlier_sequence():
    """The sharpest statement of the guarantee: the future must not exist."""
    rows = [("a", "b", 2, 0), ("a", "c", 1, 1), ("b", "c", 0, 3)]
    s1, m1 = build_sequences(frame(rows))
    s2, m2 = build_sequences(frame(rows + [("a", "b", 5, 0), ("c", "a", 2, 2)]))
    assert np.array_equal(s1, s2[: len(rows)])
    assert np.array_equal(m1, m2[: len(rows)])


def test_sequence_is_ordered_oldest_to_newest():
    """The recurrent final state must correspond to the most recent match, so
    the newest entry has to sit last."""
    rows = [("a", f"o{i}", i, 0) for i in range(4)] + [("a", "z", 0, 0)]
    seq, mask = build_sequences(frame(rows))
    gf = seq[-1, 0, :, IDX["gf"]][mask[-1, 0]]
    assert list(gf) == [0.0, 1.0, 2.0, 3.0], "expected oldest-first ordering"


def test_window_is_capped_and_keeps_the_most_recent():
    rows = [("a", f"o{i}", i, 0) for i in range(15)] + [("a", "z", 0, 0)]
    seq, mask = build_sequences(frame(rows), SeqParams(seq_len=10))
    assert mask[-1, 0].sum() == 10
    gf = seq[-1, 0, :, IDX["gf"]]
    assert gf[-1] == 14.0, "the newest match must be last"
    assert gf[0] == 5.0, "the window must drop the oldest, not the newest"


# --------------------------------------------------------------------------
# Perspective: each team sees its own view
# --------------------------------------------------------------------------

def test_goals_are_recorded_from_each_teams_own_point_of_view():
    rows = [("a", "b", 3, 1), ("a", "z", 0, 0), ("b", "y", 0, 0)]
    seq, mask = build_sequences(frame(rows))
    # 'a' won 3-1 at home
    assert seq[1, 0, -1, IDX["gf"]] == 3.0
    assert seq[1, 0, -1, IDX["ga"]] == 1.0
    assert seq[1, 0, -1, IDX["points"]] == 3.0
    assert seq[1, 0, -1, IDX["was_home"]] == 1.0
    # 'b' lost 1-3 away -- the same match, mirrored
    assert seq[2, 0, -1, IDX["gf"]] == 1.0
    assert seq[2, 0, -1, IDX["ga"]] == 3.0
    assert seq[2, 0, -1, IDX["points"]] == 0.0
    assert seq[2, 0, -1, IDX["was_home"]] == 0.0


def test_a_draw_scores_one_point_for_both_sides():
    rows = [("a", "b", 2, 2), ("a", "z", 0, 0), ("b", "y", 0, 0)]
    seq, _ = build_sequences(frame(rows))
    assert seq[1, 0, -1, IDX["points"]] == 1.0
    assert seq[2, 0, -1, IDX["points"]] == 1.0


def test_home_and_away_slots_hold_the_right_teams():
    rows = [("a", "x", 4, 0), ("b", "y", 0, 4), ("a", "b", 0, 0)]
    seq, mask = build_sequences(frame(rows))
    assert seq[2, 0, -1, IDX["gf"]] == 4.0, "side 0 must be the home team"
    assert seq[2, 1, -1, IDX["gf"]] == 0.0, "side 1 must be the away team"
    assert seq[2, 1, -1, IDX["ga"]] == 4.0


def test_goal_difference_is_consistent_with_goals():
    rows = [("a", "b", 3, 1), ("a", "z", 0, 0)]
    seq, _ = build_sequences(frame(rows))
    row = seq[1, 0, -1]
    assert row[IDX["gd"]] == row[IDX["gf"]] - row[IDX["ga"]]


# --------------------------------------------------------------------------
# Encoding details
# --------------------------------------------------------------------------

def test_opponent_elo_is_centred_and_scaled():
    df = frame([("a", "b", 1, 0), ("a", "z", 0, 0)])
    df.loc[0, "elo_away"] = 1700.0
    seq, _ = build_sequences(df, SeqParams(elo_centre=1500.0, elo_scale=200.0))
    assert seq[1, 0, -1, IDX["opp_elo_z"]] == pytest.approx(1.0)


def test_days_ago_is_in_weeks_and_capped():
    rows = [("a", "b", 1, 0)] + [("a", f"o{i}", 0, 0) for i in range(1, 3)]
    df = frame(rows)
    df.loc[2, "kickoff"] = df.loc[0, "kickoff"] + pd.Timedelta(days=400)
    seq, _ = build_sequences(df, SeqParams(max_days_ago=120.0))
    # The oldest entry is 400 days back but must clamp to 120/7 weeks.
    assert seq[2, 0, :, IDX["days_ago"]].max() == pytest.approx(120.0 / 7.0)


def test_missing_shot_counts_become_zero_not_nan():
    """Shots are absent before 2000/01 and in every extra-country file. A NaN
    would poison every gradient that touched it."""
    df = frame([("a", "b", 1, 0), ("a", "z", 0, 0)])
    df["hst"] = np.nan
    df["ast"] = np.nan
    seq, _ = build_sequences(df)
    assert np.isfinite(seq).all()
    assert seq[1, 0, -1, IDX["sot_f"]] == 0.0


def test_shapes_and_dtypes():
    seq, mask = build_sequences(frame([("a", "b", 1, 0)] * 5), SeqParams(seq_len=7))
    assert seq.shape == (5, 2, 7, len(SEQ_FEATURES))
    assert mask.shape == (5, 2, 7)
    assert seq.dtype == np.float32
    assert mask.dtype == bool


def test_unsorted_input_is_rejected():
    df = frame([("a", "b", 1, 0)] * 4).iloc[::-1]
    with pytest.raises(ValueError, match="sorted by kickoff"):
        build_sequences(df)


def test_missing_ratings_are_rejected():
    df = frame([("a", "b", 1, 0)]).drop(columns=["elo_home"])
    with pytest.raises(KeyError, match="elo_home"):
        build_sequences(df)


def test_mask_marks_exactly_the_populated_steps():
    rows = [("a", f"o{i}", 1, 0) for i in range(3)] + [("a", "z", 0, 0)]
    seq, mask = build_sequences(frame(rows), SeqParams(seq_len=10))
    assert mask[-1, 0].sum() == 3
    # Padding sits at the FRONT, so the populated steps are the last three.
    assert not mask[-1, 0, :7].any()
    assert mask[-1, 0, 7:].all()
    assert (seq[-1, 0, :7] == 0).all()
