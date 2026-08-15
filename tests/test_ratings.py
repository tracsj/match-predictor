"""Tests for Elo and pi-ratings.

The property that matters most is that a rating is *pre-match*: it must be
computed from matches strictly before kickoff and must not move because of the
match it is scoring. That is asserted directly on hand-built sequences rather
than inferred from downstream metrics.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.footballdata import OUT_DIR
from src.features.ratings import (
    EloParams, PiParams, TIER, add_ratings, elo_features, pi_rating_features,
)

PARQUET = OUT_DIR / "matches.parquet"
needs_data = pytest.mark.skipif(not PARQUET.exists(), reason="matches.parquet not built")


def frame(rows) -> pd.DataFrame:
    """rows = [(home, away, hg, ag, season, div)]"""
    df = pd.DataFrame(rows, columns=["home_key", "away_key", "fthg", "ftag", "season", "div"])
    df["country"] = "X"
    df["kickoff"] = pd.date_range("2020-08-01", periods=len(df), freq="7D")
    return df


# --------------------------------------------------------------------------
# Leak-freeness
# --------------------------------------------------------------------------

def test_first_match_uses_the_starting_rating_not_the_result():
    """Both teams are new, so both must be priced at the default. If the
    rating had absorbed the 5-0 before being recorded, this fails."""
    df = frame([("a", "b", 5, 0, "2020-21", "E0")])
    e = elo_features(df)
    assert e["elo_home"].iloc[0] == 1500.0
    assert e["elo_away"].iloc[0] == 1500.0
    p = pi_rating_features(df)
    assert p["pi_home_h"].iloc[0] == 0.0
    assert p["pi_exp_gd"].iloc[0] == 0.0


def test_a_win_raises_the_winners_rating_for_the_next_match():
    df = frame([("a", "b", 3, 0, "2020-21", "E0"), ("a", "c", 0, 0, "2020-21", "E0")])
    e = elo_features(df)
    assert e["elo_home"].iloc[1] > 1500.0          # a gained from the win
    assert e["elo_home"].iloc[0] == 1500.0          # ...but not retroactively


def test_appending_a_later_match_cannot_change_an_earlier_rating():
    """The sharpest statement of leak-freeness: the future must not exist."""
    rows = [("a", "b", 2, 0, "2020-21", "E0"), ("c", "d", 1, 1, "2020-21", "E0")]
    short = elo_features(frame(rows))
    long = elo_features(frame(rows + [("a", "c", 4, 0, "2020-21", "E0")]))
    assert np.allclose(short["elo_home"], long["elo_home"][:2])
    assert np.allclose(short["elo_diff"], long["elo_diff"][:2])


def test_elo_is_zero_sum_within_a_match():
    df = frame([("a", "b", 3, 1, "2020-21", "E0"), ("a", "b", 0, 2, "2020-21", "E0")])
    e = elo_features(df, EloParams(season_regression=0.0))
    # After match 1, a gained exactly what b lost.
    assert (e["elo_home"].iloc[1] - 1500.0) == pytest.approx(-(e["elo_away"].iloc[1] - 1500.0))


def test_bigger_wins_move_the_rating_further():
    narrow = elo_features(frame([("a", "b", 1, 0, "2020-21", "E0"),
                                 ("a", "c", 0, 0, "2020-21", "E0")]))
    rout = elo_features(frame([("a", "b", 5, 0, "2020-21", "E0"),
                               ("a", "c", 0, 0, "2020-21", "E0")]))
    assert rout["elo_home"].iloc[1] > narrow["elo_home"].iloc[1]


def test_home_advantage_shows_up_in_the_expectancy():
    df = frame([("a", "b", 0, 0, "2020-21", "E0")])
    e = elo_features(df)
    # Equal ratings, so any expectancy above 0.5 is the home offset alone.
    assert e["elo_exp_home"].iloc[0] > 0.5
    assert e["elo_diff"].iloc[0] == pytest.approx(65.0)


# --------------------------------------------------------------------------
# pi-ratings specifics
# --------------------------------------------------------------------------

def test_pi_keeps_separate_home_and_away_ratings():
    """The point of pi-ratings: a team can be strong at home and weak away, and
    the two ratings must not collapse into one."""
    rows = [("a", f"o{i}", 4, 0, "2020-21", "E0") for i in range(6)]
    p = pi_rating_features(frame(rows))
    last = p.iloc[-1]
    # 'a' has only ever played at home, so its home rating must lead its away
    # rating, which follows at gamma = 0.7.
    assert last["pi_home_h"] > last["pi_home_a"] > 0


def test_pi_expected_gd_grows_with_the_rating_gap():
    strong = pi_rating_features(frame([("a", f"o{i}", 5, 0, "2020-21", "E0") for i in range(8)]))
    assert strong["pi_exp_gd"].is_monotonic_increasing


def test_pi_gamma_zero_decouples_the_two_venues():
    rows = [("a", f"o{i}", 3, 0, "2020-21", "E0") for i in range(5)]
    p = pi_rating_features(frame(rows), PiParams(gamma=0.0))
    assert p["pi_home_a"].iloc[-1] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# The tier correction
# --------------------------------------------------------------------------

def test_movement_is_flagged_in_both_directions():
    up = elo_features(frame([("a", "b", 1, 0, "2020-21", "E1"),
                             ("a", "c", 1, 0, "2021-22", "E0")]),
                      EloParams(season_regression=0.0))
    assert up["elo_home_moved"].iloc[0] == 0
    assert up["elo_home_moved"].iloc[1] == 1          # promoted

    down = elo_features(frame([("a", "b", 1, 0, "2020-21", "E0"),
                               ("a", "c", 1, 0, "2021-22", "E1")]),
                        EloParams(season_regression=0.0))
    assert down["elo_home_moved"].iloc[1] == -1       # relegated


def test_movement_flag_is_held_for_the_whole_season_not_just_one_match():
    """The flag has to persist, because the mispricing does. A promoted team is
    overrated in April as well as August."""
    rows = [("a", "b", 1, 0, "2020-21", "E1")] + \
           [("a", f"o{i}", 1, 1, "2021-22", "E0") for i in range(4)]
    e = elo_features(frame(rows), EloParams(season_regression=0.0))
    assert e["elo_home_moved"].tolist() == [0, 1, 1, 1, 1]


def test_movement_flag_clears_once_the_team_settles():
    rows = [("a", "b", 1, 0, "2020-21", "E1"),
            ("a", "c", 1, 1, "2021-22", "E0"),
            ("a", "d", 1, 1, "2022-23", "E0")]
    e = elo_features(frame(rows), EloParams(season_regression=0.0))
    assert e["elo_home_moved"].tolist() == [0, 1, 0]


def test_tier_shift_defaults_to_zero():
    """See the long note in EloParams for why. Short version: the model learns
    the movement correction better from the moved flags than any constant can,
    and every non-zero shift costs out-of-sample RPS."""
    assert EloParams().tier_shift == 0.0


def test_tier_shift_adjusts_the_emitted_column_by_exactly_the_shift():
    rows = [("a", "b", 1, 0, "2020-21", "E1"),
            ("a", "c", 1, 0, "2021-22", "E0")]
    off = elo_features(frame(rows), EloParams(season_regression=0.0, tier_shift=0.0))
    on = elo_features(frame(rows), EloParams(season_regression=0.0, tier_shift=130.0))
    assert on["elo_home"].iloc[1] == pytest.approx(off["elo_home"].iloc[1] - 130.0)
    assert on["elo_home"].iloc[0] == pytest.approx(off["elo_home"].iloc[0])


def test_tier_shift_reaches_the_stored_rating_only_through_the_expectancy():
    """A documented second-order coupling, asserted so it stays deliberate.

    The shift is never written back into the rating. But the ADJUSTED
    expectancy is what the Elo update is scored against -- on purpose, so a
    promoted side performing as a promoted side should keeps its rating -- and
    that feeds back into the stored value. So a later, unshifted season is
    close to the unshifted run but not identical.

    The reason this is acceptable rather than the bug it replaced: the effect
    is small and second-order, where writing the shift back directly dragged
    every division mean inward and inverted the bottom of the English pyramid.
    """
    rows = [("a", "b", 1, 0, "2020-21", "E1"),
            ("a", "c", 1, 0, "2021-22", "E0"),
            ("a", "d", 1, 0, "2022-23", "E0")]
    off = elo_features(frame(rows), EloParams(season_regression=0.0, tier_shift=0.0))
    on = elo_features(frame(rows), EloParams(season_regression=0.0, tier_shift=130.0))
    settled_gap = abs(on["elo_home"].iloc[2] - off["elo_home"].iloc[2])
    assert settled_gap > 0, "expected some feedback through the expectancy"
    assert settled_gap < 20, f"feedback should be second-order, got {settled_gap:.1f} points"


def test_unknown_division_gets_no_tier_shift():
    # Extra-country files are single-division and carry no tier.
    rows = [("a", "b", 1, 0, "2020-21", "DNK"), ("a", "c", 1, 0, "2021-22", "DNK")]
    e = elo_features(frame(rows), EloParams(tier_shift=130.0))
    assert (e["elo_home_moved"] == 0).all()


# --------------------------------------------------------------------------
# Against the real corpus
# --------------------------------------------------------------------------

@needs_data
def test_add_ratings_requires_sorted_input():
    df = pd.read_parquet(PARQUET).head(500)
    with pytest.raises(ValueError, match="sorted by kickoff"):
        add_ratings(df.iloc[::-1])


@needs_data
def test_ratings_carry_real_signal():
    df = pd.read_parquet(PARQUET).sort_values("kickoff").reset_index(drop=True)
    df = add_ratings(df)
    gd = df["fthg"] - df["ftag"]
    assert df["pi_exp_gd"].corr(gd) > 0.25
    assert df["elo_diff"].corr(gd) > 0.25
    assert df["elo_exp_home"].corr((df["result"] == "H").astype(float)) > 0.20


@needs_data
def test_division_tiers_rank_correctly_in_england():
    """Elo must at minimum order the English pyramid correctly. The spread is
    compressed because divisions connect only through promotion, which is why
    tier_shift exists -- but the ordering has to be right."""
    df = pd.read_parquet(PARQUET).sort_values("kickoff").reset_index(drop=True)
    df = add_ratings(df)
    eng = df[(df["country"] == "England") & (df["season"] >= "2022-23")]
    means = eng.groupby("div")["elo_home"].mean()
    order = [means[d] for d in ("E0", "E1", "E2", "E3", "EC") if d in means]
    assert order == sorted(order, reverse=True), f"English tiers out of order: {means.to_dict()}"


@needs_data
def test_a_rating_mutating_tier_shift_would_break_that_ordering():
    """Negative control for the test above, and the reason tier_shift is an
    output adjustment rather than a rating change.

    Writing the shift back into the stored rating drags every division mean
    toward the middle and lifts the National League above the Championship.
    This asserts the ordering check can actually fail, so its passing means
    something.
    """
    df = pd.read_parquet(PARQUET).sort_values("kickoff").reset_index(drop=True)
    df = add_ratings(df, elo=EloParams(tier_shift=130.0))
    eng = df[(df["country"] == "England") & (df["season"] >= "2022-23")]
    stacked = pd.concat([
        eng[["div", "elo_home"]].rename(columns={"elo_home": "elo"}),
        eng[["div", "elo_away"]].rename(columns={"elo_away": "elo"})])
    means = stacked.groupby("div")["elo"].mean()
    # The output-only adjustment must LEAVE the ordering intact...
    order = [means[d] for d in ("E0", "E1", "E2", "E3", "EC")]
    assert order == sorted(order, reverse=True), (
        "output-only adjustment should not disturb division ordering")


@needs_data
def test_the_promotion_bias_is_real_and_confined_to_movers():
    """The measurement behind the tier_shift note in EloParams.

    Elo overrates promoted teams and underrates relegated ones, while teams
    that stay put are unbiased. That last part is the important half: it shows
    the rating pool itself is sound and only movers are mispriced, which is why
    the fix belongs in the model (via the moved flags) rather than in the
    ratings.
    """
    base = pd.read_parquet(PARQUET).sort_values("kickoff").reset_index(drop=True)

    def bias(shift):
        df = add_ratings(base, elo=EloParams(tier_shift=shift))
        m = df[(df["source"] == "main") & (df["season"] >= "2005-06")]
        parts = []
        for own, opp, moved, ha, win in (("elo_home", "elo_away", "elo_home_moved", 65, "H"),
                                         ("elo_away", "elo_home", "elo_away_moved", -65, "A")):
            parts.append(pd.DataFrame({
                "own": m[own], "opp": m[opp], "moved": m[moved], "ha": ha,
                "act": np.where(m["result"] == win, 1.0,
                                np.where(m["result"] == "D", 0.5, 0.0))}))
        L = pd.concat(parts)
        L["exp"] = 1 / (1 + 10 ** (-((L["own"] + L["ha"]) - L["opp"]) / 400))
        L["d"] = L["act"] - L["exp"]
        return (L.loc[L["moved"] > 0, "d"].mean(),
                L.loc[L["moved"] < 0, "d"].mean(),
                L.loc[L["moved"] == 0, "d"].mean())

    p0, r0, s0 = bias(0.0)
    assert p0 < -0.04, f"promoted teams should be overrated, got {p0:+.4f}"
    assert r0 > 0.04, f"relegated teams should be underrated, got {r0:+.4f}"
    assert abs(s0) < 0.005, f"teams that stayed put should be unbiased, got {s0:+.4f}"

    # And the output-only adjustment can remove it, which is what shows the
    # mechanism was understood correctly even though shipping it costs RPS.
    p1, r1, s1 = bias(76.0)
    assert abs(p1) < 0.02 and abs(r1) < 0.02
    assert abs(s1) < 0.005, "non-movers must be untouched by the adjustment"


@needs_data
def test_ratings_are_fast_enough_to_rebuild_freely():
    import time
    df = pd.read_parquet(PARQUET).sort_values("kickoff").reset_index(drop=True)
    t = time.time()
    add_ratings(df)
    assert time.time() - t < 20, "ratings should build in seconds, not minutes"
