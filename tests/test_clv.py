"""Closing-line value, which the plan makes the headline betting metric.

CLV is the ratio (price you took / closing price of that same selection). Its
value is speed: distinguishing a 2% ROI edge from zero needs tens of thousands
of bets, while Buchdahl detected a genuine ~6-7% edge from 26 tips, because
the ratio predicts realised level-stakes yield with a slope of about 1.00 --
measured over 87,960 pre-close/close pairs.

Two things have to be true for any of that to mean anything here: you must bet
at a PRE-close price and grade at the CLOSE (betting and grading at the same
price makes the ratio identically 1), and the closing price must be looked up
for the exact selection backed, not the row's favourite. The second is v1's
home/away bug in a different costume, so it gets its own test.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.footballdata import OUT_DIR
from src.eval.betting import (
    B365_PRE, CLOSE_FOR, MARKET_MAX_PRE, PINNACLE_CLOSE, PINNACLE_PRE, BetRule,
    clv_report, closing_price_for_bets, simulate,
)
from src.eval.devig import devig

PARQUET = OUT_DIR / "matches.parquet"
needs_data = pytest.mark.skipif(not PARQUET.exists(), reason="matches.parquet not built")


@pytest.fixture(scope="module")
def panel():
    df = pd.read_parquet(PARQUET)
    cols = PINNACLE_PRE.cols + PINNACLE_CLOSE.cols + B365_PRE.cols + MARKET_MAX_PRE.cols
    df = df[(df["source"] == "main")
            & df["season"].between("2016-17", "2024-25")
            & df[cols].notna().all(axis=1)]
    return df.sort_values("kickoff").reset_index(drop=True)


# --------------------------------------------------------------------------
# Hand-built cases: the arithmetic, with no data involved
# --------------------------------------------------------------------------

def _bets(odds, selections, match_ids):
    return pd.DataFrame({
        "match_id": match_ids, "selection": selections, "odds": odds,
        "date": pd.date_range("2024-01-01", periods=len(odds), freq="D"),
    })


def test_clv_ratio_is_taken_over_closing():
    # Took 2.20, closed 2.00 -> ratio 1.10, i.e. beat the close by 10%.
    bets = _bets([2.20], ["H"], ["m1"])
    r = clv_report(bets, pd.Series([2.00]))
    assert r["mean_ratio"] == pytest.approx(1.10)
    assert r["pct_shortened"] == pytest.approx(1.0)


def test_clv_detects_a_price_that_drifted_out():
    # Took 2.00, closed 2.50: the market moved against the bet.
    r = clv_report(_bets([2.00], ["H"], ["m1"]), pd.Series([2.50]))
    assert r["mean_ratio"] == pytest.approx(0.8)
    assert r["pct_shortened"] == pytest.approx(0.0)


def test_clv_of_betting_at_the_close_is_exactly_one():
    # The tautology the pre-close price sets exist to avoid.
    odds = [2.0, 3.4, 5.5, 1.7]
    r = clv_report(_bets(odds, ["H", "D", "A", "H"], list("abcd")), pd.Series(odds))
    assert r["mean_ratio"] == pytest.approx(1.0)
    assert r["pct_shortened"] == pytest.approx(0.0)


def test_clv_ignores_unusable_closing_prices_rather_than_inventing_them():
    r = clv_report(_bets([2.2, 2.2], ["H", "H"], ["a", "b"]), pd.Series([2.0, np.nan]))
    assert r["n"] == 1
    assert r["mean_ratio"] == pytest.approx(1.10)


def test_clv_on_empty_bets_is_not_an_error():
    assert clv_report(pd.DataFrame(), pd.Series(dtype=float))["n"] == 0


# --------------------------------------------------------------------------
# The lookup: right selection, right column
# --------------------------------------------------------------------------

def test_closing_lookup_follows_the_selection_not_the_favourite():
    """v1's bug in a new costume.

    Row m1 is a heavy home favourite. A bet on AWAY must be graded against the
    AWAY closing price (7.00), never the home one (1.40). If this ever picks
    positionally, every CLV number silently becomes noise.
    """
    df = pd.DataFrame({
        "match_id": ["m1", "m2"],
        "psch": [1.40, 3.00], "pscd": [4.50, 3.40], "psca": [7.00, 2.40],
    })
    bets = _bets([8.0, 3.6], ["A", "D"], ["m1", "m2"])
    got = closing_price_for_bets(bets, df)
    assert got.tolist() == [7.00, 3.40]


def test_closing_lookup_maps_every_outcome_distinctly():
    assert set(CLOSE_FOR) == {"H", "D", "A"}
    assert len(set(CLOSE_FOR.values())) == 3


# --------------------------------------------------------------------------
# Against the real corpus
# --------------------------------------------------------------------------

@needs_data
def test_pinnacle_pre_close_beats_its_own_close_about_half_the_time(panel):
    """The control with a known answer.

    Pinnacle's own pre-close price against Pinnacle's close is the closest
    thing to a null CLV distribution: no selection skill, just market drift
    between Friday and kickoff. The mean ratio must sit near 1, and the
    shortening rate near half. A large deviation means the pre-close and close
    columns are not the same market.
    """
    p = devig(panel[PINNACLE_PRE.cols].to_numpy(float), method="shin")
    bets = simulate(panel, p, PINNACLE_PRE,
                    BetRule(min_ev=-1.0, min_odds=1.0, max_odds=1e6, name="all"))
    close = closing_price_for_bets(bets, panel)
    r = clv_report(bets, close)

    assert r["n"] > 40_000
    assert r["mean_ratio"] == pytest.approx(1.0, abs=0.03)
    assert 0.30 < r["pct_shortened"] < 0.70
    # and there IS real spread -- the two columns are not duplicates
    assert (bets["odds"].to_numpy() != close.to_numpy()).mean() > 0.5


@needs_data
def test_taking_the_best_pre_close_price_produces_positive_clv(panel):
    """Backing the market-max pre-close price against the Pinnacle close must
    show CLV above 1 -- you are by construction taking the best price on offer
    and grading against a sharp consensus.

    This is the mechanism behind the Kaunitz et al. result, and seeing it in
    our own data is what tells us the CLV machinery is wired up correctly. It
    is a property of price shopping, not of any forecast.
    """
    p = devig(panel[PINNACLE_PRE.cols].to_numpy(float), method="shin")
    bets = simulate(panel, p, MARKET_MAX_PRE,
                    BetRule(min_ev=-1.0, min_odds=1.0, max_odds=1e6, name="all"))
    r = clv_report(bets, closing_price_for_bets(bets, panel))

    assert r["mean_ratio"] > 1.01, "best available price should beat the sharp close"
    assert r["pct_shortened"] > 0.5
    assert r["binom_pvalue"] < 1e-6


@needs_data
def test_clv_and_roi_point_the_same_way_on_the_price_shopping_strategy(panel):
    """The claim CLV rests on: beating the close predicts realised yield.

    The same rule should show both positive CLV and positive ROI. If they
    disagreed here, one of the two pipelines would be wrong -- this is the
    cross-check that neither is.
    """
    p = devig(panel[PINNACLE_CLOSE.cols].to_numpy(float), method="shin")
    bets = simulate(panel, p, MARKET_MAX_PRE, BetRule(min_ev=0.05))
    assert len(bets) > 1000

    roi = bets["pnl"].sum() / bets["stake"].sum()
    r = clv_report(bets, closing_price_for_bets(bets, panel))
    assert r["mean_ratio"] > 1.0
    assert roi > 0.0


@needs_data
def test_home_away_regression_test(panel):
    """The named test from the plan. This is v1's exact defect.

    Five v1 scripts assumed the first-listed team was home; measured, it was
    home in 407 of 640 fixtures (63.6%), so roughly a third of simulated bets
    were graded against the wrong side's price.

    The invariant, stated without reference to any code: when the bookmaker
    prices the HOME side shorter than the away side, the home team must win
    more often than when it prices the home side longer. If home and away were
    transposed anywhere between the CSV and the simulator, this inverts.
    """
    home_fav = panel[panel["psch"] < panel["psca"]]
    away_fav = panel[panel["psch"] > panel["psca"]]
    assert len(home_fav) > 10_000 and len(away_fav) > 10_000

    h_when_home_fav = (home_fav["result"] == "H").mean()
    h_when_away_fav = (away_fav["result"] == "H").mean()
    assert h_when_home_fav > h_when_away_fav + 0.20, (
        f"home win rate {h_when_home_fav:.3f} when home is favourite vs "
        f"{h_when_away_fav:.3f} when away is -- home/away may be transposed")

    # And the mirror, so a symmetric transposition cannot pass both.
    a_when_away_fav = (away_fav["result"] == "A").mean()
    a_when_home_fav = (home_fav["result"] == "A").mean()
    assert a_when_away_fav > a_when_home_fav + 0.20


@needs_data
def test_bets_are_graded_against_the_price_of_the_side_backed(panel):
    """End-to-end version of the same guard, through the simulator.

    Reconstruct each bet's payout by hand from the raw frame and check it
    matches what simulate() recorded. A positional slip would show up as a
    payout computed from the wrong column.
    """
    p = devig(panel[PINNACLE_CLOSE.cols].to_numpy(float), method="shin")
    bets = simulate(panel.head(5000), p[:5000], PINNACLE_CLOSE,
                    BetRule(min_ev=-1.0, min_odds=1.0, max_odds=1e6, name="all"))
    lookup = panel.head(5000).set_index("match_id")
    for row in bets.head(400).itertuples():
        expected_odds = lookup.loc[row.match_id, CLOSE_FOR[row.selection]]
        assert row.odds == pytest.approx(expected_odds)
        won = lookup.loc[row.match_id, "result"] == row.selection
        assert row.won == won
        assert row.pnl == pytest.approx(expected_odds - 1 if won else -1.0)
