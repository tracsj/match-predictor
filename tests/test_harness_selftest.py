"""Harness self-test: can this thing tell good from bad, and cheating from real?

The plan gates every model on this file. A backtest that cannot detect a
look-ahead cheater cannot validate an honest model either, and the failure is
silent -- leakage shows up as a model that looks unusually good, which is the
one result nobody interrogates.

Four probe models, in known order of merit:

    cheater    sees the result. Must look impossible.
    market     de-vigged Pinnacle closing. The ceiling for anything honest.
    base_rate  training-set class frequencies. A real, weak model.
    uniform    (1/3, 1/3, 1/3). The floor.

Expected ordering on RPS:  cheater << market < base_rate < uniform

Every threshold below is a number derived from theory or from published
measurement, written here independently -- never read off a run of this code.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.footballdata import OUT_DIR
from src.eval.betting import (
    PINNACLE_CLOSE, BetRule, bootstrap_ci, random_bet_null,
    required_sample_size, simulate, summarize,
)
from src.eval.devig import devig
from src.eval.metrics import OUTCOMES, log_loss, rps, rps_per_match
from src.eval.split import assert_no_leakage, season_walk_forward

PARQUET = OUT_DIR / "matches.parquet"
pytestmark = pytest.mark.skipif(
    not PARQUET.exists(), reason="matches.parquet not built; run build_matches()"
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """Matches with a complete Pinnacle closing price, seasons 2016-17..2024-25.

    Stops at 2024-25 deliberately: Pinnacle closing coverage decays from
    October 2025 and is gone from February 2026, so 2025-26 cannot supply the
    sharp benchmark. See docs/research/00-measured-facts.md.
    """
    df = pd.read_parquet(PARQUET)
    df = df[(df["source"] == "main")
            & df["season"].between("2016-17", "2024-25")
            & df[PINNACLE_CLOSE.cols].notna().all(axis=1)]
    return df.sort_values("kickoff").reset_index(drop=True)


@pytest.fixture(scope="module")
def probs(panel) -> dict[str, np.ndarray]:
    n = len(panel)
    y = panel["result"].to_numpy()

    uniform = np.full((n, 3), 1 / 3)

    base = np.array([(y == o).mean() for o in OUTCOMES])
    base_rate = np.tile(base / base.sum(), (n, 1))

    market = devig(panel[PINNACLE_CLOSE.cols].to_numpy(float), method="shin")

    cheater = np.full((n, 3), 0.001)
    cheater[np.arange(n), [OUTCOMES.index(v) for v in y]] = 0.998

    return {"uniform": uniform, "base_rate": base_rate,
            "market": market, "cheater": cheater}


# --------------------------------------------------------------------------
# Does the scoreboard rank models correctly?
# --------------------------------------------------------------------------

def test_panel_is_big_enough_to_mean_anything(panel):
    assert len(panel) > 50_000
    assert panel["season"].nunique() == 9


def test_uniform_log_loss_is_exactly_ln3(probs, panel):
    # -ln(1/3) = 1.0986122886681098, independent of the outcome distribution.
    assert log_loss(probs["uniform"], panel["result"]) == pytest.approx(1.0986122886681098)


def test_uniform_rps_matches_the_base_rates(probs, panel):
    # For a uniform forecast, RPS is 5/18 on a home win, 1/9 on a draw, 5/18
    # on an away win. Expected value = (5/18)(pH + pA) + (1/9)pD, computed
    # from the observed frequencies rather than from the metric under test.
    y = panel["result"]
    pH, pD, pA = [(y == o).mean() for o in OUTCOMES]
    expected = (5 / 18) * (pH + pA) + (1 / 9) * pD
    assert rps(probs["uniform"], y) == pytest.approx(expected, abs=1e-9)


def test_the_market_lands_where_the_literature_says_it_should(probs, panel):
    """Bookmaker closing odds score RPS ~0.19-0.21 and log loss ~0.96-0.99.

    If this fails, the pipeline is wrong -- not the model. Published anchors:
    market RPS 0.1905 on 19 Serie A seasons (Pitcan 2026); bookmaker consensus
    0.2063 on the 2023 Soccer Prediction Challenge (Yeung et al. 2024).
    """
    r = rps(probs["market"], panel["result"])
    ll = log_loss(probs["market"], panel["result"])
    assert 0.185 < r < 0.215, f"market RPS {r:.4f} outside the published band"
    assert 0.94 < ll < 1.00, f"market log loss {ll:.4f} outside the published band"


def test_models_rank_in_the_known_order(probs, panel):
    y = panel["result"]
    scores = {k: rps(v, y) for k, v in probs.items()}
    assert scores["cheater"] < scores["market"] < scores["base_rate"] < scores["uniform"], scores


def test_cheater_is_flagged_as_impossible(probs, panel):
    """The whole point of this file.

    Nothing honest gets near this. The best published deep model scored RPS
    0.2195 and the market 0.2063; a score below 0.05 is not a good model, it
    is a leak.
    """
    r = rps(probs["cheater"], panel["result"])
    assert r < 0.01, f"cheater RPS {r:.4f} -- the cheater is not cheating properly"
    market = rps(probs["market"], panel["result"])
    assert r < market / 10, "harness cannot separate a leak from a good model"


def test_base_rate_beats_uniform_but_only_slightly(probs, panel):
    """The total learnable signal on this task is about 0.1 nats. Knowing the
    class frequencies should help a little and nowhere near enough."""
    y = panel["result"]
    gain = rps(probs["uniform"], y) - rps(probs["base_rate"], y)
    assert 0.0 < gain < 0.02, f"base-rate gain {gain:.4f} is implausible"


# --------------------------------------------------------------------------
# Does the betting simulator behave?
# --------------------------------------------------------------------------

def test_cheater_makes_absurd_money(probs, panel):
    bets = simulate(panel, probs["cheater"], PINNACLE_CLOSE, BetRule())
    s = summarize(bets, with_ci=False)
    assert s["hit_rate"] > 0.99
    assert s["roi"] > 0.5, "a model that knows the answer must print money"


def test_market_bet_against_its_own_price_places_no_bets(panel):
    """A self-consistency check with a provable answer.

    Under multiplicative de-vigging, p_i = q_i / sum(q), so
    p_i * d_i = 1 / sum(q) for EVERY outcome -- a constant, equal to the
    reciprocal of the overround. With a ~3% margin that is EV = -0.029 on all
    three, so a rule demanding EV >= +0.05 must place exactly zero bets.

    If this ever places a bet, the de-vigger, the price columns, or the EV
    arithmetic is wrong.
    """
    p = devig(panel[PINNACLE_CLOSE.cols].to_numpy(float), method="multiplicative")
    odds = panel[PINNACLE_CLOSE.cols].to_numpy(float)
    ev = p * odds - 1.0
    # every outcome in a row carries the same EV, and it is negative
    assert np.allclose(ev.max(axis=1), ev.min(axis=1), atol=1e-9)
    assert ev.max() < 0.0

    bets = simulate(panel, p, PINNACLE_CLOSE, BetRule(min_ev=0.05))
    assert len(bets) == 0


def test_forced_market_betting_loses_approximately_the_vig(panel):
    """Same setup, but bet everything. The loss must equal the margin.

    With multiplicative de-vig the EV on every selection is 1/overround - 1,
    so flat-staking every match has an expected ROI of exactly that. Observed
    ROI is noisy around it, but must land close over 50k+ matches.
    """
    p = devig(panel[PINNACLE_CLOSE.cols].to_numpy(float), method="multiplicative")
    odds = panel[PINNACLE_CLOSE.cols].to_numpy(float)
    overround = (1 / odds).sum(axis=1)
    expected_roi = float((1 / overround - 1).mean())

    bets = simulate(panel, p, PINNACLE_CLOSE, BetRule(min_ev=-1.0, min_odds=1.0, max_odds=1000))
    observed = bets["pnl"].sum() / bets["stake"].sum()
    assert observed == pytest.approx(expected_roi, abs=0.02), (
        f"forced market betting returned {observed:.4f}, expected ~{expected_roi:.4f}")
    assert -0.06 < expected_roi < -0.01, "Pinnacle 1X2 margin should be ~2-5%"


def test_random_betting_loses_and_the_null_says_so(panel):
    null = random_bet_null(panel, PINNACLE_CLOSE, n_bets=2000, n_sims=300, seed=1)
    assert null["mean_roi"] < 0, "random betting into a margin must lose"
    assert null["hi"] > null["mean_roi"] > null["lo"]


def test_bootstrap_ci_brackets_the_point_estimate(probs, panel):
    bets = simulate(panel, probs["cheater"], PINNACLE_CLOSE, BetRule())
    ci = bootstrap_ci(bets, n_boot=400, seed=0)
    assert ci["lo"] <= ci["roi"] <= ci["hi"]
    assert ci["n_blocks"] > 100


def test_bootstrap_ci_on_a_fair_coin_covers_zero():
    """A strategy with no edge must produce an interval containing zero.

    Constructed by hand: even-money bets won exactly half the time. True ROI
    is 0, and the interval must not exclude it.
    """
    rng = np.random.default_rng(7)
    n = 4000
    won = rng.random(n) < 0.5
    bets = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="6h"),
        "pnl": np.where(won, 1.0, -1.0),
        "stake": 1.0,
        "won": won,
        "odds": 2.0,
    })
    ci = bootstrap_ci(bets, n_boot=1000, seed=3)
    assert ci["lo"] < 0 < ci["hi"], f"interval {ci['lo']:.4f}..{ci['hi']:.4f} excludes zero"


def test_required_sample_size_matches_the_published_figures():
    """Hand-check against the numbers in docs/research/02.

    A 2% edge at even money needs ~19,600 bets at 95%/80%; a mixed 1X2
    portfolio at average odds 3.2 needs ~43,500. Both computed from
    n = ((z_a + z_b) * sigma / mu)^2 with sigma = d*sqrt(p(1-p)).
    """
    assert required_sample_size(2.0, edge=0.02) == pytest.approx(19_600, rel=0.05)
    assert required_sample_size(3.2, edge=0.02) == pytest.approx(43_500, rel=0.06)
    # A bigger edge is cheaper to prove, quadratically.
    assert required_sample_size(2.0, edge=0.05) == pytest.approx(19_600 / 6.25, rel=0.06)


# --------------------------------------------------------------------------
# Does the leakage guard actually guard?
# --------------------------------------------------------------------------

def test_walk_forward_splits_never_leak(panel):
    splits = list(season_walk_forward(panel, min_train_seasons=3))
    assert len(splits) >= 5
    for s in splits:
        assert_no_leakage(panel, s)          # must not raise
        assert s.train_end < s.test_start


def test_leakage_guard_catches_a_deliberately_leaky_split(panel):
    """Negative control for the guard itself.

    Hand-build a split whose training set includes a match from the middle of
    the test window. If this does not raise, the guard is decorative.
    """
    from src.eval.split import Split
    good = next(season_walk_forward(panel, min_train_seasons=3))
    poisoned = np.concatenate([good.train_idx, good.test_idx[len(good.test_idx) // 2:]])
    bad = Split(
        label="poisoned",
        train_idx=poisoned,
        test_idx=good.test_idx,
        train_end=good.train_end,
        test_start=good.test_start,
        test_end=good.test_end,
    )
    with pytest.raises(AssertionError, match="leakage in split"):
        assert_no_leakage(panel, bad)


def test_shuffling_outcomes_destroys_the_markets_edge(panel, probs):
    """A model with real signal must lose all of it when outcomes are shuffled.

    The test is that the market can no longer beat a naive constant forecast --
    NOT that it scores the same as one. A confident model shuffled against
    random outcomes scores strictly *worse* than a hedged base-rate model,
    because its confidence now points in random directions and RPS punishes
    that. Expecting equality here was wrong, and this test caught it.
    """
    rng = np.random.default_rng(11)
    y = panel["result"].to_numpy()
    shuffled = y[rng.permutation(len(y))]

    intact = rps(probs["market"], y)
    broken = rps(probs["market"], shuffled)
    naive = rps(probs["base_rate"], shuffled)

    assert broken > intact + 0.02, "shuffling the outcomes did not degrade the market model"
    assert broken >= naive, "shuffled market still beats a constant forecast -- signal survived a shuffle"
    # And the intact model comfortably beats that same constant, which is what
    # makes the collapse meaningful rather than an artefact of the metric.
    assert intact < rps(probs["base_rate"], y) - 0.01


def test_per_match_rps_supports_paired_comparison(probs, panel):
    """Phase 5 compares two models on the same fixtures and needs a paired
    difference, not two independent means. Check the machinery exists and the
    pairing is aligned."""
    y = panel["result"]
    d = rps_per_match(probs["base_rate"], y) - rps_per_match(probs["market"], y)
    assert len(d) == len(panel)
    assert d.mean() > 0                       # market is better
    se = d.std(ddof=1) / np.sqrt(len(d))
    assert d.mean() / se > 5, "market-vs-base-rate should be overwhelmingly significant"
