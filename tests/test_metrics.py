"""Tests for scoring rules and calibration diagnostics.

Every expected number below is derived by hand from the definition of the
scoring rule, written out in the comment above the assertion, so the test does
not agree with the code merely because both came from the same place.
"""

import numpy as np
import pytest

from src.eval.metrics import (
    OUTCOMES, accuracy, brier, ece, log_loss, log_loss_per_match,
    reliability_table, rps, rps_per_match, summary, to_onehot,
)


# --------------------------------------------------------------------------
# RPS -- hand-computed from  1/(r-1) * sum_i (cumsum(p)_i - cumsum(a)_i)^2
# --------------------------------------------------------------------------

def test_rps_hand_computed_home_win():
    # p = (.5, .3, .2) -> cumsum (.5, .8);  y = H -> a = (1,0,0) -> cumsum (1, 1)
    # ((.5-1)^2 + (.8-1)^2) / 2 = (0.25 + 0.04) / 2 = 0.145
    assert rps([[0.5, 0.3, 0.2]], ["H"]) == pytest.approx(0.145)


def test_rps_hand_computed_draw():
    # same p, y = D -> a = (0,1,0) -> cumsum (0, 1)
    # ((.5-0)^2 + (.8-1)^2) / 2 = (0.25 + 0.04) / 2 = 0.145
    assert rps([[0.5, 0.3, 0.2]], ["D"]) == pytest.approx(0.145)


def test_rps_hand_computed_away():
    # same p, y = A -> a = (0,0,1) -> cumsum (0, 0)
    # ((.5-0)^2 + (.8-0)^2) / 2 = (0.25 + 0.64) / 2 = 0.445
    assert rps([[0.5, 0.3, 0.2]], ["A"]) == pytest.approx(0.445)


def test_rps_perfect_forecast_is_zero():
    p = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert rps(p, ["H", "D", "A"]) == pytest.approx(0.0)


def test_rps_worst_forecast_is_one():
    # All mass on A, outcome is H: cumsum(p) = (0,0), cumsum(a) = (1,1)
    # (1 + 1)/2 = 1.0 -- the maximum RPS can take.
    assert rps([[0.0, 0.0, 1.0]], ["H"]) == pytest.approx(1.0)


def test_rps_uniform_under_equiprobable_outcomes_is_two_ninths():
    # Averaged over H, D, A once each: (5/18 + 1/9 + 5/18)/3 = (2/9)
    u = [[1 / 3, 1 / 3, 1 / 3]] * 3
    assert rps(u, ["H", "D", "A"]) == pytest.approx(2 / 9)


def test_rps_is_distance_sensitive():
    # The whole reason RPS was chosen over Brier for 1X2. A confident home
    # forecast must be punished LESS by a draw than by an away win.
    p = [[0.7, 0.2, 0.1]]
    assert rps(p, ["D"]) < rps(p, ["A"])


def test_rps_ordering_of_columns_is_load_bearing():
    # H/D/A is an ordinal scale. Swapping D and A is not a relabelling, it
    # changes the metric -- this test exists so a future reorder fails loudly.
    assert rps([[0.5, 0.3, 0.2]], ["A"]) != pytest.approx(rps([[0.5, 0.2, 0.3]], ["A"]))


def test_rps_per_match_averages_to_rps():
    p = [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [1 / 3, 1 / 3, 1 / 3]]
    y = ["H", "A", "D"]
    assert rps_per_match(p, y).mean() == pytest.approx(rps(p, y))
    # and the individual values are the hand-computed ones
    assert rps_per_match(p, y)[0] == pytest.approx(0.145)


# --------------------------------------------------------------------------
# Log loss
# --------------------------------------------------------------------------

def test_log_loss_hand_computed():
    # -ln(0.5) = 0.6931471805599453
    assert log_loss([[0.5, 0.3, 0.2]], ["H"]) == pytest.approx(0.6931471805599453)


def test_log_loss_uniform_is_ln3():
    # -ln(1/3) = 1.0986122886681098, regardless of which outcome occurs
    for y in ("H", "D", "A"):
        assert log_loss([[1 / 3, 1 / 3, 1 / 3]], [y]) == pytest.approx(1.0986122886681098)


def test_log_loss_per_match_averages_to_log_loss():
    p = [[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]]
    y = ["H", "A"]
    assert log_loss_per_match(p, y).mean() == pytest.approx(log_loss(p, y))


def test_log_loss_does_not_return_inf_on_a_confident_miss():
    # Clipping matters: one zero-probability hit would otherwise make the whole
    # scoreboard inf and hide every other number.
    v = log_loss([[1.0, 0.0, 0.0]], ["A"])
    assert np.isfinite(v) and v > 30


# --------------------------------------------------------------------------
# Brier and accuracy
# --------------------------------------------------------------------------

def test_brier_hand_computed():
    # (0.5-1)^2 + (0.3-0)^2 + (0.2-0)^2 = 0.25 + 0.09 + 0.04 = 0.38
    assert brier([[0.5, 0.3, 0.2]], ["H"]) == pytest.approx(0.38)


def test_accuracy_is_argmax_hit_rate():
    p = [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.3, 0.4, 0.3]]
    assert accuracy(p, ["H", "A", "D"]) == pytest.approx(1.0)
    assert accuracy(p, ["A", "A", "D"]) == pytest.approx(2 / 3)


def test_accuracy_and_rps_can_disagree():
    # The reason accuracy is reported "for orientation only": a model can be
    # more accurate and worse-calibrated, and calibration is what decides a bet.
    sharp_but_wrong = [[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]]
    humble = [[0.4, 0.35, 0.25], [0.4, 0.35, 0.25]]
    y = ["H", "A"]
    assert accuracy(sharp_but_wrong, y) == accuracy(humble, y) == pytest.approx(0.5)
    assert rps(sharp_but_wrong, y) > rps(humble, y)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def test_ece_is_zero_for_a_perfectly_calibrated_forecast():
    # 100 matches forecast at exactly (0.5, 0.3, 0.2), with outcomes occurring
    # at exactly those frequencies: 50 H, 30 D, 20 A.
    rng = np.random.default_rng(0)
    p = np.tile([0.5, 0.3, 0.2], (100, 1))
    y = np.array(["H"] * 50 + ["D"] * 30 + ["A"] * 20)
    rng.shuffle(y)
    assert ece(p, y) == pytest.approx(0.0, abs=1e-12)


def test_ece_detects_overconfidence():
    # Forecast 0.9 home every time; home actually wins half the time.
    p = np.tile([0.9, 0.05, 0.05], (100, 1))
    y = np.array(["H"] * 50 + ["A"] * 50)
    # Home class alone is off by 0.4; averaged over 3 classes with the other
    # two off by 0.05 and 0.45 respectively.
    assert ece(p, y) > 0.25


def test_reliability_table_bins_and_counts():
    p = np.tile([0.5, 0.3, 0.2], (10, 1))
    y = ["H"] * 5 + ["D"] * 3 + ["A"] * 2
    tab = reliability_table(p, y, bins=10)
    assert set(tab["outcome"]) == set(OUTCOMES)
    assert tab["n"].sum() == 30          # 10 matches x 3 classes
    home = tab[tab["outcome"] == "H"].iloc[0]
    assert home["mean_forecast"] == pytest.approx(0.5)
    assert home["observed_freq"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Input validation -- these guard the failure modes that would corrupt a run
# --------------------------------------------------------------------------

def test_probabilities_must_sum_to_one():
    with pytest.raises(ValueError, match="do not sum to 1"):
        rps([[0.5, 0.3, 0.1]], ["H"])


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        rps([[0.5, 0.5]], ["H"])


def test_unknown_outcome_label_is_rejected():
    with pytest.raises(ValueError, match="not one of"):
        rps([[0.5, 0.3, 0.2]], ["X"])


def test_onehot_is_case_and_whitespace_tolerant():
    assert np.array_equal(to_onehot([" h ", "d", "A"]), np.eye(3))


def test_summary_carries_every_headline_metric():
    s = summary([[0.5, 0.3, 0.2]], ["H"], label="test")
    assert set(s) == {"model", "n", "rps", "log_loss", "brier", "ece", "accuracy"}
    assert s["rps"] == pytest.approx(0.145)
