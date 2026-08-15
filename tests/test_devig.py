"""Tests for odds -> probability conversion.

Expected values here are written independently of the implementation: either
computed by hand from the published formula, or asserted as a property the
method is documented to have. A test whose expectation is produced by calling
the code under test cannot fail.
"""

import numpy as np
import pytest

from src.eval.devig import additive, devig, multiplicative, overround, power, shin


# --------------------------------------------------------------------------
# Hand-computed values
# --------------------------------------------------------------------------

def test_overround_hand_computed():
    # 1/2.0 + 1/3.5 + 1/4.0 = 0.5 + 0.2857142857 + 0.25 = 1.0357142857
    assert overround([2.0, 3.5, 4.0]) == pytest.approx(1.0357142857, abs=1e-9)


def test_multiplicative_hand_computed():
    # q = (0.5, 0.2857142857, 0.25); Q = 1.0357142857
    # p = q/Q = (0.48275862, 0.27586207, 0.24137931)
    p = multiplicative([2.0, 3.5, 4.0])
    assert p == pytest.approx([0.48275862, 0.27586207, 0.24137931], abs=1e-7)
    assert p.sum() == pytest.approx(1.0)


def test_additive_hand_computed():
    # q = (0.5, 0.2857142857, 0.25); excess = 0.0357142857; per outcome 0.0119047619
    # p = (0.4880952381, 0.2738095238, 0.2380952381)
    p = additive([2.0, 3.5, 4.0])
    assert p == pytest.approx([0.4880952381, 0.2738095238, 0.2380952381], abs=1e-9)


def test_additive_differs_from_multiplicative_in_known_direction():
    # Additive takes an equal absolute slice from each outcome, so relative to
    # proportional removal it leaves the favourite HIGHER and the longshot
    # LOWER. This follows from the formulae, not from running them.
    mult = multiplicative([2.0, 3.5, 4.0])
    add = additive([2.0, 3.5, 4.0])
    assert add[0] > mult[0]      # favourite
    assert add[2] < mult[2]      # longshot


# --------------------------------------------------------------------------
# Properties every method must satisfy
# --------------------------------------------------------------------------

ODDS_CASES = [
    [2.0, 3.5, 4.0],
    [1.25, 6.0, 12.0],       # heavy favourite
    [4.5, 3.6, 1.85],        # away favourite
    [2.62, 3.40, 2.75],      # near-even three-way
    [1.05, 15.0, 40.0],      # extreme favourite / big longshot
]
METHODS = ["shin", "multiplicative", "additive", "power"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("odds", ODDS_CASES)
def test_sums_to_one(method, odds):
    p = devig(odds, method=method)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(p > 0) and np.all(p < 1)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("odds", ODDS_CASES)
def test_preserves_ordering(method, odds):
    # Shorter odds must always mean higher probability, whatever the method.
    p = devig(odds, method=method)
    assert list(np.argsort(-p)) == list(np.argsort(odds))


@pytest.mark.parametrize("method", METHODS)
def test_no_margin_is_a_noop(method):
    # 1/3 + 1/3 + 1/3 = 1 exactly: there is nothing to remove.
    p = devig([3.0, 3.0, 3.0], method=method)
    assert p == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_symmetric_odds_give_uniform(method):
    # Any sane method must map equal prices to equal probabilities.
    p = devig([2.9, 2.9, 2.9], method=method)
    assert p == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-9)


def test_power_solves_its_own_defining_equation():
    # The method is defined by: p_i = q_i**k with sum(p) == 1. Recover k from
    # the output and check the defining equation holds -- this checks the
    # solver, not the arithmetic that produced the answer.
    odds = np.array([2.0, 3.5, 4.0])
    q = 1.0 / odds
    p = power(odds)
    k = np.log(p[0]) / np.log(q[0])
    assert np.sum(q ** k) == pytest.approx(1.0, abs=1e-9)
    assert k > 1.0          # margin present, so exponent must exceed 1


def test_shin_reduces_to_additive_for_two_outcomes():
    # A documented property of Shin's method, stated independently of this
    # implementation: with n = 2 it coincides with the additive method.
    for odds in ([1.90, 1.90], [1.55, 2.45], [1.20, 4.60], [1.02, 20.0]):
        assert shin(odds) == pytest.approx(additive(odds), abs=1e-7)


def test_shin_z_is_zero_without_margin_and_positive_with_it():
    _, z_fair = shin([3.0, 3.0, 3.0], return_z=True)
    assert z_fair == pytest.approx(0.0, abs=1e-9)
    _, z_vig = shin([2.0, 3.5, 4.0], return_z=True)
    assert 0.0 < z_vig < 1.0


def test_shin_z_grows_with_overround():
    # More margin must be explained by more insider proportion. Monotone by
    # construction of the estimator.
    _, z_small = shin([2.05, 3.55, 4.05], return_z=True)
    _, z_large = shin([1.90, 3.20, 3.60], return_z=True)
    assert overround([1.90, 3.20, 3.60]) > overround([2.05, 3.55, 4.05])
    assert z_large > z_small


# --------------------------------------------------------------------------
# Shapes and failure modes
# --------------------------------------------------------------------------

def test_batch_matches_row_by_row():
    batch = np.array(ODDS_CASES)
    got = shin(batch)
    assert got.shape == batch.shape
    for i, row in enumerate(ODDS_CASES):
        assert got[i] == pytest.approx(shin(row), abs=1e-10)


def test_nan_odds_propagate_rather_than_silently_normalising():
    # A missing price must NOT be quietly turned into a probability -- an
    # invented number here would flow straight into a betting decision.
    p = shin([2.0, np.nan, 4.0])
    assert np.all(np.isnan(p))


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown de-vig method"):
        devig([2.0, 3.5, 4.0], method="wishful")
