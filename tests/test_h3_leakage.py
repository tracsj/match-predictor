"""The H3 leakage gate must stay quiet on real features AND be able to speak.

A detector verified only against approved input proves nothing: a function
that returns unconditionally passes that test perfectly. So each case below
that asserts silence is paired with one that asserts a raise.

H3's label is built from the closing prices, so a closing column on the input
side turns the whole hypothesis into a lookup -- and it would surface as an
unusually good result, which is the one outcome nobody interrogates.
"""

from __future__ import annotations

import pytest

from src.h3 import FORBIDDEN, assert_no_closing_leak
from src.models.baselines import ALL_FEATURES


def _price_feature_names():
    """The eleven names add_price_features() generates, written out by hand.

    Deliberately NOT imported from src.h3 and not derived from the function
    under test. If both sides came from the same place the check would move
    with the code and stay green for every value including a wrong one.
    """
    names = []
    for o in ("H", "D", "A"):
        names += [f"h3_logp_ps_{o}", f"h3_logp_b365_{o}", f"h3_disagree_{o}"]
    return names + ["h3_overround_ps", "h3_overround_b365"]


def test_the_real_h3_feature_list_is_clean():
    assert_no_closing_leak(ALL_FEATURES + _price_feature_names())


@pytest.mark.parametrize("leak", sorted(FORBIDDEN))
def test_every_known_closing_column_is_caught(leak):
    with pytest.raises(AssertionError, match="CLOSING COLUMNS"):
        assert_no_closing_leak(ALL_FEATURES + _price_feature_names() + [leak])


def test_a_closing_column_this_corpus_has_never_carried_is_still_caught():
    """The suffix scan, which is the half that survives a vendor adding a
    column nobody thought to list. `bwch` is not in FORBIDDEN and must still
    raise."""
    assert "bwch" not in FORBIDDEN
    with pytest.raises(AssertionError, match="CLOSING COLUMNS"):
        assert_no_closing_leak(ALL_FEATURES + ["bwch", "bwcd", "bwca"])


def test_the_pre_close_legs_are_not_mistaken_for_closing_columns():
    """The gate must not fire on the prices H3 is allowed to use. A detector
    that blocks the legitimate inputs would abort every run, which is a
    different failure and just as fatal."""
    assert_no_closing_leak(["psh", "psd", "psa", "b365h", "b365d", "b365a"])


def test_the_label_source_columns_are_all_in_the_forbidden_set():
    """Whatever else FORBIDDEN contains, it must contain the exact columns the
    label is built from, or the gate has a hole where it matters most."""
    for col in ("psch", "pscd", "psca"):
        assert col in FORBIDDEN
