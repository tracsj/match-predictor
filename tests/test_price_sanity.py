"""Odds columns must contain odds. Measured, one of them does not.

`src/h3.py` found 10 rows carrying a Bet365 price of exactly 0.0. `notna()`
does not catch that, and `np.nan_to_num` turns the resulting infinite
log-implied-probability into a huge finite feature -- no error, no NaN, just a
garbage row read as informative.

That fact was originally a prose note in a docstring and a handoff paragraph.
This is the repo's own rule applied to it: a mechanical fact with a right
answer belongs in a checker, because the next zero -- in a column nobody
thought to probe, in a season not yet published -- arrives silently otherwise.
The original probe only swept H3's six columns; this sweeps every odds column
in the corpus.

These skip without the corpus, like the rest of the data-dependent suite.
"""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

FEATURES = "data/processed/features.parquet"

# Decimal odds are strictly greater than 1.0: a price of exactly 1.0 pays
# nothing and a price at or below 0 is not a price at all.
FLOOR = 1.0

# Every non-positive price in the corpus as of 2026-08-17, measured by the
# sweep below rather than assumed. The 1X2 CLOSING entries (b365ca, maxca,
# avgca) are the ones worth noticing: H1 and Phase 6 both priced ROI against
# those columns. `test_a_non_positive_price_can_never_be_selected_as_a_bet`
# is why no reported number moved.
KNOWN = {
    "b365h": 10, "b365ca": 1, "maxca": 3, "avgca": 1,
    "b365_ahh": 3, "b365_aha": 3, "avg_aha": 1,
    "b365c_ahh": 1, "b365c_aha": 2, "avgc_ahh": 1, "avgc_aha": 2,
    "maxc_aha": 1,
}


def _corpus():
    import os
    if not os.path.exists(FEATURES):
        pytest.skip("no corpus on disk")
    return pd.read_parquet(FEATURES)


def _odds_columns(df):
    """Columns that hold a decimal price, found by naming convention.

    Written as a prefix/suffix rule rather than a hand-listed set so a book
    added by the vendor later is swept automatically -- the failure mode this
    test exists for is a column nobody thought to name.
    """
    books = ("b365", "ps", "max", "avg", "bfe", "bw", "iw", "wh", "vc", "gb",
             "ls", "sj", "sb", "bs", "1xb")
    out = []
    for c in df.columns:
        lc = c.lower()
        if not any(lc.startswith(b) for b in books):
            continue
        if lc.endswith(("h", "d", "a")) and df[c].dtype.kind == "f":
            out.append(c)
    return out


def test_the_sweep_actually_finds_the_odds_columns():
    """Guard on the guard. A naming rule that matched nothing would make every
    assertion below vacuously true, which is the failure mode of a check that
    can only pass."""
    df = _corpus()
    cols = _odds_columns(df)
    assert len(cols) >= 12, f"only found {len(cols)} odds columns: {cols}"
    for expected in ("psh", "psch", "b365h"):
        assert expected in cols, f"{expected} should have been swept"


def test_no_odds_column_carries_a_non_positive_price_outside_the_known_fault():
    """Every non-null price must exceed 1.0.

    The known faults are recorded explicitly rather than excluded by a blanket
    tolerance. If any count GROWS, or the fault appears in a column not listed,
    this fails -- which is the whole point.
    """
    df = _corpus()
    offenders = {}
    for c in _odds_columns(df):
        v = df[c].to_numpy(float)
        bad = int((np.isfinite(v) & (v <= FLOOR)).sum())
        if bad:
            offenders[c] = bad

    unexpected = {c: n for c, n in offenders.items()
                  if c not in KNOWN or n > KNOWN[c]}
    assert not unexpected, (
        "odds columns carrying a price <= 1.0 beyond the known fault:\n"
        f"  {unexpected}\n"
        "A price of 0 or 1 is missing data wearing a number. Anything building "
        "features from these columns must filter on > 1.0 -- see build_frame() "
        "in src/h3.py -- or nan_to_num will convert the infinity into a large "
        "finite feature value and nothing will fail."
    )


@pytest.mark.parametrize("col,count", sorted(KNOWN.items()))
def test_each_known_fault_is_still_exactly_as_measured(col, count):
    """If a fault disappears -- a vendor fix, a loader change -- this fails,
    and that is deliberate. A stale allowance left in KNOWN would silently
    absorb a fresh fault of the same size in the same column."""
    df = _corpus()
    if col not in df.columns:
        pytest.skip(f"{col} not in this corpus")
    v = df[col].to_numpy(float)
    got = int((np.isfinite(v) & (v <= FLOOR)).sum())
    assert got == count, (
        f"{col}: measured {got} non-positive prices, KNOWN says {count}. "
        "If upstream fixed it, drop the entry; if it grew, something is wrong."
    )


def test_a_non_positive_price_can_never_be_selected_as_a_bet():
    """The reason the faults above did not corrupt any reported number.

    Three of them (b365ca, maxca, avgca) are 1X2 CLOSING columns, which H1 and
    Phase 6 both priced ROI against -- so "it is only a few rows" is not
    sufficient reassurance on its own. This asserts the mechanism instead:
    simulate() computes EV = prob * odds - 1, so a price of 0 scores -1 and can
    never be the argmax, and the rule's min_odds floor of 1.5 excludes it
    again. Two independent reasons, and this proves the composite rather than
    trusting either.
    """
    import numpy as _np
    from src.eval.betting import PriceSet, BetRule, simulate

    prices = PriceSet("synthetic", "oh", "od", "oa")
    rule = BetRule(min_ev=0.05, min_odds=1.5, max_odds=5.0, stake=1.0)
    # A zero-priced home outcome the model is extremely confident in -- the
    # most favourable possible case for it slipping through.
    df = pd.DataFrame({
        "match_id": ["m1", "m2"],
        "kickoff": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "result": ["H", "H"],
        "oh": [0.0, 3.0], "od": [4.0, 4.0], "oa": [4.0, 4.0],
    })
    probs = _np.array([[0.99, 0.005, 0.005], [0.90, 0.05, 0.05]])
    bets = simulate(df, probs, prices, rule)
    assert "m1" not in set(bets["match_id"]), (
        "a zero-priced outcome was selected as a bet -- the EV and min_odds "
        "guards are not doing what this test assumed"
    )
    assert "m2" in set(bets["match_id"]), (
        "the control row was not bet either, so this test proves nothing "
        "about the zero-price case"
    )


def test_h3_frame_excludes_them():
    """The consumer-side guarantee. Whatever the corpus carries, the frame H3
    fits on must be clean -- this is what actually protects a result."""
    pytest.importorskip("catboost")
    from src.eval.betting import PINNACLE_CLOSE, PINNACLE_PRE
    from src.h3 import B365_PRE_COLS, build_frame

    frame = build_frame()
    cols = PINNACLE_PRE.cols + PINNACLE_CLOSE.cols + B365_PRE_COLS
    arr = frame[cols].to_numpy(float)
    assert np.isfinite(arr).all(), "H3's frame carries a non-finite price"
    assert (arr > FLOOR).all(), "H3's frame carries a price <= 1.0"
