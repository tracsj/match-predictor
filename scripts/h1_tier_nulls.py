"""Per-tier CLV against each tier's OWN measured drift, not against 50%.

    uv run python scripts/h1_tier_nulls.py

A control, not a candidate. It re-reads a table H1 already printed; it scores
no new configuration and does not move the registry count.

WHY IT HAD TO BE RUN. `docs/H1_RESULT.md` leads with "tier 5 -- the thinnest
market in the panel -- shows nothing, 49.25% shortened, below a coin flip",
and draws the conclusion that H1's proposed mechanism is not monotone in
market thinness. That reading compares 49.25% against 50%.

scripts/h1_odds_matched_null.py then established that 50% is NOT the null:
Pinnacle's overround tightens toward kickoff, prices lengthen by default, and
a randomly chosen band-eligible selection shortens only 45-48% of the time. So
every per-tier number in that table was read against the wrong baseline, and
the tier-5 anomaly may be an artifact of the comparison rather than a fact
about the National League.

This asks the question properly: each tier's model bets against a null matched
to that tier's own eligible selections and that tier's own odds mix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.betting import (
    PINNACLE_PRE, clv_report, closing_price_for_bets, simulate,
)
from src.features.ratings import TIER
from src.h1 import CACHE, LOWER_DIVS, RULE, build_panel

# scripts/ is not a package, so the sibling is imported by path rather than
# copied. Copying `matched_null` would let the null used here drift silently
# away from the null the odds-matched arm reported, which is the one thing
# these two files must agree on.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h1_odds_matched_null import matched_null  # noqa: E402

N_SIMS = 200


def z_vs_null(observed: float, null: float, n: int) -> float:
    """One-proportion z of the model's shortening rate against a measured null.

    The null is itself estimated, but from far more selections than the model
    bets, so its own error is the smaller term and is ignored here. That makes
    this z slightly optimistic, which is worth stating rather than hiding.
    """
    if not np.isfinite(null) or n == 0:
        return float("nan")
    se = np.sqrt(null * (1 - null) / n)
    return (observed - null) / se if se else float("nan")


def main() -> None:
    panel, _, _ = build_panel()
    if not Path(CACHE).exists():
        raise SystemExit(f"{CACHE} missing -- run `uv run python -m src.h1` first")
    z = np.load(CACHE, allow_pickle=False)
    graded = panel.iloc[z["test_idx"]].reset_index(drop=True)
    p = z["probs"]
    tier_of = graded["div"].map(TIER)

    print("=" * 78)
    print("PER-TIER CLV AGAINST EACH TIER'S OWN DRIFT")
    print("=" * 78)
    print(f"  {N_SIMS} sims per tier, odds deciles taken from that tier's own bets")
    print()

    rows = []
    for t in sorted(tier_of.dropna().unique()):
        m = (tier_of == t).to_numpy()
        sub = graded[m].reset_index(drop=True)
        bets = simulate(sub, p[m], PINNACLE_PRE, RULE)
        if bets.empty:
            continue
        close = closing_price_for_bets(bets, sub)
        rep = clv_report(bets, close)
        ok = np.isfinite(close.to_numpy(float)) & (close.to_numpy(float) > 0)
        bet_odds = bets["odds"].to_numpy(float)[ok]
        sims = np.array([matched_null(sub, bet_odds, seed=s) for s in range(N_SIMS)])
        sims = sims[np.isfinite(sims)]
        null = float(sims.mean())
        rows.append({
            "tier": int(t),
            "divisions": ",".join(sorted(sub["div"].unique())),
            "n_bets": rep["n"],
            "observed": rep["pct_shortened"],
            "null": null,
            "margin": rep["pct_shortened"] - null,
            "z": z_vs_null(rep["pct_shortened"], null, rep["n"]),
        })

    tbl = pd.DataFrame(rows)
    tbl["p_two_sided"] = 2 * (1 - stats.norm.cdf(np.abs(tbl["z"])))
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print()
    print("-" * 78)
    print("THE SAME FOR THE TWO STRATA")
    print("-" * 78)
    for name, m in (("lower (3-5)", graded["div"].isin(LOWER_DIVS).to_numpy()),
                    ("upper (1-2)", ~graded["div"].isin(LOWER_DIVS).to_numpy())):
        sub = graded[m].reset_index(drop=True)
        bets = simulate(sub, p[m], PINNACLE_PRE, RULE)
        close = closing_price_for_bets(bets, sub)
        rep = clv_report(bets, close)
        arr = close.to_numpy(float)
        bet_odds = bets["odds"].to_numpy(float)[np.isfinite(arr) & (arr > 0)]
        sims = np.array([matched_null(sub, bet_odds, seed=s) for s in range(N_SIMS)])
        null = float(sims[np.isfinite(sims)].mean())
        zz = z_vs_null(rep["pct_shortened"], null, rep["n"])
        print(f"  {name}: observed {rep['pct_shortened']:.4f}  null {null:.4f}  "
              f"margin {rep['pct_shortened'] - null:+.4f}  z {zz:.2f}  "
              f"p {2 * (1 - stats.norm.cdf(abs(zz))):.2e}")

    print()
    print("  If tier 5's margin lines up with tiers 3 and 4, then H1_RESULT.md's")
    print("  headline -- that the mechanism is not monotone in thinness --")
    print("  was an artifact of comparing against 50%, and the diagnostics")
    print("  section has to say so plainly rather than let the recorded")
    print("  reading stand.")


if __name__ == "__main__":
    main()
