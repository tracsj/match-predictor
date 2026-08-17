"""H1's finding, re-asked on the only data the project had not already seen.

    uv run python scripts/h1_holdout_tiers.py

This is a control, not a candidate, and it does not move the registry count.
It re-slices the Phase 6 settled run -- same configuration, same holdout, same
rule -- by tier, which is a question that run never asked.

WHY IT MATTERS MORE THAN THE OTHER CONTROLS. H1's panel is data this project
has already seen; the pre-registration said so in advance. Model selection
optimised RPS against outcomes on that panel, and the closing line also
correlates with outcomes, so an overfitted model's selections correlate with
the close through a pathway that is not skill. No amount of "never selected
on betting PnL" closes that pathway. Season 2025-26 is outside it.

WHY IT CANNOT SETTLE ANYTHING. Pinnacle's 2025-26 coverage is 38% and stops
after 2026-01-14, which leaves roughly 273 lower-stratum bets against the
3,250-bet floor the H1 pre-registration fixed. That floor applies here
unchanged: this stratum is INCONCLUSIVE BY FLOOR whichever way it points, and
the floor is not lowered to let it speak. Direction is all it can offer.

The exchange cannot rescue the sample. `bfeh/bfed/bfea` -- the exchange
pre-close -- are ABSENT from the historical corpus rather than sparse
(measured with scripts/h1_holdout_coverage.py), so there is no second ladder
to fall back on for a backward-looking CLV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.betting import (
    PINNACLE_CLOSE, PINNACLE_PRE, clv_report, closing_price_for_bets, simulate,
)
from src.features.ratings import TIER
from src.h1 import LOWER_DIVS, MIN_BETS, RULE
from src.phase6 import build_holdout, fit_and_predict


def main() -> None:
    train, test, seq_all, cutoff = build_holdout()
    legs = PINNACLE_PRE.cols + PINNACLE_CLOSE.cols
    keep = (test["div"].isin(TIER) & test[legs].notna().all(axis=1)).to_numpy()

    print("=" * 78)
    print("H1 CONTROL -- the 2025-26 holdout, stratified by tier")
    print("=" * 78)
    print(f"  training  {len(train):,} matches, all before {pd.Timestamp(cutoff).date()}")
    print(f"  holdout   {len(test):,} matches; {int(keep.sum()):,} carry a tier")
    print("            and both Pinnacle legs")
    print(f"  floor     {MIN_BETS:,} bets -- inherited from the pre-registration")
    print()
    print("  fitting (3 seeds, the settled configuration)...")
    p_all = fit_and_predict(train, test, seq_all)

    graded = test[keep].reset_index(drop=True)
    p = p_all[keep]
    is_lower = graded["div"].isin(LOWER_DIVS).to_numpy()

    print()
    print("-" * 78)
    print("CLV BY STRATUM, OUT OF SAMPLE")
    print("-" * 78)
    rows = []
    for name, mask in (("lower (3-5)", is_lower), ("upper (1-2)", ~is_lower)):
        sub = graded[mask].reset_index(drop=True)
        bets = simulate(sub, p[mask], PINNACLE_PRE, RULE)
        if bets.empty:
            rows.append({"stratum": name, "n_bets": 0})
            continue
        rep = clv_report(bets, closing_price_for_bets(bets, sub))
        rows.append({"stratum": name, "n_bets": rep["n"],
                     "mean_ratio": rep["mean_ratio"],
                     "median_ratio": rep["median_ratio"],
                     "pct_shortened": rep["pct_shortened"],
                     "binom_p": rep["binom_pvalue"],
                     "verdict": ("INCONCLUSIVE BY FLOOR"
                                 if rep["n"] < MIN_BETS else "at floor")})
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda v: f"{v:.4f}"))

    print()
    print("  In-sample, for comparison only -- H1 measured lower 1.0083 at")
    print("  52.53% shortened, upper 1.0046 at 50.22%. If the out-of-sample")
    print("  direction matches, the finding survives its hardest available")
    print("  test without being confirmed by it. If it does not, that is the")
    print("  more informative outcome and is reported the same way.")


if __name__ == "__main__":
    main()
