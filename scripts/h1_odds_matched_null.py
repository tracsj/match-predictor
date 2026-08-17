"""Is 50% the right null for "% shortened"? Measure it, odds-matched.

    uv run python scripts/h1_odds_matched_null.py

A control, not a candidate. It does not move the registry count.

WHY THIS ARM EXISTS. scripts/h1_diagnostics.py measured the unconditional
drift among band-eligible selections at 45.5% shortened in the lower stratum
and 47.7% in the upper -- both BELOW 50%. If that holds, the pre-registered
binomial test against 0.5 was measuring the model against a null the market
does not obey, and the model's true margin over chance is LARGER than the
headline suggests.

That conclusion is convenient, which is the precise reason it needs a second
derivation before it is written down. The obvious confound: the random control
sampled uniformly across eligible cells, while the rule bets the maximum-EV
outcome and lands at mean odds 3.26. If longshots drift differently from
short prices -- and the favourite-longshot literature says they do -- then the
45.5% could be an odds-mix artifact and nothing more.

So this arm matches the null's odds distribution to the model's actual bets,
decile by decile, and asks the same question again. If the drift survives
matching, the reframing is real. If it disappears, the pre-registered bar
against 0.5 was accidentally right, and that is what gets reported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.betting import (
    PINNACLE_CLOSE, PINNACLE_PRE, closing_price_for_bets, simulate,
)
from src.h1 import CACHE, LOWER_DIVS, RULE, build_panel

N_SIMS = 200
N_DECILES = 10


def eligible_cells(df: pd.DataFrame):
    """Every (row, outcome) the rule could legally have bet, with its ratio."""
    pre = df[PINNACLE_PRE.cols].to_numpy(float)
    close = df[PINNACLE_CLOSE.cols].to_numpy(float)
    ok = (np.isfinite(pre) & np.isfinite(close) & (close > 0)
          & (pre >= RULE.min_odds) & (pre <= RULE.max_odds))
    r, c = np.nonzero(ok)
    return pre[r, c], pre[r, c] / close[r, c]


def matched_null(df: pd.DataFrame, bet_odds: np.ndarray, seed: int) -> float:
    """% shortened for a random draw whose odds histogram matches the model's.

    Deciles are taken from the MODEL's bets, then each decile is filled from
    eligible cells whose price falls in that same decile, in the same
    proportion the model bet it. A decile with no eligible cells is skipped
    rather than filled from a neighbour, because borrowing prices from an
    adjacent bucket is the very confound this arm exists to remove.
    """
    odds, ratio = eligible_cells(df)
    edges = np.quantile(bet_odds, np.linspace(0, 1, N_DECILES + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rng = np.random.default_rng(seed)

    picked = []
    for i in range(N_DECILES):
        want = int(((bet_odds >= edges[i]) & (bet_odds < edges[i + 1])).sum())
        pool = np.nonzero((odds >= edges[i]) & (odds < edges[i + 1]))[0]
        if want == 0 or len(pool) == 0:
            continue
        picked.append(ratio[rng.choice(pool, size=want, replace=True)])
    if not picked:
        return float("nan")
    r = np.concatenate(picked)
    return float((r > 1.0).mean())


def main() -> None:
    panel, _, _ = build_panel()
    if not Path(CACHE).exists():
        raise SystemExit(f"{CACHE} missing -- run `uv run python -m src.h1` first")
    z = np.load(CACHE, allow_pickle=False)
    graded = panel.iloc[z["test_idx"]].reset_index(drop=True)
    p = z["probs"]
    is_lower = graded["div"].isin(LOWER_DIVS).to_numpy()

    print("=" * 78)
    print("ODDS-MATCHED NULL -- what % shortened means when nobody is forecasting")
    print("=" * 78)
    print(f"  {N_SIMS} sims, {N_DECILES} odds deciles taken from the model's own bets")
    print()

    observed = {"lower (3-5)": 0.5253, "upper (1-2)": 0.5022}   # the H1 run
    unmatched = {"lower (3-5)": 0.4548, "upper (1-2)": 0.4773}  # h1_diagnostics

    for name, mask in (("lower (3-5)", is_lower), ("upper (1-2)", ~is_lower)):
        sub = graded[mask].reset_index(drop=True)
        bets = simulate(sub, p[mask], PINNACLE_PRE, RULE)
        # Only bets whose closing price actually resolved contribute to CLV,
        # so the odds histogram must be taken from those and not from every
        # bet placed -- otherwise the null is matched to the wrong population.
        close = closing_price_for_bets(bets, sub).to_numpy(float)
        keep = np.isfinite(close) & (close > 0)
        bet_odds = bets["odds"].to_numpy(float)[keep]

        sims = np.array([matched_null(sub, bet_odds, seed=s) for s in range(N_SIMS)])
        sims = sims[np.isfinite(sims)]
        lo, hi = np.quantile(sims, [0.025, 0.975])
        print(f"  {name}: {len(bet_odds):,} bets, mean odds {bet_odds.mean():.3f}")
        print(f"    odds-matched null   {sims.mean():.4f}  95% [{lo:.4f}, {hi:.4f}]")
        print(f"    unmatched null      {unmatched[name]:.4f}")
        print(f"    model observed      {observed[name]:.4f}")
        print(f"    margin over matched null  {observed[name] - sims.mean():+.4f}")
        print(f"    is 0.5 inside the matched null's interval? "
              f"{bool(lo <= 0.5 <= hi)}")
        print()

    print("-" * 78)
    print("THE SAME NULL PER SEASON -- context for the 2025-26 holdout control")
    print("-" * 78)
    print("  scripts/h1_holdout_tiers.py measured 38.19% shortened in the lower")
    print("  stratum out of sample, against 52.53% in sample. That number is")
    print("  uninterpretable until the era's own drift is known: if 2025-26's")
    print("  band-eligible prices lengthened for everybody, 38% may be normal")
    print("  for the season rather than a collapse of the finding.")
    print()
    print("  No model is involved here -- these are prices only, so no fit is")
    print("  needed and nothing is being scored.")
    print()
    full_panel, _, _ = build_panel()
    from src.features.build import load as _load
    from src.features.ratings import TIER
    everything = _load()
    everything = everything[(everything["source"] == "main")
                            & everything["result"].notna()
                            & everything["div"].isin(TIER)]
    rows = []
    for season, chunk in everything.groupby("season"):
        if season < "2015-16":
            continue
        rec = {"season": season}
        for label, divs in (("lower", True), ("upper", False)):
            m = chunk["div"].isin(LOWER_DIVS) if divs else ~chunk["div"].isin(LOWER_DIVS)
            _, ratio = eligible_cells(chunk[m])
            rec[f"{label}_n"] = len(ratio)
            rec[f"{label}_pct_short"] = float((ratio > 1.0).mean()) if len(ratio) else float("nan")
        rows.append(rec)
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda v: f"{v:.4f}"))
    print()

    print("  If 0.5 sits OUTSIDE these intervals, the pre-registered binomial")
    print("  test was run against a null the market does not obey. The H1")
    print("  verdict still stands as pre-registered -- the bar was inherited")
    print("  and committed in advance, and it is not rewritten after the fact.")
    print("  What changes is the INTERPRETATION, which is where a measured")
    print("  null belongs.")


if __name__ == "__main__":
    main()
