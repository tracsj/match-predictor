"""Phase 6's CLV, re-read against a measured null. Executes the pre-reg at ac50004.

    uv run python scripts/phase6_null_reanalysis.py

Fixed by `docs/PREREG_PHASE6_NULL.md`, committed before the bet population was
re-derived and before any matched null existed for it. This file chooses
nothing.

It does NOT increment the registry count: it recomputes a statistic on a fixed
bet population from a run already counted. Same category as re-slicing Phase 6
by tier.

THE GATE COMES FIRST. Before any null is computed, the regenerated Pinnacle
pre-close population must reproduce the published row in
`docs/PHASE6_RESULT.md` -- 1,337 bets, mean ratio 0.9952, 42.4% shortened --
within tolerances fixed in the pre-registration. If it does not, this stops.
A null measured against a population that is not Phase 6's would answer a
question nobody asked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.betting import (
    B365_PRE, MARKET_AVG_PRE, MARKET_MAX_PRE, PINNACLE_CLOSE, PINNACLE_PRE,
    BetRule, clv_report, closing_price_for_bets, simulate,
)
from src.phase6 import build_holdout, fit_and_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h1_odds_matched_null import matched_null  # noqa: E402

RULE = BetRule(min_ev=0.05, min_odds=1.5, max_odds=5.0, stake=1.0,
               name="pre-registered: ev>=0.05, odds 1.5-5.0")
N_SIMS = 200
CACHE = Path("data/processed/h1_holdout_predictions.npy")

# The published row this must reproduce, and the tolerances fixed in advance.
PUBLISHED = {"n_bets": 1337, "mean_ratio": 0.9952, "pct_shortened": 0.424}
TOL = {"n_bets_frac": 0.02, "mean_ratio": 0.002, "pct_shortened": 0.01}


def matched_null_for(df: pd.DataFrame, prices, bet_odds: np.ndarray) -> np.ndarray:
    """Null shortening rates for one price set, matched to its own bets' odds.

    `matched_null` reads the pre-close from PINNACLE_PRE, so for the non-
    Pinnacle ladders the frame is handed over with that price set's columns
    renamed into those slots. Renaming rather than re-implementing keeps this
    null identical to the one the H1 diagnostics reported, which is the single
    thing these files must agree on.
    """
    if prices is PINNACLE_PRE:
        sub = df
    else:
        sub = df.copy()
        for src_col, dst_col in zip(prices.cols, PINNACLE_PRE.cols):
            sub[dst_col] = df[src_col].to_numpy(float)
    return np.array([matched_null(sub, bet_odds, seed=s) for s in range(N_SIMS)])


def main() -> None:
    train, test, seq_all, cutoff = build_holdout()
    print("=" * 78)
    print("PHASE 6 CLV RE-ANALYSIS -- executing docs/PREREG_PHASE6_NULL.md")
    print("=" * 78)
    print(f"  holdout   {len(test):,} matches, season 2025-26")
    print(f"  training  {len(train):,} matches, all before {pd.Timestamp(cutoff).date()}")
    print()

    if CACHE.exists():
        print(f"  reusing predictions from {CACHE}")
        print("  (produced by phase6's own build_holdout + fit_and_predict --")
        print("   the gate below is what proves they are the right ones)")
        p = np.load(CACHE)
    else:
        print("  fitting (3 seeds, the settled configuration)...")
        p = fit_and_predict(train, test, seq_all)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE, p)

    # ---- the gate ----
    need = PINNACLE_PRE.cols + PINNACLE_CLOSE.cols
    mask = test[need].notna().all(axis=1).to_numpy()
    sub = test[mask].reset_index(drop=True)
    bets = simulate(sub, p[mask], PINNACLE_PRE, RULE)
    rep = clv_report(bets, closing_price_for_bets(bets, sub))

    print()
    print("-" * 78)
    print("REPRODUCTION GATE -- does this population match the published row?")
    print("-" * 78)
    checks = [
        ("n_bets", rep["n"], PUBLISHED["n_bets"],
         abs(rep["n"] - PUBLISHED["n_bets"]) <= TOL["n_bets_frac"] * PUBLISHED["n_bets"]),
        ("mean_ratio", rep["mean_ratio"], PUBLISHED["mean_ratio"],
         abs(rep["mean_ratio"] - PUBLISHED["mean_ratio"]) <= TOL["mean_ratio"]),
        ("pct_shortened", rep["pct_shortened"], PUBLISHED["pct_shortened"],
         abs(rep["pct_shortened"] - PUBLISHED["pct_shortened"]) <= TOL["pct_shortened"]),
    ]
    for name, got, want, ok in checks:
        print(f"  {name:16} regenerated {got:<12.4f} published {want:<10.4f} "
              f"{'PASS' if ok else 'FAIL'}")

    if not all(ok for *_, ok in checks):
        print()
        print("  >>> GATE FAILED. The re-analysis stops here, as pre-registered.")
        print("      The regenerated population is not Phase 6's, so a null")
        print("      measured against it would answer a different question.")
        raise SystemExit(1)

    print()
    print("  >>> GATE PASSED. This is Phase 6's population.")

    # ---- the measured nulls ----
    print()
    print("-" * 78)
    print("CLV AGAINST A MATCHED NULL, ALL FOUR PRE-CLOSE LADDERS")
    print("-" * 78)
    print(f"  {N_SIMS} sims, ten odds deciles taken from each ladder's own bets.")
    print("  Pinnacle pre-close is PRIMARY. The other three are descriptive and")
    print("  can neither overturn nor rescue it.")
    print()

    rows = []
    for prices, role in ((PINNACLE_PRE, "PRIMARY"), (B365_PRE, "descriptive"),
                         (MARKET_AVG_PRE, "descriptive"), (MARKET_MAX_PRE, "descriptive")):
        need = prices.cols + PINNACLE_CLOSE.cols
        if not all(c in test.columns for c in need):
            continue
        m = test[need].notna().all(axis=1).to_numpy()
        if m.sum() < 50:
            continue
        s = test[m].reset_index(drop=True)
        b = simulate(s, p[m], prices, RULE)
        if b.empty:
            continue
        close = closing_price_for_bets(b, s)
        r = clv_report(b, close)
        arr = close.to_numpy(float)
        bet_odds = b["odds"].to_numpy(float)[np.isfinite(arr) & (arr > 0)]

        sims = matched_null_for(s, prices, bet_odds)
        sims = sims[np.isfinite(sims)]
        null, lo, hi = sims.mean(), *np.quantile(sims, [0.025, 0.975])
        se = np.sqrt(null * (1 - null) / r["n"]) if r["n"] else np.nan
        zz = (r["pct_shortened"] - null) / se if se else np.nan

        if prices is PINNACLE_PRE:
            verdict = ("OVERTURNED" if r["pct_shortened"] > hi else
                       "UPHELD" if r["pct_shortened"] < lo else "INCONCLUSIVE")
        else:
            verdict = "—"

        rows.append({"ladder": prices.label, "role": role, "n_bets": r["n"],
                     "observed": r["pct_shortened"], "null": null,
                     "null_lo": lo, "null_hi": hi,
                     "margin": r["pct_shortened"] - null, "z": zz,
                     "mean_ratio": r["mean_ratio"], "verdict": verdict})

    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    primary = tbl[tbl["role"] == "PRIMARY"].iloc[0]
    print()
    print("=" * 78)
    print("THE ANSWER, BY THE PRE-REGISTERED RULE")
    print("=" * 78)
    print(f"  Pinnacle pre-close: {primary['verdict']}")
    print(f"    observed {primary['observed']:.4f} shortened against a matched null of "
          f"{primary['null']:.4f} 95% [{primary['null_lo']:.4f}, {primary['null_hi']:.4f}]")
    print(f"    margin {primary['margin']:+.4f}, z {primary['z']:.2f}, "
          f"p {2 * (1 - stats.norm.cdf(abs(primary['z']))):.4g}")
    print()
    if primary["verdict"] == "OVERTURNED":
        print("  docs/PHASE6_RESULT.md's reading -- that the selections sat on the")
        print("  wrong side of the market's own movement -- does not survive. It")
        print("  gets a correction rather than a flag.")
    elif primary["verdict"] == "UPHELD":
        print("  The published reading survives a correctly specified null.")
    else:
        print("  Phase 6's CLV distinguishes nothing about direction. That is")
        print("  still a correction to a page that read it as a clear negative.")
    print()
    print("  WHAT THIS DOES NOT DO, per the pre-registration: it does not make")
    print("  Phase 6 profitable and must not be read as softening it. The ROI")
    print("  tables stand, the rule lost money in every price column, and a CLV")
    print("  direction is not a yield.")


if __name__ == "__main__":
    main()
