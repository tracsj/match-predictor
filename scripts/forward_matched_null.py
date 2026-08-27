"""Does odds-matching move the forward ledger's CLV null?

    PYTHONPATH=. uv run python scripts/forward_matched_null.py

A control, not a candidate. It fits nothing and does not move the registry
count -- it measures the market, in the same sense as the exchange-vs-Pinnacle
benchmark and `scripts/h1_odds_matched_null.py`, whose method this copies.

WHY THIS ARM EXISTS. `src/grade.py` tests the model's shortening rate against a
null measured from the settled forward rows themselves, taken over every cell
the rule could legally have bet. That null is UNMATCHED on odds: it samples the
whole eligible band, while the rule bets the maximum-EV outcome and lands at
longer prices than the band's average. The favourite-longshot literature says
long and short prices drift differently, so an unmatched null could be an
odds-mix artifact wearing a measurement's clothes.

H1 answered this for the Pinnacle ladder -- matched and unmatched agreed to
0.002 there and 0.004 on Phase 6's window -- but the exchange pre-close is a
different instrument, taken about a day out with a far wider book, and an
agreement measured on Pinnacle is not evidence about this ladder. Importing it
would be the same borrowing the grade.py docstring refuses to do.

The claim this script exists to keep honest is the one in
`measured_shortening_null`: that matched and unmatched agree to 0.0012 here.
Re-run it whenever the forward corpus grows. If the two ever separate, the
ledger's null needs matching and that docstring is wrong.
"""

from __future__ import annotations

import numpy as np

import src.grade as grade
from src.eval.betting import (
    EXCHANGE_CLOSE, EXCHANGE_PRE, closing_price_for_bets, simulate,
)

N_SIMS = 200
N_DECILES = 10


def eligible_cells(df):
    """Every (row, outcome) the rule could legally have bet, with its ratio.

    Both legs filtered `> 1.0`, not `> 0`: this feed carries prices at or below
    1.0 that are missing data wearing a number, invisible to `notna()`.
    """
    pre = df[EXCHANGE_PRE.cols].to_numpy(float)
    close = df[EXCHANGE_CLOSE.cols].to_numpy(float)
    ok = (np.isfinite(pre) & np.isfinite(close) & (pre > 1.0) & (close > 1.0)
          & (pre >= grade.RULE.min_odds) & (pre <= grade.RULE.max_odds))
    r, c = np.nonzero(ok)
    return pre[r, c], pre[r, c] / close[r, c]


def matched_null(cell_odds, cell_ratio, bet_odds, seed: int) -> float:
    """% shortened for a random draw whose odds histogram matches the model's.

    Deciles come from the MODEL's bets, then each decile is filled from
    eligible cells priced inside that same decile, in the proportion the model
    bet it. A decile with no eligible cells is skipped rather than filled from
    a neighbour, because borrowing prices from an adjacent bucket is the exact
    confound this arm exists to remove.
    """
    edges = np.quantile(bet_odds, np.linspace(0, 1, N_DECILES + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rng = np.random.default_rng(seed)

    picked = []
    for i in range(N_DECILES):
        want = int(((bet_odds >= edges[i]) & (bet_odds < edges[i + 1])).sum())
        pool = np.nonzero((cell_odds >= edges[i]) & (cell_odds < edges[i + 1]))[0]
        if want == 0 or len(pool) == 0:
            continue
        picked.append(cell_ratio[rng.choice(pool, size=want, replace=True)])
    if not picked:
        return float("nan")
    return float((np.concatenate(picked) > 1.0).mean())


def main() -> None:
    preds, _ = grade.load_predictions(verbose=False)
    settled = grade.join_results(preds)
    settled = settled[settled["result"].notna()].reset_index(drop=True)

    need = EXCHANGE_PRE.cols + EXCHANGE_CLOSE.cols
    sub = settled[settled[need].notna().all(axis=1).to_numpy()].reset_index(drop=True)
    if len(sub) < 30:
        raise SystemExit(f"only {len(sub)} settled rows carry both exchange ladders")

    cell_odds, cell_ratio = eligible_cells(sub)
    probs = sub[["p_home", "p_draw", "p_away"]].to_numpy(float)
    bets = simulate(sub, probs, EXCHANGE_PRE, grade.RULE)

    # The odds histogram must come from the bets whose closing price actually
    # resolved, not from every bet placed -- otherwise the null is matched to a
    # population the CLV number was never computed over.
    close = closing_price_for_bets(bets, sub, grade.CLOSE_FOR_EXCHANGE).to_numpy(float)
    keep = np.isfinite(close) & (close > 1.0)
    bet_odds = bets["odds"].to_numpy(float)[keep]
    observed = float((bet_odds / close[keep] > 1.0).mean())

    unmatched = float((cell_ratio > 1.0).mean())
    sims = np.array([matched_null(cell_odds, cell_ratio, bet_odds, s)
                     for s in range(N_SIMS)])
    sims = sims[np.isfinite(sims)]
    lo, hi = np.quantile(sims, [0.025, 0.975])

    print("=" * 78)
    print("ODDS-MATCHED NULL, FORWARD EXCHANGE LADDER")
    print("=" * 78)
    print(f"  {len(sub)} settled rows, {len(cell_odds)} eligible cells, "
          f"{len(bet_odds)} graded bets")
    print(f"  {N_SIMS} sims over {N_DECILES} odds deciles taken from the model's own bets")
    print()
    print(f"  observed (the model)   {observed:.4f}")
    print(f"  unmatched null         {unmatched:.4f}   <- what src/grade.py uses")
    print(f"  odds-matched null      {sims.mean():.4f}   95% [{lo:.4f}, {hi:.4f}]")
    print(f"  matched - unmatched   {sims.mean() - unmatched:+.4f}")
    print()
    print(f"  model bets mean odds {bet_odds.mean():.3f} against eligible cells "
          f"at {cell_odds.mean():.3f},")
    print("  so there IS a mix difference for matching to correct for.")
    print()
    if abs(sims.mean() - unmatched) < 0.01:
        print("  The two agree. The unmatched null in src/grade.py stands, and the")
        print("  margin over it is not an odds-mix artifact.")
    else:
        print("  ⚠️  The two have separated. src/grade.py's unmatched null is no longer")
        print("  defensible and `measured_shortening_null` needs matching -- its")
        print("  docstring's 0.0012 claim is stale.")


if __name__ == "__main__":
    main()
