"""Post-hoc controls for the H1 result. Run AFTER 4bc56bc recorded it.

    uv run python scripts/h1_diagnostics.py

These are controls, not candidates. They place no new configuration in the
search and do not move the registry count, in the same sense as the grading
dry run already ruled on in docs/PROGRAMME.md: the "model" in a control is a
deliberate null, not something anyone hopes will win.

THE QUESTION. H1's lower stratum returned CLV 1.0083 with 52.53% of prices
shortening. There is a mechanism that produces exactly that with ZERO model
skill. The rule bets the maximum-EV outcome, which preferentially selects
outcomes whose pre-close price is long relative to the model's opinion. In a
thin market the pre-close price is NOISIER, so "long relative to the model" is
more often simply "wrong", and the close corrects it. Prices shorten, CLV
exceeds 1.0, and the model contributed nothing beyond disagreeing.

If that is what happened, it is still a finding -- but it is H3's finding
(line movement is predictable) rather than evidence the network knows
anything about football, and it belongs to the market rather than the model.

Three controls discriminate:

  A. ORDERED LOGIT through the identical rule, rows and strata. A real
     forecasting model with none of the network's capacity. If it shows the
     same effect, the network is incidental.
  B. RANDOM selection among band-eligible outcomes, bet counts matched. If
     even this shortens, the drift is unconditional and no selection rule --
     model-driven or not -- deserves credit for it.
  C. ANTI-MODEL: bet the MINIMUM-EV outcome instead of the maximum. If
     disagreement-with-the-price is what shortens prices, this should show
     the effect INVERTED. If it shows the same effect, the direction of
     disagreement is irrelevant and the model certainly is.

Plus two omissions from the headline run, repaired here rather than by
re-running it: the MEDIAN ratio (clv_report always computed it; the table
simply failed to print it), and ROI at the prices actually TAKEN, which is
the Buchdahl check on whether the CLV survives the margin as yield.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.betting import (
    PINNACLE_CLOSE, PINNACLE_PRE, BetRule, clv_report, closing_price_for_bets,
    simulate,
)
from src.eval.devig import devig
from src.eval.metrics import OUTCOMES
from src.eval.split import season_walk_forward
from src.experiments import baseline_predictions
from src.models.baselines import ALL_FEATURES
from src.h1 import CACHE, LOWER_DIVS, RULE, build_panel

N_RANDOM_SIMS = 200


def strata_of(graded: pd.DataFrame) -> dict:
    is_lower = graded["div"].isin(LOWER_DIVS).to_numpy()
    return {"lower (3-5)": is_lower, "upper (1-2)": ~is_lower}


def clv_row(name: str, df: pd.DataFrame, probs: np.ndarray, pick_min=False) -> dict:
    sub = df.reset_index(drop=True)
    if pick_min:
        # The anti-model: invert the EV ordering so argmax selects the outcome
        # the rule would normally like LEAST, while every other part of the
        # rule -- the +5% threshold, the odds band -- is untouched.
        odds = sub[PINNACLE_PRE.cols].to_numpy(float)
        ev = probs * odds - 1.0
        probs = np.where(np.isfinite(ev), -ev, -np.inf)
        probs = (probs - probs.min(axis=1, keepdims=True) + 1e-9)
        probs = probs / probs.sum(axis=1, keepdims=True)
    bets = simulate(sub, probs, PINNACLE_PRE, RULE)
    if bets.empty:
        return {"arm": name, "n_bets": 0}
    # Legacy null, stated explicitly. These numbers are recorded in docs/ and
    # 0.5/1.0 reproduces them exactly. The measured drift that supersedes this
    # null as a standard is in docs/H1_RESULT.md.
    rep = clv_report(bets, closing_price_for_bets(bets, sub),
                     null_rate=0.5, null_ratio=1.0)
    return {"arm": name, "n_bets": rep["n"], "mean_ratio": rep["mean_ratio"],
            "median_ratio": rep["median_ratio"], "pct_shortened": rep["pct_shortened"],
            "binom_p": rep["binom_pvalue"]}


def random_in_band(df: pd.DataFrame, n_bets: int, seed: int) -> dict:
    """Pick a random eligible outcome, `n_bets` times, and measure its CLV.

    Eligible means exactly what the rule means by it -- a finite pre-close and
    closing price for that selection, and a pre-close price inside [1.5, 5.0].
    The EV threshold cannot apply, because a random pick has no model behind
    it to compute EV from; that is the one respect in which this control is
    not the rule, and it is why it answers "does the market drift?" rather
    than "does the rule work?".
    """
    pre = df[PINNACLE_PRE.cols].to_numpy(float)
    close = df[PINNACLE_CLOSE.cols].to_numpy(float)
    ok = (np.isfinite(pre) & np.isfinite(close) & (close > 0)
          & (pre >= RULE.min_odds) & (pre <= RULE.max_odds))
    rows, cols = np.nonzero(ok)
    if len(rows) == 0:
        return {}
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(rows), size=min(n_bets, len(rows)))
    r, c = rows[pick], cols[pick]
    ratio = pre[r, c] / close[r, c]
    shortened = ratio > 1.0
    return {"mean_ratio": float(ratio.mean()),
            "median_ratio": float(np.median(ratio)),
            "pct_shortened": float(shortened.mean())}


def main() -> None:
    panel, full, _ = build_panel()
    if not Path(CACHE).exists():
        raise SystemExit(f"{CACHE} missing -- run `uv run python -m src.h1` first")
    z = np.load(CACHE, allow_pickle=False)
    test_idx, p = z["test_idx"], z["probs"]
    graded = panel.iloc[test_idx].reset_index(drop=True)
    strata = strata_of(graded)

    print("=" * 78)
    print("H1 POST-HOC CONTROLS -- run after the result was committed at 4bc56bc")
    print("=" * 78)
    print(f"  graded {len(graded):,} rows, "
          f"{int(strata['lower (3-5)'].sum()):,} lower / "
          f"{int(strata['upper (1-2)'].sum()):,} upper")
    print()

    print("-" * 78)
    print("REPAIR 1 -- the median ratio the headline table failed to print")
    print("-" * 78)
    print("  The mean of taken/close is biased upward by Jensen's inequality:")
    print("  the ratio is bounded below at 0 and unbounded above, so symmetric")
    print("  price noise alone lifts the mean above 1.0. The median does not")
    print("  have that problem, and neither does % shortened.")
    print()
    rows = [clv_row(f"the net, {name}", graded[m], p[m]) for name, m in strata.items()]
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda v: f"{v:.4f}"))

    print()
    print("-" * 78)
    print("CONTROL C -- the anti-model: bet the MINIMUM-EV outcome")
    print("-" * 78)
    print("  If selecting on disagreement-with-the-price is what makes prices")
    print("  shorten, inverting the selection should invert the effect.")
    print()
    rows = [clv_row(f"anti-model, {name}", graded[m], p[m], pick_min=True)
            for name, m in strata.items()]
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda v: f"{v:.4f}"))

    print()
    print("-" * 78)
    print(f"CONTROL B -- random eligible selections, {N_RANDOM_SIMS} sims, counts matched")
    print("-" * 78)
    print("  Does the lower-division pre-close simply drift shorter for")
    print("  everybody, regardless of who picks?")
    print()
    target = {"lower (3-5)": 9920, "upper (1-2)": 21459}   # the run's bet counts
    for name, m in strata.items():
        sims = [random_in_band(graded[m], target[name], seed=s)
                for s in range(N_RANDOM_SIMS)]
        sims = [s for s in sims if s]
        pct = np.array([s["pct_shortened"] for s in sims])
        mr = np.array([s["mean_ratio"] for s in sims])
        print(f"  {name}: % shortened {pct.mean():.4f} "
              f"95% [{np.quantile(pct, 0.025):.4f}, {np.quantile(pct, 0.975):.4f}]   "
              f"mean ratio {mr.mean():.4f}")

    print()
    print("-" * 78)
    print("CONTROL A -- ordered logit, same rows, same rule, same strata")
    print("-" * 78)
    print("  A real model with none of the network's capacity. Fitting...")
    print()
    P_lr, y_lr = baseline_predictions(panel, features=ALL_FEATURES, train_pool=full)
    assert len(P_lr) == len(graded), f"{len(P_lr)} logit rows vs {len(graded)} graded"
    assert (y_lr == graded["result"].to_numpy()).all(), "logit rows are misaligned"
    rows = [clv_row(f"ordered logit, {name}", graded[m], P_lr[m])
            for name, m in strata.items()]
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda v: f"{v:.4f}"))

    print()
    print("-" * 78)
    print("REPAIR 2 -- ROI at the prices actually TAKEN (the Buchdahl check)")
    print("-" * 78)
    print("  CLV predicts level-stakes yield with a slope near 1.0. So the CLV")
    print("  bets, settled at the pre-close prices they were struck at, are the")
    print("  direct test of whether +0.83% CLV survives the margin as money.")
    print()
    for name, m in strata.items():
        sub = graded[m].reset_index(drop=True)
        bets = simulate(sub, p[m], PINNACLE_PRE, RULE)
        if bets.empty:
            continue
        roi = bets["pnl"].sum() / bets["stake"].sum()
        print(f"  {name}: {len(bets):,} bets, ROI at the taken price "
              f"{roi:+.4f}, hit {bets['won'].mean():.4f}, "
              f"avg odds {bets['odds'].mean():.3f}")

    print()
    print("=" * 78)
    print("HOW TO READ THESE")
    print("=" * 78)
    print("  If A and C also show the lower stratum shortening, the effect is")
    print("  in the market and the rule, not in the network -- which makes it")
    print("  H3's territory (line movement is predictable) and means H1's")
    print("  verdict stands while H1's CLAIM about model edge does not.")
    print("  If only the net shows it, the finding is much more interesting")
    print("  and still has to survive out-of-sample data.")


if __name__ == "__main__":
    main()
