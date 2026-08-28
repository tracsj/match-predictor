"""Do the two marginal CLV results survive treating a matchday as one draw?

    PYTHONPATH=. uv run python scripts/day_clustered_clv.py

A control, not a candidate. It fits nothing new and does not move the registry
count -- it recomputes a statistic on two FIXED, already-counted bet
populations, exactly as `scripts/phase6_null_reanalysis.py` does, and neither
population can change in response to what it finds.

WHY THIS ARM EXISTS. Every shortening-rate z this project has reported uses
`sqrt(p(1-p)/n)`, which assumes bets are independent draws. They are not, and
this repo has known it since the beginning: `bootstrap_ci` resamples MATCHDAYS
rather than bets precisely because same-day bets share news cycles and
market-wide moves. Ignoring that inflates z by the square root of the design
effect.

That cannot overturn z of 7 to 14. It can decide a result sitting at z = 2.4,
and this project has two of those, both already labelled narrow:

  Phase 6, 2025-26 all divisions   0.4241 vs a matched null of 0.3926, z 2.36
  H1 out of sample, lower tiers    0.3819 vs a matched null of 0.3145, z 2.55

H1's IN-SAMPLE lower stratum runs as a third arm and is not marginal at all
(z 14.2 on 9,920 bets). It is the control: a correction strong enough to
overturn that one would be too strong to believe, so it is here to fail to
matter. Without it, the two attenuations below have nothing to be read against.

THE GATE COMES FIRST. Each arm reproduces its recorded INDEPENDENT z before any
clustering runs. If the rebuilt population does not reproduce the published
number, the clustered number is not comparable to it and reporting one would be
worse than reporting nothing -- the same discipline `phase6_null_reanalysis.py`
applies, and for the same reason.

The null is treated as exact here. It is not: it comes from 200 matched
simulations and carries its own error. Folding that in would push p FURTHER
from significance, never toward it, so it cannot rescue either result and the
headline does not need it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.betting import (
    PINNACLE_CLOSE, PINNACLE_PRE, clv_report, closing_price_for_bets,
    day_clustered_shortening_test, simulate,
)
from src.features.ratings import TIER
from src.h1 import LOWER_DIVS, RULE
from src.phase6 import build_holdout, fit_and_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h1_odds_matched_null import matched_null  # noqa: E402

N_SIMS = 200
N_BOOT = 10_000
SEED = 0

P6_CACHE = Path("data/processed/phase6_holdout_predictions.npy")
H1_CACHE = Path("data/processed/h1_holdout_predictions.npy")

# What the docs record, and what each arm must reproduce before it is allowed
# to say anything. Written here by hand from docs/PHASE6_RESULT.md and
# docs/H1_RESULT.md rather than recomputed, so the gate can actually fail.
PUBLISHED = {
    "phase6": {"n_bets": 1337, "pct_shortened": 0.4241, "null": 0.3926, "z": 2.36},
    "h1_oos_lower": {"n_bets": 309, "pct_shortened": 0.3819, "null": 0.3145, "z": 2.55},
    # In sample, and NOT marginal -- included as the control that shows what
    # clustering can and cannot reach. Its recorded z is 14.2.
    "h1_in_sample_lower": {"n_bets": 9920, "pct_shortened": 0.5253,
                           "null": 0.4544, "z": 14.2},
}
TOL = {"n_bets_frac": 0.02, "pct_shortened": 0.005, "null": 0.02, "z": 0.15}


def arm(name: str, sub: pd.DataFrame, probs: np.ndarray) -> dict | None:
    """One population: rebuild the bets, gate on the recorded z, then cluster."""
    want = PUBLISHED[name]
    bets = simulate(sub, probs, PINNACLE_PRE, RULE)
    close = closing_price_for_bets(bets, sub)
    # Legacy null, stated explicitly: the gate compares against a published row
    # computed this way. The matched null follows immediately below.
    rep = clv_report(bets, close, null_rate=0.5, null_ratio=1.0)

    arr = close.to_numpy(float)
    keep = np.isfinite(arr) & (arr > 0)
    bet_odds = bets["odds"].to_numpy(float)[keep]
    sims = np.array([matched_null(sub, bet_odds, seed=s) for s in range(N_SIMS)])
    sims = sims[np.isfinite(sims)]
    null = float(sims.mean())

    se_ind = float(np.sqrt(null * (1 - null) / rep["n"]))
    z_ind = (rep["pct_shortened"] - null) / se_ind

    checks = [
        ("n_bets", rep["n"], want["n_bets"],
         abs(rep["n"] - want["n_bets"]) <= TOL["n_bets_frac"] * want["n_bets"]),
        ("pct_shortened", rep["pct_shortened"], want["pct_shortened"],
         abs(rep["pct_shortened"] - want["pct_shortened"]) <= TOL["pct_shortened"]),
        ("matched null", null, want["null"], abs(null - want["null"]) <= TOL["null"]),
        ("z, independent", z_ind, want["z"], abs(z_ind - want["z"]) <= TOL["z"]),
    ]
    print(f"  REPRODUCTION GATE -- {name}")
    for label, got, published, ok in checks:
        print(f"    {label:16} rebuilt {got:<11.4f} published {published:<10.4f} "
              f"{'PASS' if ok else 'FAIL'}")
    if not all(ok for *_, ok in checks):
        print("    >>> GATE FAILED. This population is not the recorded one, so a")
        print("        clustered z computed on it would answer a different question.")
        return None
    print("    >>> GATE PASSED.")
    print()

    clustered = day_clustered_shortening_test(bets, close, null_rate=null,
                                              n_boot=N_BOOT, seed=SEED)
    return {
        "result": name, "n_bets": rep["n"], "n_days": clustered["n_blocks"],
        "observed": rep["pct_shortened"], "null": null,
        "z_indep": z_ind,
        "p_indep": float(2 * (1 - stats.norm.cdf(abs(z_ind)))),
        "z_clustered": clustered["z"], "p_clustered": clustered["pvalue"],
        "se_indep": se_ind, "se_clustered": clustered["se_boot"],
    }


def main() -> None:
    print("=" * 78)
    print("DAY-CLUSTERED SHORTENING TEST -- programme item 2")
    print("=" * 78)
    print(f"  {N_BOOT:,} block-bootstrap resamples of whole matchdays, seed {SEED}")
    print()

    train, test, seq_all, cutoff = build_holdout()

    # Both arms share this fit -- it is the same holdout, the same settled
    # configuration and the same three seeds. Reusing it changes no number and
    # is what the two source scripts already do.
    if H1_CACHE.exists():
        p_all = np.load(H1_CACHE)
    else:
        p_all = fit_and_predict(train, test, seq_all)
        H1_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(H1_CACHE, p_all)

    rows = []

    # ---- Phase 6: the whole 2025-26 holdout, every division ----
    need = PINNACLE_PRE.cols + PINNACLE_CLOSE.cols
    m = test[need].notna().all(axis=1).to_numpy()
    got = arm("phase6", test[m].reset_index(drop=True), p_all[m])
    if got:
        rows.append(got)

    # ---- H1 out of sample: the same holdout, lower tiers only ----
    keep = (test["div"].isin(TIER) & test[need].notna().all(axis=1)).to_numpy()
    graded = test[keep].reset_index(drop=True)
    is_lower = graded["div"].isin(LOWER_DIVS).to_numpy()
    got = arm("h1_oos_lower", graded[is_lower].reset_index(drop=True),
              p_all[keep][is_lower])
    if got:
        rows.append(got)

    # ---- H1 in sample: the control. If clustering could overturn z = 14.2 the
    # correction would be too strong to believe, so this arm is here to fail
    # to matter.
    from src.h1 import CACHE as H1_PANEL_CACHE, build_panel
    if H1_PANEL_CACHE.exists():
        panel, _, _ = build_panel()
        z = np.load(H1_PANEL_CACHE, allow_pickle=False)
        ins = panel.iloc[z["test_idx"]].reset_index(drop=True)
        ins_lower = ins["div"].isin(LOWER_DIVS).to_numpy()
        got = arm("h1_in_sample_lower", ins[ins_lower].reset_index(drop=True),
                  z["probs"][ins_lower])
        if got:
            rows.append(got)
    else:
        print(f"  (skipping the in-sample control: {H1_PANEL_CACHE} absent)")
        print()

    if not rows:
        raise SystemExit("no arm passed its gate; nothing to report")

    print("-" * 78)
    print("RESULT")
    print("-" * 78)
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda v: f"{v:.4f}"))
    print()
    for r in rows:
        infl = r["se_clustered"] / r["se_indep"]
        print(f"  {r['result']}: {r['n_bets']:,} bets over {r['n_days']} days, "
              f"error inflated {infl:.2f}x (design effect {infl ** 2:.2f})")
    print()
    print("  A design effect near 1 would mean matchdays carry no shared")
    print("  structure and the original z stood. Well above 1 means the")
    print("  original z counted correlated bets as independent evidence.")


if __name__ == "__main__":
    main()
