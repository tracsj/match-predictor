"""H1: is the lower-division market priced less efficiently than the top tier?

    uv run python -m src.h1

Everything here is fixed by `docs/hypotheses/H1-lower-division-inefficiency.md`,
committed at 9212a3c before any tier-stratified number existed. This file
executes what was written down and prints whatever comes out. It chooses no
threshold, no market, no price column and no stratum boundary.

The one thing worth saying about the design. The PRIMARY test is absolute --
does the pooled lower stratum's CLV mean ratio exceed 1.0 with binomial p
below 0.01 -- and not relative. A relative result, lower above upper with both
under 1.0, is a fact about market microstructure rather than an edge, and the
pre-registration says so in advance precisely so it cannot be reinterpreted as
a win afterwards. The lower-minus-upper difference is computed and printed
because it is the shape of the claim, and it is labelled secondary.

Two further tables -- CLV per individual tier, and a sensitivity re-run on the
2016-17 window where SC2/SC3 carry Pinnacle prices -- are DESCRIPTIVE. They
rank nothing and decide nothing. Testing five tiers and reporting the best is
a five-way search; printing all five and deciding on neither is a description.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.betting import (
    B365_CLOSE, MARKET_MAX_CLOSE, PINNACLE_CLOSE, PINNACLE_PRE, BetRule,
    bootstrap_ci, clv_report, closing_price_for_bets, random_bet_null,
    simulate, summarize,
)
from src.eval.split import season_walk_forward
from src.experiments import run_walk_forward
from src.features.build import load as load_features, load_sequences
from src.models.baselines import ALL_FEATURES
from src.models.net import NetConfig

# ---- fixed by the pre-registration; do not tune ----
RULE = BetRule(min_ev=0.05, min_odds=1.5, max_odds=5.0, stake=1.0,
               name="pre-registered: ev>=0.05, odds 1.5-5.0")
SEEDS = (0, 1, 2)
PANEL_FIRST, PANEL_LAST = "2012-13", "2024-25"
LOWER_DIVS = {"E2", "E3", "EC", "SC2", "SC3"}     # tiers 3-5
MIN_BETS = 3250                                   # derived; see the pre-reg
SENSITIVITY_FIRST = "2016-17"                     # descriptive only

CACHE = Path("data/processed/h1_predictions.npz")


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """The graded panel, the full training corpus, and the sequence tensor.

    The panel needs BOTH Pinnacle legs, because CLV is defined only where the
    pre-close it was bet at and the close it is graded against both exist.
    Requiring only the close would silently widen the panel and then drop the
    same rows later, inside the CLV step, where the loss would not be counted.
    """
    full = load_features().sort_values("kickoff").reset_index(drop=True)
    full = full[full["result"].notna()].reset_index(drop=True)
    seq_all, _ = load_sequences()

    from src.features.ratings import TIER
    legs = PINNACLE_PRE.cols + PINNACLE_CLOSE.cols
    panel = full[
        (full["source"] == "main")
        & full["div"].isin(TIER)                       # untiered divisions are out
        & full["season"].between(PANEL_FIRST, PANEL_LAST)
        & full[legs].notna().all(axis=1)
    ].sort_values("kickoff").reset_index(drop=True)
    return panel, full, seq_all


def graded_predictions(panel, full, seq_all, verbose=True) -> tuple[pd.DataFrame, np.ndarray]:
    """Walk forward across the panel; return the graded rows and their probs.

    `run_walk_forward` is called unmodified, which is the point -- it is the
    same function the ablation campaign and the scoreboard used, so the model
    configuration here is provably the settled one rather than a re-typed
    approximation of it. Its hardcoded `min_train_seasons=3` is what makes the
    graded window 2015-16 onward while the panel opens at 2012-13.
    """
    out = run_walk_forward(
        panel, NetConfig(), features=ALL_FEATURES, calibrate=True,
        seeds=SEEDS, verbose=verbose, train_pool=full, sequences=seq_all,
    )
    # run_walk_forward stacks its test blocks in split order, so the panel rows
    # have to be gathered in exactly that order to stay aligned.
    test_idx = np.concatenate(
        [s.test_idx for s in season_walk_forward(panel, min_train_seasons=3)])
    graded = panel.iloc[test_idx].reset_index(drop=True)
    p = out["hda_calibrated"]
    assert len(graded) == len(p), f"{len(graded)} graded rows vs {len(p)} predictions"
    # The y vector run_walk_forward returns independently must agree with the
    # rows we gathered. If the ordering assumption above were wrong this is
    # where it would show, rather than as a quietly shifted result.
    assert (graded["result"].to_numpy() == out["y"]).all(), \
        "gathered panel rows do not match run_walk_forward's own y -- ordering is wrong"
    return graded, p


def clv_for(df: pd.DataFrame, probs: np.ndarray) -> tuple[dict, np.ndarray]:
    """CLV for one slice: bet at the Pinnacle pre-close, grade at its close."""
    sub = df.reset_index(drop=True)
    bets = simulate(sub, probs, PINNACLE_PRE, RULE)
    if bets.empty:
        return {"n": 0, "mean_ratio": float("nan")}, np.array([])
    close = closing_price_for_bets(bets, sub)
    rep = clv_report(bets, close)
    ratio = (bets["odds"].to_numpy(float) / np.asarray(close, float))
    return rep, ratio[np.isfinite(ratio)]


def _verdict(rep: dict) -> str:
    """The pre-registered decision rule, applied mechanically.

    Written as one function so the bar cannot drift between the place it is
    stated and the place it is applied.
    """
    if rep.get("n", 0) < MIN_BETS:
        return f"INCONCLUSIVE -- {rep.get('n', 0):,} bets, below the {MIN_BETS:,} floor"
    if rep["mean_ratio"] > 1.0 and rep["binom_pvalue"] < 0.01:
        return "SUPPORTED -- ratio above 1.0 and binomial p below 0.01"
    return "NOT SUPPORTED -- fails ratio > 1.0 and/or p < 0.01"


def _fmt(rows) -> str:
    return pd.DataFrame(rows).to_string(index=False,
                                        float_format=lambda v: f"{v:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse cached walk-forward predictions if present. The "
                         "model output is unchanged either way; this only avoids "
                         "refitting when the reporting code is edited.")
    args = ap.parse_args()

    panel, full, seq_all = build_panel()
    print("=" * 78)
    print("H1 -- tier-stratified closing-line value, run once against the pre-reg")
    print("=" * 78)
    print(f"  rule        {RULE.describe()}")
    print(f"  panel       {len(panel):,} matches, {panel['season'].nunique()} seasons, "
          f"{panel['div'].nunique()} divisions, both Pinnacle legs present")
    print(f"  lower       {sorted(LOWER_DIVS)}")
    print(f"  floor       {MIN_BETS:,} bets per stratum")
    print()

    if args.reuse and CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        idx, p = z["test_idx"], z["probs"]
        graded = panel.iloc[idx].reset_index(drop=True)
        print(f"  reusing cached predictions from {CACHE}")
    else:
        print("  fitting walk-forward (3 seeds per season, full-corpus training)...")
        graded, p = graded_predictions(panel, full, seq_all)
        test_idx = np.concatenate(
            [s.test_idx for s in season_walk_forward(panel, min_train_seasons=3)])
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, test_idx=test_idx, probs=p)

    print(f"  graded      {len(graded):,} matches, "
          f"{graded['season'].min()} -> {graded['season'].max()}, "
          f"{graded['season'].nunique()} test seasons")

    is_lower = graded["div"].isin(LOWER_DIVS).to_numpy()
    strata = {"lower (tiers 3-5)": is_lower, "upper (tiers 1-2)": ~is_lower}

    print()
    print("-" * 78)
    print("PRIMARY -- closing-line value by stratum")
    print("-" * 78)
    print("  Bet at the Pinnacle pre-close, grade against the Pinnacle close of")
    print("  the same selection. The pre-registered bar is ratio > 1.0 with a")
    print("  binomial p below 0.01, in the LOWER stratum. Absolute, not relative.")
    print()
    reps, ratios, rows = {}, {}, []
    for name, mask in strata.items():
        rep, ratio = clv_for(graded[mask], p[mask])
        reps[name], ratios[name] = rep, ratio
        rows.append({"stratum": name, "n_bets": rep["n"],
                     "mean_ratio": rep.get("mean_ratio", float("nan")),
                     "pct_shortened": rep.get("pct_shortened", float("nan")),
                     "binom_p": rep.get("binom_pvalue", float("nan"))})
    print(_fmt(rows))
    print()
    for name in strata:
        print(f"  {name}: {_verdict(reps[name])}")

    print()
    print("-" * 78)
    print("SECONDARY -- the difference between strata")
    print("-" * 78)
    print("  The shape of the claim, reported because omitting it would be")
    print("  evasive. It cannot promote H1 to supported on its own.")
    print()
    lo_r, up_r = ratios["lower (tiers 3-5)"], ratios["upper (tiers 1-2)"]
    if len(lo_r) and len(up_r):
        diff = lo_r.mean() - up_r.mean()
        t = stats.ttest_ind(lo_r, up_r, equal_var=False)
        # Welch interval on the difference of means.
        se = np.sqrt(lo_r.var(ddof=1) / len(lo_r) + up_r.var(ddof=1) / len(up_r))
        print(f"    lower - upper = {diff:+.5f}  "
              f"95% [{diff - 1.96 * se:+.5f}, {diff + 1.96 * se:+.5f}]  "
              f"Welch p = {t.pvalue:.4f}")
        print(f"    spans zero: {bool(diff - 1.96 * se < 0 < diff + 1.96 * se)}")
    else:
        print("    not computable -- a stratum placed no bets")

    print()
    print("-" * 78)
    print("DESCRIPTIVE -- CLV per individual tier (no verdict, ranked by nothing)")
    print("-" * 78)
    from src.features.ratings import TIER
    tier_of = graded["div"].map(TIER)
    rows = []
    for t in sorted(tier_of.dropna().unique()):
        m = (tier_of == t).to_numpy()
        rep, _ = clv_for(graded[m], p[m])
        rows.append({"tier": int(t), "divisions": ",".join(sorted(graded.loc[m, "div"].unique())),
                     "n_bets": rep["n"], "mean_ratio": rep.get("mean_ratio", float("nan")),
                     "pct_shortened": rep.get("pct_shortened", float("nan"))})
    print(_fmt(rows))
    print()
    print("  Printed in tier order, not sorted by result. An ordering visible")
    print("  here is a lead for a future pre-registration, not a finding of")
    print("  this one -- five tiers tested and the best reported is a five-way")
    print("  search, which is what the primary test above exists to avoid.")

    print()
    print("-" * 78)
    print(f"DESCRIPTIVE -- sensitivity, {SENSITIVITY_FIRST} onward (stable composition)")
    print("-" * 78)
    print("  SC2 and SC3 carry no Pinnacle price before 2016-17, so the lower")
    print("  stratum gains two divisions partway through the graded window.")
    print("  This re-run holds composition fixed. It cannot overturn the primary.")
    print()
    sens = (graded["season"] >= SENSITIVITY_FIRST).to_numpy()
    rows = []
    for name, mask in strata.items():
        m = mask & sens
        rep, _ = clv_for(graded[m], p[m])
        rows.append({"stratum": name, "n_bets": rep["n"],
                     "mean_ratio": rep.get("mean_ratio", float("nan")),
                     "pct_shortened": rep.get("pct_shortened", float("nan")),
                     "binom_p": rep.get("binom_pvalue", float("nan"))})
    print(_fmt(rows))

    print()
    print("-" * 78)
    print("SECONDARY -- ROI, three price columns, led by the sharpest")
    print("-" * 78)
    print("  Under-powered by design and reported for completeness. CLV above")
    print("  is the headline; a column-3-only positive is an odds screen.")
    print()
    rows = []
    for name, mask in strata.items():
        for ps, note in ((PINNACLE_CLOSE, "truth test"),
                         (B365_CLOSE, "an account you could hold"),
                         (MARKET_MAX_CLOSE, "optimistic bound")):
            sub = graded[mask].reset_index(drop=True)
            ok = sub[ps.cols].notna().all(axis=1).to_numpy()
            if ok.sum() < 50:
                rows.append({"stratum": name, "price_set": ps.label,
                             "n_eligible": int(ok.sum()), "n_bets": 0})
                continue
            s2 = sub[ok].reset_index(drop=True)
            bets = simulate(s2, p[mask][ok], ps, RULE)
            row = summarize(bets, with_ci=False)
            row["stratum"], row["note"] = name, note
            row["n_eligible"] = int(ok.sum())
            if len(bets):
                ci = bootstrap_ci(bets, n_boot=args.boot, seed=0)
                row["roi_lo"], row["roi_hi"] = ci["lo"], ci["hi"]
                row["excl_0"] = bool(ci["lo"] > 0 or ci["hi"] < 0)
                row["null_roi"] = random_bet_null(
                    s2, ps, n_bets=len(bets), n_sims=400, seed=0)["mean_roi"]
            rows.append(row)
    tbl = pd.DataFrame(rows)
    cols = ["stratum", "price_set", "n_eligible", "n_bets", "roi", "roi_lo",
            "roi_hi", "excl_0", "null_roi", "hit_rate", "avg_odds"]
    print(tbl[[c for c in cols if c in tbl.columns]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    print()
    print("=" * 78)
    print("THE ANSWER TO H1, BY THE PRE-REGISTERED RULE")
    print("=" * 78)
    print(f"  {_verdict(reps['lower (tiers 3-5)'])}")
    print()
    print("  Increment the configuration count in docs/PROGRAMME.md to 48 and")
    print("  record the result in the hypothesis file, whichever way it went.")


if __name__ == "__main__":
    main()
