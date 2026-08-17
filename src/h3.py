"""H3: can the direction of line movement be forecast from the snapshot?

    uv run python -m src.h3

Executes `docs/hypotheses/H3-line-movement.md`, pre-registered at 3b6e021
before any movement model existed. This file chooses no threshold, no feature
set, no model and no holdout.

THE NARROWED QUESTION. H1's diagnostics already showed that a match-outcome
model's disagreement with the price predicts movement direction. So H3 is not
asking whether movement is predictable at all -- that is partly answered.
It asks whether fitting the movement label DIRECTLY beats what a match model
achieves incidentally. A null is the interesting result: it would mean the
market moves toward what a decent match model already thinks, leaving no
separate microstructure signal to harvest.

THE ONE PLACE THIS CAN CHEAT is the feature side. The label is built from the
closing prices, so any closing column reaching the input matrix turns H3 into
a lookup. `assert_no_closing_leak` aborts the run rather than trusting care,
because a leak surfaces as an unusually good result and that is the one
outcome nobody interrogates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.betting import PINNACLE_CLOSE, PINNACLE_PRE
from src.eval.metrics import OUTCOMES
from src.eval.split import assert_no_leakage, season_walk_forward
from src.features.build import load as load_features
from src.models.baselines import ALL_FEATURES, CatBoostBaseline

# ---- fixed by the pre-registration; do not tune ----
DEV_FIRST, DEV_LAST = "2012-13", "2023-24"
HOLDOUT_SEASON = "2024-25"
MIN_ODDS, MAX_ODDS = 1.5, 5.0
MIN_BETS = 3250
N_SIMS = 200
B365_PRE_COLS = ["b365h", "b365d", "b365a"]

# Every closing column known to the corpus. Anything here appearing on the
# input side is the label leaking back into its own features.
FORBIDDEN = {
    "psch", "pscd", "psca", "b365ch", "b365cd", "b365ca",
    "maxch", "maxcd", "maxca", "avgch", "avgcd", "avgca",
    "bfech", "bfecd", "bfeca",
}


def assert_no_closing_leak(feature_names: list[str]) -> None:
    """Abort if any closing column reached the feature matrix.

    Deliberately checks two ways. The explicit set catches the columns this
    corpus is known to carry; the suffix scan catches a column added later by
    a vendor that nobody thought to add here. A future feed introducing
    `bwch` would sail past the first check and be caught by the second.
    """
    hits = sorted(c for c in feature_names if c.lower() in FORBIDDEN)
    suspicious = sorted(
        c for c in feature_names
        if c.lower() not in FORBIDDEN
        and any(c.lower().endswith(f"c{o}") for o in ("h", "d", "a"))
        and any(c.lower().startswith(b) for b in ("ps", "b365", "max", "avg", "bfe", "bw", "iw", "wh", "vc"))
    )
    if hits or suspicious:
        raise AssertionError(
            "CLOSING COLUMNS REACHED THE FEATURE MATRIX -- H3 aborted.\n"
            f"  known closing columns: {hits}\n"
            f"  suffix-matched suspects: {suspicious}\n"
            "The label is built from these, so their presence makes H3 a lookup. "
            "Per the pre-registration an aborted run is not an evaluation and "
            "does not increment the registry count."
        )


def build_frame() -> pd.DataFrame:
    """Gradable rows: both Pinnacle legs, plus Bet365 pre-close for the
    cross-book disagreement features."""
    df = load_features().sort_values("kickoff").reset_index(drop=True)
    need = PINNACLE_PRE.cols + PINNACLE_CLOSE.cols + B365_PRE_COLS
    # A price of 0.0 is missing data encoded as a number, and notna() does not
    # catch it. Left in, it becomes an infinite log-implied-probability that
    # nan_to_num silently turns into a huge finite feature -- no error, no NaN,
    # just a garbage row treated as informative. Measured at 10 rows in b365h,
    # all in the training window and none in the holdout
    # (scripts/h3_zero_price_check.py), so this changes no reported number; it
    # is fixed because the next season's data has no reason to be as kind.
    positive = (df[need].to_numpy(float) > 0).all(axis=1)
    df = df[(df["source"] == "main")
            & df["result"].notna()
            & df[need].notna().all(axis=1)
            & positive
            & df["season"].between(DEV_FIRST, HOLDOUT_SEASON)]
    return df.sort_values("kickoff").reset_index(drop=True)


def add_price_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The eleven pre-close price features fixed by the pre-registration.

    All are functions of the SNAPSHOT only. The closing legs are used further
    down to build the label and never here.
    """
    out = df.copy()
    ps = df[PINNACLE_PRE.cols].to_numpy(float)
    b3 = df[B365_PRE_COLS].to_numpy(float)
    names = []
    for j, o in enumerate(OUTCOMES):
        out[f"h3_logp_ps_{o}"] = np.log(1.0 / ps[:, j])
        out[f"h3_logp_b365_{o}"] = np.log(1.0 / b3[:, j])
        # Cross-book disagreement: where Bet365 and Pinnacle differ at the
        # snapshot, one of them is more likely to be the one that moves.
        out[f"h3_disagree_{o}"] = np.log(ps[:, j] / b3[:, j])
        names += [f"h3_logp_ps_{o}", f"h3_logp_b365_{o}", f"h3_disagree_{o}"]
    out["h3_overround_ps"] = (1.0 / ps).sum(axis=1)
    out["h3_overround_b365"] = (1.0 / b3).sum(axis=1)
    names += ["h3_overround_ps", "h3_overround_b365"]
    return out, names


def movement_label(df: pd.DataFrame) -> np.ndarray:
    """Which outcome's price shortened most: argmax of pre_close / close.

    Labelled with the same H/D/A strings the rest of the repo uses, so
    CatBoostBaseline's by-name class mapping is reused rather than
    re-implemented -- that mapping exists because getting it wrong scrambles
    every metric silently.
    """
    ratio = (df[PINNACLE_PRE.cols].to_numpy(float)
             / df[PINNACLE_CLOSE.cols].to_numpy(float))
    return np.array(OUTCOMES)[np.argmax(ratio, axis=1)]


def matched_null_shortening(df: pd.DataFrame, bet_odds: np.ndarray,
                            n_sims: int = N_SIMS) -> np.ndarray:
    """Null shortening rate, odds-matched to H3's own bets.

    Same construction as scripts/h1_odds_matched_null.py: deciles taken from
    the model's bets, each filled only from eligible cells inside that decile,
    a decile with no eligible cells skipped rather than borrowed from a
    neighbour.
    """
    pre = df[PINNACLE_PRE.cols].to_numpy(float)
    close = df[PINNACLE_CLOSE.cols].to_numpy(float)
    ok = (np.isfinite(pre) & np.isfinite(close) & (close > 0)
          & (pre >= MIN_ODDS) & (pre <= MAX_ODDS))
    r, c = np.nonzero(ok)
    odds, ratio = pre[r, c], pre[r, c] / close[r, c]
    edges = np.quantile(bet_odds, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf

    sims = []
    for s in range(n_sims):
        rng = np.random.default_rng(s)
        picked = []
        for i in range(10):
            want = int(((bet_odds >= edges[i]) & (bet_odds < edges[i + 1])).sum())
            pool = np.nonzero((odds >= edges[i]) & (odds < edges[i + 1]))[0]
            if want == 0 or len(pool) == 0:
                continue
            picked.append(ratio[rng.choice(pool, size=want, replace=True)])
        if picked:
            sims.append(float((np.concatenate(picked) > 1.0).mean()))
    return np.array(sims)


def bets_from(df: pd.DataFrame, pred: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Bet the predicted argmax wherever its pre-close price is in band."""
    idx = {o: j for j, o in enumerate(OUTCOMES)}
    j = np.array([idx[p] for p in pred])
    rows = np.arange(len(df))
    pre = df[PINNACLE_PRE.cols].to_numpy(float)[rows, j]
    close = df[PINNACLE_CLOSE.cols].to_numpy(float)[rows, j]
    take = (np.isfinite(pre) & np.isfinite(close) & (close > 0)
            & (pre >= MIN_ODDS) & (pre <= MAX_ODDS))
    return pd.DataFrame({
        "selection": pred[take], "pre": pre[take], "close": close[take],
        "ratio": pre[take] / close[take],
        "confidence": proba[rows, j][take],
        "season": df["season"].to_numpy()[take],
    })


def report(bets: pd.DataFrame, pool: pd.DataFrame, label: str) -> dict:
    sims = matched_null_shortening(pool, bets["pre"].to_numpy())
    null, lo, hi = sims.mean(), *np.quantile(sims, [0.025, 0.975])
    obs = float((bets["ratio"] > 1.0).mean())
    se = np.sqrt(null * (1 - null) / len(bets))
    z = (obs - null) / se if se else np.nan
    return {"arm": label, "n_bets": len(bets), "observed": obs,
            "null": null, "null_lo": lo, "null_hi": hi,
            "margin": obs - null, "z": z,
            "p": 2 * (1 - stats.norm.cdf(abs(z))),
            "mean_ratio": float(bets["ratio"].mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", action="store_true",
                    help="also run the descriptive walk-forward over the "
                         "development window. Slow, and decides nothing.")
    args = ap.parse_args()

    df = build_frame()
    df, price_names = add_price_features(df)
    features = ALL_FEATURES + price_names

    # The gate. Before anything is fitted.
    assert_no_closing_leak(features)

    df["h3_label"] = movement_label(df)

    print("=" * 78)
    print("H3 -- forecasting the direction of line movement")
    print("=" * 78)
    print(f"  frame     {len(df):,} gradable rows, "
          f"{df['season'].min()} -> {df['season'].max()}")
    print(f"  features  {len(features)} = {len(ALL_FEATURES)} existing "
          f"+ {len(price_names)} pre-close price")
    print("  leakage   no closing column reached the feature matrix (asserted)")
    print(f"  label     3-way argmax of pre/close, base rates "
          f"{dict(pd.Series(df['h3_label']).value_counts(normalize=True).round(4))}")
    print()

    train = df[df["season"] <= DEV_LAST].reset_index(drop=True)
    test = df[df["season"] == HOLDOUT_SEASON].reset_index(drop=True)
    assert pd.to_datetime(train["kickoff"]).max() < pd.to_datetime(test["kickoff"]).min(), \
        "development window leaks into the holdout"

    print("-" * 78)
    print(f"HOLDOUT {HOLDOUT_SEASON} -- the only thing that decides H3")
    print("-" * 78)
    print(f"  train {len(train):,} rows, test {len(test):,} rows")
    print("  fitting CatBoost at repo defaults (400 iters, depth 4, lr 0.05)...")
    model = CatBoostBaseline().fit(train[features].to_numpy(float),
                                   train["h3_label"].to_numpy())
    proba = model.predict_proba(test[features].to_numpy(float))
    pred = np.array(OUTCOMES)[np.argmax(proba, axis=1)]

    y = test["h3_label"].to_numpy()
    acc = float((pred == y).mean())
    base_lab, base_share = pd.Series(y).value_counts(normalize=True).idxmax(), \
        pd.Series(y).value_counts(normalize=True).max()
    print()
    print(f"  directional accuracy   {acc:.4f}")
    print(f"  majority baseline      {base_share:.4f}  (always predict {base_lab})")
    print(f"  lift over baseline     {acc - base_share:+.4f}")

    bets = bets_from(test, pred, proba)
    rows = [report(bets, test, "H3, every match")]

    # Pre-specified selective variant. Descriptive. Decides nothing.
    cut = bets["confidence"].quantile(0.75)
    top = bets[bets["confidence"] >= cut]
    if len(top) > 50:
        rows.append(report(top, test, "H3, top quartile (descriptive)"))

    print()
    print("-" * 78)
    print("CLV AGAINST AN ODDS-MATCHED NULL")
    print("-" * 78)
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    primary = rows[0]
    inconclusive = primary["n_bets"] < MIN_BETS
    supported = (not inconclusive
                 and primary["observed"] > primary["null_hi"]
                 and primary["p"] < 0.01)

    print()
    print("-" * 78)
    print("THE STAKEABILITY TEST -- part of the falsifier, not an afterthought")
    print("-" * 78)
    orr = float((1.0 / test[PINNACLE_PRE.cols].to_numpy(float)).sum(axis=1).mean())
    gain = primary["mean_ratio"] - 1.0
    print(f"  CLV ratio gain on the bets taken   {gain:+.4f}  ({gain:+.2%} of price)")
    print(f"  Pinnacle pre-close overround       {orr:.4f}  ({orr - 1:.2%} margin)")
    print(f"  gain clears the margin?            {gain > (orr - 1)}")

    print()
    print("=" * 78)
    print("THE ANSWER, BY THE PRE-REGISTERED RULE")
    print("=" * 78)
    if inconclusive:
        print(f"  INCONCLUSIVE BY FLOOR -- {primary['n_bets']:,} bets "
              f"against the {MIN_BETS:,} floor")
    elif supported:
        print("  SUPPORTED -- shortening rate above the matched null's interval "
              "and p < 0.01")
    else:
        print("  NOT SUPPORTED -- fails the interval and/or p < 0.01")
    print(f"    observed {primary['observed']:.4f}  null {primary['null']:.4f} "
          f"95% [{primary['null_lo']:.4f}, {primary['null_hi']:.4f}]  "
          f"margin {primary['margin']:+.4f}  z {primary['z']:.2f}  p {primary['p']:.4g}")
    print()
    print("  For comparison, and NOT a criterion: H1's diagnostics measured the")
    print("  settled match-outcome net at +2.7pp (tiers 1-2) to +8.7pp (tier 4)")
    print("  over the same kind of null. H3 beats the incidental signal only if")
    print("  it lands above that band.")

    if args.dev:
        print()
        print("-" * 78)
        print("DESCRIPTIVE -- walk-forward over the development window")
        print("-" * 78)
        dev = df[df["season"] <= DEV_LAST].reset_index(drop=True)
        out = []
        for s in season_walk_forward(dev, min_train_seasons=3):
            assert_no_leakage(dev, s)
            tr, te = dev.iloc[s.train_idx], dev.iloc[s.test_idx]
            m = CatBoostBaseline().fit(tr[features].to_numpy(float),
                                       tr["h3_label"].to_numpy())
            pr = m.predict_proba(te[features].to_numpy(float))
            pd_ = np.array(OUTCOMES)[np.argmax(pr, axis=1)]
            yy = te["h3_label"].to_numpy()
            b = bets_from(te.reset_index(drop=True), pd_, pr)
            r = report(b, te.reset_index(drop=True), s.label)
            r["accuracy"] = float((pd_ == yy).mean())
            r["baseline"] = float(pd.Series(yy).value_counts(normalize=True).max())
            out.append(r)
            print(f"    {s.label} done", flush=True)
        print()
        cols = ["arm", "n_bets", "accuracy", "baseline", "observed", "null",
                "margin", "z", "p"]
        print(pd.DataFrame(out)[cols].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))
        print()
        print("  Descriptive. The holdout above is what decides H3.")


if __name__ == "__main__":
    main()
