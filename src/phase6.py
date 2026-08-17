"""Phase 6: run the pre-registered betting rule, once, on the untouched season.

    uv run python -m src.phase6

Everything here is fixed by `docs/PREREGISTRATION.md`, committed before any
model PnL existed. Nothing in this file chooses a threshold, a market, or a
price column -- it executes what was written down and reports whatever comes
out. That is the entire point: looking at several rules and reporting the best
is how a backtest manufactures an edge, and Constantinou's own threshold sweep
moves 1X2 ROI from -9% to +23% purely by shrinking the sample to 37 bets.

Holdout: season 2025-26, which no model in this project has been evaluated on.
Pinnacle closing coverage decays from October 2025 and is gone from February
2026, so results are reported separately for the subset that has a Pinnacle
close and for the full season graded against Bet365 and market-average close,
with the latter labelled as the softer benchmark it is.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.eval.betting import (
    B365_CLOSE, B365_PRE, MARKET_AVG_PRE, MARKET_MAX_CLOSE, MARKET_MAX_PRE,
    PINNACLE_CLOSE, PINNACLE_PRE, BetRule, bootstrap_ci, clv_report,
    closing_price_for_bets, random_bet_null, required_sample_size, simulate,
    summarize,
)
from src.eval.devig import devig
from src.eval.metrics import log_loss, rps, summary
from src.features.build import load as load_features, load_sequences
from src.models.baselines import ALL_FEATURES, OrderedLogit
from src.models.net import (
    NetConfig, TemperatureScaler, build_vocab, predict, train_net,
)

# ---- fixed by the pre-registration; do not tune ----
RULE = BetRule(min_ev=0.05, min_odds=1.5, max_odds=5.0, stake=1.0,
               name="pre-registered: ev>=0.05, odds 1.5-5.0")
HOLDOUT_SEASON = "2025-26"
SEEDS = (0, 1, 2)

MARKET_AVG_CLOSE_COLS = ["avgch", "avgcd", "avgca"]


def build_holdout():
    """Train on everything strictly before 2025-26; predict 2025-26."""
    full = load_features().sort_values("kickoff").reset_index(drop=True)
    full = full[full["result"].notna()].reset_index(drop=True)
    seq_all, _ = load_sequences()

    test = full[(full["source"] == "main") & (full["season"] == HOLDOUT_SEASON)]
    test = test.reset_index(drop=True)
    cutoff = pd.to_datetime(test["kickoff"]).min()
    train = full[pd.to_datetime(full["kickoff"]) < cutoff].reset_index(drop=True)

    assert pd.to_datetime(train["kickoff"]).max() < cutoff, "training data leaks into the holdout"
    return train, test, seq_all, cutoff


def fit_and_predict(train, test, seq_all, verbose=True):
    X_tr = train[ALL_FEATURES].to_numpy(float)
    X_te = test[ALL_FEATURES].to_numpy(float)
    seq_tr = seq_all[train["corpus_row"].to_numpy()]
    seq_te = seq_all[test["corpus_row"].to_numpy()]
    vocab = build_vocab(train)

    preds = []
    for seed in SEEDS:
        cfg = NetConfig(seed=seed)
        model, meta = train_net(train, X_tr, vocab, cfg, seq_train=seq_tr)
        out = predict(model, test, X_te, vocab, meta, seq=seq_te)
        cut = int(len(train) * 0.85)
        val = train.iloc[cut:]
        val_out = predict(model, val, X_tr[cut:], vocab, meta, seq=seq_tr[cut:])
        scaler = TemperatureScaler().fit(val_out["logits"], val["result"])
        preds.append(scaler.transform(out["logits"]))
        if verbose:
            print(f"    seed {seed}: best epoch {meta['best_epoch']}, "
                  f"temperature {scaler.temperature:.3f}", flush=True)
    return np.mean(preds, axis=0)


def _fmt(df):
    return df.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()

    train, test, seq_all, cutoff = build_holdout()
    print("=" * 74)
    print("PHASE 6 -- the pre-registered rule, run once")
    print("=" * 74)
    print(f"  rule       {RULE.describe()}")
    print(f"  holdout    {HOLDOUT_SEASON}, {len(test):,} matches, "
          f"{test['div'].nunique()} divisions")
    print(f"  training   {len(train):,} matches, all strictly before "
          f"{pd.Timestamp(cutoff).date()}")
    print()
    print("  fitting (3 seeds, full corpus)...")
    p = fit_and_predict(train, test, seq_all)
    y = test["result"].to_numpy()

    print()
    print("-" * 74)
    print("FORECAST QUALITY ON THE HOLDOUT")
    print("-" * 74)
    rows = [summary(p, y, "the net (pre-registered config)")]
    has_pinn = test[PINNACLE_CLOSE.cols].notna().all(axis=1).to_numpy()
    if has_pinn.any():
        mkt = devig(test.loc[has_pinn, PINNACLE_CLOSE.cols].to_numpy(float), method="shin")
        rows.append(summary(mkt, y[has_pinn], f"market, Pinnacle close (n={has_pinn.sum()})"))
        rows.append(summary(p[has_pinn], y[has_pinn], "the net, same subset"))
    print(_fmt(pd.DataFrame(rows)))
    print()
    print(f"  Pinnacle close available for {has_pinn.sum():,} of {len(test):,} "
          f"({has_pinn.mean():.1%}) -- the feed stops in February 2026.")

    print()
    print("-" * 74)
    print("CLOSING-LINE VALUE   (the headline metric)")
    print("-" * 74)
    print("  Bet at a pre-close price, grade against the Pinnacle close of the")
    print("  same selection. Only defined where both exist.")
    print()
    clv_rows = []
    for ps in (PINNACLE_PRE, B365_PRE, MARKET_AVG_PRE, MARKET_MAX_PRE):
        need = ps.cols + PINNACLE_CLOSE.cols
        if not all(c in test.columns for c in need):
            continue
        sub_mask = test[need].notna().all(axis=1).to_numpy()
        if sub_mask.sum() < 50:
            continue
        sub = test[sub_mask].reset_index(drop=True)
        bets = simulate(sub, p[sub_mask], ps, RULE)
        if bets.empty:
            clv_rows.append({"taken_at": ps.label, "n_bets": 0})
            continue
        r = clv_report(bets, closing_price_for_bets(bets, sub))
        clv_rows.append({"taken_at": ps.label, "n_bets": len(bets),
                         "mean_ratio": r["mean_ratio"],
                         "pct_shortened": r["pct_shortened"],
                         "binom_p": r["binom_pvalue"]})
    print(_fmt(pd.DataFrame(clv_rows)) if clv_rows else "  no usable pre-close/close pairs")

    print()
    print("-" * 74)
    print("ROI, THREE PRICE COLUMNS, LED BY THE SHARPEST")
    print("-" * 74)
    price_sets = [
        (PINNACLE_CLOSE, "truth test"),
        (B365_CLOSE, "a book you could hold an account with"),
        (MARKET_MAX_CLOSE, "optimistic bound"),
    ]
    rows = []
    for ps, note in price_sets:
        if not all(c in test.columns for c in ps.cols):
            continue
        m = test[ps.cols].notna().all(axis=1).to_numpy()
        if m.sum() < 50:
            rows.append({"price_set": ps.label, "note": note, "n_eligible": int(m.sum()),
                         "n_bets": 0})
            continue
        sub = test[m].reset_index(drop=True)
        bets = simulate(sub, p[m], ps, RULE)
        row = summarize(bets, with_ci=False)
        row["note"] = note
        row["n_eligible"] = int(m.sum())
        if len(bets):
            ci = bootstrap_ci(bets, n_boot=args.boot, seed=0)
            row["roi_lo"], row["roi_hi"] = ci["lo"], ci["hi"]
            row["excludes_zero"] = bool(ci["lo"] > 0 or ci["hi"] < 0)
            null = random_bet_null(sub, ps, n_bets=len(bets), n_sims=400, seed=0)
            row["null_roi"] = null["mean_roi"]
        rows.append(row)
    cols = ["price_set", "n_eligible", "n_bets", "roi", "roi_lo", "roi_hi",
            "excludes_zero", "null_roi", "hit_rate", "avg_odds", "n_needed_for_2pct"]
    tbl = pd.DataFrame(rows)
    print(_fmt(tbl[[c for c in cols if c in tbl.columns]]))

    # The softer benchmark for the part of the season Pinnacle does not cover.
    if all(c in test.columns for c in MARKET_AVG_CLOSE_COLS):
        from src.eval.betting import PriceSet
        avg_close = PriceSet("market_avg_close", *MARKET_AVG_CLOSE_COLS)
        m = test[avg_close.cols].notna().all(axis=1).to_numpy()
        if m.sum() > 50:
            sub = test[m].reset_index(drop=True)
            bets = simulate(sub, p[m], avg_close, RULE)
            print()
            print("  Softer benchmark for the whole season, since Pinnacle close")
            print("  covers only part of it. Market-average close is an easier")
            print("  bar than a sharp book and is labelled as such.")
            if len(bets):
                ci = bootstrap_ci(bets, n_boot=args.boot, seed=0)
                s = summarize(bets, with_ci=False)
                print(f"    market_avg_close: {s['n_bets']} bets, ROI {s['roi']:+.4f} "
                      f"95% [{ci['lo']:+.4f}, {ci['hi']:+.4f}], hit {s['hit_rate']:.3f}")
            else:
                print(f"    market_avg_close: 0 bets from {m.sum():,} eligible matches")

    print()
    print("-" * 74)
    print("HOW TO READ THIS")
    print("-" * 74)
    n_needed = required_sample_size(3.2, 0.02)
    print(f"  Distinguishing a 2% edge from zero at average odds 3.2 needs")
    print(f"  ~{n_needed:,} bets. One season cannot do it, which is why the")
    print("  pre-registration named CLV as the headline and ROI as secondary.")
    print()
    print("  A result positive only in the market-maximum column is price")
    print("  shopping, not forecasting. This project already measured that at")
    print("  +4.8% using the market's own de-vigged opinion as the model.")


if __name__ == "__main__":
    main()
