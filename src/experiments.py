"""Walk-forward experiments for the network, with paired comparisons.

    uv run python -m src.experiments ablate
    uv run python -m src.experiments scoreboard

Everything here reports a PAIRED difference against a named reference on the
same fixtures, not two independent means. At ~0.1 nats of total learnable
signal, the effects worth having are the size of the differences between model
families in the literature (0.0002-0.002), and an unpaired comparison at that
scale is noise.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.eval.metrics import OUTCOMES, log_loss, rps, rps_per_match, summary
from src.eval.split import assert_no_leakage, season_walk_forward
from src.models.baselines import ALL_FEATURES, OrderedLogit, RATING_FEATURES
from src.models.net import (
    NetConfig, TemperatureScaler, build_vocab, predict, train_net,
)
from src.scoreboard import load_panel


def paired(a: np.ndarray, b: np.ndarray, y) -> dict:
    """RPS difference (a - b) on identical fixtures, with a t statistic.

    Positive means b is better. Paired because both models saw the same
    matches, which removes match difficulty from the comparison entirely.
    """
    d = rps_per_match(a, y) - rps_per_match(b, y)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"delta": float(d.mean()), "se": float(se),
            "t": float(d.mean() / se) if se else float("nan"), "n": int(len(d))}


def run_walk_forward(panel, cfg: NetConfig, features=RATING_FEATURES,
                     calibrate: bool = True, seeds=(0,), verbose=False,
                     train_pool: pd.DataFrame | None = None):
    """Fit the net on each walk-forward split; return stacked OOS predictions.

    `train_pool` lets the model train on MORE matches than it is evaluated on.
    The test set has to carry Pinnacle closing odds, because that is the
    benchmark -- but the training set does not. Restricting both to the priced
    panel throws away 85% of the corpus, and the literature is explicit that
    deep models only become competitive somewhere between 100k and 300k
    matches across many leagues. Training rows are still taken strictly before
    the test window opens, so nothing leaks.

    Averaging over seeds is not decoration. Yeung et al.'s defence of their
    deep model against CatBoost was lower loss variance rather than lower
    loss, so a single-seed number on this task is not a measurement.
    """
    pool = panel if train_pool is None else train_pool
    vocab = build_vocab(pool)
    X = panel[features].to_numpy(float)
    X_pool = pool[features].to_numpy(float)
    pool_kick = pd.to_datetime(pool["kickoff"]).to_numpy()
    y_all = panel["result"].to_numpy()

    hda, hda_cal, from_goals, ys = [], [], [], []
    for s in season_walk_forward(panel, min_train_seasons=3):
        assert_no_leakage(panel, s)
        te = panel.iloc[s.test_idx]
        if train_pool is None:
            tr, X_tr = panel.iloc[s.train_idx], X[s.train_idx]
        else:
            keep = pool_kick < np.datetime64(s.test_start)
            tr, X_tr = pool[keep], X_pool[keep]
            assert pd.to_datetime(tr["kickoff"]).max() < s.test_start

        seed_p, seed_g, seed_logits = [], [], []
        for seed in seeds:
            model, meta = train_net(tr, X_tr, vocab,
                                    NetConfig(**{**cfg.__dict__, "seed": seed}))
            out = predict(model, te, X[s.test_idx], vocab, meta)
            seed_p.append(out["hda"])
            seed_g.append(out["hda_from_goals"])

            if calibrate:
                # Fit the temperature on the TAIL of the training window, which
                # the model early-stopped on but never fitted weights to. Fitting
                # it on the test season would manufacture the result outright.
                cut = int(len(tr) * 0.85)
                val = tr.iloc[cut:]
                val_out = predict(model, val, X_tr[cut:], vocab, meta)
                scaler = TemperatureScaler().fit(val_out["logits"], val["result"])
                seed_logits.append(scaler.transform(out["logits"]))

        hda.append(np.mean(seed_p, axis=0))
        from_goals.append(np.mean(seed_g, axis=0))
        if calibrate:
            hda_cal.append(np.mean(seed_logits, axis=0))
        ys.append(y_all[s.test_idx])
        if verbose:
            print(f"    {s.label}: train {len(tr):,} test {len(te):,}", flush=True)

    out = {"hda": np.vstack(hda), "hda_from_goals": np.vstack(from_goals),
           "y": np.concatenate(ys)}
    if calibrate:
        out["hda_calibrated"] = np.vstack(hda_cal)
    return out


def baseline_predictions(panel, features=RATING_FEATURES, train_pool=None):
    """Ordered logit on the same splits. `train_pool` mirrors the net's option
    so both models can be given identical training data -- otherwise the
    comparison measures the data, not the model."""
    X = panel[features].to_numpy(float)
    y_all = panel["result"].to_numpy()
    if train_pool is not None:
        X_pool = train_pool[features].to_numpy(float)
        y_pool = train_pool["result"].to_numpy()
        pool_kick = pd.to_datetime(train_pool["kickoff"]).to_numpy()

    P, Y = [], []
    for s in season_walk_forward(panel, min_train_seasons=3):
        if train_pool is None:
            xt, yt = X[s.train_idx], y_all[s.train_idx]
        else:
            keep = pool_kick < np.datetime64(s.test_start)
            xt, yt = X_pool[keep], y_pool[keep]
        P.append(OrderedLogit().fit(xt, yt).predict_proba(X[s.test_idx]))
        Y.append(y_all[s.test_idx])
    return np.vstack(P), np.concatenate(Y)


VARIANTS = {
    "full":            NetConfig(),
    "no team emb":     NetConfig(team_dim=0),
    "no league emb":   NetConfig(league_dim=0),
    "no embeddings":   NetConfig(team_dim=0, league_dim=0),
    "no goals head":   NetConfig(goal_loss_weight=0.0),
    "single member":   NetConfig(members=1),
    "wide (h=256)":    NetConfig(hidden=256),
    "no dropout":      NetConfig(dropout=0.0),
}


def cmd_ablate(args) -> None:
    panel = load_panel(args.first_season, args.last_season)
    features = ALL_FEATURES if args.features == "all" else RATING_FEATURES
    print(f"panel {len(panel):,} matches, {panel['season'].nunique()} seasons, "
          f"{panel['div'].nunique()} divisions")
    print(f"seeds {list(args.seeds)}   features {len(features)} ({args.features})")
    print()

    # The reference MUST see the same features as the variant. Comparing a
    # 49-feature net against a 7-feature baseline measures the features, not
    # the model, and it flatters the net by exactly the amount the extra
    # columns are worth -- which here is more than the model difference.
    ref_p, ref_y = baseline_predictions(panel, features)
    print(f"reference: ordered logit on the SAME {len(features)} features   "
          f"RPS {rps(ref_p, ref_y):.5f}   log loss {log_loss(ref_p, ref_y):.5f}")
    print()

    names = args.only.split(",") if args.only else list(VARIANTS)
    rows = []
    full_pred = None
    for name in names:
        cfg = VARIANTS[name]
        t = time.time()
        out = run_walk_forward(panel, cfg, features=features, seeds=args.seeds)
        y = out["y"]
        assert (y == ref_y).all(), "variants must be scored on identical fixtures"

        best_key = "hda_calibrated" if "hda_calibrated" in out else "hda"
        row = {
            "variant": name,
            "rps": rps(out["hda"], y),
            "rps_cal": rps(out[best_key], y),
            "logloss_cal": log_loss(out[best_key], y),
            "rps_goals_head": rps(out["hda_from_goals"], y),
            "secs": round(time.time() - t),
        }
        vs_ref = paired(ref_p, out[best_key], y)
        row["vs_baseline"] = vs_ref["delta"]
        row["t_vs_baseline"] = vs_ref["t"]
        if name == "full":
            full_pred = out[best_key]
        elif full_pred is not None:
            vs_full = paired(out[best_key], full_pred, y)
            row["vs_full"] = vs_full["delta"]
            row["t_vs_full"] = vs_full["t"]
        rows.append(row)
        print(f"  {name:<16} rps {row['rps_cal']:.5f}  "
              f"vs baseline {row['vs_baseline']:+.5f} (t {row['t_vs_baseline']:+.2f})  "
              f"[{row['secs']}s]", flush=True)

    print()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print()
    print("  vs_baseline > 0 means the net beats the ordered logit.")
    print("  vs_full     > 0 means the variant beats the full net, so a")
    print("              NEGATIVE value is a component earning its place.")


def cmd_scoreboard(args) -> None:
    panel = load_panel(args.first_season, args.last_season)
    features = ALL_FEATURES if args.features == "all" else RATING_FEATURES
    ref_ratings, y = baseline_predictions(panel, RATING_FEATURES)
    ref_p, _ = baseline_predictions(panel, features)
    out = run_walk_forward(panel, VARIANTS["full"], features=features,
                           seeds=args.seeds, verbose=True)

    from src.eval.betting import PINNACLE_CLOSE
    from src.eval.devig import devig
    mkt = devig(panel[PINNACLE_CLOSE.cols].to_numpy(float), method="shin")
    test_idx = np.concatenate([s.test_idx for s in season_walk_forward(panel, min_train_seasons=3)])

    rows = [
        summary(mkt[test_idx], y, "market (Pinnacle close)"),
        summary(ref_ratings, y, "ordered logit (7 rating feats)"),
        summary(ref_p, y, f"ordered logit ({len(features)} feats)"),
        summary(out["hda"], y, "net (uncalibrated)"),
        summary(out["hda_calibrated"], y, "net (temperature-scaled)"),
        summary(out["hda_from_goals"], y, "net, Poisson head -> 1X2"),
    ]
    print()
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print()
    r = paired(ref_p, out["hda_calibrated"], y)
    print(f"  net vs logit (same {len(features)} features): {r['delta']:+.5f}  "
          f"t {r['t']:+.2f}  n {r['n']:,}   (positive = the net wins)")
    r = paired(out["hda_calibrated"], mkt[test_idx], y)
    print(f"  market vs net  : {r['delta']:+.5f}  t {r['t']:+.2f}  "
          "(positive = the market is still ahead)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["ablate", "scoreboard"])
    ap.add_argument("--first-season", default="2016-17")
    ap.add_argument("--last-season", default="2024-25")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", default="", help="comma-separated variant names")
    ap.add_argument("--features", choices=["all", "ratings"], default="all")
    args = ap.parse_args()
    {"ablate": cmd_ablate, "scoreboard": cmd_scoreboard}[args.command](args)


if __name__ == "__main__":
    main()
