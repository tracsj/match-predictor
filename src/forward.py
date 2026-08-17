"""Forward validation: predict upcoming fixtures and commit the file.

    uv run python -m src.forward

Every other number in this project is backtested — computed over history by
someone who could see that history. This produces the other kind: a prediction
written down before kickoff, with the price that was available when it was
written, committed to a repository whose timestamps are not ours to set.

It is the only artifact that would make a future positive result believable,
and it accumulates only in real time. That is why it runs on a schedule rather
than on demand.

**One prediction per fixture, and it is the earliest.** A fixture can appear in
several consecutive `fixtures.csv` snapshots, so a later run would happily
predict it again — at a shorter horizon, with a price closer to the close, and
with a model trained on more data. Restating a prediction as kickoff approaches
is how a forward ledger quietly turns back into a backtest, so a match already
present in any committed prediction file is skipped here rather than updated.

**The model is odds-free.** `ALL_FEATURES` carries ratings and rolling form and
no price column, so the network is unaffected by Pinnacle's removal from the
feed. What changed is the benchmark it gets graded against, not its inputs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.footballdata import REPO_ROOT, build_matches, refresh_current
from src.data.fixtures import build_fixtures, uk_now_naive
from src.features.build import build_forward
from src.features.horizon import UNPLAYED_COL
from src.models.baselines import ALL_FEATURES
from src.models.net import NetConfig, TemperatureScaler, build_vocab, predict, train_net

PREDICTIONS_DIR = REPO_ROOT / "predictions"

# Fixed by docs/PREREGISTRATION.md. The forward run uses the same configuration
# as the settled study, which is what makes the two comparable -- and is why a
# scheduled retrain is not a new configuration for the registry's purposes.
SEEDS = (0, 1, 2)

# Pre-close prices recorded alongside each prediction. Led by the exchange,
# which is the sharpest price still in the feed now Pinnacle is gone.
PRICE_COLS = ["bfeh", "bfed", "bfea", "b365h", "b365d", "b365a",
              "maxh", "maxd", "maxa", "avgh", "avgd", "avga"]

OUT_COLS = (["predicted_at", "match_id", "kickoff", "div", "league", "season",
             "home_raw", "away_raw", "home_key", "away_key",
             "p_home", "p_draw", "p_away"] + PRICE_COLS)


def already_predicted() -> set[str]:
    """Every match_id in every committed prediction file."""
    seen: set[str] = set()
    for p in sorted(PREDICTIONS_DIR.glob("*.csv")):
        try:
            seen.update(pd.read_csv(p, usecols=["match_id"])["match_id"].astype(str))
        except (ValueError, KeyError, pd.errors.EmptyDataError):
            continue
    return seen


def fit_and_predict_forward(train: pd.DataFrame, horizon: pd.DataFrame,
                            seq_all: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Train on completed matches, predict the horizon.

    This is `phase6.fit_and_predict` with the holdout swapped for a set of rows
    that have no labels at all. The temperature scaler is fitted on the tail of
    the TRAINING window, never on anything at or after the horizon.
    """
    X_tr = train[ALL_FEATURES].to_numpy(float)
    X_te = horizon[ALL_FEATURES].to_numpy(float)
    seq_tr = seq_all[train["corpus_row"].to_numpy()]
    seq_te = seq_all[horizon["corpus_row"].to_numpy()]
    vocab = build_vocab(train)

    preds = []
    for seed in SEEDS:
        cfg = NetConfig(seed=seed)
        model, meta = train_net(train, X_tr, vocab, cfg, seq_train=seq_tr)
        out = predict(model, horizon, X_te, vocab, meta, seq=seq_te)
        cut = int(len(train) * 0.85)
        val = train.iloc[cut:]
        val_out = predict(model, val, X_tr[cut:], vocab, meta, seq=seq_tr[cut:])
        scaler = TemperatureScaler().fit(val_out["logits"], val["result"])
        preds.append(scaler.transform(out["logits"]))
        if verbose:
            print(f"    seed {seed}: best epoch {meta['best_epoch']}, "
                  f"temperature {scaler.temperature:.3f}", flush=True)
    return np.mean(preds, axis=0)


def run(refresh: bool = True, verbose: bool = True,
        as_of: pd.Timestamp | None = None) -> Path | None:
    """`as_of` back-dates the 'now' used to select upcoming fixtures.

    DRY RUNS ONLY. It exists so the whole path can be exercised and timed
    against a window that has already resolved, which is the only way to check
    the grader before trusting it live. The scheduled job never passes it.
    """
    now = uk_now_naive() if as_of is None else pd.Timestamp(as_of)

    if refresh:
        t = time.time()
        rep = refresh_current()
        if verbose:
            print(f"  refresh   {time.time() - t:5.1f}s  "
                  f"downloaded {rep.downloaded}, cached {rep.cached}, missing {rep.missing}")
        if rep.errors:
            raise RuntimeError(f"refresh failed on {len(rep.errors)} targets: {rep.errors[:3]}")
        build_matches(write=True)

    fixtures = build_fixtures(refresh=refresh, now=now)
    if fixtures.empty:
        print("no upcoming fixtures in the feed window; nothing to predict")
        return None

    seen = already_predicted()
    fresh = fixtures[~fixtures["match_id"].isin(seen)].copy()
    if len(fresh) < len(fixtures):
        print(f"  skipping {len(fixtures) - len(fresh)} fixtures already predicted "
              "in an earlier run; a prediction is never restated")
    if fresh.empty:
        print("every fixture in the window has already been predicted")
        return None

    df, seq_all, _ = build_forward(fresh, verbose=verbose)

    horizon = df[df[UNPLAYED_COL]].reset_index(drop=True)
    if len(horizon) != len(fresh):
        raise RuntimeError(f"expected {len(fresh)} unplayed rows, found {len(horizon)}")

    first_kick = pd.to_datetime(horizon["kickoff"]).min()
    train = df[df["result"].notna() & (pd.to_datetime(df["kickoff"]) < first_kick)]
    train = train.reset_index(drop=True)
    assert pd.to_datetime(train["kickoff"]).max() < first_kick, \
        "training data reaches into the horizon"

    if verbose:
        print(f"  training on {len(train):,} completed matches, all before {first_kick}")
        print(f"  predicting {len(horizon)} fixtures across "
              f"{horizon['div'].nunique()} divisions")
    p = fit_and_predict_forward(train, horizon, seq_all, verbose=verbose)

    out = pd.DataFrame({
        "predicted_at": now.isoformat(timespec="seconds"),
        "match_id": horizon["match_id"],
        "kickoff": pd.to_datetime(horizon["kickoff"]).dt.strftime("%Y-%m-%d %H:%M"),
        "div": horizon["div"], "league": horizon["league"], "season": horizon["season"],
        "home_raw": horizon["home_raw"], "away_raw": horizon["away_raw"],
        "home_key": horizon["home_key"], "away_key": horizon["away_key"],
        "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2],
    })
    for c in PRICE_COLS:
        out[c] = horizon[c].to_numpy() if c in horizon.columns else np.nan

    # A prediction file containing a fixture that has already kicked off is
    # worthless, and would not otherwise announce itself. Training takes
    # minutes, so this is re-checked against the clock as it stands NOW rather
    # than as it stood when the horizon was selected.
    deadline = now if as_of is not None else uk_now_naive()
    late = pd.to_datetime(out["kickoff"]) <= deadline
    if late.any() and as_of is None:
        raise RuntimeError(
            f"{int(late.sum())} fixtures kicked off during the run; refusing to write "
            "a prediction file that claims to precede them"
        )

    if as_of is not None:
        # A back-dated run is not evidence of anything and must never land in
        # predictions/, where the only thing distinguishing real forward
        # predictions from replayed ones is that replayed ones are not there.
        print(f"DRY RUN (as-of {now}): {len(out)} predictions, not written")
        print(out[["match_id", "kickoff", "p_home", "p_draw", "p_away",
                   "bfeh", "b365h"]].head(10).to_string(index=False))
        return None

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    dest = PREDICTIONS_DIR / f"{now.date()}.csv"
    if dest.exists():
        prior = pd.read_csv(dest)
        out = pd.concat([prior, out[~out["match_id"].isin(prior["match_id"])]],
                        ignore_index=True)
    out[OUT_COLS].to_csv(dest, index=False)
    print(f"wrote {dest.relative_to(REPO_ROOT)} -- {len(out)} predictions")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-refresh", action="store_true",
                    help="use the cached CSVs as they stand (for a dry run)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--as-of", default=None,
                    help="DRY RUN ONLY: back-date 'now' to replay a resolved "
                         "window. Writes nothing.")
    args = ap.parse_args()
    run(refresh=not args.no_refresh, verbose=not args.quiet, as_of=args.as_of)


if __name__ == "__main__":
    main()
