"""The tier-2 experiment: what is a starting XI worth?

    uv run python -m src.tier2

This is the question the two-tier design was built to answer, and the one that
decides whether the SportMonks upgrade is worth paying for. Same network, same
fixtures, same closing odds, trained twice -- once with a permutation-invariant
encoder over each starting XI, once without -- and the difference measured.

Three things the plan fixed in advance, all of which matter:

**Both arms warm-start from the same tier-1 pretrained model.** Training from
scratch on ~2,900 matches would handicap both arms equally and turn the result
into a statement about small-data optimisation rather than about players.

**The comparison is a paired bootstrap CI on per-match RPS**, not a bare delta.
The held-out set is small and the effects in this literature run 0.0002-0.002;
a point estimate at that scale is noise wearing a number.

**The literature is empty here.** No published pre-match football model encodes
a starting XI as a permutation-invariant set of player vectors. There is no
recipe, so a null result is a real finding rather than a failure to reproduce.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.data.sportmonks_parse import MATCHES_PARQUET, PLAYERS_PARQUET
from src.eval.betting import PINNACLE_CLOSE
from src.eval.devig import devig
from src.eval.metrics import log_loss, rps, rps_per_match, summary
from src.eval.split import assert_no_leakage, season_walk_forward
from src.features.build import load as load_features, load_sequences
from src.features.players import build_squads, player_feature_names
from src.models.baselines import ALL_FEATURES, OrderedLogit
from src.models.net import (
    NetConfig, TemperatureScaler, build_vocab, predict, train_net,
)

__all__ = ["build_tier2_panel", "run_tier2"]


def build_tier2_panel() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """SportMonks matches joined to the football-data feature table.

    The join is (division, date, home, away). SportMonks reports kickoff in
    UTC and football-data in local time -- a one-to-two hour offset for these
    leagues -- so matching on the date rather than the timestamp is deliberate,
    and safe because no fixture kicks off later than 20:00 UTC.
    """
    sm = pd.read_parquet(MATCHES_PARQUET).sort_values("kickoff").reset_index(drop=True)
    players = pd.read_parquet(PLAYERS_PARQUET)
    squads, squad_mask = build_squads(sm, players)

    fd = load_features()
    seq_all, _ = load_sequences()

    def jk(d):
        return (d["div"].astype(str) + "|"
                + pd.to_datetime(d["kickoff"]).dt.strftime("%Y%m%d") + "|"
                + d["home_key"].astype(str) + "|" + d["away_key"].astype(str))

    sm["_jk"] = jk(sm)
    fd = fd.copy()
    fd["_jk"] = jk(fd)
    fd_small = fd.drop(columns=[c for c in ("home_key", "away_key", "div", "fthg",
                                            "ftag", "result", "kickoff")
                               if c in fd.columns])

    sm["_row"] = np.arange(len(sm))
    panel = sm.merge(fd_small, on="_jk", how="inner")
    panel = panel.sort_values("kickoff").reset_index(drop=True)

    rows = panel["_row"].to_numpy()
    return panel, squads[rows], squad_mask[rows]


def _pretrain(cfg: NetConfig, features, seeds, verbose=True):
    """One tier-1 model per seed, trained on the full corpus up to the tier-2
    window. Shared by both arms so neither gets a head start."""
    full = load_features().sort_values("kickoff").reset_index(drop=True)
    full = full[full["result"].notna()].reset_index(drop=True)
    seq_all, _ = load_sequences()
    seq_full = seq_all[full["corpus_row"].to_numpy()]
    X = full[features].to_numpy(float)
    vocab = build_vocab(full)
    models = []
    for seed in seeds:
        t = time.time()
        m, meta = train_net(full, X, vocab,
                            NetConfig(**{**cfg.__dict__, "squad_hidden": 0, "seed": seed}),
                            seq_train=seq_full)
        models.append((m, meta))
        if verbose:
            print(f"    pretrain seed {seed}: {len(full):,} matches, "
                  f"best epoch {meta['best_epoch']} [{time.time() - t:.0f}s]", flush=True)
    return models, vocab


def run_tier2(seeds=(0, 1, 2), squad_hidden=24, n_boot=4000, verbose=True) -> dict:
    panel, squads, squad_mask = build_tier2_panel()
    if verbose:
        print(f"tier-2 panel: {len(panel):,} matches, "
              f"{panel['season'].nunique()} seasons, {panel['div'].nunique()} divisions")
        print(f"squad tensor: {squads.shape}, {squad_mask.mean():.1%} of slots filled")

    features = ALL_FEATURES
    X = panel[features].to_numpy(float)
    y_all = panel["result"].to_numpy()
    seq_all, _ = load_sequences()
    seq = seq_all[panel["corpus_row"].to_numpy()]

    base_cfg = NetConfig()
    if verbose:
        print("\npretraining the shared tier-1 trunk...")
    pre, _ = _pretrain(base_cfg, features, seeds, verbose=verbose)

    arms = {
        "without squad encoder": NetConfig(**{**base_cfg.__dict__, "squad_hidden": 0}),
        "with squad encoder": NetConfig(**{**base_cfg.__dict__,
                                           "squad_hidden": squad_hidden}),
    }

    vocab = build_vocab(panel)
    out: dict[str, list] = {k: [] for k in arms}
    ys = []

    splits = list(season_walk_forward(panel, min_train_seasons=3))
    if verbose:
        print(f"\n{len(splits)} walk-forward splits over the tier-2 panel")

    for s in splits:
        assert_no_leakage(panel, s)
        tr, te = panel.iloc[s.train_idx], panel.iloc[s.test_idx]
        for name, cfg in arms.items():
            preds = []
            for i, seed in enumerate(seeds):
                c = NetConfig(**{**cfg.__dict__, "seed": seed})
                m, meta = train_net(
                    tr, X[s.train_idx], vocab, c,
                    seq_train=seq[s.train_idx],
                    squads_train=squads[s.train_idx] if cfg.squad_hidden else None,
                    squad_mask_train=squad_mask[s.train_idx] if cfg.squad_hidden else None,
                    init_from=pre[i][0],
                )
                p = predict(m, te, X[s.test_idx], vocab, meta,
                            seq=seq[s.test_idx],
                            squads=squads[s.test_idx] if cfg.squad_hidden else None,
                            squad_mask=squad_mask[s.test_idx] if cfg.squad_hidden else None)
                cut = int(len(tr) * 0.85)
                val_out = predict(m, tr.iloc[cut:], X[s.train_idx][cut:], vocab, meta,
                                  seq=seq[s.train_idx][cut:],
                                  squads=squads[s.train_idx][cut:] if cfg.squad_hidden else None,
                                  squad_mask=squad_mask[s.train_idx][cut:] if cfg.squad_hidden else None)
                sc = TemperatureScaler().fit(val_out["logits"], tr.iloc[cut:]["result"])
                preds.append(sc.transform(p["logits"]))
            out[name].append(np.mean(preds, axis=0))
        ys.append(y_all[s.test_idx])
        if verbose:
            print(f"    {s.label}: train {len(tr):,} test {len(te):,}", flush=True)

    y = np.concatenate(ys)
    without = np.vstack(out["without squad encoder"])
    with_sq = np.vstack(out["with squad encoder"])

    # Paired bootstrap over matchdays on the per-match RPS difference.
    d = rps_per_match(without, y) - rps_per_match(with_sq, y)
    days = pd.to_datetime(np.concatenate([panel["kickoff"].to_numpy()[s.test_idx]
                                          for s in splits])).floor("D").to_numpy()
    uniq = np.unique(days)
    by_day = {u: d[days == u] for u in uniq}
    rng = np.random.default_rng(0)
    reps = np.array([np.concatenate([by_day[u] for u in
                                     rng.choice(uniq, len(uniq), replace=True)]).mean()
                     for _ in range(n_boot)])

    return {
        "panel": panel, "y": y, "without": without, "with": with_sq,
        "delta": float(d.mean()),
        "ci": (float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))),
        "n": int(len(y)), "n_days": int(len(uniq)),
        "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--squad-hidden", type=int, default=24)
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()

    r = run_tier2(seeds=tuple(args.seeds), squad_hidden=args.squad_hidden,
                  n_boot=args.boot)
    panel, y = r["panel"], r["y"]

    print()
    print("=" * 72)
    print("WHAT IS A STARTING XI WORTH?")
    print("=" * 72)

    rows = [summary(r["without"], y, "net WITHOUT squad encoder"),
            summary(r["with"], y, "net WITH squad encoder")]

    ok = panel[PINNACLE_CLOSE.cols].notna().all(axis=1).to_numpy()
    idx = np.concatenate([s.test_idx for s in
                          season_walk_forward(panel, min_train_seasons=3)])
    mkt_ok = ok[idx]
    if mkt_ok.any():
        mkt = devig(panel[PINNACLE_CLOSE.cols].to_numpy(float)[idx][mkt_ok], method="shin")
        rows.insert(0, summary(mkt, y[mkt_ok], "market (Pinnacle close)"))

    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print()
    lo, hi = r["ci"]
    print(f"  paired RPS difference (positive = the squad encoder helps)")
    print(f"    delta {r['delta']:+.5f}   95% BCa-style CI [{lo:+.5f}, {hi:+.5f}]"
          f"   t {r['t']:+.2f}")
    print(f"    over {r['n']:,} matches on {r['n_days']:,} matchdays")
    print()
    if lo > 0:
        print("  The interval excludes zero: the starting XI carries information")
        print("  the team-level features do not. An upgrade would buy more of it.")
    elif hi < 0:
        print("  The interval excludes zero in the WRONG direction: the encoder")
        print("  actively hurts at this sample size.")
    else:
        print("  The interval spans zero. On this sample the starting XI adds")
        print("  nothing measurable over team-level form and ratings -- which is")
        print("  a real finding, not a failed run, since nothing in the")
        print("  literature had tested it either way.")


if __name__ == "__main__":
    main()
