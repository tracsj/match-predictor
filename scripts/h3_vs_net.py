"""Does fitting the movement label directly beat the match model's accident?

    uv run python scripts/h3_vs_net.py

This is H3's pre-registered "genuinely uncertain part", and it is a control
rather than a candidate: it compares two already-scored arms on a shared row
set and introduces no new configuration. The registry count does not move.

WHY IT NEEDS ITS OWN SCRIPT. The H3 runner prints the net's tier band from
H1's diagnostics as a rough comparator, and that comparison is not sound: the
net's +2.7pp to +8.7pp was measured over 2015-16 -> 2024-25 pooled by tier and
restricted to tiered divisions, while H3's +4.27pp is 2024-25 alone across
every main division carrying a Bet365 price. Different seasons, different
divisions, different odds mixes. Eyeballing one against the other is exactly
the sort of comparison that reads as decisive and means nothing.

So this restricts BOTH arms to the same rows -- 2024-25 matches present in
both the H1 panel and H3's holdout -- and scores them against the same
odds-matched null.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.betting import PINNACLE_PRE, BetRule, simulate
from src.eval.metrics import OUTCOMES
from src.h1 import CACHE as H1_CACHE, build_panel
from src.h3 import (
    B365_PRE_COLS, HOLDOUT_SEASON, add_price_features, bets_from, build_frame,
    matched_null_shortening, movement_label,
)
from src.models.baselines import ALL_FEATURES, CatBoostBaseline

RULE = BetRule(min_ev=0.05, min_odds=1.5, max_odds=5.0, stake=1.0)


def score(name: str, pool: pd.DataFrame, ratios: np.ndarray,
          bet_odds: np.ndarray) -> dict:
    sims = matched_null_shortening(pool, bet_odds)
    null, lo, hi = sims.mean(), *np.quantile(sims, [0.025, 0.975])
    obs = float((ratios > 1.0).mean())
    se = np.sqrt(null * (1 - null) / len(ratios))
    z = (obs - null) / se if se else np.nan
    return {"arm": name, "n_bets": len(ratios), "observed": obs, "null": null,
            "null_lo": lo, "null_hi": hi, "margin": obs - null, "z": z,
            "p": 2 * (1 - stats.norm.cdf(abs(z))),
            "mean_ratio": float(ratios.mean())}


def main() -> None:
    # ---- H3's holdout, refitted exactly as src.h3 does ----
    h3 = build_frame()
    h3, price_names = add_price_features(h3)
    h3["h3_label"] = movement_label(h3)
    feats = ALL_FEATURES + price_names
    tr = h3[h3["season"] < HOLDOUT_SEASON]
    te = h3[h3["season"] == HOLDOUT_SEASON].reset_index(drop=True)

    print("=" * 78)
    print("H3 vs THE MATCH MODEL, ON IDENTICAL ROWS")
    print("=" * 78)
    print(f"  fitting H3 (CatBoost, repo defaults) on {len(tr):,} rows...")
    m = CatBoostBaseline().fit(tr[feats].to_numpy(float), tr["h3_label"].to_numpy())
    proba = m.predict_proba(te[feats].to_numpy(float))
    pred = np.array(OUTCOMES)[np.argmax(proba, axis=1)]

    # ---- the net's 2024-25 predictions, from H1's cache ----
    if not Path(H1_CACHE).exists():
        raise SystemExit(f"{H1_CACHE} missing -- run `uv run python -m src.h1` first")
    panel, _, _ = build_panel()
    z = np.load(H1_CACHE, allow_pickle=False)
    graded = panel.iloc[z["test_idx"]].reset_index(drop=True)
    net_p = z["probs"]
    keep = (graded["season"] == HOLDOUT_SEASON).to_numpy()
    net_rows, net_probs = graded[keep].reset_index(drop=True), net_p[keep]

    # ---- the shared row set ----
    shared = set(te["match_id"]) & set(net_rows["match_id"])
    print(f"  H3 holdout rows        {len(te):,}")
    print(f"  net 2024-25 rows       {len(net_rows):,}")
    print(f"  shared (both arms)     {len(shared):,}")
    print()
    print("  The net's panel is tiered divisions only, so the intersection is")
    print("  smaller than either arm. Both are scored on it and nothing else.")

    te_m = te["match_id"].isin(shared).to_numpy()
    net_m = net_rows["match_id"].isin(shared).to_numpy()
    pool = te[te_m].reset_index(drop=True)

    rows = []

    # H3 on the shared rows
    b = bets_from(pool, pred[te_m], proba[te_m])
    rows.append(score("H3 (movement label, direct)", pool,
                      b["ratio"].to_numpy(), b["pre"].to_numpy()))

    # The net's max-EV selections on the same rows, via the same simulate()
    # the pre-registered rule uses, so the comparison is against what the net
    # ACTUALLY did rather than a re-derivation of it.
    npool = net_rows[net_m].reset_index(drop=True)
    nbets = simulate(npool, net_probs[net_m], PINNACLE_PRE, RULE)
    idx = npool.set_index("match_id")
    close_map = {"H": "psch", "D": "pscd", "A": "psca"}
    nclose = np.array([float(idx.loc[mid, close_map[sel]])
                       for mid, sel in zip(nbets["match_id"], nbets["selection"])])
    nratio = nbets["odds"].to_numpy(float) / nclose
    ok = np.isfinite(nratio)
    rows.append(score("the net (match model, incidental)", pool,
                      nratio[ok], nbets["odds"].to_numpy(float)[ok]))

    print()
    print("-" * 78)
    print("BOTH ARMS, SAME ROWS, SAME NULL CONSTRUCTION")
    print("-" * 78)
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    a, bnet = rows[0], rows[1]
    diff = a["margin"] - bnet["margin"]
    se = np.sqrt(a["observed"] * (1 - a["observed"]) / a["n_bets"]
                 + bnet["observed"] * (1 - bnet["observed"]) / bnet["n_bets"])
    zz = diff / se if se else np.nan

    print()
    print("=" * 78)
    print("THE PRE-REGISTERED UNCERTAIN PART")
    print("=" * 78)
    print(f"  H3 margin over null      {a['margin']:+.4f}")
    print(f"  net margin over null     {bnet['margin']:+.4f}")
    print(f"  difference               {diff:+.4f}   z {zz:.2f}   "
          f"p {2 * (1 - stats.norm.cdf(abs(zz))):.4f}")
    print()
    print("  CAVEAT that keeps this honest: the two arms do not bet the same")
    print("  matches. H3 bets every row whose predicted outcome is in band; the")
    print("  net bets only where its EV filter fires. The bet counts differ and")
    print("  the selections overlap only partly, so this is a comparison of two")
    print("  strategies on a shared universe, not a paired test.")
    print()
    if abs(zz) < 1.96:
        print("  >>> No detectable difference. Fitting the movement label")
        print("      directly does about as well as a match model's")
        print("      disagreement with the price -- which is the interesting")
        print("      negative H3 was written to look for: the market moves")
        print("      toward what a decent match model already thinks, and there")
        print("      is no separate microstructure signal on top of it.")
    elif diff > 0:
        print("  >>> H3 beats the incidental signal. There IS structure in the")
        print("      movement beyond what a match model extracts.")
    else:
        print("  >>> The match model beats H3. Fitting the label directly is")
        print("      worse than predicting the match, which would say the")
        print("      movement is mostly about match quality after all.")


if __name__ == "__main__":
    main()
