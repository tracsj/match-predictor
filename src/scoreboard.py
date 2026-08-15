"""Print the scoreboard: the ceiling, the baselines, and (later) the network.

    uv run python -m src.scoreboard

Three rows are always visible, because a number without its ceiling and its
floor is not interpretable:

    de-vigged Pinnacle closing   the ceiling -- what the market knew
    tuned Dixon-Coles            the baseline the network must pass
    the network                  once it exists

Until the network exists this prints the ceiling, the naive floors, and the
betting-side reality check, which is enough to tell whether the pipeline is
sane before any modelling effort is spent on it.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data.footballdata import OUT_DIR
from src.eval.betting import (
    B365_CLOSE, MARKET_AVG_PRE, MARKET_MAX_CLOSE, MARKET_MAX_PRE,
    PINNACLE_CLOSE, PINNACLE_PRE, BetRule, bootstrap_ci, clv_report,
    closing_price_for_bets, random_bet_null, required_sample_size, simulate,
    summarize,
)
from src.eval.devig import devig, overround
from src.eval.metrics import OUTCOMES, summary
from src.eval.split import assert_no_leakage, season_walk_forward

PRICE_SETS = (PINNACLE_CLOSE, B365_CLOSE, MARKET_MAX_CLOSE)


def load_panel(first: str = "2016-17", last: str = "2024-25") -> pd.DataFrame:
    """Main divisions with a complete Pinnacle closing price.

    Ends at 2024-25 by default. Pinnacle closing coverage decays from October
    2025 and is absent from February 2026, so 2025-26 cannot supply the sharp
    benchmark -- see docs/research/00-measured-facts.md.
    """
    df = pd.read_parquet(OUT_DIR / "matches.parquet")
    df = df[(df["source"] == "main")
            & df["season"].between(first, last)
            & df[PINNACLE_CLOSE.cols].notna().all(axis=1)]
    return df.sort_values("kickoff").reset_index(drop=True)


def reference_models(panel: pd.DataFrame, train_mask: np.ndarray | None = None) -> dict:
    """The forecasts that need no fitting, plus one that needs only counting.

    `train_mask` scopes the base-rate fit so it never sees the test period.
    """
    n = len(panel)
    y = panel["result"].to_numpy()
    fit_on = y if train_mask is None else y[train_mask]

    base = np.array([(fit_on == o).mean() for o in OUTCOMES], dtype=float)
    base = base / base.sum()

    return {
        "uniform": np.full((n, 3), 1 / 3),
        "base_rate": np.tile(base, (n, 1)),
        "market (Pinnacle close, Shin)": devig(
            panel[PINNACLE_CLOSE.cols].to_numpy(float), method="shin"),
        "market (Pinnacle close, mult)": devig(
            panel[PINNACLE_CLOSE.cols].to_numpy(float), method="multiplicative"),
    }


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first-season", default="2016-17")
    ap.add_argument("--last-season", default="2024-25")
    ap.add_argument("--min-ev", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    panel = load_panel(args.first_season, args.last_season)
    y = panel["result"]

    print("=" * 78)
    print("CORPUS")
    print("=" * 78)
    ov = overround(panel[PINNACLE_CLOSE.cols].to_numpy(float))
    print(f"  matches            {len(panel):,}")
    print(f"  seasons            {panel['season'].nunique()}  "
          f"({panel['season'].min()} .. {panel['season'].max()})")
    print(f"  divisions          {panel['div'].nunique()}")
    print(f"  outcome mix        H {(y=='H').mean():.3f}  D {(y=='D').mean():.3f}  "
          f"A {(y=='A').mean():.3f}")
    print(f"  Pinnacle overround median {np.median(ov):.4f}  "
          f"(mean {ov.mean():.4f})  -> the vig you must clear")

    print()
    print("=" * 78)
    print("FORECAST QUALITY   (lower is better for rps / log_loss / brier / ece)")
    print("=" * 78)
    print("  Walk-forward. Anything that needs fitting is fitted on prior")
    print("  seasons only, then scored on the held-out season. The market rows")
    print("  need no fitting, so they are unaffected -- but they are scored on")
    print("  the same held-out rows, so every row here is comparable.")
    print()

    splits = list(season_walk_forward(panel, min_train_seasons=3))
    for s in splits:
        assert_no_leakage(panel, s)

    oos: dict[str, list[np.ndarray]] = {}
    oos_y: list[np.ndarray] = []
    for s in splits:
        mask = np.zeros(len(panel), dtype=bool)
        mask[s.train_idx] = True
        models = reference_models(panel, train_mask=mask)
        for name, p in models.items():
            oos.setdefault(name, []).append(p[s.test_idx])
        oos_y.append(panel["result"].to_numpy()[s.test_idx])

    y_oos = np.concatenate(oos_y)
    rows = [summary(np.vstack(v), y_oos, label=k) for k, v in oos.items()]
    print(_fmt(pd.DataFrame(rows)))
    print()
    print("  Published anchors: market RPS 0.1905 over 19 Serie A seasons")
    print("  (Pitcan 2026); bookmaker consensus 0.2063 on the 2023 Soccer")
    print("  Prediction Challenge, where the best deep model managed 0.2195.")
    print("  A uniform forecast scores log loss ln(3) = 1.0986 exactly.")

    print()
    print("=" * 78)
    print("WALK-FORWARD SPLITS   (train on all prior seasons, test on one)")
    print("=" * 78)
    print(f"  {len(splits)} splits, all pass assert_no_leakage; "
          f"{len(y_oos):,} out-of-sample matches scored above")
    srows = [{"test_season": s.label, "train_n": len(s.train_idx),
              "test_n": len(s.test_idx), "train_ends": s.train_end.date(),
              "test_starts": s.test_start.date()} for s in splits]
    print(_fmt(pd.DataFrame(srows)))

    print()
    print("=" * 78)
    print("BETTING REALITY CHECK")
    print("=" * 78)
    print("  Betting the de-vigged market back into its own prices. Under")
    print("  multiplicative de-vig every selection carries EV = 1/overround - 1,")
    print("  so this is a closed-form check that the price join is correct.")
    print()
    mult = devig(panel[PINNACLE_CLOSE.cols].to_numpy(float), method="multiplicative")
    expected = float((1 / overround(panel[PINNACLE_CLOSE.cols].to_numpy(float)) - 1).mean())
    forced = simulate(panel, mult, PINNACLE_CLOSE,
                      BetRule(min_ev=-1.0, min_odds=1.0, max_odds=1e6, name="bet everything"))
    got = forced["pnl"].sum() / forced["stake"].sum()
    print(f"  expected ROI {expected:+.4f}   observed {got:+.4f}   "
          f"delta {got - expected:+.4f}")

    print()
    print("  Same forecasts, EV threshold applied, across the three price sets.")
    print("  Lead with Pinnacle close. A strategy that is profitable only in")
    print("  the market-max column is an odds-comparison screen, not a model.")
    print()
    rule = BetRule(min_ev=args.min_ev)
    brows = []
    for ps in PRICE_SETS:
        if not all(c in panel.columns for c in ps.cols):
            continue
        sub = panel[panel[ps.cols].notna().all(axis=1)].reset_index(drop=True)
        if sub.empty:
            continue
        p = devig(sub[PINNACLE_CLOSE.cols].to_numpy(float), method="shin")
        bets = simulate(sub, p, ps, rule)
        row = summarize(bets, with_ci=False)
        row["n_eligible"] = len(sub)
        if len(bets):
            ci = bootstrap_ci(bets, n_boot=args.boot, seed=0)
            row["roi_lo"], row["roi_hi"] = ci["lo"], ci["hi"]
            row["excludes_zero"] = bool(ci["lo"] > 0 or ci["hi"] < 0)
            null = random_bet_null(sub, ps, n_bets=len(bets), n_sims=300, seed=0)
            row["null_roi"] = null["mean_roi"]
        brows.append(row)
    if brows:
        cols = ["price_set", "n_bets", "n_eligible", "roi", "roi_lo", "roi_hi",
                "excludes_zero", "null_roi", "hit_rate", "avg_odds",
                "n_needed_for_2pct"]
        tbl = pd.DataFrame(brows)
        print(_fmt(tbl[[c for c in cols if c in tbl.columns]]))
        print()
        print("  Read this as a control, not a result. The forecast IS the")
        print("  Pinnacle market, so it contains no forecasting skill by")
        print("  construction -- any profit is one book disagreeing with")
        print("  another. That is a price-shopping edge, and it is exactly the")
        print("  strategy Kaunitz et al. (2017) ran profitably until the")
        print("  bookmakers limited their stakes to as little as $1.25.")
        print()
        print("  Also note what the market-max column is: an aggregator's")
        print("  record that SOME book showed that price. Not necessarily one")
        print("  you could have taken, in size, at that moment. No commission")
        print("  is modelled. 3 price sets were tried, which is the")
        print("  multiple-comparisons disclosure.")

    print()
    print("-" * 78)
    print("  CLOSING-LINE VALUE   (the headline metric -- report before ROI)")
    print("-" * 78)
    print("  Bet at a PRE-close price, grade against the Pinnacle close.")
    print("  Ratio > 1 means you took a bigger price than the market settled")
    print("  at. Betting and grading at the same price makes this identically")
    print("  1.0, which is why the pre-close columns exist.")
    print()
    clv_panel = panel[panel[PINNACLE_PRE.cols + MARKET_MAX_PRE.cols]
                      .notna().all(axis=1)].reset_index(drop=True)
    if len(clv_panel):
        sharp = devig(clv_panel[PINNACLE_PRE.cols].to_numpy(float), method="shin")
        all_in = BetRule(min_ev=-1.0, min_odds=1.0, max_odds=1e6, name="every match")
        crows = []
        for ps in (PINNACLE_PRE, MARKET_AVG_PRE, MARKET_MAX_PRE):
            if not all(c in clv_panel.columns for c in ps.cols):
                continue
            b = simulate(clv_panel, sharp, ps, all_in)
            if b.empty:
                continue
            r = clv_report(b, closing_price_for_bets(b, clv_panel))
            crows.append({"taken_at": ps.label, "n": r["n"],
                          "mean_ratio": r["mean_ratio"],
                          "median_ratio": r["median_ratio"],
                          "pct_shortened": r["pct_shortened"],
                          "binom_p": r["binom_pvalue"]})
        if crows:
            print(_fmt(pd.DataFrame(crows)))
            print()
            print("  Pinnacle pre-close against Pinnacle close is the null: no")
            print("  selection skill, only drift between Friday and kickoff, so")
            print("  it should sit at ~1.0. Anything above it is price shopping.")

    print()
    null = random_bet_null(panel, PINNACLE_CLOSE, n_bets=2000, n_sims=500, seed=0)
    print(f"  random-bet null (2,000 bets)   mean ROI {null['mean_roi']:+.4f}   "
          f"95% [{null['lo']:+.4f}, {null['hi']:+.4f}]")
    print(f"  bets needed to prove a 2% edge at avg odds 3.2: "
          f"{required_sample_size(3.2, 0.02):,}")
    print(f"  ...at even money: {required_sample_size(2.0, 0.02):,}")
    print()
    print("  Report closing-line value before ROI. CLV converges roughly a")
    print("  hundred times faster, which is the only reason a real edge is")
    print("  detectable inside a human lifetime.")


if __name__ == "__main__":
    main()
