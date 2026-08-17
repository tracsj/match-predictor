"""What H1 would actually have to work with, measured rather than assumed.

    uv run python scripts/h1_coverage_probe.py

H1 contrasts CLV between pooled lower tiers and pooled upper tiers, and CLV
needs BOTH legs of the same price -- the pre-close it would have been bet at
and the close it is graded against. `docs/hypotheses/H1-lower-division-inefficiency.md`
verified the CLOSING leg only (~100% in every tier). The pre-close leg is the
one that has never been counted, and if it is thin in tiers 3-5 the whole
contrast loses its power before a model is ever fitted.

This counts coverage. It fits nothing, scores nothing, and places no bets, so
it does not move the registry count in `docs/PROGRAMME.md`.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from src.eval.betting import PINNACLE_CLOSE, PINNACLE_PRE
from src.features.build import load as load_features
from src.features.ratings import TIER

# H1's intended strata, from the hypothesis file.
LOWER = {"E2", "E3", "EC", "SC2", "SC3"}


def main() -> None:
    df = load_features()
    df = df[(df["source"] == "main") & df["result"].notna()].copy()
    df["tier"] = df["div"].map(TIER)

    pre = df[PINNACLE_PRE.cols].notna().all(axis=1)
    close = df[PINNACLE_CLOSE.cols].notna().all(axis=1)
    df["has_pre"] = pre.to_numpy()
    df["has_close"] = close.to_numpy()
    df["has_pair"] = (pre & close).to_numpy()

    print("=" * 78)
    print("H1 COVERAGE PROBE -- Pinnacle pre-close AND close, per tier per season")
    print("=" * 78)
    print(f"  corpus: {len(df):,} main-division matches with a result")
    print(f"  divisions carrying a tier: {df['tier'].notna().sum():,}")
    print(f"  divisions with NO tier (excluded from any tier contrast): "
          f"{sorted(df.loc[df['tier'].isna(), 'div'].unique())}")
    print()

    tiered = df[df["tier"].notna()].copy()
    tiered["tier"] = tiered["tier"].astype(int)

    print("-" * 78)
    print("PAIR COVERAGE BY SEASON x TIER  (both pre-close and close present)")
    print("-" * 78)
    pivot_n = tiered.pivot_table(index="season", columns="tier",
                                 values="has_pair", aggfunc="sum", fill_value=0)
    pivot_rate = tiered.pivot_table(index="season", columns="tier",
                                    values="has_pair", aggfunc="mean")
    print("\n  matches with both legs:")
    print(pivot_n.to_string())
    print("\n  as a share of matches played:")
    print(pivot_rate.to_string(float_format=lambda v: f"{v:.3f}"))

    print()
    print("-" * 78)
    print("EACH LEG SEPARATELY, BY TIER, over the whole corpus")
    print("-" * 78)
    by_tier = tiered.groupby("tier").agg(
        matches=("has_pair", "size"),
        has_pre=("has_pre", "sum"),
        has_close=("has_close", "sum"),
        has_pair=("has_pair", "sum"),
    )
    by_tier["pre_rate"] = by_tier["has_pre"] / by_tier["matches"]
    by_tier["close_rate"] = by_tier["has_close"] / by_tier["matches"]
    by_tier["pair_rate"] = by_tier["has_pair"] / by_tier["matches"]
    print(by_tier.to_string(float_format=lambda v: f"{v:.3f}"))

    print()
    print("-" * 78)
    print("BY DIVISION -- first and last season carrying a usable pair")
    print("-" * 78)
    usable = tiered[tiered["has_pair"]]
    span = usable.groupby(["tier", "div"]).agg(
        pairs=("has_pair", "size"),
        first_season=("season", "min"),
        last_season=("season", "max"),
    ).sort_index()
    print(span.to_string())

    print()
    print("-" * 78)
    print("THE STRATA H1 PROPOSES")
    print("-" * 78)
    tiered["stratum"] = tiered["div"].map(
        lambda d: "lower (3-5)" if d in LOWER else "upper (1-2)")
    # 2015-16 is the GRADED window's first season, not 2012-13. The panel opens
    # at 2012-13 (Pinnacle's first season on both legs), but
    # `run_walk_forward` hardcodes min_train_seasons=3, so the panel's first
    # three seasons are training-only and are never graded.
    for window in ("2012-13", "2015-16", "2016-17"):
        print(f"\n  window {window} -> 2024-25 inclusive:")
        sub = tiered[(tiered["season"] >= window) & (tiered["season"] <= "2024-25")]
        g = sub.groupby("stratum").agg(
            matches=("has_pair", "size"), pairs=("has_pair", "sum"))
        g["pair_rate"] = g["pairs"] / g["matches"]
        g["bets_at_phase6_rate"] = (g["pairs"] * (1337 / 2964)).round().astype(int)
        print(g.to_string(float_format=lambda v: f"{v:.3f}"))

    print()
    print("  The bet-rate column extrapolates phase6's observed 1,337 CLV bets")
    print("  from 2,964 eligible Pinnacle matches (45.1%). It is an estimate of")
    print("  volume, not a claim about the rate holding per tier.")

    print()
    print("-" * 78)
    print("HOW MANY BETS A STRATUM NEEDS BEFORE IT MAY BE CALLED A RESULT")
    print("-" * 78)
    print("  CLV's binomial test asks whether the share of prices that SHORTENED")
    print("  differs from 50%. Two-sided alpha = 0.01 to match the bar already")
    print("  set in docs/PREREGISTRATION.md, 80% power, sd 0.5 under both.")
    print()
    z_a, z_b = stats.norm.ppf(1 - 0.01 / 2), stats.norm.ppf(0.80)
    for delta in (0.02, 0.03, 0.05):
        n = ((z_a * 0.5 + z_b * 0.5) / delta) ** 2
        print(f"    detect a {delta:.0%} shift in % shortened -> {n:,.0f} bets")


if __name__ == "__main__":
    main()
