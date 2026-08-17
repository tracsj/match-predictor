"""Which price ladder can grade a tier-stratified CLV on the 2025/26 holdout?

    uv run python scripts/h1_holdout_coverage.py

H1 ran on data the project had already seen, and said so. The only untouched
data available is season 2025-26, consumed once by the Phase 6 pre-registered
run and never stratified by tier. Before paying for a fit, this establishes
what that season can actually support.

Two candidate ladders and a real doubt about each. Pinnacle's pair coverage in
2025-26 is 29-48% and stops after 2026-01-14. The Betfair Exchange is ~100%
covered from 2024/25 -- but src/eval/betting.py records that the exchange
PRE-close (BFEH/BFED/BFEA) arrives in fixtures.csv while only the CLOSE
(BFEC*) is in the results files, and CLV needs both. If that is right, the
exchange cannot grade a historical CLV at all, and the note is worth
confirming rather than repeating.
"""

from __future__ import annotations

import pandas as pd

from src.eval.betting import (
    EXCHANGE_CLOSE, EXCHANGE_PRE, PINNACLE_CLOSE, PINNACLE_PRE,
)
from src.features.build import load as load_features
from src.features.ratings import TIER
from src.h1 import LOWER_DIVS

HOLDOUT = "2025-26"


def main() -> None:
    df = load_features()
    df = df[(df["source"] == "main") & df["result"].notna()
            & df["div"].isin(TIER)].copy()
    df["stratum"] = df["div"].map(
        lambda d: "lower (3-5)" if d in LOWER_DIVS else "upper (1-2)")

    print("=" * 78)
    print(f"WHAT THE {HOLDOUT} HOLDOUT CAN GRADE")
    print("=" * 78)

    for label, pre, close in (("pinnacle", PINNACLE_PRE, PINNACLE_CLOSE),
                              ("exchange", EXCHANGE_PRE, EXCHANGE_CLOSE)):
        missing = [c for c in pre.cols + close.cols if c not in df.columns]
        print(f"\n  {label}: pre={pre.cols} close={close.cols}")
        if missing:
            print(f"    COLUMNS ABSENT FROM THE CORPUS: {missing}")
            print("    -> cannot grade CLV on this ladder at all")
            continue
        h = df[df["season"] == HOLDOUT]
        has_pre = h[pre.cols].notna().all(axis=1)
        has_close = h[close.cols].notna().all(axis=1)
        pair = (has_pre & has_close)
        print(f"    {HOLDOUT}: {len(h):,} matches, pre {has_pre.sum():,} "
              f"({has_pre.mean():.1%}), close {has_close.sum():,} "
              f"({has_close.mean():.1%}), PAIR {pair.sum():,} ({pair.mean():.1%})")
        if pair.any():
            g = h[pair].groupby("stratum").size()
            print(f"    by stratum: {g.to_dict()}")
            # phase6 observed 1,337 CLV bets from 2,964 eligible (45.1%).
            print("    estimated bets at phase6's 45.1% rate: "
                  f"{ {k: round(v * 1337 / 2964) for k, v in g.to_dict().items()} }")

    print()
    print("  The 3,250-bet floor from the H1 pre-registration applies to any")
    print("  stratum reported here. A stratum below it is INCONCLUSIVE BY")
    print("  FLOOR and directionally informative at best -- which is a fact")
    print("  about 2025-26's thin coverage, not a reason to lower the floor.")


if __name__ == "__main__":
    main()
