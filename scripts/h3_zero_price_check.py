"""How many rows carry a price of 0.0, and does excluding them change H3?

    uv run python scripts/h3_zero_price_check.py

The H3 run emitted `divide by zero encountered` while building its pre-close
features, which means some Bet365 or Pinnacle price in the corpus is literally
0.0. A price of zero is not a price -- it is missing data encoded as a number,
and `notna()` does not catch it.

This is the repo's own standing rule biting from the other direction: never
fill a missing value with zero. Here the FEED supplied the zero and the filter
let it through, which produces an infinite log-implied-probability that
`np.nan_to_num` then silently converts into a huge finite feature value. No
error, no NaN, just a garbage row treated as informative.

A control. It fits nothing and does not move the registry count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.betting import PINNACLE_CLOSE, PINNACLE_PRE
from src.features.build import load as load_features
from src.h3 import B365_PRE_COLS, DEV_FIRST, HOLDOUT_SEASON

ALL_PRICE_COLS = PINNACLE_PRE.cols + PINNACLE_CLOSE.cols + B365_PRE_COLS


def main() -> None:
    df = load_features()
    df = df[(df["source"] == "main") & df["result"].notna()
            & df[ALL_PRICE_COLS].notna().all(axis=1)
            & df["season"].between(DEV_FIRST, HOLDOUT_SEASON)]

    print("=" * 78)
    print("ZERO AND NON-POSITIVE PRICES IN H3's FRAME")
    print("=" * 78)
    print(f"  rows passing the notna() filter H3 used: {len(df):,}")
    print()

    arr = df[ALL_PRICE_COLS].to_numpy(float)
    bad_cell = ~np.isfinite(arr) | (arr <= 0)
    bad_row = bad_cell.any(axis=1)
    print(f"  rows with at least one non-positive price: {int(bad_row.sum()):,} "
          f"({bad_row.mean():.4%})")
    print()
    print("  by column:")
    for j, c in enumerate(ALL_PRICE_COLS):
        n = int(bad_cell[:, j].sum())
        if n:
            print(f"    {c:8} {n:,}")
    if not bad_cell.any():
        print("    none")

    print()
    print("  by season:")
    s = df.assign(_bad=bad_row).groupby("season")["_bad"].agg(["sum", "size"])
    s = s[s["sum"] > 0]
    if len(s):
        s["rate"] = s["sum"] / s["size"]
        print(s.to_string(float_format=lambda v: f"{v:.4f}"))
    else:
        print("    none")

    print()
    print("  A price of 1.0 is also suspect -- it implies certainty and pays")
    print("  nothing. Counting those separately:")
    ones = (arr == 1.0).any(axis=1)
    print(f"    rows with a price of exactly 1.0: {int(ones.sum()):,}")

    print()
    print("=" * 78)
    print("WHAT IT MEANS FOR THE HOLDOUT")
    print("=" * 78)
    hold = df[df["season"] == HOLDOUT_SEASON]
    a = hold[ALL_PRICE_COLS].to_numpy(float)
    hb = (~np.isfinite(a) | (a <= 0)).any(axis=1)
    print(f"  {HOLDOUT_SEASON}: {len(hold):,} rows, "
          f"{int(hb.sum()):,} with a non-positive price ({hb.mean():.4%})")
    print()
    if hb.sum() == 0:
        print("  The holdout is clean, so the H3 result is unaffected and the")
        print("  contamination is confined to training rows -- which is worth")
        print("  fixing but cannot have manufactured the holdout number.")
    else:
        print("  The holdout is NOT clean. The H3 result has to be re-run with")
        print("  these rows excluded before it can be reported.")


if __name__ == "__main__":
    main()
