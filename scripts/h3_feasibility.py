"""What H3 has to work with, and whether its label is worth modelling at all.

    uv run python scripts/h3_feasibility.py

A control, not a candidate. It fits nothing and scores no configuration, so it
does not move the registry count. Its job is to answer three questions before
a pre-registration is written, because each one could change the design:

  1. WHICH PRE-CLOSE COLUMNS EXIST as candidate features. H3's most promising
     input is cross-book disagreement at the snapshot -- if the books disagree
     about a match and one of them is about to move, that disagreement is
     visible before the move. That only works if several books' pre-close
     prices are actually populated on the same rows.

  2. HOW MUCH THE LABEL MOVES. If the pre-close is already within a whisker of
     the close, there is nothing to forecast and H3 is dead before it starts.
     football-data's snapshot is taken one to three days out, so a large part
     of the open-to-close path has already happened by then.

  3. HOW CONCENTRATED THE LABEL IS. If one outcome shortens most in, say, 80%
     of matches, then a constant predictor scores 80% and any accuracy number
     has to be read against that, not against 33%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.betting import PINNACLE_CLOSE, PINNACLE_PRE
from src.features.build import load as load_features

# Every pre-close book triple football-data has carried at some point. Closing
# columns are deliberately NOT listed: on the input side they are the leak.
PRE_BOOKS = {
    "b365": ["b365h", "b365d", "b365a"],
    "bw":   ["bwh", "bwd", "bwa"],
    "iw":   ["iwh", "iwd", "iwa"],
    "ps":   ["psh", "psd", "psa"],
    "wh":   ["whh", "whd", "wha"],
    "vc":   ["vch", "vcd", "vca"],
    "max":  ["maxh", "maxd", "maxa"],
    "avg":  ["avgh", "avgd", "avga"],
    "bfe":  ["bfeh", "bfed", "bfea"],
}


def main() -> None:
    df = load_features()
    df = df[(df["source"] == "main") & df["result"].notna()].copy()
    cols = set(df.columns)

    print("=" * 78)
    print("1. CANDIDATE PRE-CLOSE FEATURE COLUMNS")
    print("=" * 78)
    print("  Coverage measured on rows that carry BOTH Pinnacle legs, since")
    print("  those are the only rows H3 can be graded on anyway.")
    print()
    legs = PINNACLE_PRE.cols + PINNACLE_CLOSE.cols
    gradable = df[df[legs].notna().all(axis=1)] if all(c in cols for c in legs) else df.iloc[:0]
    print(f"  gradable rows: {len(gradable):,}")
    print()
    for book, trio in PRE_BOOKS.items():
        missing = [c for c in trio if c not in cols]
        if missing:
            print(f"    {book:5} ABSENT from the corpus ({missing})")
            continue
        cov = gradable[trio].notna().all(axis=1).mean() if len(gradable) else float("nan")
        print(f"    {book:5} present, {cov:.1%} of gradable rows")

    usable = [b for b, t in PRE_BOOKS.items()
              if all(c in cols for c in t)
              and len(gradable) and gradable[t].notna().all(axis=1).mean() > 0.90]
    print()
    print(f"  books above 90% coverage: {usable}")
    print("  Cross-book disagreement needs at least two of these on the same row.")

    if len(gradable) == 0:
        raise SystemExit("no gradable rows -- nothing further to measure")

    pre = gradable[PINNACLE_PRE.cols].to_numpy(float)
    close = gradable[PINNACLE_CLOSE.cols].to_numpy(float)
    ratio = pre / close
    logr = np.log(ratio)

    print()
    print("=" * 78)
    print("2. HOW MUCH DOES THE LABEL ACTUALLY MOVE?")
    print("=" * 78)
    print("  |log(pre/close)| per outcome, across all gradable rows.")
    print()
    flat = np.abs(logr).ravel()
    flat = flat[np.isfinite(flat)]
    for q in (0.25, 0.50, 0.75, 0.90, 0.99):
        print(f"    {q:.0%} quantile   {np.quantile(flat, q):.4f}")
    print(f"    mean          {flat.mean():.4f}")
    print(f"    share of outcomes moving <0.5%   {(flat < 0.005).mean():.1%}")
    print(f"    share of outcomes moving <1%     {(flat < 0.010).mean():.1%}")
    print()
    print("  For scale, Pinnacle's overround is ~3-5%, so a move has to clear")
    print("  a couple of points before it is worth anything to a bettor.")

    print()
    print("=" * 78)
    print("3. HOW CONCENTRATED IS THE 3-WAY LABEL?")
    print("=" * 78)
    print("  Label = which outcome's price SHORTENED most, i.e. argmax of")
    print("  pre/close. A constant predictor scores the largest share below,")
    print("  and that -- not 33% -- is the number any accuracy must beat.")
    print()
    ok = np.isfinite(ratio).all(axis=1)
    lab = np.argmax(np.where(np.isfinite(ratio), ratio, -np.inf), axis=1)[ok]
    names = ["H", "D", "A"]
    counts = pd.Series(lab).value_counts(normalize=True).sort_index()
    for i, share in counts.items():
        print(f"    {names[i]}   {share:.4f}")
    print(f"\n    majority-class baseline: {counts.max():.4f}")

    print()
    print("  Same, split by season, to see whether the majority class is stable")
    print("  or whether a walk-forward model would be chasing a moving target:")
    g = gradable[ok].copy()
    g["_lab"] = [names[i] for i in lab]
    tab = pd.crosstab(g["season"], g["_lab"], normalize="index")
    print(tab[tab.index >= "2015-16"].to_string(float_format=lambda v: f"{v:.3f}"))

    print()
    print("=" * 78)
    print("WHAT THIS DECIDES")
    print("=" * 78)
    print("  If the label barely moves, or one class dominates and is stable,")
    print("  then H3's free form is answering a question with very little")
    print("  signal in it, and the pre-registration should say so and set its")
    print("  expectations accordingly rather than discovering it afterwards.")


if __name__ == "__main__":
    main()
