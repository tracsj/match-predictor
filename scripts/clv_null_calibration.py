"""What does a CLV ratio of 1.0 actually mean here? Not "no movement".

    uv run python scripts/clv_null_calibration.py

A control, not a candidate. No model is fitted and nothing is scored -- this
is prices only.

THE PROBLEM IT MEASURES. Both `docs/PREREGISTRATION.md` and the H1
pre-registration test closing-line value against ratio > 1.0 and % shortened
> 50%. That null assumes the pre-close and the close are, on average, the same
price. They are not. Pinnacle's margin tightens toward kickoff as limits rise,
so nearly every price LENGTHENS a little regardless of information, and a
selection picked at random from the rule's odds band shortens well under half
the time.

If that is right, then a ratio just under 1.0 is not evidence that a model sat
on the wrong side of the market's movement. It may be evidence of nothing at
all -- or, above the drift, of the right side.

This matters beyond H1. `docs/PHASE6_RESULT.md` reads its 0.9952 / 42.4% as
"the wrong side of the market's own movement", and that reading is only valid
if the correct null is 1.0 / 50%.

Two measurements, both model-free:
  1. The overround of the pre-close against the close, per season. This is the
     mechanism, and if it does not tighten the story is wrong.
  2. The unconditional band-eligible drift on PHASE 6's population -- season
     2025-26, all main divisions, not just the tiered ones H1 used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.betting import PINNACLE_CLOSE, PINNACLE_PRE, BetRule
from src.features.build import load as load_features

RULE = BetRule()
PHASE6_SEASON = "2025-26"
PHASE6_OBSERVED_PCT_SHORTENED = 0.424      # docs/PHASE6_RESULT.md
PHASE6_OBSERVED_MEAN_RATIO = 0.9952        # docs/PHASE6_RESULT.md


def main() -> None:
    df = load_features()
    df = df[(df["source"] == "main") & df["result"].notna()].copy()
    pre_cols, close_cols = PINNACLE_PRE.cols, PINNACLE_CLOSE.cols
    both = df[pre_cols + close_cols].notna().all(axis=1)
    df = df[both]

    pre = df[pre_cols].to_numpy(float)
    close = df[close_cols].to_numpy(float)

    print("=" * 78)
    print("1. THE MECHANISM -- does Pinnacle's margin tighten toward the close?")
    print("=" * 78)
    print("  Overround = sum of implied probabilities. If the close carries a")
    print("  SMALLER overround than the pre-close, then prices lengthen on")
    print("  average for reasons that have nothing to do with information.")
    print()
    df["_or_pre"] = (1.0 / pre).sum(axis=1)
    df["_or_close"] = (1.0 / close).sum(axis=1)
    tbl = df.groupby("season").agg(n=("_or_pre", "size"),
                                   overround_pre=("_or_pre", "mean"),
                                   overround_close=("_or_close", "mean"))
    tbl = tbl[tbl.index >= "2015-16"]
    tbl["tightening"] = tbl["overround_pre"] - tbl["overround_close"]
    print(tbl.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print("  A positive `tightening` column means the close is the sharper,")
    print("  lower-margin price -- so the average selection gets LONGER, and a")
    print("  ratio of taken/close below 1.0 is the default rather than a")
    print("  failure.")

    print()
    print("=" * 78)
    print("2. THE NULL ON PHASE 6's OWN POPULATION")
    print("=" * 78)
    print(f"  Season {PHASE6_SEASON}, all main divisions, every (match, outcome)")
    print("  the pre-registered rule could legally have bet -- price in")
    print(f"  [{RULE.min_odds}, {RULE.max_odds}] with both legs present. No model.")
    print()
    h = df[df["season"] == PHASE6_SEASON]
    p_h = h[pre_cols].to_numpy(float)
    c_h = h[close_cols].to_numpy(float)
    ok = (np.isfinite(p_h) & np.isfinite(c_h) & (c_h > 0)
          & (p_h >= RULE.min_odds) & (p_h <= RULE.max_odds))
    r, c = np.nonzero(ok)
    ratio = p_h[r, c] / c_h[r, c]
    null_pct = float((ratio > 1.0).mean())
    null_mean = float(ratio.mean())
    print(f"    eligible selections   {len(ratio):,}")
    print(f"    NULL % shortened      {null_pct:.4f}")
    print(f"    NULL mean ratio       {null_mean:.4f}")
    print()
    print(f"    phase 6 observed      {PHASE6_OBSERVED_PCT_SHORTENED:.4f} shortened, "
          f"mean ratio {PHASE6_OBSERVED_MEAN_RATIO:.4f}")
    print(f"    margin over the null  "
          f"{PHASE6_OBSERVED_PCT_SHORTENED - null_pct:+.4f} on % shortened, "
          f"{PHASE6_OBSERVED_MEAN_RATIO - null_mean:+.4f} on mean ratio")
    print()
    if PHASE6_OBSERVED_PCT_SHORTENED > null_pct:
        print("    >>> Phase 6's selections sat ABOVE the season's own drift.")
        print("        docs/PHASE6_RESULT.md reads 0.9952 as 'the wrong side of")
        print("        the market's own movement'. Against a correctly specified")
        print("        null that reading does not hold, and the settled study's")
        print("        headline needs re-examining rather than quiet editing.")
    else:
        print("    >>> Phase 6's selections sat BELOW the season's own drift, so")
        print("        its published reading survives the correction.")

    print()
    print("  CAVEAT, and it is not small. This null is unconditional across")
    print("  eligible selections, while phase 6's bets were chosen by a model")
    print("  and carry a different odds mix. For H1 the odds-matched and")
    print("  unmatched nulls agreed to within 0.002 (see")
    print("  scripts/h1_odds_matched_null.py), which suggests the mix does")
    print("  little work here -- but that was measured on a different season")
    print("  and different divisions, so it is a reason to expect agreement")
    print("  rather than a demonstration of it. Phase 6's own bet population")
    print("  would have to be re-derived to settle it, which is a re-analysis")
    print("  of a settled study and belongs in its own pre-registration.")


if __name__ == "__main__":
    main()
