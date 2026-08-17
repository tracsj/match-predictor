"""The `unplayed` flag, and why the feature builders need it.

A forward prediction needs a feature row for a match that has not been played.
Every builder in this package is already leak-free by construction -- each
scores the current row from history and only then absorbs that row's result --
so producing features for a future fixture is, in principle, just a matter of
appending it to the frame and running the same forward pass.

**What goes wrong if you do only that.** The appended row has no score, so
`fthg`/`ftag` are NaN, and the absorb step runs anyway on NaN inputs:

  - `elo_features` RAISES, at `int(nan)`.
  - `pi_rating_features` writes NaN into the team's stored rating, so every
    later row involving that team gets NaN pi features.
  - `rolling_features` computes `gd = nan`, and `3.0 if gd > 0 else (1.0 if
    gd == 0 else 0.0)` takes the final branch on NaN -- recording a phantom
    0-0 DEFEAT. It also stamps `last_played`, corrupting `rest_days`.
  - `build_sequences` appends the same fabricated loss, then `np.nan_to_num`
    erases the evidence that the goals were never known.

The three silent failures matter more than the crash. A prediction horizon
routinely holds a midweek and a weekend fixture for the same club, so a team's
second fixture in a horizon would read its own invented first result. That is
not an edge case, it is the default.

**The contract.** A row flagged `unplayed` is SCORED from history exactly as
any other row, and is never ABSORBED into it. Nothing is written back: no
rating update, no history append, no `last_played` stamp, no league aggregate.
The flag is explicit rather than derived from `fthg.isna()` so that a genuinely
missing score in historical data keeps failing loudly instead of being
silently treated as a fixture.

Because unplayed rows sort after every played row, this also guarantees that
appending a horizon cannot change any completed row's features -- the property
`tests/test_forward.py` asserts bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["UNPLAYED_COL", "unplayed_flags"]

UNPLAYED_COL = "unplayed"


def unplayed_flags(df: pd.DataFrame) -> np.ndarray:
    """Boolean array over `df`, True where the row is a fixture with no result.

    Absent column means an all-played frame, which is what every caller
    predating the forward path passes.
    """
    n = len(df)
    if UNPLAYED_COL not in df.columns:
        return np.zeros(n, dtype=bool)
    flags = df[UNPLAYED_COL].fillna(False).to_numpy(dtype=bool)
    scored = df["result"].notna().to_numpy() if "result" in df.columns else np.zeros(n, bool)
    both = flags & scored
    if both.any():
        raise ValueError(
            f"{int(both.sum())} rows are flagged {UNPLAYED_COL} yet carry a result; "
            "the flag and the data disagree and one of them is wrong"
        )
    return flags
