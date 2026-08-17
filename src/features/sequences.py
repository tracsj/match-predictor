"""Per-team match sequences for the recurrent branch.

The rolling features in `rolling.py` are means over a window. A mean throws
away order: a team that lost four then won six looks identical to one that won
six then lost four. A sequence model sees the difference, and the Kaggle
Football Match Probability Prediction competition -- 150k matches, 860 leagues,
each row carrying both teams' previous 10 matches, odds excluded -- is the
best public evidence that the ordering carries signal.

**Built in one forward chronological pass, deliberately.** The obvious
implementation is a per-team groupby then `.tail(10)` per match, and it is a
leak waiting to happen: any grouping that sees the whole frame can pick up
matches after the target kickoff, and the bug is invisible because the output
looks perfectly reasonable. Here each match reads the deque as it stands and
only then appends itself, exactly as the ratings do.

Shape: `(n_matches, 2, seq_len, n_features)` with side 0 = home, side 1 = away,
plus a `(n_matches, 2, seq_len)` mask that is False where a team has not yet
played enough matches. Steps are ordered oldest to newest, so the recurrent
final state corresponds to the most recent match.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.horizon import unplayed_flags

__all__ = ["SeqParams", "build_sequences", "SEQ_FEATURES"]

# Per past match, from the perspective of the team whose sequence it is.
SEQ_FEATURES = (
    "gf",           # goals scored
    "ga",           # goals conceded
    "gd",           # goal difference
    "points",       # 3 / 1 / 0
    "was_home",
    "opp_elo_z",    # opponent strength, centred and scaled
    "sot_f",        # shots on target for  (NaN before 2000/01 and in extra files)
    "sot_a",
    "days_ago",     # recency of this past match, in weeks, capped
)


@dataclass(frozen=True)
class SeqParams:
    seq_len: int = 10
    max_days_ago: float = 120.0
    elo_centre: float = 1500.0
    elo_scale: float = 200.0


def build_sequences(df: pd.DataFrame, params: SeqParams = SeqParams()
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return (sequences, mask).

    `df` must be sorted by kickoff and carry elo_home / elo_away.
    """
    if not df["kickoff"].is_monotonic_increasing:
        raise ValueError("df must be sorted by kickoff before building sequences")
    for need in ("elo_home", "elo_away"):
        if need not in df.columns:
            raise KeyError(f"sequences need {need!r}; call add_ratings first")

    L, F = params.seq_len, len(SEQ_FEATURES)
    n = len(df)
    seqs = np.zeros((n, 2, L, F), dtype=np.float32)
    mask = np.zeros((n, 2, L), dtype=bool)

    hist: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=L))

    def col(name):
        return df[name].to_numpy() if name in df.columns else np.full(n, np.nan)

    country = col("country")
    hk, ak = col("home_key"), col("away_key")
    gh, ga_ = col("fthg"), col("ftag")
    hst, ast = col("hst"), col("ast")
    eh, ea = col("elo_home"), col("elo_away")
    kick = pd.to_datetime(df["kickoff"]).to_numpy()
    unplayed = unplayed_flags(df)

    for i in range(n):
        now = kick[i]
        for side, key in ((0, (country[i], hk[i])), (1, (country[i], ak[i]))):
            past = hist[key]
            # Oldest first, so the final recurrent state is the most recent
            # match. Left-pad by writing into the tail of the window.
            offset = L - len(past)
            for j, rec in enumerate(past):
                days = (now - rec["when"]) / np.timedelta64(1, "D")
                seqs[i, side, offset + j] = (
                    rec["gf"], rec["ga"], rec["gf"] - rec["ga"], rec["points"],
                    rec["was_home"], rec["opp_elo_z"],
                    rec["sot_f"], rec["sot_a"],
                    min(float(days), params.max_days_ago) / 7.0,
                )
                mask[i, side, offset + j] = True

        # A fixture with no result appends nothing to either team's deque. The
        # naive version records a fabricated 0-0 loss and then nan_to_num wipes
        # the trace of it -- see src/features/horizon.py.
        if unplayed[i]:
            continue

        gd = gh[i] - ga_[i]
        z = lambda e: (float(e) - params.elo_centre) / params.elo_scale
        hist[(country[i], hk[i])].append({
            "when": now, "gf": gh[i], "ga": ga_[i],
            "points": 3.0 if gd > 0 else (1.0 if gd == 0 else 0.0),
            "was_home": 1.0, "opp_elo_z": z(ea[i]),
            "sot_f": hst[i], "sot_a": ast[i],
        })
        hist[(country[i], ak[i])].append({
            "when": now, "gf": ga_[i], "ga": gh[i],
            "points": 3.0 if gd < 0 else (1.0 if gd == 0 else 0.0),
            "was_home": 0.0, "opp_elo_z": z(eh[i]),
            "sot_f": ast[i], "sot_a": hst[i],
        })

    # Shots are missing before 2000/01 and absent from the extra-country files.
    # Zero is the neutral value once the mask is applied; a NaN would poison
    # every gradient that touched it.
    np.nan_to_num(seqs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return seqs, mask
