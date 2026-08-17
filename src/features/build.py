"""Build and cache the full feature table.

    uv run python -m src.features.build

Ratings take under a second; rolling form takes about a minute over the whole
296k-match corpus. Rebuilding that on every experiment is the kind of friction
that quietly reduces how many experiments get run, so it is cached.

Features are built over the WHOLE corpus in chronological order and only then
filtered. Every row still sees only matches strictly before it, so this is not
leakage -- but filtering first would throw away the history that gives a team
in 2016 any form at all.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.footballdata import OUT_DIR
from src.features.ratings import add_ratings
from src.features.rolling import add_rolling
from src.features.sequences import SeqParams, build_sequences

FEATURES_PARQUET = OUT_DIR / "features.parquet"
SEQ_NPY = OUT_DIR / "sequences.npy"
SEQ_MASK_NPY = OUT_DIR / "sequences_mask.npy"

# Everything the models may draw on. Membership here does not mean a model
# uses it -- see RATING_FEATURES and NET_FEATURES in src.models.
RATING_COLS = [
    "elo_home", "elo_away", "elo_diff", "elo_exp_home",
    "elo_home_moved", "elo_away_moved",
    "pi_home_h", "pi_home_a", "pi_away_h", "pi_away_a", "pi_exp_gd",
]


def build_frame(df: pd.DataFrame, verbose: bool = True) -> tuple:
    """Run the whole feature pass over one chronological frame.

    Shared by the cached corpus build and the forward build, so a horizon of
    upcoming fixtures goes through exactly the same code as history rather than
    a parallel implementation that could drift from it.
    """
    df = df.sort_values("kickoff").reset_index(drop=True)

    t = time.time()
    df = add_ratings(df)
    if verbose:
        print(f"  ratings   {time.time() - t:5.1f}s")

    t = time.time()
    df = add_rolling(df)
    if verbose:
        print(f"  rolling   {time.time() - t:5.1f}s")

    # Row position in the full chronological corpus, so the sequence arrays
    # (which are row-aligned with this frame) can be indexed after filtering.
    df["corpus_row"] = np.arange(len(df), dtype=np.int64)

    t = time.time()
    seq, mask = build_sequences(df)
    if verbose:
        print(f"  sequences {time.time() - t:5.1f}s  shape {seq.shape}")
    return df, seq, mask


def build(force: bool = False) -> pd.DataFrame:
    if FEATURES_PARQUET.exists() and not force:
        return pd.read_parquet(FEATURES_PARQUET)

    df = pd.read_parquet(OUT_DIR / "matches.parquet")
    df, seq, mask = build_frame(df)

    FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PARQUET, index=False)
    np.save(SEQ_NPY, seq)
    np.save(SEQ_MASK_NPY, mask)
    return df


def build_forward(fixtures: pd.DataFrame | None = None, verbose: bool = True) -> tuple:
    """Features for the corpus PLUS a horizon of unplayed fixtures.

    Not cached. The horizon changes every run, and a stale forward feature
    table is worse than no table -- it would predict last week's fixtures while
    looking entirely healthy.

    The unplayed rows are scored from history and absorbed into nothing (see
    src/features/horizon.py), and they sort after every played row because
    their kickoffs are in the future. Together those two facts mean appending a
    horizon cannot change any completed row's features, which
    tests/test_forward.py asserts bit-for-bit rather than trusting.
    """
    from src.data.fixtures import FIXTURES_PARQUET, load_fixtures
    from src.features.horizon import UNPLAYED_COL

    if fixtures is None:
        fixtures = (pd.read_parquet(FIXTURES_PARQUET)
                    if FIXTURES_PARQUET.exists() else load_fixtures())

    corpus = pd.read_parquet(OUT_DIR / "matches.parquet")
    corpus[UNPLAYED_COL] = False

    # The feed retains fixtures that have already been played, so a horizon row
    # can collide with a corpus row for the same match. The kickoff filter in
    # load_fixtures normally prevents it; this catches the case where a result
    # has landed since the snapshot was taken. A duplicate would enter history
    # twice -- once as a real match and once as a fixture absorbed into nothing.
    dupes = fixtures["match_id"].isin(corpus["match_id"])
    if dupes.any():
        if verbose:
            print(f"  dropping {int(dupes.sum())} horizon fixtures already in the corpus")
        fixtures = fixtures[~dupes].copy()

    combined = pd.concat([corpus, fixtures], ignore_index=True)
    combined[UNPLAYED_COL] = combined[UNPLAYED_COL].fillna(False).astype(bool)
    # Concatenating an int16 score column onto a nullable one lands in object
    # dtype, where `int(value)` meets pd.NA. Float carries the missing scores
    # as NaN, which every builder already expects.
    for c in ("fthg", "ftag"):
        combined[c] = pd.to_numeric(combined[c], errors="coerce").astype("float64")

    if verbose:
        print(f"  corpus {len(corpus):,} + horizon {len(fixtures)} matches")
    return build_frame(combined, verbose=verbose)


def load_sequences() -> tuple:
    """The (n, 2, L, F) sequence tensor and its mask, aligned row-for-row with
    the features parquet. Built in the same chronological pass, so a row's
    sequence contains only matches strictly before its own kickoff."""
    import numpy as _np
    if not SEQ_NPY.exists():
        build(force=True)
    return _np.load(SEQ_NPY), _np.load(SEQ_MASK_NPY)


def load(force_rebuild: bool = False) -> pd.DataFrame:
    return build(force=force_rebuild)


if __name__ == "__main__":
    df = build(force=True)
    print(f"wrote {FEATURES_PARQUET} -- {len(df):,} matches, {len(df.columns)} columns")
