"""Time-respecting splits, and the guards that prove they are time-respecting.

Random k-fold is invalid on match data and it is invalid in three separate
ways, only one of which is obvious:

  1. Temporal   -- team strength is autocorrelated, so a fold that trains on
                   May and tests on the previous November has seen the future
                   state of the same teams.
  2. Feature    -- rolling-form features must be built from matches whose
                   kickoff is strictly before the target kickoff. The classic
                   bug is a per-season groupby-then-shift, which silently
                   includes same-day fixtures.
  3. Market     -- if odds are a feature and you then bet against those odds,
                   the "edge" is a rearrangement of the bookmaker's opinion.

This module handles (1) and gives you the tools to assert (2). (3) is a design
decision made in the feature builder: odds are excluded from features.

A note on the pre-2019/20 era. football-data.co.uk only added a `Time` column
from 2019/20, so before that same-day fixtures cannot be ordered. `purge_days`
exists for exactly this: a match cannot use any fixture from its own matchday
as a feature when the ordering within that day is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

__all__ = ["Split", "season_walk_forward", "rolling_origin", "assert_no_leakage"]


@dataclass(frozen=True)
class Split:
    """One train/test window. Indices are positional into the sorted frame."""
    label: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"Split({self.label}: train n={len(self.train_idx)} to "
                f"{self.train_end:%Y-%m-%d}, test n={len(self.test_idx)} "
                f"{self.test_start:%Y-%m-%d}..{self.test_end:%Y-%m-%d})")


def _kickoff(df: pd.DataFrame, time_col: str) -> pd.Series:
    if time_col not in df.columns:
        raise KeyError(f"frame has no {time_col!r} column")
    k = pd.to_datetime(df[time_col])
    if k.isna().any():
        raise ValueError(f"{int(k.isna().sum())} rows have no {time_col}; drop or fix them "
                         "before splitting -- a NaT cannot be ordered")
    if not k.is_monotonic_increasing:
        raise ValueError(f"frame must be sorted by {time_col} before splitting")
    return k


def season_walk_forward(
    df: pd.DataFrame,
    season_col: str = "season",
    time_col: str = "kickoff",
    min_train_seasons: int = 3,
    purge_days: float = 0.0,
) -> Iterator[Split]:
    """Train on every season before season *k*, test on season *k*.

    The coarse, standard protocol -- what the credible papers in this field
    use. Constantinou (2022) evaluates 13 EPL seasons this way and warns that
    tuning a betting threshold *per season* is fantasy, because the optimal
    value cannot be known before the season starts. Tune on train, never on
    the test season.
    """
    k = _kickoff(df, time_col)
    seasons = df[season_col].to_numpy()
    order = pd.unique(pd.Series(seasons))     # already time-sorted

    for i, season in enumerate(order):
        if i < min_train_seasons:
            continue
        test_mask = seasons == season
        test_start = k[test_mask].min()
        cutoff = test_start - pd.Timedelta(days=purge_days)
        train_mask = (k < cutoff).to_numpy() & ~test_mask
        if not train_mask.any():
            continue
        yield Split(
            label=str(season),
            train_idx=np.flatnonzero(train_mask),
            test_idx=np.flatnonzero(test_mask),
            train_end=k[train_mask].max(),
            test_start=test_start,
            test_end=k[test_mask].max(),
        )


def rolling_origin(
    df: pd.DataFrame,
    time_col: str = "kickoff",
    step_days: int = 7,
    min_train_days: int = 365 * 3,
    start: pd.Timestamp | str | None = None,
    purge_days: float = 0.0,
    expanding: bool = True,
    train_window_days: int | None = None,
) -> Iterator[Split]:
    """Refit every `step_days` and predict the window that follows.

    The fine-grained protocol, and the one that matches how the model would
    actually have been used. `expanding=True` trains on all history; set it
    False with `train_window_days` for a sliding window instead.

    Temporal weighting is worth roughly ten times more than the choice of model
    family on this task (penaltyblog's Eredivisie comparison: 0.0002 spread
    across six distributions, 0.0023 from tuning lookback and decay), so the
    sliding-window option is not a footnote -- it is a lever worth sweeping.
    """
    k = _kickoff(df, time_col)
    first, last = k.iloc[0], k.iloc[-1]

    origin = pd.Timestamp(start) if start is not None else first + pd.Timedelta(days=min_train_days)
    step = pd.Timedelta(days=step_days)
    purge = pd.Timedelta(days=purge_days)

    while origin < last:
        stop = origin + step
        test_mask = ((k >= origin) & (k < stop)).to_numpy()
        if test_mask.any():
            train_mask = (k < origin - purge).to_numpy()
            if train_window_days is not None and not expanding:
                lo = origin - pd.Timedelta(days=train_window_days)
                train_mask &= (k >= lo).to_numpy()
            if train_mask.any():
                yield Split(
                    label=f"{origin:%Y-%m-%d}",
                    train_idx=np.flatnonzero(train_mask),
                    test_idx=np.flatnonzero(test_mask),
                    train_end=k[train_mask].max(),
                    test_start=k[test_mask].min(),
                    test_end=k[test_mask].max(),
                )
        origin = stop


def assert_no_leakage(df: pd.DataFrame, split: Split, time_col: str = "kickoff") -> None:
    """Fail loudly if any training row is not strictly before every test row.

    Cheap, and worth calling on every split of every run. A leak here does not
    announce itself -- it shows up as a model that looks unusually good, which
    is the one result nobody interrogates.
    """
    k = pd.to_datetime(df[time_col])
    train_max = k.iloc[split.train_idx].max()
    test_min = k.iloc[split.test_idx].min()
    if train_max >= test_min:
        raise AssertionError(
            f"leakage in split {split.label!r}: last training kickoff {train_max} "
            f"is not before first test kickoff {test_min}"
        )
    overlap = np.intersect1d(split.train_idx, split.test_idx)
    if overlap.size:
        raise AssertionError(
            f"split {split.label!r} has {overlap.size} rows in both train and test"
        )
