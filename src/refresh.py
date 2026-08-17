"""Refresh the corpus: re-fetch the current season and rebuild matches.parquet.

    uv run python -m src.refresh

A separate entry point from `src.forward` so a scheduled run can refresh the
data, then verify the harness against it, and only then predict. Bundling the
refresh inside the prediction step would mean the self-tests either run against
last week's corpus or do not run at all.

`download_all` cannot do this job -- it skips anything already on disk and
anything memoised in `_missing.json`, with no force flag. See
`footballdata.refresh_current` for what that costs and why the second cache is
the dangerous one.
"""

from __future__ import annotations

import time

import pandas as pd

from src.data.footballdata import (
    MAIN_DIVISIONS, RAW_DIR, build_matches, current_season_code, refresh_current,
)


def main() -> None:
    code = current_season_code()
    season = f"20{code[:2]}-{code[2:]}"
    before = len(list((RAW_DIR / "main").glob(f"{code}_*.csv")))

    t = time.time()
    rep = refresh_current()
    print(f"refresh   {time.time() - t:5.1f}s  downloaded={rep.downloaded} "
          f"cached={rep.cached} missing={rep.missing} errors={len(rep.errors)}")
    if rep.errors:
        raise SystemExit(f"refresh failed on {len(rep.errors)} targets: {rep.errors[:5]}")

    after = len(list((RAW_DIR / "main").glob(f"{code}_*.csv")))
    print(f"          current-season files {before} -> {after} of {len(MAIN_DIVISIONS)}")
    if after < before:
        raise SystemExit(
            f"current-season file count FELL from {before} to {after}. The refresh "
            "deletes before it downloads, so this means fetches failed after the "
            "delete and the corpus is now missing data it had."
        )

    t = time.time()
    df = build_matches(write=True)
    cur = df[df["season"] == season]
    print(f"corpus    {time.time() - t:5.1f}s  {len(df):,} matches total")
    print(f"          {season}: {len(cur):,} matches, {cur['div'].nunique()} divisions, "
          f"latest result {pd.Timestamp(cur['date'].max()).date() if len(cur) else '-'}")


if __name__ == "__main__":
    main()
