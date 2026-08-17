"""Cheap pre-flight for src.h1 -- everything except the expensive fit.

    uv run python scripts/h1_panel_check.py

The walk-forward fit costs 30-75 minutes, so any plumbing bug that only
surfaces downstream of it is paid for twice. This exercises the panel, the
split ordering, and the whole CLV/ROI reporting path against RANDOM
probabilities, which cost nothing.

Random probabilities are a plumbing check and nothing else. The numbers it
prints are noise by construction and must not be read as a result, which is
why it prints no verdict.
"""

from __future__ import annotations

import numpy as np

from src.eval.betting import PINNACLE_CLOSE, PINNACLE_PRE
from src.eval.split import season_walk_forward
from src.h1 import LOWER_DIVS, build_panel, clv_for


def main() -> None:
    panel, full, seq_all = build_panel()
    print(f"panel      {len(panel):,} rows, {panel['season'].nunique()} seasons "
          f"({panel['season'].min()} -> {panel['season'].max()}), "
          f"{panel['div'].nunique()} divisions")
    print(f"corpus     {len(full):,} rows for training")
    print(f"sequences  {seq_all.shape}")

    assert "match_id" in panel.columns, "closing_price_for_bets indexes on match_id"
    dupes = int(panel["match_id"].duplicated().sum())
    print(f"match_id   unique: {dupes == 0} ({dupes} duplicates)")
    assert dupes == 0, "a duplicated match_id would make the CLV close lookup ambiguous"

    legs = PINNACLE_PRE.cols + PINNACLE_CLOSE.cols
    assert panel[legs].notna().all().all(), "panel carries a row missing a price leg"
    print(f"price legs complete on all {len(panel):,} rows: True")

    splits = list(season_walk_forward(panel, min_train_seasons=3))
    test_idx = np.concatenate([s.test_idx for s in splits])
    graded = panel.iloc[test_idx].reset_index(drop=True)
    print(f"graded     {len(graded):,} rows across {len(splits)} test seasons "
          f"({graded['season'].min()} -> {graded['season'].max()})")
    print(f"           seasons: {list(pd.unique(graded['season']))}")

    is_lower = graded["div"].isin(LOWER_DIVS).to_numpy()
    print(f"lower      {is_lower.sum():,} rows   upper {(~is_lower).sum():,} rows")

    # Random probabilities: exercises simulate -> closing_price_for_bets ->
    # clv_report end to end. The output is noise and is labelled as such.
    rng = np.random.default_rng(0)
    p = rng.dirichlet((4.0, 3.0, 3.0), size=len(graded))
    for name, mask in (("lower", is_lower), ("upper", ~is_lower)):
        rep, ratio = clv_for(graded[mask], p[mask])
        print(f"  [noise] {name}: {rep['n']:,} bets, mean ratio "
              f"{rep.get('mean_ratio', float('nan')):.4f}, "
              f"ratios returned {len(ratio):,}")

    print()
    print("Plumbing OK. The ratios above are random-probability noise, not a")
    print("result -- they exist only to prove the path runs end to end.")


if __name__ == "__main__":
    import pandas as pd  # noqa: E402  (used in the seasons print above)
    main()
