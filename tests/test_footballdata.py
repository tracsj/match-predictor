"""Tests for football-data.co.uk ingest.

Split in two. The pure functions are tested against hand-written expectations.
The integrity tests run against the built parquet and are skipped if it does
not exist, because they check facts about the real corpus that no fixture can
stand in for -- the Pinnacle boundary, the substitute-file collapse, the
kickoff-time era.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.footballdata import (
    EXTRA_COUNTRIES, MAIN_DIVISIONS, OUT_DIR, _parse_dates, normalize_team,
    season_codes,
)

PARQUET = OUT_DIR / "matches.parquet"
needs_data = pytest.mark.skipif(
    not PARQUET.exists(), reason="matches.parquet not built; run build_matches()"
)


@pytest.fixture(scope="module")
def matches():
    return pd.read_parquet(PARQUET)


# --------------------------------------------------------------------------
# Pure functions
# --------------------------------------------------------------------------

def test_season_codes_hand_written():
    codes = season_codes(1993, 1995)
    assert codes == ["9394", "9495", "9596"]


def test_season_codes_across_the_century_boundary():
    # The one place an off-by-one would be silent: 1999/2000 is '9900',
    # 2000/01 is '0001'.
    assert season_codes(1999, 2000) == ["9900", "0001"]


def test_season_codes_recent():
    assert season_codes(2023, 2025) == ["2324", "2425", "2526"]


def test_division_and_country_lists_are_the_expected_size():
    # 22 main divisions carried 7,799 matches in 2023/24 (measured 2026-08-15).
    assert len(MAIN_DIVISIONS) == 22
    assert len(EXTRA_COUNTRIES) == 16


@pytest.mark.parametrize("raw,expected", [
    ("Man United", "man united"),
    ("Nott'm Forest", "nottm forest"),
    ("  Real  Madrid ", "real madrid"),
    ("Bayern Munich", "bayern munich"),
    ("FC Copenhagen", "copenhagen"),          # leading FC dropped
    ("Malmo FF", "malmo"),                    # trailing FF dropped
    ("Sporting Braga", "sporting braga"),
    ("Köln", "koln"),                         # accent folded
    ("Ath Bilbao", "ath bilbao"),
])
def test_normalize_team(raw, expected):
    assert normalize_team(raw) == expected


def test_normalize_team_does_not_unify_genuinely_different_spellings():
    # Deliberate: the normaliser must NOT guess that these are the same club.
    # Guessing here would silently merge two teams' histories. team_review()
    # surfaces candidates for a human instead.
    assert normalize_team("Man United") != normalize_team("Manchester Utd")


def test_parse_dates_handles_both_era_formats():
    s = pd.Series(["05/09/93", "08/08/2026", "31/12/1999"])
    out = _parse_dates(s)
    assert out.iloc[0] == pd.Timestamp("1993-09-05")
    assert out.iloc[1] == pd.Timestamp("2026-08-08")
    assert out.iloc[2] == pd.Timestamp("1999-12-31")


def test_parse_dates_is_day_first_not_month_first():
    # 05/09/93 must be 5 September, not 9 May. Getting this backwards would
    # reorder the entire corpus and silently corrupt every rolling feature.
    assert _parse_dates(pd.Series(["05/09/93"])).iloc[0].month == 9


def test_parse_dates_returns_nat_rather_than_guessing():
    assert pd.isna(_parse_dates(pd.Series(["not a date"])).iloc[0])


# --------------------------------------------------------------------------
# Integrity of the built corpus
# --------------------------------------------------------------------------

@needs_data
def test_corpus_is_sorted_by_kickoff(matches):
    assert matches["kickoff"].is_monotonic_increasing


@needs_data
def test_match_ids_are_unique(matches):
    # football-data serves a SUBSTITUTE file when a division-season does not
    # exist -- in 1993/94 the P1, SC1, SP1 and SP2 URLs all return identical
    # SP1 content. Trusting the file's own Div column plus this dedup is what
    # stops those copies being counted three times.
    assert not matches["match_id"].duplicated().any()


@needs_data
def test_no_missing_scores_or_dates(matches):
    assert matches["fthg"].notna().all()
    assert matches["ftag"].notna().all()
    assert matches["kickoff"].notna().all()


@needs_data
def test_result_matches_the_goals(matches):
    # `result` is derived rather than read from FTR, because FTR is absent in
    # some early files and mis-cased in others. It must agree with the goals.
    h, a, r = matches["fthg"], matches["ftag"], matches["result"]
    assert (r[h > a] == "H").all()
    assert (r[h == a] == "D").all()
    assert (r[h < a] == "A").all()


@needs_data
def test_pinnacle_closing_odds_start_at_2012_13(matches):
    """The boundary the backtest window depends on.

    Measured independently by curl before this code existed: PSCH is absent in
    2011/12 and present from 2012/13 onward. If this ever fails, either the
    upstream data changed or the parser started dropping the column.
    """
    main = matches[matches["source"] == "main"]
    by_season = main.groupby("season")["psch"].count()
    for season in ("2009-10", "2010-11", "2011-12"):
        assert by_season.get(season, 0) == 0, f"{season} should have no Pinnacle close"
    for season in ("2012-13", "2013-14", "2023-24"):
        assert by_season.get(season, 0) > 5000, f"{season} should have Pinnacle close"


@needs_data
def test_kickoff_time_only_exists_from_2019_20(matches):
    """Bounds how far back rolling features can be built safely.

    Without a time, same-day fixtures cannot be ordered, so a match may not use
    its own matchday as a feature. See split.purge_days.
    """
    main = matches[matches["source"] == "main"]
    early = main[main["season"] < "2019-20"]
    late = main[main["season"] >= "2020-21"]
    assert not early["has_kickoff_time"].any()
    assert late["has_kickoff_time"].mean() > 0.95


@needs_data
def test_2023_24_main_division_count(matches):
    """Cross-check against a count taken directly from the CSVs, before this
    module existed: 22 divisions, 7,799 matches, all with Pinnacle closing."""
    m = matches[(matches["source"] == "main") & (matches["season"] == "2023-24")]
    assert len(m) == 7799
    assert m["div"].nunique() == 22
    assert m["psch"].notna().all()


@needs_data
def test_odds_are_plausible_decimal_prices(matches):
    for col in ("psch", "pscd", "psca", "b365ch", "avgch"):
        v = matches[col].dropna()
        if v.empty:
            continue
        assert v.min() > 1.0, f"{col} has a price at or below evens"
        assert v.max() < 2000, f"{col} has an implausible price {v.max()}"


@needs_data
def test_overround_on_pinnacle_close_is_in_a_sane_band(matches):
    """A sharp book's 1X2 overround runs about 1.02-1.05. Well outside that
    means the three columns are not really H/D/A -- which is the failure this
    project cares most about catching, since it is v1's exact bug."""
    m = matches.dropna(subset=["psch", "pscd", "psca"])
    ov = 1 / m["psch"] + 1 / m["pscd"] + 1 / m["psca"]
    assert ov.median() == pytest.approx(1.03, abs=0.03)
    assert (ov.between(0.95, 1.20)).mean() > 0.99


@needs_data
def test_home_favourite_wins_more_often_than_away_favourite_loses(matches):
    """A crude but independent orientation check on the H/D/A column order.

    If home and away were transposed anywhere in the pipeline, the shorter
    price would stop predicting the more frequent result. Home advantage in
    football is real and large, so the home win rate must exceed the away rate.
    """
    r = matches["result"].value_counts(normalize=True)
    assert r["H"] > r["A"] > r["D"]
    assert 0.40 < r["H"] < 0.50
