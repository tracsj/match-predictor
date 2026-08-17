"""Tests for the SportMonks fixture parser.

The load-bearing one is home/away resolution. v1's defining bug was assuming
the first-listed team was home -- measured, that held for only 407 of 640
fixtures (63.6%), and roughly a third of its simulated bets were graded
against the wrong side's price. This parser reads `meta.location` and skips
the fixture rather than guessing.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.data.sportmonks_parse import (
    COUNT_STATS, MATCHES_PARQUET, PLAYERS_PARQUET, RATE_STATS, STAT_BY_ID,
    STAT_DEFS, _parse_fixture,
)

HAS_DATA = MATCHES_PARQUET.exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="SportMonks not parsed yet")


def fixture(home_first=True, home_goals=2, away_goals=1, with_meta=True,
            with_score=True, lineups=None):
    home = {"id": 100, "name": "Celtic", "meta": {"location": "home"} if with_meta else {}}
    away = {"id": 200, "name": "Rangers", "meta": {"location": "away"} if with_meta else {}}
    parts = [home, away] if home_first else [away, home]
    scores = []
    if with_score:
        scores = [
            {"description": "CURRENT", "score": {"participant": "home", "goals": home_goals}},
            {"description": "CURRENT", "score": {"participant": "away", "goals": away_goals}},
        ]
    return {
        "id": 1, "league_id": 501, "season_id": 21787,
        "starting_at": "2024-01-02 15:00:00",
        "participants": parts, "scores": scores,
        "lineups": lineups if lineups is not None else [],
    }


# --------------------------------------------------------------------------
# Home / away
# --------------------------------------------------------------------------

def test_home_away_come_from_meta_location_not_list_order():
    """v1's exact bug, asserted directly. Reversing the participant order must
    not change which team is home."""
    a, _ = _parse_fixture(fixture(home_first=True))
    b, _ = _parse_fixture(fixture(home_first=False))
    assert a["home_key"] == b["home_key"] == "celtic"
    assert a["away_key"] == b["away_key"] == "rangers"
    assert a["fthg"] == b["fthg"] == 2


def test_a_fixture_without_location_metadata_is_skipped_not_guessed():
    assert _parse_fixture(fixture(with_meta=False)) is None


def test_players_are_flagged_home_or_away_by_team_id():
    lineups = [
        {"player_id": 1, "team_id": 100, "type_id": 11, "formation_field": "2:1",
         "details": [{"type_id": 119, "data": {"value": 90}}]},
        {"player_id": 2, "team_id": 200, "type_id": 11, "formation_field": "4:2",
         "details": [{"type_id": 119, "data": {"value": 90}}]},
    ]
    _, players = _parse_fixture(fixture(lineups=lineups))
    by_id = {p["player_id"]: p for p in players}
    assert by_id[1]["is_home"] == 1
    assert by_id[2]["is_home"] == 0


# --------------------------------------------------------------------------
# Result and score
# --------------------------------------------------------------------------

@pytest.mark.parametrize("hg,ag,expected", [(2, 1, "H"), (1, 1, "D"), (0, 3, "A")])
def test_result_follows_the_goals(hg, ag, expected):
    m, _ = _parse_fixture(fixture(home_goals=hg, away_goals=ag))
    assert m["result"] == expected


def test_a_fixture_with_no_score_is_skipped():
    """These are real: 19 of the fetched fixtures are the COVID-abandoned
    2019/20 Scottish season and were never played."""
    assert _parse_fixture(fixture(with_score=False)) is None


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def test_stat_ids_are_unique_and_named():
    ids = [tid for _, tid, _ in STAT_DEFS]
    assert len(ids) == len(set(ids)), "duplicate stat type id"
    names = [n for n, _, _ in STAT_DEFS]
    assert len(names) == len(set(names)), "duplicate stat name"
    assert len(STAT_DEFS) == 37
    assert set(COUNT_STATS) | set(RATE_STATS) == set(names)


def test_stats_are_read_from_the_nested_value():
    lineups = [{
        "player_id": 1, "team_id": 100, "type_id": 11, "formation_field": "4:1",
        "details": [
            {"type_id": 119, "data": {"value": 78}},     # minutes
            {"type_id": 52, "data": {"value": 2}},       # goals
            {"type_id": 118, "data": {"value": 8.4}},    # rating
        ],
    }]
    _, players = _parse_fixture(fixture(lineups=lineups))
    p = players[0]
    assert p["minutes"] == 78
    assert p["goals"] == 2
    assert p["rating"] == pytest.approx(8.4)


def test_a_string_valued_stat_is_coerced():
    lineups = [{"player_id": 1, "team_id": 100, "type_id": 11, "formation_field": "3:2",
                "details": [{"type_id": 1584, "data": {"value": "87.5"}}]}]
    _, players = _parse_fixture(fixture(lineups=lineups))
    assert players[0]["accurate_passes_pct"] == pytest.approx(87.5)


def test_an_unparseable_stat_becomes_nan_not_zero():
    """Zero would read as 'happened zero times', which is a different and
    false claim from 'not recorded'."""
    lineups = [{"player_id": 1, "team_id": 100, "type_id": 11, "formation_field": "3:2",
                "details": [{"type_id": 52, "data": {"value": None}}]}]
    _, players = _parse_fixture(fixture(lineups=lineups))
    assert np.isnan(players[0]["goals"])


def test_position_group_comes_from_the_formation_field():
    """The leading digit encodes the line: 1 keeper, 2 defence, 3 midfield,
    4 attack."""
    lineups = [
        {"player_id": i, "team_id": 100, "type_id": 11, "formation_field": ff, "details": []}
        for i, ff in enumerate(["1:1", "2:3", "3:2", "4:1"])
    ]
    _, players = _parse_fixture(fixture(lineups=lineups))
    assert [p["position_group"] for p in players] == [1, 2, 3, 4]


def test_a_missing_formation_field_does_not_crash():
    lineups = [{"player_id": 1, "team_id": 100, "type_id": 12,
                "formation_field": None, "details": []}]
    _, players = _parse_fixture(fixture(lineups=lineups))
    assert players[0]["position_group"] == 0
    assert players[0]["is_starter"] == 0


def test_starters_and_bench_are_distinguished():
    lineups = [
        {"player_id": 1, "team_id": 100, "type_id": 11, "formation_field": "3:1", "details": []},
        {"player_id": 2, "team_id": 100, "type_id": 12, "formation_field": None, "details": []},
    ]
    _, players = _parse_fixture(fixture(lineups=lineups))
    assert players[0]["is_starter"] == 1
    assert players[1]["is_starter"] == 0


# --------------------------------------------------------------------------
# Against the real parse
# --------------------------------------------------------------------------

@needs_data
def test_parsed_corpus_is_sane():
    m = pd.read_parquet(MATCHES_PARQUET)
    p = pd.read_parquet(PLAYERS_PARQUET)
    assert len(m) > 1500
    assert m["sm_fixture_id"].is_unique
    assert m["kickoff"].notna().all()
    assert set(m["result"]) <= {"H", "D", "A"}
    assert set(m["div"]) <= {"DNK", "SC0"}
    # Roughly 36-40 lineup entries per fixture (starters plus bench, both sides)
    assert 30 < len(p) / len(m) < 50


@needs_data
def test_home_win_rate_is_football_like():
    """A crude but independent check that home and away were not transposed
    somewhere in the parse."""
    m = pd.read_parquet(MATCHES_PARQUET)
    rates = m["result"].value_counts(normalize=True)
    assert 0.38 < rates["H"] < 0.52
    assert rates["H"] > rates["A"]


@needs_data
def test_every_match_joins_to_football_data_by_date_and_teams():
    """The join the whole player-level experiment depends on.

    A handful of late-season Danish playoff fixtures exist in SportMonks and
    not in football-data, so this allows a small shortfall rather than
    demanding perfection -- but a large drop means the alias map or the date
    handling has broken.
    """
    from src.features.build import load
    sm = pd.read_parquet(MATCHES_PARQUET)
    fd = load()
    fd = fd[fd["div"].isin(["DNK", "SC0"])]

    def key(d):
        return (d["div"].astype(str) + "|"
                + pd.to_datetime(d["kickoff"]).dt.strftime("%Y%m%d") + "|"
                + d["home_key"].astype(str) + "|" + d["away_key"].astype(str))

    rate = key(sm).isin(set(key(fd))).mean()
    assert rate > 0.98, f"only {rate:.1%} of SportMonks matches join to football-data"


@needs_data
def test_no_kickoff_late_enough_for_a_timezone_date_rollover():
    """SportMonks reports UTC and football-data reports local time, a +1/+2h
    offset for these leagues. The join is by DATE, which is only safe while no
    fixture sits near midnight UTC. Measured: the latest is 20:00."""
    m = pd.read_parquet(MATCHES_PARQUET)
    hours = pd.to_datetime(m["kickoff"]).dt.hour
    assert hours.max() <= 21, "a late kickoff could roll the local date"
