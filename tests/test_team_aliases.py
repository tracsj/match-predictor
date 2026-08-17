"""Team-name normalisation and the SportMonks crosswalk.

The two-tier design only works if the same club resolves to the same key in
both sources. A club that fails to join does not error -- it silently drops
its fixtures from the player-level experiment, shrinking the test set in a way
that looks like nothing happened.
"""

import json
import pathlib

import pytest

from src.data.footballdata import normalize_team
from src.data.team_aliases import (
    SPORTMONKS_TO_FOOTBALLDATA, resolve_team, unmatched_report,
)

SM_FIXTURES = pathlib.Path("data/raw/sportmonks/fixtures")


# --------------------------------------------------------------------------
# Non-decomposable letters
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Brøndby", "brondby"),          # o-slash: NOT decomposable by NFKD
    ("København", "kobenhavn"),
    ("Nordsjælland", "nordsjaelland"),   # ae ligature
    ("Sønderjyske", "sonderjyske"),
    ("Beşiktaş", "besiktas"),        # Turkish cedilla
    ("Köln", "koln"),                # decomposable umlaut, the easy case
    ("Borussia M'gladbach", "borussia mgladbach"),
])
def test_non_ascii_letters_survive_normalisation(raw, expected):
    """Regression test for a real bug.

    NFKD decomposes a-umlaut into a plus a combining mark, but o-slash and the
    ae ligature are distinct letters with no decomposition. They survived
    accent-stripping and were then destroyed by the a-z filter, so "Brøndby"
    normalised to "br ndby" and failed to join. Same class of failure for
    German sharp-s, Polish l-stroke and Turkish dotless i.
    """
    assert normalize_team(raw) == expected


def test_club_type_suffixes_are_stripped():
    assert normalize_team("Lyngby Boldklub") == "lyngby"
    assert normalize_team("Vejle Boldklub") == "vejle"
    assert normalize_team("FC Copenhagen") == "copenhagen"


def test_normalisation_still_does_not_merge_different_clubs():
    """The guard on the fix. Folding harder must not start merging clubs that
    are genuinely distinct."""
    assert normalize_team("Man United") != normalize_team("Man City")
    assert normalize_team("Dundee") != normalize_team("Dundee United")
    assert normalize_team("Bristol City") != normalize_team("Bristol Rovers")
    assert normalize_team("Sheffield United") != normalize_team("Sheffield Weds")


# --------------------------------------------------------------------------
# The alias map
# --------------------------------------------------------------------------

def test_aliases_resolve_to_football_data_keys():
    assert resolve_team("AGF") == "aarhus"
    assert resolve_team("FC København") == "copenhagen"
    assert resolve_team("SønderjyskE Fodbold") == "sonderjyske"
    assert resolve_team("Hamilton Academical") == "hamilton"


def test_alias_table_is_stored_already_normalised():
    """Keys and values are both post-normalize_team, so the table only carries
    genuine naming differences. A raw un-normalised entry would never match."""
    for k, v in SPORTMONKS_TO_FOOTBALLDATA.items():
        assert normalize_team(k) == k, f"alias key {k!r} is not normalised"
        assert normalize_team(v) == v, f"alias value {v!r} is not normalised"


def test_alias_map_has_no_cycles_or_self_maps():
    for k, v in SPORTMONKS_TO_FOOTBALLDATA.items():
        assert k != v, f"{k!r} maps to itself"
        assert v not in SPORTMONKS_TO_FOOTBALLDATA, f"{k!r} -> {v!r} -> chains further"


def test_a_name_needing_no_alias_passes_through():
    assert resolve_team("Celtic") == "celtic"
    assert resolve_team("Midtjylland") == "midtjylland"


def test_unmatched_report_finds_both_directions():
    r = unmatched_report(["Celtic", "Rangers", "Made Up FC"], ["celtic", "rangers", "hearts"])
    assert r["matched"] == ["celtic", "rangers"]
    assert r["unmatched_sportmonks"] == ["made up"]
    assert r["unmatched_footballdata"] == ["hearts"]
    assert r["match_rate"] == pytest.approx(2 / 3)


def test_unmatched_report_handles_an_empty_input():
    assert unmatched_report([], ["celtic"])["match_rate"] == 0.0


# --------------------------------------------------------------------------
# Against the real fetch
# --------------------------------------------------------------------------

@pytest.mark.skipif(not SM_FIXTURES.exists() or not any(SM_FIXTURES.glob("*.json")),
                    reason="SportMonks fixtures not fetched")
def test_every_fetched_sportmonks_team_joins_to_football_data():
    """The check that matters, run against whatever has actually been fetched.

    If this fails, a club is silently missing from the player-level tier --
    add it to SPORTMONKS_TO_FOOTBALLDATA rather than loosening the normaliser,
    because loosening risks merges nobody reviewed.
    """
    from src.features.build import load

    by_league = {}
    for f in SM_FIXTURES.glob("*.json"):
        d = json.loads(f.read_text())
        by_league.setdefault(d.get("league_id"), set()).update(
            p.get("name") for p in (d.get("participants") or []))

    fd = load()
    for league, div in ((271, "DNK"), (501, "SC0")):
        if league not in by_league:
            continue
        keys = set(fd[fd["div"] == div].query('season >= "2019-20"')["home_key"])
        r = unmatched_report(by_league[league], keys)
        assert not r["unmatched_sportmonks"], (
            f"league {league}: {r['unmatched_sportmonks']} do not join to "
            f"football-data. Add them to SPORTMONKS_TO_FOOTBALLDATA.")
