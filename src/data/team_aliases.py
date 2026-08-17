"""Explicit crosswalk between SportMonks team names and football-data's.

Two different problems live here and only one of them is solvable by string
folding.

**Folding problems** are handled in `footballdata.normalize_team`: accents,
non-decomposable letters (o-slash, ae, sharp-s, l-stroke, dotless i), club
suffixes. That got the Danish Superliga join from 11/18 to 15/18.

**Alias problems** are not string problems at all. "AGF" and "Aarhus" are the
same club under two naming conventions, and no normaliser turns one into the
other without also merging things that should stay apart. Those need a human
decision, recorded here, with the reasoning attached.

The map is deliberately small and deliberately explicit. A fuzzy matcher would
close these three cases and silently invent others -- and a wrong merge here
does not fail loudly, it quietly averages two clubs' histories together.
"""

from __future__ import annotations

from src.data.footballdata import normalize_team

__all__ = ["SPORTMONKS_TO_FOOTBALLDATA", "resolve_team", "unmatched_report"]


# Keys and values are BOTH post-normalize_team, so this table only has to
# carry genuine naming differences rather than punctuation variants.
SPORTMONKS_TO_FOOTBALLDATA: dict[str, str] = {
    # Denmark. AGF is Aarhus Gymnastikforening; football-data lists the city.
    "agf": "aarhus",
    # FC Kobenhavn is FC Copenhagen -- Danish vs English name for the club.
    "kobenhavn": "copenhagen",
    # "Fodbold" is Danish for football; a club-type suffix like "Boldklub",
    # but rare enough that stripping it globally risks more than it fixes.
    "sonderjyske fodbold": "sonderjyske",

    # Scotland. Hamilton Academical FC; football-data uses the short form.
    "hamilton academical": "hamilton",
}


def resolve_team(name: str) -> str:
    """SportMonks name -> football-data key. Normalises first, then aliases."""
    key = normalize_team(name)
    return SPORTMONKS_TO_FOOTBALLDATA.get(key, key)


def unmatched_report(sportmonks_names, footballdata_keys) -> dict:
    """Which teams still fail to join, in both directions.

    Called after every SportMonks ingest. An unmatched team is not a cosmetic
    problem: it silently drops that club's fixtures from the player-level
    experiment, which shrinks the test set in a way that looks like nothing
    at all.
    """
    resolved = {resolve_team(n) for n in sportmonks_names}
    fd = set(footballdata_keys)
    return {
        "matched": sorted(resolved & fd),
        "unmatched_sportmonks": sorted(resolved - fd),
        "unmatched_footballdata": sorted(fd - resolved),
        "match_rate": len(resolved & fd) / len(resolved) if resolved else 0.0,
    }
