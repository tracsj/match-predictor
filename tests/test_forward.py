"""Tests for the forward path: unplayed fixtures through the feature builders.

The property that matters is the same one `test_ratings.py` asserts for history,
extended one step: appending a match that has NOT been played must change
nothing about the matches that have. That is checked bit-for-bit rather than
inferred, because every failure mode here is silent.

The naive implementation — append the row, let the NaN score flow through —
passes a casual eye and fails these. It records a phantom 0-0 defeat for both
sides in rolling form and sequences, and writes NaN into the stored pi-rating.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.footballdata import OUT_DIR, current_season_code, normalize_team
from src.data.fixtures import season_of
from src.features.horizon import UNPLAYED_COL, unplayed_flags
from src.features.ratings import add_ratings, elo_features, pi_rating_features
from src.features.rolling import rolling_features
from src.features.sequences import SEQ_FEATURES, build_sequences

PARQUET = OUT_DIR / "matches.parquet"
needs_data = pytest.mark.skipif(not PARQUET.exists(), reason="matches.parquet not built")


def frame(rows) -> pd.DataFrame:
    """rows = [(home, away, hg, ag)]. A None score marks an unplayed fixture."""
    df = pd.DataFrame(rows, columns=["home_key", "away_key", "fthg", "ftag"])
    df["country"] = "X"
    df["div"] = "E0"
    df["season"] = "2020-21"
    df["kickoff"] = pd.date_range("2020-08-01", periods=len(df), freq="7D")
    df["fthg"] = pd.to_numeric(df["fthg"], errors="coerce").astype(float)
    df["ftag"] = pd.to_numeric(df["ftag"], errors="coerce").astype(float)
    df[UNPLAYED_COL] = df["fthg"].isna()
    df["result"] = pd.Series(pd.NA, index=df.index, dtype="string").mask(
        df["fthg"] > df["ftag"], "H").mask(df["fthg"] == df["ftag"], "D").mask(
        df["fthg"] < df["ftag"], "A")
    return df


PLAYED = [("a", "b", 2, 0), ("c", "d", 1, 1), ("a", "c", 0, 3), ("b", "d", 2, 2)]


# --------------------------------------------------------------------------
# The horizon must not reach backwards
# --------------------------------------------------------------------------

def test_appending_an_unplayed_fixture_changes_no_completed_row():
    """The forward analogue of test_appending_a_later_match_cannot_change_an
    _earlier_rating.

    Worth being precise about what this does and does not catch, because it is
    tempting to read it as the test that guards the whole design. It is not.
    Measured against a deliberately un-guarded build, this assertion still
    passes: the horizon sorts after every played row, so absorbing it corrupts
    only later rows, and there are none. The tests below are the ones that go
    red on the real bug.

    It is kept because it locks the ordering property itself — a future change
    that sorted the frame differently, or absorbed history backwards, would
    break here and nowhere else.
    """
    short = frame(PLAYED)
    long = frame(PLAYED + [("a", "d", None, None), ("b", "c", None, None)])

    for fn in (elo_features, pi_rating_features):
        s, ll = fn(short), fn(long)
        for col in s.columns:
            assert np.allclose(s[col], ll[col][: len(s)], equal_nan=True), col

    s = rolling_features(add_ratings(short))
    ll = rolling_features(add_ratings(long))
    for col in s.columns:
        assert np.allclose(s[col], ll[col][: len(s)], equal_nan=True), col

    s_seq, s_mask = build_sequences(add_ratings(short))
    l_seq, l_mask = build_sequences(add_ratings(long))
    assert np.array_equal(s_seq, l_seq[: len(short)])
    assert np.array_equal(s_mask, l_mask[: len(short)])


def test_elo_does_not_raise_on_a_fixture_with_no_score():
    """int(nan) is the loud failure. Without the guard this is a ValueError."""
    e = elo_features(frame(PLAYED + [("a", "d", None, None)]))
    assert np.isfinite(e["elo_home"].iloc[-1])
    assert np.isfinite(e["elo_away"].iloc[-1])


# --------------------------------------------------------------------------
# The quiet failures: state must not absorb a non-result
# --------------------------------------------------------------------------

def test_an_unplayed_fixture_does_not_move_the_pi_rating():
    """The naive version writes NaN into the stored rating, so every later row
    for either team comes out NaN."""
    rows = PLAYED + [("a", "d", None, None), ("a", "b", None, None)]
    p = pi_rating_features(frame(rows))
    # Team a's home rating going into its second unplayed fixture must equal
    # the one it had going into the first: nothing happened in between.
    assert p["pi_home_h"].iloc[-1] == pytest.approx(p["pi_home_h"].iloc[-2])
    assert np.isfinite(p[["pi_home_h", "pi_away_a", "pi_exp_gd"]].to_numpy()).all()


def test_an_unplayed_fixture_records_no_phantom_defeat():
    """`3.0 if gd > 0 else (1.0 if gd == 0 else 0.0)` takes the last branch on
    NaN, so the naive version credits both sides with a 0-0 loss."""
    base = frame(PLAYED + [("a", "b", None, None)])
    r = rolling_features(add_ratings(base))
    # 'a' has played twice in PLAYED; its played-count must stay at 2 for a
    # later row, and its points average must be unchanged by the fixture.
    played_before = r["h_played"].iloc[2]          # a's second appearance
    rows = PLAYED + [("a", "b", None, None), ("a", "c", None, None)]
    r2 = rolling_features(add_ratings(frame(rows)))
    assert r2["h_played"].iloc[-1] == r2["h_played"].iloc[-2]
    assert r2["h_pts_5"].iloc[-1] == pytest.approx(r2["h_pts_5"].iloc[-2], nan_ok=True)
    assert played_before == r["h_played"].iloc[2]


def test_two_unplayed_fixtures_for_one_team_are_independent():
    """A horizon routinely holds a midweek and a weekend fixture for one club.
    The first must not feed the second -- the within-horizon contamination case,
    which is the default rather than an edge case."""
    rows = PLAYED + [("a", "d", None, None), ("d", "a", None, None)]
    seq, mask = build_sequences(add_ratings(frame(rows)))

    # 'a' played exactly twice in PLAYED, so both unplayed rows must see two
    # past matches -- not three, which is what absorbing the first would give.
    assert mask[-2, 0].sum() == 2
    assert mask[-1, 1].sum() == 2

    # 'a' is home in the first unplayed row and away in the second, so its
    # history sits at side 0 then side 1. Every past-match feature must agree
    # EXCEPT days_ago, which is measured from each fixture's own kickoff and is
    # legitimately seven days apart.
    days_ago = SEQ_FEATURES.index("days_ago")
    keep = [i for i in range(len(SEQ_FEATURES)) if i != days_ago]
    assert np.array_equal(seq[-2, 0][:, keep], seq[-1, 1][:, keep])


def test_unplayed_rest_days_do_not_shift_between_horizon_fixtures():
    """last_played must not be stamped by an unplayed row, or the second
    fixture's rest_days is measured from a match that never happened."""
    rows = PLAYED + [("a", "d", None, None), ("a", "c", None, None)]
    r = rolling_features(add_ratings(frame(rows)))
    # Both read from a's last PLAYED match, seven and fourteen days earlier.
    assert r["h_rest_days"].iloc[-1] > r["h_rest_days"].iloc[-2]
    assert r["h_rest_days"].iloc[-1] == pytest.approx(r["h_rest_days"].iloc[-2] + 7.0)


# --------------------------------------------------------------------------
# The flag itself
# --------------------------------------------------------------------------

def test_flag_absent_means_every_row_is_played():
    df = frame(PLAYED).drop(columns=[UNPLAYED_COL])
    assert not unplayed_flags(df).any()


def test_a_flag_that_contradicts_the_data_raises():
    """The flag is explicit rather than derived from fthg.isna() so that a
    genuinely missing historical score keeps failing loudly. If the two ever
    disagree, one of them is wrong and neither should be guessed at."""
    df = frame(PLAYED)
    df.loc[0, UNPLAYED_COL] = True          # row 0 has a result
    with pytest.raises(ValueError, match="flagged unplayed yet carry a result"):
        unplayed_flags(df)


# --------------------------------------------------------------------------
# Season labelling and the join key
# --------------------------------------------------------------------------

@pytest.mark.parametrize("date,expected", [
    ("2026-07-31", "2026-27"),      # the cached 2026/27 SC0 file opens here
    ("2026-08-15", "2026-27"),
    ("2026-12-31", "2026-27"),
    ("2027-01-01", "2026-27"),
    ("2027-05-31", "2026-27"),
    ("2026-06-30", "2025-26"),
])
def test_season_of_puts_july_in_the_new_season(date, expected):
    assert season_of(pd.Timestamp(date)) == expected


def test_current_season_code_matches_season_of():
    assert current_season_code(pd.Timestamp("2026-08-17")) == "2627"
    assert current_season_code(pd.Timestamp("2026-06-30")) == "2526"


def test_no_refresh_still_fetches_the_fixtures_feed(monkeypatch):
    """CI regression. The workflow passes --no-refresh because src.refresh has
    already pulled the corpus, but src.refresh does NOT fetch fixtures.csv. Tying
    both to one flag left a cold runner with no horizon file at all.

    A live run must always fetch the feed; only a back-dated replay may use the
    cached copy.
    """
    import src.forward as fwd

    calls = {}

    def fake_build_fixtures(refresh, now=None):
        calls["refresh"] = refresh
        return pd.DataFrame()          # empty horizon ends run() early

    monkeypatch.setattr(fwd, "build_fixtures", fake_build_fixtures)

    fwd.run(refresh=False, verbose=False)
    assert calls["refresh"] is True, "a live run must fetch the fixtures feed"

    fwd.run(refresh=False, verbose=False, as_of="2026-08-14 12:00")
    assert calls["refresh"] is False, "a back-dated replay must use the cache"


def test_saving_the_missing_manifest_creates_its_directory(tmp_path, monkeypatch):
    """Cold-start regression. refresh_current() writes _missing.json before
    download_all() creates the tree, and data/ is gitignored -- so on every CI
    run this raised FileNotFoundError before a single request went out. Found by
    a real runner, not locally, because the directory always exists here."""
    import src.data.footballdata as fd

    monkeypatch.setattr(fd, "RAW_DIR", tmp_path / "does" / "not" / "exist")
    fd._save_missing({"main/2627/E1"})
    assert (tmp_path / "does" / "not" / "exist" / "_missing.json").exists()
    assert fd._load_missing() == {"main/2627/E1"}


@needs_data
def test_fixture_match_id_matches_the_corpus_formula():
    """The join key is what makes a committed prediction gradeable later. A
    mismatch orphans every prediction silently, so it is asserted against a
    real corpus row rather than against another copy of the same expression.

    The expected value is written out independently here, on purpose: deriving
    it from the code under test would keep this green for every wrong answer.
    """
    corpus = pd.read_parquet(PARQUET, columns=["match_id", "div", "date",
                                              "home_raw", "away_raw"])
    row = corpus.iloc[-1]
    d = pd.Timestamp(row["date"])
    expected = (f"{row['div']}|{d.year:04d}{d.month:02d}{d.day:02d}"
                f"|{normalize_team(row['home_raw'])}|{normalize_team(row['away_raw'])}")
    assert row["match_id"] == expected
