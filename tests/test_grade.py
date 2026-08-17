"""Tests for grading committed forward predictions.

The load-bearing test here is not the arithmetic — that is the same `simulate`
and `clv_report` the settled study used, already covered by `test_clv.py`. It is
the provenance check: a prediction file is evidence only if it existed before the
matches it predicts, and a ledger that grades a file committed afterwards is a
backtest with a more convincing directory name.
"""

import pandas as pd
import pytest

import src.grade as grade
from src.data.footballdata import OUT_DIR

PARQUET = OUT_DIR / "matches.parquet"
needs_data = pytest.mark.skipif(not PARQUET.exists(), reason="matches.parquet not built")


def write_predictions(dirpath, rows: pd.DataFrame) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "2026-01-01.csv").write_text(rows.to_csv(index=False))


def fake_rows(match_ids, kickoff="2026-02-01 15:00") -> pd.DataFrame:
    n = len(match_ids)
    return pd.DataFrame({
        "predicted_at": "2026-01-01T09:00:00",
        "match_id": match_ids,
        "kickoff": kickoff,
        "div": "E0", "league": "England Premier League", "season": "2025-26",
        "home_raw": "H", "away_raw": "A", "home_key": "h", "away_key": "a",
        "p_home": [0.5] * n, "p_draw": [0.25] * n, "p_away": [0.25] * n,
        "bfeh": [2.5] * n, "bfed": [3.5] * n, "bfea": [4.0] * n,
        "b365h": [2.4] * n, "b365d": [3.4] * n, "b365a": [3.8] * n,
        "maxh": [2.6] * n, "maxd": [3.6] * n, "maxa": [4.1] * n,
        "avgh": [2.45] * n, "avgd": [3.45] * n, "avga": [3.9] * n,
    })


@pytest.fixture
def predictions_dir(tmp_path, monkeypatch):
    d = tmp_path / "predictions"
    monkeypatch.setattr(grade, "PREDICTIONS_DIR", d)
    return d


def test_a_shallow_clone_is_refused_not_silently_ungraded(predictions_dir, monkeypatch):
    """A shallow clone makes git report no commit for any file, which would read
    as 'uncommitted' for every prediction and grade nothing while exiting zero.
    The CI default (`actions/checkout` fetch-depth 1) produces exactly that, so
    it has to be detected rather than documented."""
    predictions_dir.mkdir(parents=True)

    class Result:
        stdout = "true\n"

    monkeypatch.setattr(grade.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(SystemExit, match="shallow clone"):
        grade.build_report(verbose=False)


def test_no_prediction_files_is_reported_not_crashed(predictions_dir):
    predictions_dir.mkdir(parents=True)
    text = grade.build_report(verbose=False)
    assert "No prediction files yet" in text


def test_a_file_committed_after_kickoff_is_not_graded(predictions_dir, monkeypatch):
    """The whole point. A prediction committed after the match is not a
    prediction, and must be excluded rather than quietly averaged in."""
    write_predictions(predictions_dir, fake_rows(["E0|20260201|h|a"]))
    monkeypatch.setattr(grade, "file_committed_at",
                        lambda p: pd.Timestamp("2026-02-01 18:00"))   # after kickoff
    text = grade.build_report(verbose=False)
    assert "committed too late" in text
    assert "has passed the commit-before-kickoff check" in text


def test_an_uncommitted_file_is_not_graded(predictions_dir, monkeypatch):
    """A file on disk but not in git carries no timestamp anybody else can
    check, so it is not evidence either."""
    write_predictions(predictions_dir, fake_rows(["E0|20260201|h|a"]))
    monkeypatch.setattr(grade, "file_committed_at", lambda p: None)
    text = grade.build_report(verbose=False)
    assert "uncommitted" in text


def test_a_file_committed_before_kickoff_is_graded(predictions_dir, monkeypatch):
    write_predictions(predictions_dir, fake_rows(["E0|20260201|h|a"]))
    monkeypatch.setattr(grade, "file_committed_at",
                        lambda p: pd.Timestamp("2026-01-01 09:00"))
    text = grade.build_report(verbose=False)
    # Check the file's own provenance ROW, not the whole document -- the preamble
    # legitimately discusses both failure statuses, so a document-wide string
    # search passes or fails for the wrong reasons.
    row = next(ln for ln in text.splitlines() if "2026-01-01.csv" in ln)
    assert row.rstrip().endswith("ok"), row
    # A made-up match_id has no result, so the row is counted and awaits one.
    assert "Predictions committed: **1**" in text
    assert "Awaiting result: **1**" in text


def test_the_ledger_states_the_benchmark_change(predictions_dir, monkeypatch):
    """The exchange close replaced Pinnacle, and the measured comparison says it
    is not a downgrade. The ledger must carry that rather than implying the
    forward numbers are weaker than the backtest's."""
    predictions_dir.mkdir(parents=True)
    text = grade.build_report(verbose=False)
    assert "not softer" in text
    assert "pre-commission" in text


@needs_data
def test_a_real_settled_match_joins_and_grades(predictions_dir, monkeypatch):
    """End to end against the corpus: a prediction for a match that really has a
    result must reach the settled tables rather than sitting in 'awaiting'."""
    corpus = pd.read_parquet(PARQUET, columns=["match_id", "result", "kickoff",
                                              "bfech", "bfecd", "bfeca"])
    have = corpus[corpus["result"].notna()
                  & corpus[["bfech", "bfecd", "bfeca"]].notna().all(axis=1)]
    if have.empty:
        pytest.skip("no settled match with an exchange close in the corpus")
    row = have.iloc[-1]
    kick = pd.Timestamp(row["kickoff"])

    rows = fake_rows([row["match_id"]], kickoff=kick.strftime("%Y-%m-%d %H:%M"))
    write_predictions(predictions_dir, rows)
    monkeypatch.setattr(grade, "file_committed_at", lambda p: kick - pd.Timedelta(days=1))

    text = grade.build_report(verbose=False)
    assert "Results landed: **1**" in text
    assert "Awaiting result: **0**" in text
