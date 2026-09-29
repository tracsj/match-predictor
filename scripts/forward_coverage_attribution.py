"""Why was each missed fixture missed? Attributes the ledger's coverage gaps.

    PYTHONPATH=. uv run python scripts/forward_coverage_attribution.py

A control, not a candidate. It fits nothing and does not move the registry
count -- it audits the forward schedule, not a model.

`docs/FORWARD_LEDGER.md` counts the fixtures no forward run ever predicted, by
kickoff slot. That count says where the misses are and not why, and the why
decides the fix: a fixture the feed never contained needs a different schedule,
one that kicked off before a LATE run started needs a punctual one, and one that
was in a run's window but absent from its file may be a join bug.

For each missed fixture this finds the scheduled slot that should have caught it
(the latest slot before kickoff) and the run that actually served that slot:

    failed run         the run for that slot failed
    unreachable        kickoff within RUN_TIME of the slot's ON-TIME start
    lateness           an on-time run would have caught it; the actual one
                       started after kickoff
    feed window        no fixture on that date is in the run's file, so the
                       feed did not reach it
    absent from file   the date is in the run's file and this fixture is not.
                       Check for a team-key mismatch before blaming the feed

SLOTS is the cron as it stood 2026-08-17 -> 2026-09-29. Runs before 2026-08-27
live in the private archive repo, so for those the prediction file's own
`predicted_at` stands in for the start time. Needs `gh` and a refreshed corpus.
"""

from __future__ import annotations

import json
import subprocess

import pandas as pd

from src.forward import PREDICTIONS_DIR
from src.grade import corpus_in_window

# (weekday, UTC time) -- Tue 13:15 and Fri 17:15, the cron until 2026-09-29.
SLOTS = ((1, "13:15"), (4, "17:15"))
RUN_TIME = pd.Timedelta(minutes=25)
UK = "Europe/London"


def uk(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert(UK).tz_localize(None)


def slot_times(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Series:
    out = []
    for day in pd.date_range(lo.normalize() - pd.Timedelta(days=7), hi, freq="D"):
        for wd, hhmm in SLOTS:
            if day.weekday() == wd:
                out.append(uk(pd.Timestamp(f"{day.date()} {hhmm}", tz="UTC")))
    return pd.Series(sorted(out))


def gh_runs() -> pd.DataFrame:
    raw = subprocess.run(
        ["gh", "run", "list", "--workflow", "forecast.yml", "-L", "200",
         "--json", "createdAt,event,conclusion"],
        capture_output=True, text=True, check=True).stdout
    df = pd.DataFrame(json.loads(raw))
    df = df[df["event"] == "schedule"].copy()
    df["start"] = df["createdAt"].map(uk)
    return df


def main() -> None:
    files = {}
    for p in sorted(PREDICTIONS_DIR.glob("*.csv")):
        d = pd.read_csv(p)
        d["kickoff"] = pd.to_datetime(d["kickoff"])
        d["predicted_at"] = pd.to_datetime(d["predicted_at"])
        files[p.name] = d
    allp = pd.concat(files.values(), ignore_index=True)
    divs = sorted(allp["div"].dropna().unique())
    lo = allp["kickoff"].min()
    hi = uk(pd.Timestamp.now(tz="UTC")).normalize() - pd.Timedelta(seconds=1)

    scope = corpus_in_window(divs, lo, hi)
    scope["kickoff"] = pd.to_datetime(scope["kickoff"])
    missed = scope[~scope["match_id"].isin(set(allp["match_id"]))]
    print(f"scope {lo} -> {hi}: {len(scope):,} fixtures in {len(divs)} divisions, "
          f"{len(missed)} never predicted\n")

    # Slots run to NOW, not to `hi`: a run serving the latest slot must map to
    # that slot, not fall back onto the one before it.
    slots = slot_times(lo, uk(pd.Timestamp.now(tz="UTC")))
    runs = gh_runs()

    def served_by(slot: pd.Timestamp):
        """(start, conclusion, file) of the run that served this slot."""
        later = slots[slots > slot]
        nxt = later.min() if len(later) else pd.Timestamp.max
        r = runs[(runs["start"] >= slot) & (runs["start"] < nxt)]
        f = [n for n, d in files.items()
             if slot <= d["predicted_at"].min() < nxt]
        fname = f[0] if f else None
        if len(r):
            first = r.sort_values("start").iloc[0]
            return first["start"], first["conclusion"], fname
        if fname:
            return files[fname]["predicted_at"].min(), "success", fname
        return None, "unknown", None

    rows = []
    for _, m in missed.iterrows():
        slot = slots[slots < m["kickoff"]].max()
        start, conclusion, fname = served_by(slot)
        if conclusion == "failure":
            why = "failed run"
        elif m["kickoff"] <= slot + RUN_TIME:
            why = "unreachable"
        elif start is not None and m["kickoff"] <= start:
            why = "lateness"
        elif fname and (files[fname]["kickoff"].dt.date == m["kickoff"].date()).any():
            why = "absent from file"
        elif conclusion == "unknown":
            why = "no run found"
        else:
            why = "feed window"
        rows.append({"match_id": m["match_id"], "kickoff": m["kickoff"],
                     "slot": slot, "run_start": start, "why": why})
    r = pd.DataFrame(rows)
    if r.empty:
        print("nothing missed")
        return

    print(r["why"].value_counts().to_string(), "\n")
    r["kick_slot"] = (r["kickoff"].dt.day_name().str[:3] + " "
                      + r["kickoff"].dt.hour.astype(str).str.zfill(2))
    print(pd.crosstab(r["kick_slot"], r["why"]).to_string(), "\n")

    delay = (runs.assign(slot=runs["start"].map(lambda t: slots[slots <= t].max()))
                 .assign(late=lambda d: d["start"] - d["slot"]))
    print("start delay per scheduled run")
    print(delay[["slot", "start", "late", "conclusion"]]
          .sort_values("slot").to_string(index=False), "\n")

    print("misses not explained by lateness or a failed run")
    print(r[~r["why"].isin(["lateness", "failed run"])]
          .sort_values("kickoff")[["match_id", "kickoff", "run_start", "why"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
