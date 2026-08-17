"""Grade committed forward predictions as results land.

    uv run python -m src.grade

Reads every file under `predictions/`, joins them to the corpus on `match_id`,
and reports CLV first and ROI second, per the standing rule. Rewrites
`docs/FORWARD_LEDGER.md` from scratch each run, so the ledger is always
reproducible from committed data rather than accumulated by appending — an
appended ledger cannot be checked against anything.

**The check that makes this mean something.** A prediction file is evidence only
if it existed before the matches it predicts. So each file's git commit time is
read, and any file committed at or after the kickoff of any fixture inside it is
refused rather than graded. Without that, this is a backtest with extra steps
and a more convincing directory name.

The commit time used is the LAST commit touching the file, not the first. A
second run on the same day appends rows, and those rows are committed later than
the file was created; taking the latest possible commit time is the conservative
reading, and it under-claims rather than over-claims.

**The benchmark changed, and it is NOT softer.** The settled study graded against
Pinnacle closing; football-data dropped Pinnacle in 2026/27, so forward CLV is
measured against the Betfair Exchange close. Measured on the 16,875 matches that
carry both (2026-08-17):

    book              mean overround   de-vigged RPS
    pinnacle close        1.0389          0.20408
    exchange close        1.0089          0.20404

The exchange close is an equally accurate estimate of the truth on a quarter of
the margin, and its prices run 3.9% longer. So CLV against it is at least as
demanding as CLV against Pinnacle — there is less vig to hide in. A positive
result here would be stronger evidence than the same result against Pinnacle,
not weaker, and the ledger should not apologise for the change.

Exchange ROI, separately, IS flattered: it is PRE-COMMISSION, and commission of
roughly 2-5% of net winnings would absorb most of that 3.9% price advantage. CLV
is immune, because both legs are exchange prices and commission cancels in the
ratio. Hence CLV first, as always.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from src.data.footballdata import OUT_DIR, REPO_ROOT
from src.eval.betting import (
    B365_CLOSE, CLOSE_FOR_EXCHANGE, EXCHANGE_CLOSE, EXCHANGE_PRE, MARKET_AVG_CLOSE,
    MARKET_MAX_CLOSE, bootstrap_ci, clv_report, closing_price_for_bets,
    required_sample_size, simulate, summarize,
)
from src.eval.devig import devig
from src.eval.metrics import summary
from src.forward import PREDICTIONS_DIR
from src.phase6 import RULE

LEDGER = REPO_ROOT / "docs" / "FORWARD_LEDGER.md"

CLOSE_SETS = [
    (EXCHANGE_CLOSE, "the sharpest price still in the feed"),
    (B365_CLOSE, "a book you could hold an account with"),
    (MARKET_MAX_CLOSE, "optimistic bound"),
    (MARKET_AVG_CLOSE, "softer benchmark, for coverage"),
]


def file_committed_at(path: Path) -> pd.Timestamp | None:
    """Commit time of the last commit touching `path`, or None if uncommitted."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", str(path)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    # Compare in UK local, which is what the kickoff column is in.
    return pd.Timestamp(out).tz_convert("Europe/London").tz_localize(None)


def load_predictions(verbose: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """Every committed prediction, one row per fixture, earliest prediction kept.

    Returns the usable rows plus a per-file provenance report.
    """
    frames, provenance = [], []
    for p in sorted(PREDICTIONS_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df["kickoff"] = pd.to_datetime(df["kickoff"])
        committed = file_committed_at(p)
        latest_kick = df["kickoff"].max()
        if committed is None:
            status = "uncommitted"
        elif committed >= df["kickoff"].min():
            status = "committed too late"
        else:
            status = "ok"
        provenance.append({"file": p.name, "rows": len(df),
                           "committed_at": committed, "first_kickoff": df["kickoff"].min(),
                           "last_kickoff": latest_kick, "status": status})
        if status == "ok":
            df["source_file"] = p.name
            frames.append(df)
        elif verbose:
            print(f"  ! {p.name}: {status} -- {len(df)} predictions not graded")

    if not frames:
        return pd.DataFrame(), provenance
    allp = pd.concat(frames, ignore_index=True)
    # A fixture should appear once, but keep the earliest if it ever does not.
    allp = allp.sort_values("predicted_at").drop_duplicates("match_id", keep="first")
    return allp.reset_index(drop=True), provenance


def join_results(preds: pd.DataFrame) -> pd.DataFrame:
    """Attach the landed result and every closing price to each prediction."""
    corpus = pd.read_parquet(OUT_DIR / "matches.parquet")
    close_cols = [c for s in CLOSE_SETS for c in s[0].cols if c in corpus.columns]
    keep = ["match_id", "result", "fthg", "ftag"] + sorted(set(close_cols))
    graded = preds.merge(corpus[keep], on="match_id", how="left", suffixes=("", "_corpus"))
    return graded


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def build_report(verbose: bool = True) -> str:
    """Build the ledger text. Returns it; `main` writes it."""
    preds, provenance = load_predictions(verbose=verbose)
    out: list[str] = []
    w = out.append

    w("# Forward ledger")
    w("")
    w("Predictions committed before kickoff, graded as results landed. Rewritten from")
    w("`predictions/*.csv` on every run, so nothing here is accumulated by hand.")
    w("")
    w("**Read the CLV column first.** Distinguishing a 2% edge from zero needs roughly")
    w(f"{required_sample_size(3.2, 0.02):,} bets at average odds 3.2; CLV converges about a hundred times")
    w("faster and is what correctly said stop in the backtest.")
    w("")
    w("**The benchmark changed, and it is not softer.** The settled study graded against")
    w("Pinnacle closing, which left the feed in 2026/27. The Betfair Exchange close")
    w("replaces it, and on the 16,875 matches carrying both it is an equally accurate")
    w("estimate of the truth — de-vigged RPS 0.20404 against Pinnacle's 0.20408 — on a")
    w("quarter of the margin, 1.0089 against 1.0389. Its prices run 3.9% longer, so")
    w("beating it is if anything harder. CLV below is a like-for-like exchange ratio.")
    w("")
    w("**Exchange ROI below is pre-commission.** 2–5% of net winnings is not deducted,")
    w("and that would absorb most of the price advantage. CLV is immune to it, since")
    w("both legs are exchange prices and the commission cancels in the ratio.")
    w("")

    if not provenance:
        w("_No prediction files yet._")
        return "\n".join(out) + "\n"

    prov = pd.DataFrame(provenance)
    w("## Provenance")
    w("")
    w("Each file's commit time against the earliest kickoff it predicts. A file")
    w("committed at or after any of its own kickoffs is not graded at all.")
    w("")
    w("```")
    w(_fmt(prov[["file", "rows", "committed_at", "first_kickoff", "status"]]))
    w("```")
    w("")

    if preds.empty:
        w("_No prediction file has passed the commit-before-kickoff check yet._")
        return "\n".join(out) + "\n"

    graded = join_results(preds)
    settled = graded[graded["result"].notna()].reset_index(drop=True)

    w("## Coverage")
    w("")
    w(f"- Predictions committed: **{len(graded):,}**")
    w(f"- Results landed: **{len(settled):,}**")
    w(f"- Awaiting result: **{len(graded) - len(settled):,}**")
    if len(graded):
        w(f"- Divisions: **{graded['div'].nunique()}**, "
          f"kickoffs {graded['kickoff'].min()} → {graded['kickoff'].max()}")
    w("")

    if settled.empty:
        w("_Nothing has settled yet. The tables below appear once results land._")
        return "\n".join(out) + "\n"

    p = settled[["p_home", "p_draw", "p_away"]].to_numpy(float)
    y = settled["result"].to_numpy()

    w("## Forecast quality")
    w("")
    rows = [summary(p, y, "the net (forward)")]
    mkt_mask = settled[EXCHANGE_CLOSE.cols].notna().all(axis=1).to_numpy()
    if mkt_mask.sum() >= 20:
        mkt = devig(settled.loc[mkt_mask, EXCHANGE_CLOSE.cols].to_numpy(float), method="shin")
        rows.append(summary(mkt, y[mkt_mask], f"market, exchange close (n={mkt_mask.sum()})"))
        rows.append(summary(p[mkt_mask], y[mkt_mask], "the net, same subset"))
    w("```")
    w(_fmt(pd.DataFrame(rows)))
    w("```")
    w("")
    w("The market band to sanity-check against is RPS 0.19–0.21. Outside it, suspect")
    w("the pipeline before the model.")
    w("")

    w("## Closing-line value")
    w("")
    w("Bet at the pre-close exchange price recorded at prediction time; grade against")
    w("the exchange close of the same selection. A ratio at or below 1.0 means the")
    w("selections sat on the wrong side of the market's own movement.")
    w("")
    clv_rows = []
    need = EXCHANGE_PRE.cols + EXCHANGE_CLOSE.cols
    if all(c in settled.columns for c in need):
        m = settled[need].notna().all(axis=1).to_numpy()
        if m.sum() >= 10:
            sub = settled[m].reset_index(drop=True)
            bets = simulate(sub, p[m], EXCHANGE_PRE, RULE)
            if len(bets):
                r = clv_report(bets, closing_price_for_bets(bets, sub, CLOSE_FOR_EXCHANGE))
                clv_rows.append({"taken_at": EXCHANGE_PRE.label, "n_bets": len(bets),
                                 "mean_ratio": r["mean_ratio"],
                                 "pct_shortened": r["pct_shortened"],
                                 "binom_p": r["binom_pvalue"]})
            else:
                clv_rows.append({"taken_at": EXCHANGE_PRE.label, "n_bets": 0})
    w("```")
    w(_fmt(pd.DataFrame(clv_rows)) if clv_rows
      else "  no usable pre-close/close pairs yet")
    w("```")
    w("")

    w("## ROI, led by the sharpest price")
    w("")
    roi_rows = []
    for ps, note in CLOSE_SETS:
        if not all(c in settled.columns for c in ps.cols):
            continue
        m = settled[ps.cols].notna().all(axis=1).to_numpy()
        if m.sum() < 10:
            roi_rows.append({"price_set": ps.label, "note": note,
                             "n_eligible": int(m.sum()), "n_bets": 0})
            continue
        sub = settled[m].reset_index(drop=True)
        bets = simulate(sub, p[m], ps, RULE)
        row = summarize(bets, with_ci=False)
        row["note"] = note
        row["n_eligible"] = int(m.sum())
        if len(bets) >= 20:
            ci = bootstrap_ci(bets, n_boot=2000, seed=0)
            row["roi_lo"], row["roi_hi"] = ci["lo"], ci["hi"]
        roi_rows.append(row)
    cols = ["price_set", "n_eligible", "n_bets", "roi", "roi_lo", "roi_hi",
            "hit_rate", "avg_odds", "note"]
    tbl = pd.DataFrame(roi_rows)
    w("```")
    w(_fmt(tbl[[c for c in cols if c in tbl.columns]]))
    w("```")
    w("")
    w(f"Rule: {RULE.describe()} — fixed by `docs/PREREGISTRATION.md`.")
    w("")
    w("A result positive only in the market-maximum column is price shopping rather")
    w("than forecasting, and is the strategy that got Kaunitz et al. stake-limited")
    w("into uselessness.")
    w("")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = ap.parse_args()
    text = build_report()
    if args.stdout:
        print(text)
        return
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(text)
    print(f"wrote {LEDGER.relative_to(REPO_ROOT)} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
