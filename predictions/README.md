# predictions/

One CSV per run day, written by `uv run python -m src.forward` and committed by
`.github/workflows/forecast.yml`. Graded by `uv run python -m src.grade`, which
rewrites `docs/FORWARD_LEDGER.md` from whatever is in here.

**These files are the evidence, and the commit timestamp is the part that
matters.** Every other number in this project is backtested — computed over
history by someone who could already see it. A file here says: this was the
forecast, at this price, before the match, and a machine that is not ours
recorded when.

That property is fragile in exactly three ways, so:

1. **Never edit or rewrite a committed file.** A corrected prediction is not a
   prediction. If one is wrong, the honest move is to say so in the ledger and
   leave the file alone.
2. **Never backfill.** `src/forward.py --as-of` replays a resolved window for
   testing and deliberately writes nothing at all. There is no supported way to
   produce a file here for a match that has already kicked off, and that is not
   an oversight.
3. **Never re-predict a fixture.** A match already present in any file here is
   skipped by later runs. Predicting it again closer to kickoff, with a model
   trained on more data and a price nearer the close, would quietly convert this
   directory back into a backtest.

`src/grade.py` enforces (2) independently rather than trusting any of this: a
file whose last commit does not precede every kickoff inside it is reported and
excluded, never averaged in.

## Columns

| column | |
|---|---|
| `predicted_at` | UK wall-clock time the run started |
| `match_id` | `div\|YYYYMMDD\|home_key\|away_key`, identical to the corpus formula — this is the join |
| `kickoff` | UK local, matching the feed and the corpus |
| `p_home` `p_draw` `p_away` | temperature-scaled, three seeds averaged |
| `bfeh` `bfed` `bfea` | Betfair Exchange pre-close — the sharpest price available at prediction time, and the one CLV is measured from |
| `b365h` … `avga` | Bet365, market maximum and market average pre-close |

Prices are recorded because CLV needs the price that could actually have been
taken. Without them these files could not answer the only question they exist
to answer.

## Where the first three files came from

`2026-08-18.csv`, `2026-08-21.csv` and `2026-08-25.csv` were written and
committed by scheduled runs in **`tracsj/match-predictor-archive`** — the
private repository this one was published from on 2026-08-27. Publishing
rewrote the history, so their commit SHAs here are new; they were carried
across with `git cherry-pick` and their original committer dates preserved,
which is what `src/grade.py` reads. The runs that produced them are
`32145154725`, `32509613714` and `32856896845`, and the archive is the copy
where a runner recorded those timestamps rather than a laptop. Every file
after `2026-08-25.csv` was committed by this repository's own workflow.
