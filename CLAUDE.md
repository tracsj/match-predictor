# match-predictor — root context

## What this is

A standing programme hunting for exploitable inefficiency in sports betting markets, and a harness honest enough that a positive result would be believable. It began as a football 1X2 predictor; the model is finished and the question has moved on.

**The finished result, so nobody re-derives it.** The network reaches RPS 0.20765 against de-vigged Pinnacle closing odds at 0.20291 — beating every non-neural baseline (t = +2.50), losing to the market (t = +20). The pre-registered betting rule lost in every price column with closing-line value at **0.9952**, meaning its selections sat on the wrong side of the market's own movement. Filtering by model confidence does not rescue it: the deficit is uniform across confidence buckets.

**This is the normal outcome, not a defect in the pipeline.** Wilkens (2021) ran 15 ML architectures on ATP tennis and beat no odds-implied forecast; Kovalchik (2016) found the bookmaker consensus beat 11 published models; Data Golf publishes −0.92% ROI for its own model. **Do not treat 0.2076 as a near-miss to be closed with better features.** It is the empirical ceiling for a pre-match model in a liquid top-tier market. New work goes into finding a *different* market, not a better feature.

## Where the detail lives

None of these is `@`-imported — an imported spoke is still always-loaded and saves nothing. Read one when its rule is about to bind.

| file | what is in it |
|---|---|
| `docs/PROGRAMME.md` | the hypothesis registry — status, graveyard, running count of everything ever tested |
| `docs/hypotheses/*.md` | one file per hypothesis: pre-registration inline, result when settled |
| `docs/PREREGISTRATION.md` | the football betting rule, prices and holdout, fixed before any PnL existed |
| `docs/PHASE6_RESULT.md` | the betting answer, the CLV table, and why the model loses more than random betting |
| `docs/TIER2_RESULT.md` | what a starting XI is worth (nothing measurable), and the SportMonks upgrade recommendation |
| `docs/research/00-measured-facts.md` | what each data source actually contains, with the command that established it |
| `docs/research/01-neural-nets-for-match-prediction.md` | what wins on this task, with RPS numbers tagged to their datasets |
| `docs/research/02-betting-evaluation-and-odds-data.md` | how to build a backtest that would tell you the truth if the model were bad |

## Always

- **Pre-register before running a betting rule.** Threshold, prices, holdout and model config committed *before* any PnL is computed. Searching several rules and reporting the best is how a backtest manufactures an edge — one sweep here already produced a tempting +2.4% cell that was noise on 296 bets.
- **Increment the registry count for every configuration evaluated**, including the ones that die quietly mid-session. The count is the only thing that keeps a widening search honest.
- **Report CLV before ROI.** Distinguishing a 2% edge from zero needs ~45,000 bets; CLV converges roughly a hundred times faster and is what correctly said stop.
- **Lead the sharpest price.** Three columns always: Pinnacle close, one book you could actually hold an account with, market maximum. A result positive only in the third is an odds-comparison screen.
- **Build every feature in one forward chronological pass**, reading history before appending the current match. A per-entity `groupby` then `.tail(n)` looks reasonable and silently includes the future.
- **Verify a data claim by probing it.** Every fact in `docs/research/00-measured-facts.md` carries the command that produced it, so the next session can re-run rather than trust.
- **State where a winning strategy could be staked.** Kaunitz et al. made real money and were limited to $1.25 stakes within months. An edge you cannot get filled on is a hobby.

## Never

- **Never infer home and away from list order.** v1 assumed the first-listed team was home; measured across 640 fixtures it held 63.6% of the time, so a third of its simulated bets were graded against the wrong side's price. Read the explicit location field, or skip the fixture.
- **Never fill a missing statistic with zero without checking the feed covers it.** SportMonks omits a detail row when a statistic is not collected. Filling those with zero turned "not measured" into "did it zero times" and made `touches` read as 10.9 per 90 against 38.8 passes — physically impossible, and I blamed the vendor before finding my own bug.
- **Never trust a football-data filename over the file's own `Div` column.** The server returns a *substitute* file for a division-season that does not exist: the 1993/94 P1, SC1, SP1 and SP2 URLs are byte-identical Spanish La Liga.
- **Never write `df.div`** — it resolves to pandas' division method, not the column. Use `df["div"]`.
- **Never fit a scaler, calibrator or base rate on data that includes the test period.** The temperature scaler fits on the tail of the *training* window.

## Verification that must keep passing

The harness self-tests are the reason any number here is trustworthy. `uv run pytest` — 277 tests.

- A **result-peeking cheater** must score RPS < 0.01 and be flagged.
- A **deliberately poisoned split** must raise from `assert_no_leakage`.
- Betting the de-vigged market back into its own prices must place **zero** bets at any positive EV threshold, and betting all of them must return **exactly minus the margin** — a closed-form check that only passes if the price join is correct.
- The market must land at **RPS 0.19–0.21**. Outside that band the pipeline is wrong, not the model.

A change that breaks one of these broke something real.

## Stack

Python 3.12 via `uv` (`uv sync`, `uv run pytest`). torch 2.13, catboost, penaltyblog, pandas, scipy, pyarrow. No lightgbm — its wheel needs Homebrew `libomp` and catboost is the literature's reference anyway.

`data/` and `models/` are gitignored. `.env` holds `SPORTMONKS_API_TOKEN` and is denied at user scope.

Ratings build in ~1s over 296k matches, rolling form in ~57s, sequences in ~11s — all cached to `data/processed/features.parquet` by `uv run python -m src.features.build`.

## Structure

```
src/
  data/        footballdata (22 divisions, 296k matches), sportmonks, team_aliases
  features/    ratings (Elo, pi), rolling form, sequences, squads, build
  models/      baselines (ordered logit, CatBoost, Dixon-Coles), net, heads
  eval/        devig (Shin), metrics (RPS/log-loss/ECE), split, betting, CLV
  scoreboard.py  experiments.py  tier2.py  phase6.py
v1/            the original build, frozen. Its betting numbers are unreliable —
               see the home/away rule above. v1/data is a symlink to ../data.
```

## Data boundaries that bite

- **Pinnacle closing odds run 2012/13 → January 2026.** Coverage decays from October 2025 and stops in February 2026. Anything after that grades against Bet365 or market-average closing, which is a softer benchmark and must be labelled as one.
- **football-data has no `Time` column before 2019/20**, so same-day fixtures cannot be ordered. Team-level rolling features are unaffected (nobody plays twice a day); league-wide aggregates are, and are lagged a day.
- **SportMonks free plan** reaches Danish Superliga and Scottish Premiership only, 3,000 req/hr per entity. Rich per-player statistics begin 2019/20. Filter odds to `markets/1` — 50 KB against 2.0 MB unfiltered.
- **SportMonks reports kickoff in UTC, football-data in local time.** The join is by date, which is safe only because no fixture kicks off later than 20:00 UTC.
