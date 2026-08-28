# match-predictor — root context

## What this is

A standing programme hunting for exploitable inefficiency in sports betting markets, and a harness honest enough that a positive result would be believable. It began as a football 1X2 predictor; the model is finished and the question has moved on.

**The finished result, so nobody re-derives it.** The network reaches RPS 0.20765 against de-vigged Pinnacle closing odds at 0.20291 — beating every non-neural baseline (t = +2.50), losing to the market (t = +20). The pre-registered betting rule lost in every price column, with closing-line value at **0.9952**. Filtering by model confidence does not rescue it: the deficit is uniform across confidence buckets.

⚠️ **That 0.9952 was once glossed as "its selections sat on the wrong side of the market's own movement". That gloss is withdrawn** (2026-08-17). The null is not 1.0 — see the CLV rule below. Re-analysed under `docs/PREREG_PHASE6_NULL.md`, against a regenerated bet population that reproduced the published row exactly, the observed 42.41% shortened sits **above** an odds-matched null of 39.26% — the *right* side of the market's movement, by +3.15pp at p = 0.018. **Narrow, and it changes nothing about the money**: the rule still lost in every price column and lost more than random. The corrected picture is a price edge of roughly 1–2% against a ~4% margin — good price selection, worse-than-market outcome selection.

**This is the normal outcome, not a defect in the pipeline.** Wilkens (2021) ran 15 ML architectures on ATP tennis and beat no odds-implied forecast; Kovalchik (2016) found the bookmaker consensus beat 11 published models; Data Golf publishes −0.92% ROI for its own model. **Do not treat 0.2076 as a near-miss to be closed with better features.** It is the empirical ceiling for a pre-match model in a liquid top-tier market. New work goes into finding a *different* market, not a better feature.

## Where the detail lives

**Starting a session cold? Read `docs/PROGRAMME.md` first** — its "Where we are" section is the handoff: what was finished last time, what is next in order, and the open threads. Everything else below is read on demand.

None of these is `@`-imported — an imported spoke is still always-loaded and saves nothing. Read one when its rule is about to bind.

| file | what is in it |
|---|---|
| `docs/PROGRAMME.md` | **the handoff** ("Where we are"), plus the hypothesis registry — status board, graveyard, the running count of every configuration ever tested, and the ruling on what does and does not increment it |
| `docs/hypotheses/*.md` | one file per hypothesis: pre-registration inline, result when settled. **H1 is `settled` (supported, then heavily qualified by its diagnostics); H2–H4 are `proposed`** and each names what is still open before it may run |
| `docs/PREREGISTRATION.md` | the football betting rule, prices and holdout, fixed before any PnL existed |
| `docs/PREREG_PHASE6_NULL.md` | the pre-registered re-analysis of Phase 6's CLV against a measured null, its reproduction gate, and the result that withdrew the "wrong side of the market" reading |
| `docs/FORWARD_LEDGER.md` | the forward record — predictions committed before kickoff, graded as results land. Rewritten from `predictions/*.csv` on every run, never appended |
| `docs/PHASE6_RESULT.md` | the betting answer, the CLV table, and why the model loses more than random betting — **plus a 2026-08-17 correction withdrawing its "wrong side of the market" reading**, with the superseded passages marked in place rather than rewritten |
| `docs/H3_RESULT.md` | H3's result: line movement **is** forecastable (+4.27pp over a matched null) and buys nothing — the gain is −0.15% of price against a 5.08% margin, and it does not beat a match model's incidental signal |
| `docs/H1_RESULT.md` | H1's result and its diagnostics: the tier-stratified CLV tables, the **measured CLV null** and the overround-tightening mechanism behind it, the per-tier margins over each tier's own drift, the control arms (anti-model, ordered logit, random-in-band), and the 2025-26 out-of-sample check |
| `docs/TIER2_RESULT.md` | what a starting XI is worth (nothing measurable), and the SportMonks upgrade recommendation |
| `docs/research/00-measured-facts.md` | what each data source actually contains, with the command that established it — including **Pinnacle's removal in 2026/27** and the exchange-vs-Pinnacle benchmark measurement, what `fixtures.csv` holds, and why `download_all` cannot refresh |
| `docs/research/01-neural-nets-for-match-prediction.md` | what wins on this task, with RPS numbers tagged to their datasets |
| `docs/research/02-betting-evaluation-and-odds-data.md` | how to build a backtest that would tell you the truth if the model were bad |

## Always

- **Pre-register before running a betting rule.** Threshold, prices, holdout and model config committed *before* any PnL is computed. Searching several rules and reporting the best is how a backtest manufactures an edge — one sweep here already produced a tempting +2.4% cell that was noise on 296 bets.
- **Increment the registry count for every configuration evaluated**, including the ones that die quietly mid-session. The count is the only thing that keeps a widening search honest.
- **Report CLV before ROI.** Distinguishing a 2% edge from zero needs ~45,000 bets; CLV converges roughly a hundred times faster and is what correctly said stop.
- **Never test CLV against a ratio of 1.0 or a 50% shortening rate. Measure the null, per ladder.** The pre-close and the close are not on average the same price: an overround that tightens toward kickoff means prices lengthen by default, so a randomly chosen band-eligible selection shortens well under half the time. Against 50% a real effect reads as nothing and a null tier reads as a contradiction; both happened here in one session and both were recorded before being caught, and **correcting the null flipped the sign of the founding study's CLV conclusion**. `clv_report` now enforces it — `null_rate` and `null_ratio` are required with no default, since a default is how the assumption travelled silently for two years, and both come back in the returned dict so no table can print a p-value without printing what it was tested against. **The rates differ enormously by ladder and must not be borrowed**: Pinnacle 45–48%, the forward exchange **31.8%**, because `bfe*` arrives through `fixtures.csv` a day out with a far wider book. Odds-matching has never moved any of them by more than 0.004, so the mix is not the driver. Instruments: `scripts/clv_null_calibration.py`, `scripts/h1_odds_matched_null.py`, `scripts/forward_matched_null.py`; the per-ladder numbers and their mechanism are in `docs/research/00-measured-facts.md`.
- **Cluster the shortening test by matchday, and check the block count before reading its p.** Same-day bets share news and market-wide moves, so `sqrt(p(1-p)/n)` counts correlated bets as independent evidence — `bootstrap_ci` has always resampled matchdays for ROI, and the shortening test did not until 2026-08-27. It decided both marginal results here: Phase 6 fell from p 0.018 to 0.154 and H1's out-of-sample lower stratum from 0.011 to 0.118, while the in-sample z of 14.2 barely moved. **The design effect is not a constant** — about 2.7 in 2025-26 against 1.11 over the ten-season panel, because a season whose overround tightens hardest is one where a day's prices lengthen together. And **below about 20 matchdays the correction inverts**: a bootstrap over few blocks estimates the error downward and returns a p *smaller* than the uncorrected one, which reads as the correction strengthening the result. `day_clustered_shortening_test` refuses to return one under its floor for that reason.
- **Commit a pre-registered result before running a single control.** Controls run first stop being controls: they become the reasons you hesitated to write the result down, and a surprising result is exactly when that hesitation feels most like rigour. Record the tables, then diagnose in a separate section — and when a control later overturns a reading, **correct it in the diagnostics rather than editing the recorded table.** Done both ways in one session on 2026-08-17: the H1 tables were committed at `4bc56bc` before any control ran, and three of their readings were later corrected in place below them.
- **Count confirmations only when they are independent, and they usually are not.** Four price ladders on the same matches are one result seen from four angles. Three examinations sharing a corpus, an era and overlapping models are converging, not confirming — especially when two of them have been *measured as equivalent*. The 2026-08-17 session got this right about the ladders and then made the identical claim one level up about H1, Phase 6 and H3 within the hour, so applying the rule locally is not evidence of having applied it everywhere. Say "converging" and name what they share.
- **Lead the sharpest price.** Three columns always: the sharpest close, one book you could actually hold an account with, market maximum. A result positive only in the third is an odds-comparison screen. **Which price is sharpest depends on the era** — Pinnacle close through 2024/25, **Betfair Exchange close from 2024/25 onward**, because football-data removed Pinnacle in 2026/27. The exchange is not a softer benchmark: equally accurate on a quarter of the margin, measured on 16,875 matches carrying both.
- **Report exchange ROI as pre- or post-commission, explicitly.** 2–5% of net winnings is real money and the difference is larger than most claimed edges. CLV is immune when both legs are exchange prices.
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

The harness self-tests are the reason any number here is trustworthy. `uv run pytest` — **348 tests** (2026-08-27).

**All five skip silently when `data/processed/matches.parquet` is absent**, which is the default on a fresh runner, and pytest then reports green. `uv run python scripts/assert_selftests_ran.py` asserts the five checks below actually ran; the workflow calls it after the suite. A green build without that step means nothing.

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
  data/        footballdata (22 divisions, 296k matches), fixtures (upcoming),
               sportmonks, team_aliases
  features/    ratings (Elo, pi), rolling form, sequences, squads, horizon, build
  models/      baselines (ordered logit, CatBoost, Dixon-Coles), net
               (both heads are inline in net.py — there is no heads.py)
  eval/        devig (Shin), metrics (RPS/log-loss/ECE), split, betting, CLV
  scoreboard.py  experiments.py  tier2.py  phase6.py  h1.py  h3.py
  refresh.py   re-fetch the current season and rebuild the corpus
  forward.py   predict upcoming fixtures, write predictions/YYYY-MM-DD.csv
  grade.py     grade committed predictions, rewrite docs/FORWARD_LEDGER.md
scripts/       assert_selftests_ran.py — the five self-tests must RUN, not skip
               h1_*.py, clv_null_calibration.py, forward_matched_null.py —
               H1's coverage probe, its post-hoc controls, and the forward
               ledger's odds-matching check. Each says in its docstring whether
               it is a control (no registry count) or an evaluation
predictions/   committed forward predictions. NOT gitignored; the commit is the
               evidence, so never rewrite or backfill a file here.
v1/            the original build, frozen. Its betting numbers are unreliable —
               see the home/away rule above. v1/data is a symlink to ../data.
```

## Forward validation

`.github/workflows/forecast.yml` runs Tuesday and Friday: refresh → test → assert the self-tests ran → predict → grade → commit. The commit timestamp is the evidence, which is why this runs on a GitHub runner rather than a laptop.

- **A prediction is never restated.** A fixture already present in any committed prediction file is skipped, not re-predicted at a shorter horizon with a better-trained model. Restating is how a forward ledger turns back into a backtest.
- **Never backfill or edit a file in `predictions/`.** `--as-of` exists for dry runs and deliberately writes nothing.
- **Grading requires the file's git commit time to precede every kickoff in it.** Rows failing that are reported and excluded, not averaged in.
- **An unplayed fixture is scored from history and absorbed into nothing** — see `src/features/horizon.py`. Letting a NaN score flow through the absorb step crashes Elo and, worse, silently records a phantom 0-0 defeat in rolling form and sequences.

## Data boundaries that bite

- **The exchange has no historical PRE-close.** `bfeh/bfed/bfea` are **absent from the results files entirely** — they arrive only through `fixtures.csv`, going forward. `bfec*` (the close) is there from 2024/25. So the exchange can grade forward CLV and **cannot grade a backward-looking one at all**; `psh → psch` is the only historical ladder that exists, and it ends in January 2026. Measured with `scripts/h1_holdout_coverage.py`.
- **Some "odds" are not odds.** 29 cells across 12 columns carry a price ≤ 1.0 — missing data wearing a number, which `notna()` does not catch and `np.nan_to_num` turns into a huge finite feature with no error and no NaN. Three are **1X2 closing** columns (`b365ca`, `maxca`, `avgca`). Nothing reported has moved, because a zero scores EV −1 and can never be a rule's argmax, and `min_odds` excludes it again — but **anything building features from raw prices must filter `> 1.0`**, as `build_frame()` in `src/h3.py` does. `tests/test_price_sanity.py` sweeps every odds column and fails on any new occurrence.
- **Pinnacle is gone.** Closing odds run 2012/13 → **2026-01-14**, decaying from October 2025, and the columns are **absent from the 2026/27 schema entirely** — removed, not empty. Grade forward work against the **Betfair Exchange close** (`bfec*`, from 2024/25, ~100% covered). Do not fall back to Bet365 or market-average and call it the closing line.
- **`fixtures.csv` is a rolling ~4-day window**, not a season fixture list, collected Friday ≤17:00 UK and Tuesday ≤13:00 UK. Anything reading it must run at least every four days or fixtures are silently never predicted. It retains already-played fixtures, so filter on kickoff.
- **`download_all` cannot refresh anything.** It skips files on disk *and* keys in `_missing.json`. Use `refresh_current()`, which purges the current season from both — eight 2026/27 divisions are memoised missing and would otherwise never arrive.
- **football-data has no `Time` column before 2019/20**, so same-day fixtures cannot be ordered. Team-level rolling features are unaffected (nobody plays twice a day); league-wide aggregates are, and are lagged a day.
- **SportMonks free plan** reaches Danish Superliga and Scottish Premiership only, 3,000 req/hr per entity. Rich per-player statistics begin 2019/20. Filter odds to `markets/1` — 50 KB against 2.0 MB unfiltered.
- **SportMonks reports kickoff in UTC, football-data in local time.** The join is by date, which is safe only because no fixture kicks off later than 20:00 UTC.
