# H3 — The direction of line movement is predictable

**Status:** `settled` — **supported, and useless**
**Opened:** 2026-08-17 · **Pre-registered:** 2026-08-17 · **Settled:** 2026-08-17

> **This file is now a pre-registration.** Everything below was committed before any H3 model existed and before any movement forecast had been scored. Nothing here may be changed once the run starts; a correction goes in the Result section, not in the rule.

## The claim

Stop trying to predict the match and predict the *market* instead. If the direction a price moves between the pre-close snapshot and the close can be forecast from information available at the snapshot, then betting the side the market is about to move toward captures closing-line value without ever holding a better opinion about the football.

**Why this is the strongest of the four.** Every other hypothesis measures CLV as a *proxy* for edge. Here CLV is the objective itself, which removes the whole problem of an under-powered ROI sample: the settled study needed ~43,600 bets to separate a 2% edge from zero, and CLV converges roughly a hundred times faster.

**What would falsify it.** No better-than-null prediction of movement direction, or a directional edge that exists but is smaller than the margin it would have to cross.

## ⚠️ Disclosure: this project already knows something about H3's label

**This has to come before the rule, because it changes what H1 leaves for H3 to find.** The 2026-08-17 session measured line movement extensively while diagnosing H1, and produced what is effectively a partial H3 result by accident:

- The pre-close→close drift is now measured per season, per tier and odds-matched. Prices lengthen by default; a randomly chosen band-eligible selection shortens 45–48% of the time, 31% in 2025-26.
- **A match-outcome model's disagreement with the price predicts movement direction.** The settled net's maximum-EV selections shortened **+2.7pp (tier 1–2) to +8.7pp (tier 4)** more often than that drift, and the *anti-model* — betting the minimum-EV outcome — inverted it to roughly 4pp below.
- Phase 6's selections showed the same effect out of sample, +3.15pp at p = 0.018.

**So "line movement contains recoverable structure" is already partly established, and H3 cannot claim it as a new finding.** What is genuinely open, and what this pre-registration tests:

> **Does a model fitted directly to the movement label beat what a match-outcome model already achieves incidentally?**

That is the question, and it is narrower and more honest than the one this file was opened with.

**Consequence for the holdout.** Season 2024-25 is inside the window H1 measured drift on. Drift is *null* information rather than signal information, and no model has ever been fitted to this label — but it is not a pristine holdout and is not described as one.

## The rule

Nothing inherits from `docs/PREREGISTRATION.md`: this is not a match-outcome model and `BetRule` does not describe it.

### 1. The target

**Three-way categorical: which outcome's price shortened most**, i.e. `argmax` over H/D/A of `pre_close / close` using Pinnacle's two legs (`psh/psd/psa` → `psch/pscd/psca`).

Labelled `"H"`, `"D"`, `"A"` deliberately, so `CatBoostBaseline` and `OUTCOMES` are reused **unchanged** — that class already maps CatBoost's alphabetical ordering back to H/D/A by name, and re-implementing it is how a silent scramble gets introduced.

**The baseline to beat on accuracy is 41.1%, not 33.3%** — measured with `scripts/h3_feasibility.py`: H 41.08%, D 19.82%, A 39.10%. The majority class is also drifting (draws rose from 15% in 2015-16 to 26% in 2024-25), so a walk-forward model is chasing a moving target and a per-season baseline is reported beside it.

### 2. The features

`ALL_FEATURES` (the existing 49) **plus** eleven pre-close price features, all knowable at the snapshot:

- log implied probability from Pinnacle pre-close, per outcome (3)
- log implied probability from Bet365 pre-close, per outcome (3)
- overround of each book's pre-close (2)
- `log(ps / b365)` per outcome — the **cross-book disagreement** signal (3)

Only Bet365 and Pinnacle are used. Measured coverage on gradable rows: **b365 99.9%, ps 100.0%**, while `max`/`avg` sit at 48.1% and `bw`, `iw`, `wh`, `vc`, `bfe` are absent from the parsed corpus entirely. Using `max`/`avg` would halve the sample and introduce an era confound, since their coverage is concentrated in later seasons. They are a v2 option, not part of this test.

### 3. The leakage constraint, and its assertion

**No closing column may appear on the input side.** `psch/pscd/psca`, `b365c*`, `maxc*`, `avgc*`, `bfec*` are the label's source and are forbidden as features.

This is the single place H3 can silently cheat, and it gets a **hard assertion in the runner rather than care**: the feature matrix's column list is checked against a forbidden-substring set (`"c" + outcome` patterns and every known closing column) and the run aborts if any appears. A leak here would show up as an unusually good result, which is the one outcome nobody interrogates.

### 4. The model — one, untuned

`CatBoostBaseline` at its existing repo defaults: 400 iterations, depth 4, learning rate 0.05, seed 0. **No hyperparameter is tuned and no second model is fitted.** CatBoost is the literature's reference for this task per `CLAUDE.md`, and it is already in the repo with fixed defaults, which is precisely why it is the one being used — a model chosen now cannot be a model chosen after seeing results.

### 5. Split and holdout

**Development window: 2012/13 → 2023/24**, walk-forward by season via `season_walk_forward` unchanged, training strictly before each test season.

**Holdout: 2024/25**, trained on everything strictly before it. **The holdout is the only thing that decides H3.** Walk-forward results on the development window are descriptive and are reported to show whether the signal is stable, not to establish it.

**2025/26 is excluded**: Pinnacle pair coverage is 38% and stops after 2026-01-14, so it cannot carry this label cleanly.

### 6. What counts as a bet

Bet the predicted-argmax outcome **on every holdout match** where that outcome's Pinnacle pre-close price falls in [1.5, 5.0]. **No confidence threshold** — a threshold is the classic way a backtest manufactures an edge, and this file fixes its absence rather than its value.

A single pre-specified selective variant is reported **descriptively and cannot decide anything**: the top quartile by predicted probability. One variant, named now, not a sweep.

### 7. The primary criterion

On the 2024/25 holdout, for the bets defined above:

**H3 is supported if the shortening rate lies above the odds-matched null's 95% interval AND the one-proportion z-test against that null gives p < 0.01.**

The null is `matched_null` from `scripts/h1_odds_matched_null.py`, unchanged — ten deciles from H3's own bets, 200 sims, drawn from band-eligible selections in the same holdout season.

**Minimum 3,250 bets**, the same floor H1 derived (a 3pp shift at α = 0.01, 80% power). Below it the result is inconclusive by floor.

Testing against 1.0 or 50% is **forbidden**, per the standing rule this project learned the hard way earlier the same day.

### 8. The stakeability test, which is part of the falsifier

The claim dies if the edge is real but smaller than the spread it must cross. So the write-up must report, on the holdout: the **CLV ratio gain in percentage terms** against **Pinnacle's measured pre-close overround** on the same rows. If the gain is under the margin, H3 is a forecasting result and **must be stated as one**, not as a strategy.

The prior from H1's diagnostics is that it will be: roughly 1–2% of price against ~4% of margin.

### 9. Registry treatment

**This run increments the count by one, 48 → 49.** One configuration: one target, one feature set, one model at fixed defaults, one holdout.

`scripts/h3_feasibility.py` does not increment it — it counts coverage and label balance, fits nothing.

If the leakage assertion aborts the run, that is **not** an evaluation and does not increment the count.

## Data provenance

| | |
|---|---|
| source | football-data.co.uk, already parsed into `matches.parquet` — **zero new ingest** |
| coverage | **100,584 gradable rows** carrying both Pinnacle legs, 2012/13 → Jan 2026 |
| odds timing | pre-close snapshot (Friday ≤17:00 UK / Tuesday ≤13:00 UK) and closing. Both dated, which is what a CLV claim requires |
| label movement | median `\|log(pre/close)\|` = **4.21%**, 75th percentile 8.22%; only 14.4% of outcomes move less than 1%. There is real movement to forecast |
| known gaps | Pinnacle absent from 2026/27; the exchange **pre-close does not exist historically at all**, so there is no second ladder |

**The status board once recorded H3 as "new ingest". That was wrong** — football-data carries both legs for the same match, so this costs nothing.

**What the pre-close leg is NOT.** It is not the opening line. The snapshot is one to three days out, so much of the open-to-close path has already happened. Published evidence motivating H3 concerns the *full* open-to-close move, so this tests a shorter and harder segment. A true-opening version needs `sportsbookreviewsonline` and is the right second step **only if** this shows something.

## Holdout honesty

The corpus has been used extensively, but **never for this label** — no model in this project has been fitted to price movement. The rows are not untouched; the target is. Combined with the disclosure at the top, that is a weaker position than a clean holdout and a stronger one than a free-for-all, and the write-up must not present it as the former.

## Where it could be staked

A positive H3 is the hardest of the four to convert into money and the easiest to convert into a finding.

- Capturing CLV requires getting on at the pre-close price, in size, repeatedly. A soft book restricts that quickly; the exchange will not, but exchange liquidity one to three days out is much thinner than at the close — and this project has now measured that the exchange pre-close is not even *recorded* historically, which is a hint about how thin that market is.
- **Steam-chasing is the behaviour books limit fastest**, because it is the signature of a sharp customer and costs them on every bet.

So H3 is framed from the start as: does the market's own movement contain recoverable structure *beyond what a match model already extracts*? Whether that structure is stakeable is a second question, and pretending otherwise would repeat the mistake the market-maximum column was invented to catch.

## Expected outcome

**A real but small directional signal that does not clear the margin** — and note this is a *different* prediction from the one this file carried when it was opened, which was "no edge, with the lowest confidence of the four".

The change is not a hedge, and it is on the record before the run: H1's diagnostics already established that movement direction is partly predictable from a match model's disagreement with the price. Predicting "no signal at all" would now be predicting against something already measured.

**Specifically**: the holdout shortening rate lands **above** its matched null, plausibly clearing p < 0.01 given several thousand bets — and the CLV gain lands **under** Pinnacle's pre-close overround, making H3 a forecasting result rather than a strategy. Accuracy is expected in the mid-40s against a 41.1% majority baseline.

**The genuinely uncertain part**, and the thing worth running for: whether fitting the movement label *directly* beats the incidental signal the net already produced. If it does not, that is the interesting negative — it would mean the market's move is mostly toward what a decent match model already thinks, and there is no separate market-microstructure signal to harvest.

## Result

**Run 2026-08-17. SUPPORTED by its primary test, and it buys nothing.** Full tables in `docs/H3_RESULT.md`.

On the 2024/25 holdout: directional accuracy **44.39%** against a 39.89% majority baseline, and a shortening rate of **0.4782 against an odds-matched null of 0.4356**, 95% [0.4243, 0.4457] — **+4.27pp, z = 6.96, p = 3.3e-12**, on 6,549 bets against the 3,250 floor. Line movement direction is forecastable, comfortably.

**The stakeability test failed, and it was written into the falsifier for exactly this.** The mean ratio of the prices taken is **0.9985** — a gain of −0.15% of price against a **5.08%** pre-close overround. H3 picks the shortening side more often than chance and the magnitudes do not follow. It is a forecasting result and is stated as one.

**The narrowed question came back negative, which is the finding worth keeping.** On 6,173 shared rows, fitting the movement label directly returned **+3.98pp** over the null against the settled match model's **+4.50pp** — difference −0.52pp, z = −0.45, **p = 0.65**. Building a model whose whole purpose is forecasting the market's move does about as well as one that was never asked to. The market appears to move toward what a decent match model already thinks, with no separable microstructure signal on top.

**Consequence for the second step, and it runs against the file's own plan.** This file said an ingest of true opening lines is justified *only if* the free version showed something. It did — and it also showed that what it found is not separable from a match model's existing disagreement signal. **The case for buying opening lines is weaker after this run than before it**, and that is worth stating plainly rather than treating the trigger as met on a technicality.

**The expected outcome was recorded in advance and was right**, including the revision made when H1's diagnostics narrowed the question: a real but small signal, accuracy in the mid-40s, not clearing the margin.
