# H3 — line movement is forecastable, and it buys nothing

**Run 2026-08-17 against `docs/hypotheses/H3-line-movement.md`, once.** Reproduce with `uv run python -m src.h3`. The pre-registration was committed at `3b6e021` and the runner at `e0b2ea4`, both before any movement model existed.

**Two answers, and they point opposite ways on purpose.** The pre-registered test passed: the direction of line movement is forecastable well above a measured null. The pre-registered *stakeability* test — part of the falsifier, not an afterthought — failed. And the question H3 was narrowed to actually ask came back negative.

## Setup

- **Frame**: 97,577 gradable rows, 2012/13 → 2024/25, carrying both Pinnacle legs and Bet365 pre-close.
- **Features**: 60 = the existing 49 + 11 pre-close price features (log implied probabilities from each book, both overrounds, and `log(ps/b365)` cross-book disagreement).
- **Leakage**: asserted, not assumed. No closing column reached the feature matrix; the gate is verified in both directions by 19 tests in `tests/test_h3_leakage.py`.
- **Model**: `CatBoostBaseline` at repo defaults — 400 iterations, depth 4, lr 0.05, seed 0. Untuned, one model.
- **Holdout**: 2024/25, trained on 89,909 rows strictly before it. The only thing that decides H3.

## Directional accuracy

| | |
|---|---|
| accuracy | **44.39%** |
| majority baseline (always predict H) | 39.89% |
| **lift** | **+4.50pp** |

Read against 39.89%, not 33.3%. The three-way label is H 41.1% / A 39.3% / D 19.7% across the corpus, so a constant predictor is already most of the way to a number that looks respectable.

## Primary — CLV against an odds-matched null

| arm | bets | observed | matched null | 95% interval | margin | z | p |
|---|---|---|---|---|---|---|---|
| **H3, every match** | 6,549 | **0.4782** | 0.4356 | [0.4243, **0.4457**] | **+4.27pp** | 6.96 | 3.3e-12 |
| H3, top quartile *(descriptive)* | 1,638 | 0.5073 | 0.4354 | [0.4133, 0.4598] | +7.20pp | 5.87 | <1e-8 |

**SUPPORTED.** Above the interval, p far below 0.01, on 6,549 bets against a 3,250 floor.

The top-quartile row is the single pre-specified selective variant. It is **descriptive and decides nothing** — it is reported because it was named in advance, and it would have been reported identically had it come back flat.

## The stakeability test — which H3 fails

| | |
|---|---|
| CLV ratio gain on the bets taken | **−0.15%** of price |
| Pinnacle pre-close overround on the same rows | **5.08%** margin |
| gain clears the margin? | **no** |

**This is the number that matters for whether H3 is a strategy, and it is negative.** The model picks the shortening side 4.27 points more often than chance — and the mean ratio of the prices it takes is **0.9985**, still below 1.0. It wins the direction more often than it should and the magnitudes do not follow.

Per the pre-registration, that makes H3 **a forecasting result and it must be stated as one.** It is not a strategy and no threshold rescues it, which is why no threshold was permitted.

## The question H3 was actually narrowed to ask

The pre-registration disclosed that H1's diagnostics had already shown a match-outcome model's disagreement with the price predicts movement direction. So the open question was **whether fitting the movement label directly beats that accident.**

The runner prints H1's tier band as a rough comparator, and that comparison is not sound — different seasons, divisions and odds mixes. `scripts/h3_vs_net.py` restricts both arms to the **same 6,173 shared 2024/25 rows** and scores them against the same null construction:

| arm | bets | observed | matched null | margin | z |
|---|---|---|---|---|---|
| H3 (movement label, fitted directly) | 5,429 | 0.4773 | 0.4375 | **+3.98pp** | 5.91 |
| the net (match model, incidental) | 2,819 | 0.4863 | 0.4414 | **+4.50pp** | 4.81 |

**Difference −0.52pp, z = −0.45, p = 0.65. No detectable difference.**

**This is the interesting negative, and it is the finding worth keeping.** Building a model whose entire purpose is to forecast the market's move does about as well as a match-outcome model that was never asked to. The most natural reading: **the market moves toward what a decent match model already thinks**, and there is no separate market-microstructure signal sitting on top of it to harvest.

*Caveat that keeps this honest:* the two arms do not bet the same matches. H3 bets every row whose predicted outcome is in band; the net bets only where its EV filter fires. The bet counts differ and the selections overlap only partly, so this compares two strategies on a shared universe rather than being a paired test. A paired design would be a stronger instrument and is not what was pre-registered.

## One data fault found and fixed mid-run

The first execution emitted `divide by zero` while building price features: **10 rows carry a Bet365 price of exactly 0.0**, which `notna()` does not catch. Left alone it becomes an infinite log-implied-probability that `np.nan_to_num` silently converts to a huge finite feature — no error, no NaN, just a garbage row treated as informative. This is the repo's own "never fill a missing value with zero" rule arriving from the other direction: the *feed* supplied the zero.

Measured with `scripts/h3_zero_price_check.py`: all 10 are in `b365h`, all in the training window, **none in the holdout**. So no reported number moved. The filter was fixed anyway and the run repeated; the pre-fix holdout figures were accuracy 44.54% and margin +4.45pp against the post-fix 44.39% and +4.27pp. Both are recorded so the change is visible rather than silent.

## What is settled, and what is not

**Settled.** Line movement direction is forecastable from snapshot information, comfortably and repeatedly. Fitting the label directly buys nothing over a match model. The edge does not clear the margin, so there is no strategy here.

**Not settled.** This tests the *snapshot-to-close* segment, one to three days out — not the full open-to-close path the published evidence concerns. A true-opening version needs `sportsbookreviewsonline`, and H3's own file said that step is justified **only if** the free version showed something. It did show something, and it also showed that what it found is not separable from what a match model already provides — so the case for paying for opening lines is weaker after this run than it looked before it, not stronger.

**The expected outcome, recorded in advance**, was "a real but small signal that does not clear the margin", with accuracy "in the mid-40s against a 41.1% majority baseline". Observed: 44.39% accuracy, a real signal, and a gain of −0.15% against a 5.08% margin. That is what happened.
