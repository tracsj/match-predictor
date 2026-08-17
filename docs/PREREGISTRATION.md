# Pre-registration: the betting rule

**Committed 2026-08-17, before any model PnL was computed.**

The network has never been run through `simulate()`. This file fixes the
betting rule, the prices, and the holdout *before* that happens, because the
alternative is to look at several rules and report the best one — which is
how a backtest manufactures an edge. Constantinou's own threshold sweep moves
1X2 ROI from −9% to +23% purely by shrinking the sample to 37 bets.

Phase 6 runs what is written here, once, and reports the result whatever it is.

---

## The rule

```python
BetRule(min_ev=0.05, min_odds=1.5, max_odds=5.0, stake=1.0)
```

Bet the maximum-EV outcome when its expected value is at least +5% and the
price falls between 1.5 and 5.0. Flat one unit. No compounding, no Kelly in
the headline number.

`min_ev = 0.05` is chosen ahead of time as a round threshold well above the
~3% Pinnacle margin, not tuned. Odds bounds exclude the two regimes where
staking is least realistic: prices below 1.5 where the vig dominates, and
above 5.0 where per-bet variance triples the sample size needed.

## The model

The configuration committed in `NetConfig` defaults as of `fd46f79`:
embeddings off, one trunk member, `hidden=96`, `dropout=0.2`, `lr=1e-3`,
goals head at weight 0.3, **GRU(32) sequence branch over each team's last 10
matches**, temperature-scaled on the tail of the training window. Features: `ALL_FEATURES` (49). Training pool: the full 296,208-match
corpus, matches strictly before each test window. Three seeds, averaged.

Probabilities come from the temperature-scaled softmax head. The Poisson head
is reported alongside but is **not** the betting signal.

## The prices

Three columns, reported together, led by the sharpest:

1. **Pinnacle closing** — the truth test.
2. **Bet365 closing** — a book an account could actually be held with.
3. **Market maximum closing** — the optimistic bound.

A result that is positive only in column 3 is an odds-comparison screen, not a
model, and will be reported as such.

CLV is computed by taking the **pre-close** price and grading against the
Pinnacle close for the same selection. CLV is the headline; ROI is secondary.

## The holdout, and an honest deviation

The plan said "keep the final 2–3 seasons in a locked holdout, untouched until
the end." **That did not survive.** The ablation campaign evaluated on every
panel season through 2024-25, and the shipped `NetConfig` defaults were
selected on those test seasons. Calling 2024-25 a holdout now would be false.

The genuinely untouched data is **season 2025-26**, which no model in this
project has been evaluated on.

Constraint recorded in advance: Pinnacle closing coverage decays from October
2025 and is absent from February 2026 (see `docs/research/00-measured-facts.md`).
So 2025-26 will be graded against **Bet365 closing and market-average closing**,
labelled explicitly as a softer benchmark than Pinnacle. Where Pinnacle close
exists (through roughly January 2026) it is reported separately.

Expected volume is roughly 800–1,200 bets from one season. Against the
19,000–43,500 bets needed to distinguish a 2% edge from zero, **this cannot
settle the ROI question and will not be presented as if it does.** CLV carries
the answer.

## Disclosure: configurations already evaluated on the panel

Required by the protocol in `docs/research/02-betting-evaluation-and-odds-data.md`.
Approximately **28** distinct configurations have been scored on the 2016-17 →
2024-25 panel during model selection:

| family | count | detail |
|---|---|---|
| Elo tier-shift | 7 | 0, 38, 50, 65, 76, 90, 130 |
| net ablation variants | 8 | full, no team emb, no league emb, no embeddings, no goals head, single member, wide, no dropout |
| learning rate | 3 | 3e-3, 1e-3, 3e-4 |
| feature sets | 2 | 7 rating features, 49 all features |
| training pools | 2 | panel-only, full corpus |
| sequence branch | 3 | none, GRU(32), GRU(64) |
| Dixon-Coles (Eredivisie only) | 4 | two xi values x two lookbacks |

None of these were selected on betting PnL — all on RPS and log loss, which is
the reason the betting question is still open rather than already contaminated.
That is the whole point of fixing the rule now.

## What would count as a real result

- **CLV mean ratio > 1.0 against Pinnacle close, with a binomial p below 0.01.**
  That is the only outcome here that would be evidence of genuine edge.
- ROI positive in the Pinnacle-close column with a bootstrap interval
  excluding zero would be suggestive, and still under-powered at this sample.
- Positive ROI only at market maximum is **not** a result. It is price
  shopping, already measured in this project at +4.8% using the market's own
  de-vigged opinion as the "model", and it is the strategy that got Kaunitz et
  al.'s accounts stake-limited into uselessness.

The expected outcome, stated in advance: **no edge against the closing line.**
