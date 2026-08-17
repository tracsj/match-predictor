# H1 — the lower divisions cleared the bar

**Run 2026-08-17 against `docs/hypotheses/H1-lower-division-inefficiency.md`, once.** Reproduce with `uv run python -m src.h1`. The pre-registration was committed at `9212a3c` and the runner at `ebfcf6c`, both before any tier-stratified number existed.

**This is the programme's first positive pre-registered result, and it arrived against a prior that said it would not.** That is the circumstance in which a number most deserves disbelief, so the tables below are recorded exactly as the run printed them, and the diagnostics that interrogate them are further down and clearly separated. Recording came first on purpose: controls run before the result is written become reasons for having hesitated to write it.

## Setup

- **Panel**: 79,430 matches, 13 seasons, 17 divisions, carrying **both** Pinnacle legs.
- **Graded**: 61,462 matches, **2015-16 → 2024-25**, ten test seasons. Split 18,920 lower / 42,542 upper.
- **Model**: the settled `NetConfig`, `ALL_FEATURES` (49), full-corpus training strictly before each test season, three seeds averaged, temperature-scaled on the training tail. `run_walk_forward` called unmodified.
- **Rule**: `BetRule(min_ev=0.05, min_odds=1.5, max_odds=5.0, stake=1.0)`, inherited unchanged.

## Primary — closing-line value by stratum

Bet at the Pinnacle pre-close, grade against the Pinnacle close of the same selection. The pre-registered bar: **ratio > 1.0 with binomial p < 0.01, in the lower stratum, absolute rather than relative.**

| stratum | bets | mean ratio | % shortened | binomial p |
|---|---|---|---|---|
| **lower (tiers 3–5)** | 9,920 | **1.0083** | **52.53%** | <0.0001 |
| upper (tiers 1–2) | 21,459 | 1.0046 | 50.22% | 0.53 |

**Lower: SUPPORTED.** Ratio above 1.0, binomial p below 0.01, 9,920 bets against a 3,250 floor.
**Upper: NOT SUPPORTED.**

## Secondary — the difference between strata

| | |
|---|---|
| lower − upper | **+0.00374** |
| 95% interval | [+0.00160, +0.00587] |
| Welch p | 0.0006 |
| spans zero | no |

Labelled secondary in advance and it stays secondary: it cannot promote H1 on its own.

## Descriptive — CLV per tier, ranked by nothing

| tier | divisions | bets | mean ratio | % shortened |
|---|---|---|---|---|
| 1 | D1, E0, F1, I1, SC0, SP1 | 9,246 | 1.0001 | 48.95% |
| 2 | D2, E1, F2, I2, SC1, SP2 | 12,213 | 1.0080 | 51.17% |
| 3 | E2, SC2 | 3,420 | 1.0091 | 53.51% |
| 4 | E3, SC3 | 3,515 | 1.0133 | 54.37% |
| 5 | **EC** | 2,985 | 1.0016 | **49.25%** |

**Read tier 5 before reading anything else here.** The National League is the thinnest, least-watched market in the panel, and it is the one tier that shows nothing — 49.25% shortened, below a coin flip. The effect lives in tiers 3 and 4. **That is not the shape the claim predicted.** H1's proposed mechanism was monotone — less attention, less money, looser prices — and a monotone mechanism should be strongest at the bottom. It is absent at the bottom.

Per the pre-registration's own words, this table decides nothing and is a lead for a future pre-registration. It is placed this prominently because it argues *against* the finding's stated mechanism, and burying an inconvenient descriptive table under a passing primary test is the exact move the registry exists to prevent.

## Descriptive — sensitivity, 2016-17 onward

Holding the lower stratum's composition fixed, since SC2/SC3 carry no Pinnacle price before 2016/17.

| stratum | bets | mean ratio | % shortened | binomial p |
|---|---|---|---|---|
| lower | 8,950 | 1.0078 | 52.20% | <0.0001 |
| upper | 19,129 | 1.0046 | 50.28% | 0.44 |

Unchanged. The composition wrinkle is not carrying the result.

## Secondary — ROI, three columns, led by the sharpest

| stratum | price set | eligible | bets | ROI | 95% CI | excludes 0 | random null | hit | avg odds |
|---|---|---|---|---|---|---|---|---|---|
| lower | **Pinnacle close** | 18,920 | 10,601 | **−5.43%** | [−8.39%, −2.79%] | yes | −4.38% | 30.6% | 3.30 |
| lower | Bet365 close | 11,300 | 4,995 | −6.67% | [−10.75%, −2.42%] | yes | −7.41% | 30.1% | 3.33 |
| lower | market max close | 11,319 | 7,087 | −1.81% | [−5.56%, +1.76%] | no | −1.92% | 31.8% | 3.31 |
| upper | **Pinnacle close** | 42,542 | 22,683 | −4.93% | [−6.84%, −2.92%] | yes | −3.63% | 32.4% | 3.20 |
| upper | Bet365 close | 25,250 | 10,851 | −7.95% | [−10.70%, −5.00%] | yes | −6.39% | 30.8% | 3.26 |
| upper | market max close | 25,256 | 15,872 | −2.53% | [−4.82%, −0.20%] | yes | −0.20% | 32.9% | 3.22 |

**ROI is negative everywhere, and in the lower stratum it is worse than betting at random** (−5.43% against a −4.38% null), which is the same signature the founding study found and explained: an EV filter is a magnifying glass pointed at the model's own errors.

So the two headline numbers point opposite ways — positive CLV, negative ROI, in the same stratum. They are not the same bet population (CLV bets are placed at pre-close prices, ROI bets at closing prices), so this is not a contradiction in arithmetic. It is still the central thing to explain, and the diagnostics below exist for it.

## What is settled, and what is not

**Settled**: the pre-registered test passed. Written down in advance, run once, reported as it came out. The count moves to 48.

**Not settled, and not even close**: that this is model skill, or that it could be staked. Three specific reasons, all visible in the tables above rather than discovered later.

1. **Tier 5 contradicts the mechanism**, as above.
2. **The upper stratum's mean ratio is also above 1.0** (1.0046) while its shortening rate sits at 50.22%. That is the signature of a statistical artifact, not an effect: the mean of `taken / close` is biased upward by Jensen's inequality even under symmetric price noise, because the ratio is bounded below and unbounded above. The upper stratum shows what "no effect" looks like on this metric, and it looks like a mean ratio of 1.0046. **Requiring the binomial test alongside the ratio is the only reason the primary bar was not passed by both strata**, and that requirement came from the founding pre-registration rather than from anything chosen here.
3. **The data is not unseen.** The pre-registration says so plainly. Model selection optimised RPS against outcomes on this very panel; the closing line also correlates with outcomes; so an overfitted model's selections correlate with the close through a pathway that is not skill. "Never selected on betting PnL" is true and does not close this gap.

The honest one-line summary: **something in the lower divisions moved the market's way more often than chance, the pre-registered test caught it, and nothing yet shows the model is the reason.**

## Post-hoc diagnostics

*(Everything below was run after the tables above were committed. These are controls, not candidates — they place no new configuration in the search, in the same sense as the grading dry run recorded in `docs/PROGRAMME.md`.)*

*(pending)*
