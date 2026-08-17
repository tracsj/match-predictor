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

*(Everything below was run after the tables above were committed at `4bc56bc`. These are controls, not candidates — they place no new configuration in the search, in the same sense as the grading dry run recorded in `docs/PROGRAMME.md`. The count stays at 48.)*

### The headline finding of the diagnostics: the null was wrong

**Everything above compares "% shortened" against 50% and the mean ratio against 1.0. That null is not the one this market obeys, and three of the readings recorded above do not survive the correction.**

Pinnacle's overround **tightens toward kickoff in every season measured**, because limits rise and the margin comes in. So prices lengthen by default, and a selection drawn at random from the rule's own odds band shortens well under half the time. Measured with `scripts/clv_null_calibration.py`:

| season | overround pre-close | overround close | tightening |
|---|---|---|---|
| 2018-19 | 1.0370 | 1.0318 | +0.0052 |
| 2021-22 | 1.0375 | 1.0330 | +0.0046 |
| 2023-24 | 1.0379 | 1.0351 | +0.0027 |
| **2024-25** | 1.0508 | 1.0393 | **+0.0114** |
| **2025-26** | 1.0491 | 1.0385 | **+0.0106** |

The odds-matched null, 200 sims with deciles taken from the model's own bets (`scripts/h1_odds_matched_null.py`):

| stratum | odds-matched null | 95% interval | unmatched null | is 0.5 inside? |
|---|---|---|---|---|
| lower | 0.4544 | [0.4439, 0.4644] | 0.4548 | **no** |
| upper | 0.4754 | [0.4695, 0.4818] | 0.4773 | **no** |

Matched and unmatched agree to within 0.002, so the odds mix is not producing the drift — which is the confound that arm existed to rule out, and it is ruled out.

### Three recorded readings that do not survive

**The recorded tables above are left exactly as they were run.** The corrections go here, which is the rule the hypothesis file set for itself before any of this was known.

**1. "Tier 5 shows nothing" — wrong.** Against each tier's own odds-matched drift (`scripts/h1_tier_nulls.py`):

| tier | divisions | bets | observed | its own null | **margin** | z |
|---|---|---|---|---|---|---|
| 1 | D1, E0, F1, I1, SC0, SP1 | 9,246 | 0.4895 | 0.4825 | **+0.70pp** | 1.35 — **not distinguishable** |
| 2 | D2, E1, F2, I2, SC1, SP2 | 12,213 | 0.5117 | 0.4701 | +4.16pp | 9.22 |
| 3 | E2, SC2 | 3,420 | 0.5351 | 0.4583 | +7.68pp | 9.01 |
| 4 | E3, SC3 | 3,515 | 0.5437 | 0.4572 | **+8.65pp** | 10.30 |
| 5 | EC | 2,985 | 0.4925 | 0.4500 | +4.25pp | 4.67 |

The National League is four points above its own drift at z = 4.67. It is not null. **The corrected picture is close to the opposite of the one recorded above**: the top tier is the only one indistinguishable from its own drift — which is what an efficient market is supposed to look like — and every tier beneath it is distinguishable.

The ordering is still **not monotone**. Tier 4 peaks at +8.65pp and tier 5 falls back to roughly tier 2's level. So the claim's thin-market mechanism is *qualified* rather than vindicated: something separates the top flight from everything below it, and it is not simply a gradient of thinness.

**2. "The upper stratum shows what nothing looks like" — wrong.** Against its measured null the upper stratum is **+2.67pp, z = 7.84, p ≈ 4e-15**. The Jensen's-inequality reasoning in that paragraph is correct as far as it goes — the mean of a ratio *is* biased upward — but it was the wrong explanation for this particular number. Median ratios, which the headline table failed to print: lower **1.0057**, upper **1.0025**.

**3. The finding is not really about lower divisions.** It is: **this model anticipates line movement in every tier except the top one, and does so most strongly in tiers 3–4.** H1's stratification found a real thing and mis-described it.

### Is it the model, or would any rule do this?

| arm | lower: % shortened | upper | reading |
|---|---|---|---|
| the net | 0.5253 | 0.5022 | +7.09pp / +2.67pp over null |
| ordered logit | 0.5076 | 0.4877 | roughly +5.3pp / +1.2pp over the same nulls |
| **anti-model** (min-EV) | **0.4141** | **0.4400** | **inverts — about 4pp *below* null** |
| random in band | 0.4548 | 0.4773 | the null itself |

**The anti-model inversion is the most informative arm.** Betting the outcome the rule likes *least* produces prices that lengthen relative to drift, by roughly the amount the real rule shortens them. So the direction of disagreement with the price carries information, which is not something a mechanical artifact would produce.

Both real models beat the drift, the net by more than the logit. **That gap does not establish skill.** The net is the model that was selected on this panel; the ordered logit is the one that could not overfit it. "A more flexible model shows more CLV on the data it was tuned against" is equally consistent with genuine line-anticipation and with the contamination pathway named above. In-sample the two readings are observationally equivalent, and the logit control must not be cited as ruling out contamination.

*(Caveat on the comparison: the logit and anti-model arms place different bet counts — 10,970 and 17,533 in the lower stratum against the net's 9,920 — so their odds mixes are not identical to the net's, and their margins over the net's nulls are approximate.)*

### The out-of-sample test, which is the one that matters

Season 2025-26 is the only data outside the contamination pathway. `scripts/h1_holdout_tiers.py` re-slices the settled Phase 6 run by tier — same configuration, same holdout, a question that run never asked.

| stratum | bets | observed | 2025-26 matched null | margin | z | p | in-sample margin |
|---|---|---|---|---|---|---|---|
| lower | 309 | 0.3819 | 0.3145 | **+6.74pp** | 2.55 | 0.011 | +7.09pp |
| upper | 823 | 0.4362 | 0.4182 | +1.80pp | 1.05 | 0.29 | +2.67pp |

**Both strata are INCONCLUSIVE BY FLOOR** — 309 and 823 bets against the pre-registered 3,250 — and the floor is not lowered to let them speak.

Read against 50%, that 38.19% looks like the finding collapsing. Read against the season's own drift of 31.45% — 2025-26 has by far the most extreme drift in the corpus — it is **directionally consistent with in-sample and close to it in size**. It is *not* a replication: the lower stratum is marginally significant on 309 bets, the upper is not significant at all, and both nulls come from a single anomalous season. Directionally consistent is the strongest phrase the evidence supports.

### The thing that stops all of it being a strategy

**ROI at the prices actually taken** — the CLV bets, settled at the pre-close prices they were struck at, which is the direct Buchdahl check:

| stratum | bets | ROI at taken price | hit rate | avg odds |
|---|---|---|---|---|
| lower | 9,920 | **−4.69%** | 31.21% | 3.264 |
| upper | 21,459 | **−4.91%** | 32.42% | 3.187 |

The anticipation is worth on the order of **1–2% on price** against a margin of roughly **4%**. It is real and it is smaller than the vig. **There is no staking story here under any reading of the CLV**, which is what H1's own "Where it could be staked" section predicted a positive would turn out to be: a finding about market microstructure rather than a strategy.

### This reaches past H1, into the settled study

`docs/PHASE6_RESULT.md` reports CLV 0.9952 at 42.4% shortened and reads it as *"the selections were systematically on the wrong side of the market's own movement."* That reading is only valid if the null is 1.0 / 50%.

Measured on Phase 6's own population — 2025-26, all main divisions, every band-eligible selection with both legs — the null is **0.3889 shortened, mean ratio 0.9904**. Phase 6 observed **0.4240 / 0.9952**, which is **+3.51pp above the drift** (z ≈ 2.6).

**Against a correctly specified null, the founding study's selections sat on the *right* side of the market's movement, not the wrong one.** Its ROI numbers are untouched and its conclusion that the rule lost money stands; it is the *CLV interpretation* that is in doubt.

That claim carries a real caveat and is not being acted on tonight: this null is unconditional across eligible selections, while Phase 6's bets were model-chosen with a different odds mix. In the H1 population matched and unmatched nulls agreed to within 0.002, which is a reason to expect agreement rather than a demonstration of it, and it was measured on different seasons and divisions. **Re-deriving Phase 6's own bet population against a measured null is its own pre-registered re-analysis.** Nothing in `docs/PHASE6_RESULT.md` or `CLAUDE.md` has been rewritten on the strength of this.

### What should happen next

1. **Re-analyse Phase 6's CLV against a measured null**, pre-registered, with the bet population re-derived rather than approximated.
2. **H3 inherits this entire apparatus** — the drift measurement, the requirement that CLV be tested against a measured null rather than 1.0, and the measured fact that the exchange pre-close does not exist historically so `psh → psch` is the only backward-looking ladder. H3 targets line movement directly, which is what these diagnostics say the model is actually doing.
3. **H1b, if it is worth running**: the never-graded 2012-15 seasons, which model selection also never touched. Reaching them needs `min_train_seasons` relaxed, and that is a protocol change that belongs in a pre-registration rather than an improvisation after seeing a positive.

### Every control run tonight

Each is a control, not a candidate. None increments the registry count.

| script | what it establishes |
|---|---|
| `h1_coverage_probe.py` | price-pair coverage per tier; the 3,250-bet floor's power arithmetic |
| `h1_panel_check.py` | plumbing pre-flight on random probabilities, no verdict |
| `h1_diagnostics.py` | median ratio, anti-model, random-in-band, ordered logit, ROI at taken prices |
| `h1_holdout_coverage.py` | the exchange pre-close is **absent** from the historical corpus |
| `h1_holdout_tiers.py` | 2025-26 out-of-sample, tier-stratified, with matched null |
| `h1_odds_matched_null.py` | the drift is not an odds-mix artifact; 0.5 is not the null |
| `h1_tier_nulls.py` | per-tier margins over each tier's own drift |
| `clv_null_calibration.py` | the overround mechanism, and the Phase 6 implication |
