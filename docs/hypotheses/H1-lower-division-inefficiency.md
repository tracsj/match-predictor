# H1 — Lower-division football is priced less efficiently than the top tier

**Status:** `settled` — **supported by its pre-registered test**
**Opened:** 2026-08-17 · **Pre-registered:** 2026-08-17 · **Settled:** 2026-08-17

> **This file is now a pre-registration.** Everything below was committed before any tier-stratified CLV or PnL number existed. Nothing here may be changed once the run starts; if something turns out to be wrong, the run is reported against what is written here and the correction goes in the Result section, not in the rule.

## The claim

Bookmakers price the Premier League with more attention, more money and sharper competition than they price League Two, so a model with no edge overall may still have one where the market is thinnest.

**What would falsify it.** The lower-tier stratum's CLV mean ratio at or below 1.0, or above 1.0 with a binomial p at or above 0.01. Since the programme's prior is no edge, the falsifier is the expected result and the claim is what needs the evidence.

An earlier draft of this file named *two* falsifiers — the lower-stratum ratio, and separately "no ordered relationship between tier and CLV". That is the multiplicity problem this file elsewhere warns about, so the ordering test has been demoted to a descriptive table (see "The rule", item 5) and only the single test above decides H1.

## The rule

**Inherited unchanged from `docs/PREREGISTRATION.md`,** so that H1 is not a new threshold search wearing a new question as a disguise:

```python
BetRule(min_ev=0.05, min_odds=1.5, max_odds=5.0, stake=1.0)
```

Model configuration also inherited: the `NetConfig` defaults of the settled study, `ALL_FEATURES` (49), full-corpus training, three seeds averaged, temperature-scaled on the tail of the training window.

### 1. The contrast — one comparison, fixed

**Lower = tiers 3–5 pooled** (E2, E3, EC, SC2, SC3). **Upper = tiers 1–2 pooled** (E0, SC0, D1, I1, SP1, F1, E1, SC1, D2, I2, SP2, F2). Tiers come from the `TIER` map in `src/features/ratings.py`.

**The primary test is absolute, not relative**: does the lower stratum's CLV mean ratio exceed 1.0, with binomial p below 0.01? That is the same bar `docs/PREREGISTRATION.md` set for the founding study, and it is the only outcome that would be evidence of genuine edge. A relative result — lower above upper, both below 1.0 — is a fact about market microstructure and cannot promote H1 to supported.

The lower-minus-upper difference **is** computed and reported, with a confidence interval, because it is the shape of the claim and it would be evasive to omit it. It is secondary and labelled as such.

### 2. Minimum bets per stratum

**3,250 bets.** Below that a stratum is reported as inconclusive rather than as a result, whichever direction it points.

Derived rather than chosen: CLV's binomial test asks whether the share of prices that shortened differs from 50%, and at two-sided α = 0.01 with 80% power, detecting a 3-percentage-point shift needs 3,244 bets. Reproduce with `uv run python scripts/h1_coverage_probe.py`. Measured expected volume clears it comfortably in both strata — about 8,530 lower and 19,190 upper — so this floor should not bind. It is written down anyway, because a floor set after seeing a thin stratum is not a floor.

### 3. Divisions with no tier are excluded outright

Five main-feed divisions carry no `TIER` entry because their countries have a single division in the corpus: **B1, G1, N1, P1, T1**. The extra-country files are excluded for the same reason. A division with no tier cannot enter a tier contrast, and inventing one for it would be a modelling choice made to increase sample size.

### 4. Window and protocol

**Panel**: main-division matches with a tier, carrying **both** the Pinnacle pre-close (`psh/psd/psa`) and the Pinnacle close (`psch/pscd/psca`), seasons **2012-13 → 2024-25**.

**Graded window: 2015-16 → 2024-25, ten test seasons.** The panel opens at 2012-13 because that is Pinnacle's first season on both legs, but `run_walk_forward` in `src/experiments.py` hardcodes `min_train_seasons=3`, so the panel's first three seasons serve as training data and are never graded. That harness constraint is accepted rather than worked around: relaxing it would mean editing shared code covered by 299 tests immediately before a pre-registered run, and the measured volume clears the floor without those three seasons.

**Protocol**: `season_walk_forward`, train on everything strictly before each test season, three seeds (0, 1, 2) averaged, temperature scaler fitted on the last 15% of each training window. Training pool is the **full corpus**, not the panel — the test set must carry Pinnacle prices, the training set need not. This is `run_walk_forward(panel, NetConfig(), features=ALL_FEATURES, seeds=(0,1,2), train_pool=full_corpus, sequences=...)`, called unmodified.

**Expected cost**: ten walk-forward fits × three seeds, roughly 30–75 minutes locally. Recorded here so that a slow run is not later used as an argument for cutting seeds — `docs/PROGRAMME.md` already rules that a seed count is a configuration and a runner is not.

### 5. What is reported descriptively, with no verdict attached

Two tables, printed in full, ranked by nothing and used to decide nothing:

- **CLV per individual tier**, 1 through 5. Testing each tier and reporting the best is a five-way search; printing all five and deciding on neither is a description. If an ordering appears, it is a lead for a future pre-registration, not a result of this one.
- **A sensitivity re-run on 2016-17 → 2024-25**, the window in which SC2 and SC3 carry Pinnacle prices and the lower stratum therefore has stable composition throughout (see Data provenance). It cannot rescue or overturn the primary test.

### 6. Prices

CLV, the headline: **Pinnacle pre-close, graded against the Pinnacle close** of the same selection. ROI, secondary and under-powered as always: three columns led by the sharpest — **Pinnacle close**, **Bet365 close**, **market maximum close**. A result positive only in the third column is an odds-comparison screen and will be reported as one.

### 7. The registry

This run **increments the configuration count in `docs/PROGRAMME.md` by one**, from 47 to 48. One configuration: inherited rule, inherited model, one pre-specified contrast. The count moves whatever the result says.

`scripts/h1_coverage_probe.py` does **not** increment it. It counts price coverage, fits nothing and scores nothing.

## Data provenance

| | |
|---|---|
| source | football-data.co.uk main division files |
| coverage | 2012/13 → 2024/25. `PSCH` (Pinnacle closing) is ~100% populated in **every** lower tier across that window — E1, E2, E3, EC, SC1, SC2, SC3, D2, I2, SP2, F2 |
| odds timing | **closing** for grading (`psch/pscd/psca`), **pre-close** for placing (`psh/psd/psa`). CLV takes the pre-close and grades against the close |
| known gaps | **SC2/SC3 carry no Pinnacle price before 2016/17** — the matches are in the corpus, the odds are not. 2025/26 pair coverage is 29–48% and stops after 2026-01-14. 2026/27 has no Pinnacle column at all |

Coverage of the **pair** — both legs present, which is what CLV actually needs — measured 2026-08-17 with `scripts/h1_coverage_probe.py`. Over the graded window it runs **97.3% in the lower stratum** (18,920 of 19,450 matches) and **99.4% in the upper** (42,542 of 42,787). The earlier version of this table verified the closing leg only; the pre-close leg had never been counted, and it is the leg that decides whether the contrast has any power.

The one composition wrinkle: because SC2 and SC3 have no Pinnacle price before 2016/17, the lower stratum is E2/E3/EC-only in the graded window's first season and gains the two Scottish divisions afterwards. Disclosed here, handled by the descriptive sensitivity re-run in item 5, and not corrected for in the primary test.

Coverage is **uniform across tiers** rather than worse in the lower ones, which is the fact that makes H1 answerable at all — and it is the opposite of what one would guess.

**H1 is a backward-looking test and cannot be extended forward.** Pinnacle left the feed in 2026/27, so nothing after January 2026 can carry a Pinnacle-closing CLV claim.

The exchange close is the forward replacement and is **not** a softer benchmark — measured as equally accurate on a quarter of the margin. But it only begins in 2024/25, so it cannot serve H1's window either. H1 is bounded by Pinnacle's coverage at one end and the exchange's at the other, and that is why it is a closed historical question rather than an ongoing one. Bet365 and market-average closing *are* softer and would have to be labelled as such if used.

## Holdout

**There is no untouched data left for H1, and that has to be said plainly.** The panel 2016/17 → 2024/25 was used for model selection, and 2025/26 was consumed by the Phase 6 pre-registered run. H1 re-uses data this project has already seen.

What makes it still worth running: **no configuration here was ever selected on betting PnL**, tier-stratified or otherwise — selection was on RPS and log loss only, which is recorded in `docs/PREREGISTRATION.md`. Tier-stratified CLV is a question never asked of this data. That is a weaker position than a clean holdout and a stronger one than a free-for-all, and the write-up must not present it as the former.

The 47 configurations already evaluated (`docs/PROGRAMME.md`) are the relevant disclosure, and the count moves to 48 when H1 runs.

## Where it could be staked

This is the hypothesis's real problem, and it is worse than the statistics.

- **The headwind points the wrong way.** Overround *rises* with tier (recorded on the status board in `docs/PROGRAMME.md`), so any pricing inefficiency in the lower divisions has to overcome a wider spread before it pays. The two effects are not independent: a market is loose *because* it is thin, and thin is also why the margin is fat.
- **Limits scale with attention too.** The books that price League Two loosely are the books that accept £50 on it. Kaunitz et al. made real money and were limited to $1.25 stakes within months, and that was on better-known leagues.
- **Exchange liquidity in English lower divisions is thin**, so the exchange's usual answer to stake limits — it does not ban winners — is weaker here, because the money simply is not on the other side.

A positive H1 would most likely be a finding about market microstructure rather than a strategy. That is still worth having, and it should be framed that way from the start rather than discovered at the end.

## Expected outcome

**No edge.** The programme's baseline, and here there is a specific additional reason: the settled study found the deficit to the market uniform across confidence buckets, which is not the signature of a model that is right somewhere and wrong elsewhere. A tier split is another way of slicing the same population, and the confidence split already came back flat.

Stated more precisely, so it can be wrong: the lower stratum's CLV mean ratio lands **below 1.0**, within roughly a point of the 0.9952 the founding study measured pooled, and the lower-minus-upper difference has a confidence interval spanning zero.

## Result

**Run 2026-08-17. The pre-registered test passed: SUPPORTED.** Full tables in `docs/H1_RESULT.md`.

The lower stratum (tiers 3–5) returned a CLV mean ratio of **1.0083** with **52.53%** of prices shortening across **9,920 bets**, binomial p below 0.0001 — clearing a bar of ratio > 1.0 at p < 0.01 with three times the required volume. The upper stratum did not clear it (1.0046, 50.22% shortened, p = 0.53). The secondary lower-minus-upper difference is +0.00374, 95% [+0.00160, +0.00587].

**The expected outcome recorded in advance was wrong.** This file predicted a ratio below 1.0, near the founding study's 0.9952, with the difference spanning zero. None of that happened.

**What the same run also shows, and it is not supporting evidence.** Tier 5 — the National League, the thinnest market in the panel — is the one tier with no effect at all (49.25% shortened). The proposed mechanism was monotone in thinness and the observed pattern is not. ROI in the lower stratum is **−5.43%** at Pinnacle close, worse than the −4.38% random-bet null. And the upper stratum's mean ratio also exceeds 1.0, which is what Jensen's inequality does to a mean of ratios under symmetric noise — the binomial requirement inherited from `docs/PREREGISTRATION.md` is the only thing that separated the two strata.

So H1 is settled as **supported by its own test and unexplained by its own mechanism.** The write-up carries the diagnostics.

### Addendum, same day, after the controls ran

**The paragraph above is wrong in two of its three criticisms, and it is corrected here rather than edited, per this file's own rule.**

The controls established that **50% is not the null for "% shortened"**. Pinnacle's overround tightens toward kickoff in every season measured, so prices lengthen by default and a randomly chosen band-eligible selection shortens only 45–48% of the time. The odds-matched null puts 0.5 outside both strata's intervals, and matched and unmatched nulls agree to within 0.002, so it is not an odds-mix artifact.

Against each tier's **own** drift, tier 5 is **+4.25pp at z = 4.67** — not null at all — and the upper stratum is **+2.67pp at z = 7.84**, not Jensen noise. The only tier indistinguishable from its own drift is **tier 1**, the top flight. The corrected reading is nearly the inverse of the one above: the top division looks efficient and everything below it does not, though the gradient is not monotone (tier 4 peaks, tier 5 falls back to tier 2's level).

**The claim's stratification found something real and named it wrongly.** The finding is that this model anticipates line movement in every tier but the top one, most strongly in tiers 3–4 — which is H3's subject wearing H1's stratification.

**What survives unchanged from the paragraph above**: ROI. At the prices actually taken it is −4.69% in the lower stratum. The anticipation is worth 1–2% on price against a ~4% margin, so there is no strategy here, exactly as "Where it could be staked" predicted.

**The verdict is not revised.** The bar was inherited from `docs/PREREGISTRATION.md` and committed in advance, and a bar is not rewritten once its result is known — including when the correction would arguably *strengthen* the finding. A measured null changes the interpretation, and the interpretation is where it has been put.
