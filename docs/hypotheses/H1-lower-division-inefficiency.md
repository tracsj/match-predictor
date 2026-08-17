# H1 — Lower-division football is priced less efficiently than the top tier

**Status:** `proposed`
**Opened:** 2026-08-17 · **Settled:** —

> This file is **not** a pre-registration. It states the claim, the data and the
> known problems so the rule can be fixed in one sitting later. Everything under
> "The rule" that is still open is marked as open, and H1 may not move to
> `running` until none of it is.

## The claim

Bookmakers price the Premier League with more attention, more money and sharper
competition than they price League Two, so a model with no edge overall may
still have one where the market is thinnest.

**What would falsify it.** CLV mean ratio at or below 1.0 in the lower-tier
stratum, or no ordered relationship between tier and CLV. Since the programme's
prior is no edge, the falsifier is the expected result and the claim is what
needs the evidence.

## The rule

**Inherited unchanged from `docs/PREREGISTRATION.md`,** so that H1 is not a new
threshold search wearing a new question as a disguise:

```python
BetRule(min_ev=0.05, min_odds=1.5, max_odds=5.0, stake=1.0)
```

Model configuration also inherited: the `NetConfig` defaults of the settled
study, `ALL_FEATURES` (49), full-corpus training, three seeds averaged,
temperature-scaled on the tail of the training window.

**Still open, and must be fixed before running:**

1. **The contrast.** One pre-specified comparison, not five. The intended form
   is tiers 3–5 pooled (E2, E3, EC, SC2, SC3) against tiers 1–2 pooled, using
   the `TIER` map in `src/features/ratings.py`. Testing each tier separately and
   reporting the best is a five-way search, and would need an explicit
   multiplicity correction stated in advance.
2. **Minimum bets per stratum** below which a stratum is reported as
   inconclusive rather than as a result.
3. Whether the extra-country files (single-division, no tier) are excluded
   outright — they have no tier, so they cannot enter a tier contrast.

## Data provenance

| | |
|---|---|
| source | football-data.co.uk main division files |
| coverage | 2012/13 → 2024/25. `PSCH` (Pinnacle closing) is ~100% populated in **every** lower tier across that window — E1, E2, E3, EC, SC1, SC2, SC3, D2, I2, SP2, F2 |
| odds timing | **closing** for grading (`psch/pscd/psca`), **pre-close** for placing (`psh/psd/psa`). CLV takes the pre-close and grades against the close |
| known gaps | **SC2/SC3 begin at 2016/17**, not 2012/13. 2025/26 Pinnacle coverage is 38.8% pooled and **zero after 2026-01-14**. 2026/27 has no Pinnacle column at all |

Measured 2026-08-17. Coverage is **uniform across tiers** rather than worse in
the lower ones, which is the fact that makes H1 answerable at all — and it is
the opposite of what one would guess.

**H1 is a backward-looking test and cannot be extended forward.** Pinnacle left
the feed in 2026/27, so nothing after January 2026 can carry a Pinnacle-closing
CLV claim.

The exchange close is the forward replacement and is **not** a softer benchmark —
measured as equally accurate on a quarter of the margin. But it only begins in
2024/25, so it cannot serve H1's 2012/13–2024/25 window either. H1 is bounded by
Pinnacle's coverage at one end and the exchange's at the other, and that is why
it is a closed historical question rather than an ongoing one. Bet365 and
market-average closing *are* softer and would have to be labelled as such if
used.

## Holdout

**There is no untouched data left for H1, and that has to be said plainly.** The
panel 2016/17 → 2024/25 was used for model selection, and 2025/26 was consumed
by the Phase 6 pre-registered run. H1 re-uses data this project has already
seen.

What makes it still worth running: **no configuration here was ever selected on
betting PnL**, tier-stratified or otherwise — selection was on RPS and log loss
only, which is recorded in `docs/PREREGISTRATION.md`. Tier-stratified CLV is a
question never asked of this data. That is a weaker position than a clean
holdout and a stronger one than a free-for-all, and the write-up must not
present it as the former.

The 47 configurations already evaluated (`docs/PROGRAMME.md`) are the relevant
disclosure, and the count moves again when H1 runs.

## Where it could be staked

This is the hypothesis's real problem, and it is worse than the statistics.

- **The headwind points the wrong way.** Overround *rises* with tier (recorded
  on the status board in `docs/PROGRAMME.md`), so any pricing inefficiency in
  the lower divisions has to overcome a wider spread before it pays. The two
  effects are not independent: a market is loose *because* it is thin, and thin
  is also why the margin is fat.
- **Limits scale with attention too.** The books that price League Two loosely
  are the books that accept £50 on it. Kaunitz et al. made real money and were
  limited to $1.25 stakes within months, and that was on better-known leagues.
- **Exchange liquidity in English lower divisions is thin**, so the exchange's
  usual answer to stake limits — it does not ban winners — is weaker here,
  because the money simply is not on the other side.

A positive H1 would most likely be a finding about market microstructure rather
than a strategy. That is still worth having, and it should be framed that way
from the start rather than discovered at the end.

## Expected outcome

**No edge.** The programme's baseline, and here there is a specific additional
reason: the settled study found the deficit to the market uniform across
confidence buckets, which is not the signature of a model that is right
somewhere and wrong elsewhere. A tier split is another way of slicing the same
population, and the confidence split already came back flat.

## Result

*(filled after the run, whatever it says)*
