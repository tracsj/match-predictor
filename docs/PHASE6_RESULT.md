# Would it have made money? No.

> **⚠️ Correction 2026-08-17 — one sentence in the CLV section below is wrong. The tables are not.**
>
> This page reads its 0.9952 mean ratio and 42.4% shortening rate as *"the selections were systematically on the wrong side of the market's own movement."* **That sentence is withdrawn.** It compares against a null of a 1.0 ratio and a 50% shortening rate — that the pre-close and the close are on average the same price. They are not: Pinnacle's overround tightens toward kickoff in every season measured, so prices lengthen by default.
>
> The re-analysis was pre-registered at `docs/PREREG_PHASE6_NULL.md` and run against **this page's own regenerated bet population**, which reproduced the published row exactly (1,337 bets, 0.9952, 0.4241 against 0.4240) before any null was computed. Against an odds-matched null:
>
> | | observed | matched null | 95% interval | margin |
> |---|---|---|---|---|
> | Pinnacle pre-close | 0.4241 | 0.3926 | [0.3650, 0.4196] | **+3.15pp**, z = 2.36 |
>
> **The selections were on the right side of the market's movement, not the wrong one** — narrowly, at p = 0.018, which would not clear the p < 0.01 bar this project uses for claiming edge. All four pre-close ladders point the same way, though they are overlapping selections rather than independent confirmations.
>
> **⚠️ SECOND CORRECTION, 2026-08-27 — that p = 0.018 assumed bets were independent, and they are not.** The row above stands as recorded; this is the diagnostic under it, not a rewrite of it. Every shortening-rate z in this project used `sqrt(p(1-p)/n)` while `bootstrap_ci` has always resampled *matchdays*, because same-day bets share news and market-wide moves. Block-bootstrapping days over the same 1,337 bets:
>
> | | bets | days | z | p |
> |---|---|---|---|---|
> | as recorded, independent | 1,337 | — | 2.36 | 0.018 |
> | **day-clustered** | 1,337 | **116** | **1.43** | **0.154** |
>
> The standard error inflates **1.66×**, a design effect of **2.74** — so the 1,337 bets carried closer to 490 bets' worth of independent evidence. That correction is not uniform and was checked against a control: H1's in-sample stratum, 9,920 bets over 903 days, shows a design effect of only **1.11** and its z of 14.2 barely moves. The within-day correlation is roughly 0.16 in 2025-26 against 0.01 over the ten-season panel, which is consistent with 2025-26 carrying the corpus's most extreme drift — a season where a day's prices lengthen together. **The direction is unchanged and the significance is gone.** The +3.15pp margin over the matched null is still positive and still on the right side of the market's movement; it is no longer distinguishable from noise. Gated on reproducing this page's recorded z of 2.36 before the clustered one was computed, at `scripts/day_clustered_clv.py`. The null is treated as exact, which it is not — folding in its simulation error moves p further from significance, never toward it.
>
> **Everything else on this page stands, and the correction must not be read as softening it**: every ROI number, the finding that the rule lost money in all four price columns, the random-bet-null comparison, and the conclusion that no staking rule rescues a model that has not first beaten the market on RPS. A CLV direction is not a yield. The corrected picture is that the model picked *prices* slightly better than recorded while picking *outcomes* worse than the market — a price edge of roughly 1–2% against a margin near 4%.
>
> Full result in `docs/PREREG_PHASE6_NULL.md`; the mechanism and the null's derivation in `docs/H1_RESULT.md`.

**Run 2026-08-17 against `docs/PREREGISTRATION.md`, once.** Reproduce with
`uv run python -m src.phase6`.

The rule, the prices, the model configuration and the holdout were all fixed
in advance, before the network had ever been through the betting simulator.
This is what came out.

## Setup

- **Holdout**: season 2025-26, 7,646 matches, 22 divisions. No model in this
  project had been evaluated on it.
- **Training**: 283,832 matches, all strictly before 2025-07-25.
- **Model**: the committed `NetConfig` defaults — GRU(32) sequence branch, no
  embeddings, one trunk member, goals head at 0.3, temperature-scaled on the
  tail of the training window, three seeds averaged.
- **Rule**: `BetRule(min_ev=0.05, min_odds=1.5, max_odds=5.0, stake=1.0)`.

## Forecast quality on the holdout

| model | n | RPS | log loss | ECE | accuracy |
|---|---|---|---|---|---|
| the net | 7,646 | 0.2080 | 1.0165 | 0.0083 | 49.1% |
| market (Pinnacle close) | 2,964 | **0.2038** | 1.0040 | 0.0148 | 49.7% |
| the net, same subset | 2,964 | 0.2086 | 1.0197 | 0.0110 | 48.0% |

The market wins on the untouched season too, by about the same margin it won
on every season before it. Nothing surprising, and the consistency is itself
reassuring about the harness.

Pinnacle closing is available for only 2,964 of 7,646 matches (38.8%), because
the feed decays from October 2025 and stops in February 2026 — recorded in
advance in `docs/research/00-measured-facts.md`.

## Closing-line value — the pre-registered headline

Bet at a pre-close price, grade against the Pinnacle close of the same
selection. A ratio above 1.0 means the price taken was bigger than the market
settled at.

| price taken at | bets | mean ratio | % shortened | binomial p |
|---|---|---|---|---|
| Pinnacle pre-close | 1,337 | **0.9952** | 42.4% | <0.001 |
| Bet365 pre-close | 1,131 | 0.9781 | 36.2% | <0.001 |
| market average pre-close | 988 | 0.9615 | 28.3% | <0.001 |
| market maximum pre-close | 1,532 | 1.0141 | 59.5% | <0.001 |

**Every real book comes out below 1.0, significantly.** The pre-registration
set the bar as *"CLV mean ratio > 1.0 against Pinnacle close, with a binomial
p below 0.01"*. Observed: 0.9952, with only 42.4% of prices shortening. That
is not a near miss — it is the wrong side of the line, and significantly so.

> ⚠️ **Superseded 2026-08-17 — see the correction at the top of this page.**
> The bar quoted here is the pre-registered one and stands as the bar. What
> is wrong is the inference that falling under it puts the selections on the
> wrong side of the market's movement: against a measured null they were on
> the right side of it. The text is left as written because what this study
> concluded at the time is part of the record.

The market-maximum row clears 1.0, but that is price shopping rather than
forecasting, and this project already measured the same effect at +4.0% CLV
using the market's own de-vigged opinion as the "model".

## ROI, three columns, led by the sharpest

| price set | eligible | bets | ROI | 95% CI | excludes 0 | random-bet null | hit rate | avg odds |
|---|---|---|---|---|---|---|---|---|
| **Pinnacle close** | 2,964 | 1,490 | **−8.92%** | [−17.9%, +0.9%] | no | −5.02% | 30.1% | 3.29 |
| Bet365 close | 7,646 | 3,020 | −15.28% | [−19.3%, −11.0%] | yes | −8.68% | 28.2% | 3.30 |
| market max close | 7,646 | 4,051 | −8.50% | [−12.7%, −3.7%] | yes | −4.58% | 30.6% | 3.29 |
| market avg close | 7,646 | 2,776 | −17.61% | [−22.5%, −12.5%] | yes | — | 28.6% | — |

Negative everywhere, and significantly so in three of four columns.

## The finding worth understanding

**The model loses more than betting at random.** In every column the ROI is
below the random-bet null: −8.92% against −5.02% at Pinnacle close, −15.28%
against −8.68% at Bet365.

That is not a paradox, it is the mechanism. The rule bets where the model's
probability most exceeds the price — that is what a +5% EV filter selects for.
When the model is slightly *worse*-calibrated than the market, the matches
where it disagrees most with the market are exactly the matches where it is
most wrong. The EV filter is a magnifying glass pointed at the model's own
errors.

A model that merely matched the market would lose the vig. A model marginally
behind the market, used to pick disagreements, loses considerably more.

⚠️ One caveat on that comparison: the random-bet null picks random outcomes
across eligible matches, so its odds mix is not identical to the model's
selections. The direction is clear and consistent across four price columns,
but the exact gap should not be over-read.

## What this does and does not settle

**Settled**: the pre-registered rule, applied to this model, would have lost
money over 2025-26. ~~CLV says the selections were systematically on the
wrong side of the market's own movement~~ — **withdrawn 2026-08-17**; against
a measured null they were on the *right* side of it, by +3.15pp at p = 0.018.
CLV converges roughly a hundred times faster than ROI, so it is still the
informative half, and what it now says is that the model's PRICE selection
was mildly good while its OUTCOME selection stayed worse than the market's.
**The money conclusion is unchanged.**

**Not settled**: whether a 2% edge exists somewhere in this space. One season
produced 1,490 bets against the ~45,000 needed to distinguish a 2% edge from
zero at these odds. The pre-registration said so in advance and named CLV as
the headline for exactly this reason.

**The expected outcome, recorded in advance, was "no edge against the closing
line."** That is what happened.

## What would have to change

Not a better threshold — that is the search that manufactures edges. The model
would have to beat the market on RPS *first*. It currently sits at 0.2080
against 0.2038 on the holdout, having closed most of the gap from a 0.2340
uniform baseline but not the last of it. Until the forecast is better than the
price, no staking rule rescues it, and the CLV result is the clean measurement
of that.
