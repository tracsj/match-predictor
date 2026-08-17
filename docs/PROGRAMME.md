# The programme — hypothesis registry

**Live working doc.** Updated whenever a hypothesis changes status, and reconciled at every session wrap-up.

## The prior

Most entries here will die. That is the expected outcome, not a disappointment.

The football study finished at RPS 0.20765 against a market at 0.20291, with closing-line value of 0.9952 — behind the price, and on the wrong side of the market's own movement. A research sweep found the same result reported independently for tennis (Wilkens 2021; Kovalchik 2016; Lyócsa & Výrost 2018) and golf (Data Golf's own model, −0.92% ROI at a 0% EV threshold, concluding the market deserves more weight than they do).

**So the baseline expectation for every hypothesis below is: no edge.** This registry exists to find an exception efficiently and to make the search honest, not to assume an exception is there.

The deliverable is the testing machine plus honest findings. A dozen well-killed hypotheses and a harness trustworthy enough that a surviving result would be believable is a success.

## The count

**Configurations evaluated to date: ~38.**

Seeded honestly rather than from zero: ~28 disclosed in `docs/PREREGISTRATION.md` (Elo tier-shift sweep, net ablation variants, learning rates, feature sets, training pools, sequence-branch sizes, Dixon-Coles configs) plus ~10 from the post-Phase-6 confidence analysis (8 sweep thresholds and the calibration/CLV bucket breakdowns).

This number goes up for **every** configuration scored, including ones abandoned after a single look. A search whose own count is wrong is theatre. Distinguishing a 2% edge from zero needs ~45,000 bets; the count is what stops a widening search quietly manufacturing one.

## Status board

| ID | hypothesis | status | cost to first answer | notes |
|---|---|---|---|---|
| H1 | Lower-division football is less efficiently priced | `proposed` | zero new data | ⚠️ overround *rises* with tier — the headwind points the wrong way |
| H2 | Derived/correlated markets (O/U, BTTS, correct score) | `proposed` | zero new data | reuses the existing Poisson scoreline head |
| H3 | Line movement is predictable (open → close) | `proposed` | new ingest, free | strongest published evidence; targets the market, not the match |
| H4 | Betfair AU/NZ niche leagues (AFL, NRL, NBL, BBL) | `proposed` | new ingest, free | best capacity-adjusted option — the exchange does not ban winners |

Statuses run `proposed → pre-registered → running → settled`. A hypothesis may not move to `running` until its pre-registration is committed.

## Graveyard

Nothing settled yet beyond the founding study.

| what | verdict | where |
|---|---|---|
| Football 1X2, pre-registered betting rule | **dead** — lost in every price column, CLV 0.9952 | `docs/PHASE6_RESULT.md` |
| Confidence-threshold filtering | **dead** — deficit to market uniform across confidence; one positive cell was 296 bets with a CI spanning zero | `docs/PHASE6_RESULT.md` |
| Starting-XI squad encoder (tier 2) | **no measurable effect**, from an experiment that could only have detected one ~14× larger | `docs/TIER2_RESULT.md` |
| Elo tier-shift correction for promoted teams | **not shipped** — fixed the bias, corrupted the rating pool; the model learns it better from a flag | `src/features/ratings.py` (EloParams) |

## Rules for an entry

Every hypothesis file in `docs/hypotheses/` states, before anything is run:

1. **The claim**, in one sentence, and what would falsify it.
2. **The rule** — thresholds, markets, staking, all fixed.
3. **Data provenance** — source, coverage, and whether the odds are opening, pre-close or closing. Undated odds cannot support a CLV claim.
4. **The holdout**, and whether the project has touched it before.
5. **Where it could be staked.** Kaunitz et al. made real money and were limited to $1.25 within months. An edge that cannot be filled is a finding, not a strategy.
6. **The expected outcome**, written down in advance.

Results are committed after the run whatever they say, and the count moves either way.
