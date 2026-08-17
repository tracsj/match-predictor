# The programme — hypothesis registry

**Live working doc.** Updated whenever a hypothesis changes status, and reconciled at every session wrap-up.

---

## Where we are — read this first

**Last session: 2026-08-17.** The football study is finished and settled; the programme is standing up around it. 277 tests green at close.

**Done**
- Football v2 built, measured, and its betting question answered — see the graveyard below.
- `.claude/` setup: project CLAUDE.md, settings, `/wrap-up`. This repo had none before.
- This registry, seeded with an honest configuration count.

**Next, in order**

1. **Flesh out `docs/hypotheses/` entries for H1–H4** — copy `docs/hypotheses/TEMPLATE.md` per hypothesis. Currently only listed on the status board below, with no files behind them.
2. **Forward-validation workflow** (`.github/workflows/forecast.yml`). Weekly cron: refresh CSVs, retrain (~60s), predict upcoming fixtures, commit a prediction file timestamped *before* kickoff, grade past ones as results land. Plain Python, no Claude in the loop, no secrets — `football-data.co.uk/fixtures.csv` is free and unauthenticated. **Land this early**, so the ledger accumulates while everything else proceeds.
3. **H1 pre-registered, then run.** Zero new data.
4. **n-outcome harness generalisation** + move sport-specific code under `src/sports/football/`. ~23 lines across `metrics.py`, `net.py`, `baselines.py`, `betting.py`; `devig.py` and `split.py` already generalise. H2 forces this first.
5. **H2 pre-registered, then run.**
6. **H3 ingest** (sportsbookreviewsonline — needs a browser user-agent or it 404s) and pre-registration.

**Open threads worth knowing**
- `uv run python -c` is denied in `.claude/settings.json` on purpose, to push analysis into re-runnable files. If it turns out to be more friction than it is worth, that is one line to remove.
- The `no-script-file-mutation.py` hook was **not** adopted here. Its promotion review is dated on-or-after 2026-09-07 and requires re-running corpus validation against this repo's own transcripts. Evidence for that review: the 2026-08-17 session used script-based file edits heavily.
- Forward CLV may have to grade against Bet365 or market-average closing, since Pinnacle closing stops February 2026. Establish which columns are populated in 2026-27 on the first run.

---

## The prior

Most entries here will die. That is the expected outcome, not a disappointment.

The football study finished at RPS 0.20765 against a market at 0.20291, with closing-line value of 0.9952 — behind the price, and on the wrong side of the market's own movement. A research sweep found the same result reported independently for tennis (Wilkens 2021; Kovalchik 2016; Lyócsa & Výrost 2018) and golf (Data Golf's own model, −0.92% ROI at a 0% EV threshold, concluding the market deserves more weight than they do).

**So the baseline expectation for every hypothesis below is: no edge.** This registry exists to find an exception efficiently and to make the search honest, not to assume an exception is there.

The deliverable is the testing machine plus honest findings. A dozen well-killed hypotheses and a harness trustworthy enough that a surviving result would be believable is a success.

## The count

**Configurations evaluated to date: 47.**

| family | n | detail |
|---|---|---|
| Elo tier-shift values | 11 | 0, 38, 50, 65, 71, 76, 85, 90, 100, 130, 160 |
| Net ablation variants | 8 | full, no team emb, no league emb, no embeddings, no goals head, single member, wide h=256, no dropout |
| Confidence thresholds | 8 | 0.34 … 0.70 |
| Dixon-Coles (Eredivisie) | 4 | two ξ × two lookbacks |
| Learning rates | 3 | 3e-3, 1e-3, 3e-4 |
| Feature sets | 3 | 7 rating, 49 all, 5 core without the moved flags |
| Sequence branch | 3 | none, GRU(32), GRU(64) |
| Training pools | 2 | panel-only, full 296k corpus |
| Confidence/CLV bucket analyses | 2 | RPS-vs-market by bucket, CLV by bucket |
| Tier-2 squad encoder arms | 2 | with / without |
| Phase 6 pre-registered run | 1 | |

**⚠️ This count was seeded at ~38 and corrected to 47 on its first reconciliation.** `docs/PREREGISTRATION.md` disclosed 7 tier-shift values where 11 were actually swept, and the tier-2 arms and feature-set variants were never counted at all. The pre-registration is a frozen record and is deliberately **not** edited retroactively — this registry carries the live number, and the discrepancy is recorded here rather than smoothed over.

That correction is the step working as designed. The failure mode is never a dishonest entry; it is a configuration tried casually, found uninteresting, and never written down.

This number goes up for **every** configuration scored. Distinguishing a 2% edge from zero needs ~45,000 bets; the count is what stops a widening search quietly manufacturing one.

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
