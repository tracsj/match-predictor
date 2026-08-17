# The programme — hypothesis registry

**Live working doc.** Updated whenever a hypothesis changes status, and reconciled at every session wrap-up.

---

## Where we are — read this first

**Last session: 2026-08-17 (third session that day).** H1 was pre-registered, run, and settled — **it passed, the first hypothesis in this programme to do so** — and its diagnostics then found that the test had been run against the wrong null, which reaches back into the founding study. 302 tests green at close.

**What happened, shortest version.** H1 asked whether lower-division football is priced less efficiently. Its pre-registered test came back **SUPPORTED**: the pooled lower stratum (tiers 3–5) returned CLV 1.0083 with 52.53% of prices shortening over 9,920 bets, p < 0.0001, against a floor of 3,250. The result was committed *before* any control was run, deliberately.

Then the controls found that **50% is not the null for "% shortened".** Pinnacle's overround tightens toward kickoff every season, so prices lengthen by default and a randomly chosen band-eligible selection shortens only 45–48% of the time. Three readings already committed did not survive that correction, and are corrected in `docs/H1_RESULT.md` rather than edited away.

**The corrected finding, which is not the one H1 claimed.** Against each tier's own drift, **tier 1 is the only tier indistinguishable from its own baseline** (+0.70pp, z = 1.35) — which is what an efficient market should look like — while every tier beneath it is distinguishable: tier 2 +4.16pp, tier 3 +7.68pp, tier 4 +8.65pp, tier 5 +4.25pp. So this model **anticipates line movement in every division except the top flight**, most strongly in tiers 3–4. The gradient is not monotone, so the thin-market mechanism is qualified rather than vindicated.

**It is still not a strategy, and that part survives every reading.** ROI at the prices actually taken is −4.69% (lower) and −4.91% (upper). The anticipation is worth 1–2% on price against a ~4% margin.

**The out-of-sample check, which is the one that matters.** 2025-26 re-sliced by tier: lower +6.74pp over that season's own matched drift (z = 2.55, p = 0.011) against +7.09pp in sample; upper +1.80pp (z = 1.05, not significant). **Directionally consistent and close in size — not a replication**, at 309 and 823 bets against a 3,250 floor, and both are inconclusive by that floor.

**Why this mattered beyond H1, and it is now settled too.** `docs/PHASE6_RESULT.md` read its 0.9952 CLV as "the selections sat on the wrong side of the market's own movement". That reading is **withdrawn**. The re-analysis was pre-registered at `docs/PREREG_PHASE6_NULL.md` and gated on reproducing the published row exactly — it did, to four decimals — before any null was computed. Observed **0.4241 shortened against an odds-matched null of 0.3926**, above the 95% interval, **+3.15pp at p = 0.018**. Narrow, and short of the p < 0.01 bar this project uses for claiming edge. **The ROI tables and the "rule lost money" conclusion are untouched**; the superseded passages are marked in place rather than rewritten, because what the study concluded at the time is part of the record.

**Still true from the previous session, and unchanged:** the forward workflow is live and verified on a real runner (`tracsj/match-predictor`, private, default branch `master`), Pinnacle is gone from the 2026/27 schema, and the exchange close replaces it going forward. The four runner-only bugs and the cold-cache timings are further down this file.

**New data boundary found this session.** `bfeh/bfed/bfea` — the exchange **pre-close** — are **absent from the results files entirely**, arriving only through `fixtures.csv`. The exchange can grade forward CLV and cannot grade a backward-looking one at all, so `psh → psch` is the only historical ladder and it ends 2026-01-14.

**Next, in order**

1. **Check Tuesday's scheduled run (2026-08-18, 13:15 UTC) — still outstanding.** The 2026-08-17 session could not do this: the cron had not fired yet, and the four runs `gh run list` showed were all that evening's manual dispatches. It is the first run with a non-empty horizon, so the first to exercise training on a runner and the first to write a real `predictions/YYYY-MM-DD.csv`. Concretely: `gh run list --workflow=forecast.yml --limit 3`, then `gh run view <id> --log`, and confirm three things — the predict step reports a fixture count rather than "no upcoming fixtures", training completed inside the 120-minute timeout (unmeasured on 2 vCPUs; 4m05s locally), and `predictions/` gained a file. If it timed out, raise the runner rather than cutting seeds — see the registry ruling below.

2. **H3 in its free form, now much better equipped.** It targets line movement directly, which is what the H1 diagnostics say the model is actually doing. It inherits the drift measurement, the measured-null requirement, and the fact that no exchange pre-close exists historically.

3. **Read the ledger's "Schedule coverage" table after a few weeks.** There is a known structural gap and it is measured rather than assumed. The earliest observed Friday kickoff is **17:30 UK**, while the Friday run fires 18:15 UK under BST and takes ~20 minutes — so Friday early kickoffs can only ever be reached from *Tuesday's* snapshot, and whether that snapshot spans to Friday is not something one observation could settle. **No cron change fixes this**: the feed has exactly two states a week. If the table shows Friday-evening misses accumulating, the options are a cached model fast enough to fit between the 17:00 upload and a 17:30 kickoff, or accepting the gap and saying so.

4. **n-outcome harness generalisation** + move sport-specific code under `src/sports/football/`. ~23 lines across `metrics.py`, `net.py`, `baselines.py`, `betting.py`; `devig.py` and `split.py` already generalise. H2 forces this first, H4 needs the two-outcome case.

5. **H2 pre-registered, then run.**

6. **H1b, only if it earns a slot**: the 2012-15 seasons, never graded and never used for model selection. Reaching them requires relaxing `run_walk_forward`'s hardcoded `min_train_seasons=3`, which is a protocol change and belongs in a pre-registration rather than an improvisation after seeing a positive.

**Open threads worth knowing**
- `uv run python -c` is denied in `.claude/settings.json` on purpose, to push analysis into re-runnable files. It cost three extra steps this session and was worth it each time — the scripts are re-runnable.
- The `no-script-file-mutation.py` hook was **not** adopted here. Its promotion review is dated on-or-after 2026-09-07 and requires re-running corpus validation against this repo's own transcripts. Evidence for that review: the 2026-08-17 sessions used script-based file edits heavily.
- **`HxG`/`AxG` arrived in 2026/27** and the loader parses them away. Per-division, not universal (present B1, N1; absent EC). A genuinely new feature source and the first new one in a while.
- **Eight 2026/27 divisions are still 404 upstream** (D1, E1, E2, F1, G1, I1, I2, T1) and are re-memoised in `_missing.json` on every refresh until they publish. `refresh_current()` purges and retries them each run, so this heals itself — but a division silently missing from the ledger for weeks would look exactly like this, so check the refresh line in the workflow log.
- `new_league_fixtures.csv` covers 14 of the 16 extra countries and is **not** read yet, so the forward path serves the 22 main divisions only.

**⚠️ The correction that matters most, 2026-08-17 (third session): every CLV number this project has ever reported was tested against the wrong null.**

`docs/PREREGISTRATION.md` and the H1 pre-registration both test closing-line value against a mean ratio of 1.0 and a shortening rate of 50%. That assumes the pre-close and the close are on average the same price. **Measured, they are not** — Pinnacle's overround tightens toward kickoff in every season from 2015-16, so prices lengthen by default and a randomly chosen band-eligible selection shortens only **45–48%** of the time. The odds-matched null puts 0.5 outside both strata's intervals, and matched and unmatched nulls agree to within 0.002, so it is not an odds-mix artifact. Instruments: `scripts/clv_null_calibration.py`, `scripts/h1_odds_matched_null.py`, `scripts/h1_tier_nulls.py`.

Consequences, in order of how much they change:

1. ~~**Phase 6's headline reading is in doubt.**~~ **Settled the same day — the reading is withdrawn.** Pre-registered at `docs/PREREG_PHASE6_NULL.md` and run against a regenerated bet population that reproduced the published row exactly (1,337 bets, 0.9952, 0.4241 vs 0.4240) before any null was computed. Observed **0.4241 shortened against an odds-matched null of 0.3926**, 95% [0.3650, 0.4196] — above the interval, **+3.15pp at z = 2.36, p = 0.018**. Narrow, and it would not clear the p < 0.01 bar this project uses for claiming edge. All four pre-close ladders point the same way but are overlapping selections, not independent confirmations. **The ROI tables and the "rule lost money" conclusion are untouched**, and the superseded passages in `docs/PHASE6_RESULT.md` are marked in place rather than rewritten.
2. ~~**`CLAUDE.md`'s summary needs one clause revisited**~~ — **done.** The withdrawn clause, recorded here so the change is visible rather than silent: *"closing-line value at 0.9952, meaning its selections sat on the wrong side of the market's own movement."* The number stands and the "meaning" half is gone, replaced by the measured result. The rest of that paragraph — the RPS figures, the losses in every price column, and "do not treat 0.2076 as a near-miss to be closed with better features" — was never affected.
3. **H3 inherits the whole apparatus.** The drift measurement, the requirement to test CLV against a measured null, and the measured fact that `bfeh/bfed/bfea` — the exchange pre-close — is **absent from the historical corpus** rather than sparse, so `psh → psch` is the only backward-looking ladder that exists.

**Corrections to this board made 2026-08-17**
- **H3 is not "new ingest".** football-data already carries a pre-close and a closing price for the same match, so a first version costs zero new data. The `sportsbookreviewsonline` ingest buys the *true opening* line and should follow only if the free version shows something.
- **H2 is not three markets.** Of O/U 2.5, BTTS and correct score, only **O/U 2.5 is in the feed**. The other two need new ingest, so H2's "zero new data" applies to a third of what it claimed.

---

## The prior

Most entries here will die. That is the expected outcome, not a disappointment.

The football study finished at RPS 0.20765 against a market at 0.20291, with closing-line value of 0.9952 — behind the price, and on the wrong side of the market's own movement. A research sweep found the same result reported independently for tennis (Wilkens 2021; Kovalchik 2016; Lyócsa & Výrost 2018) and golf (Data Golf's own model, −0.92% ROI at a 0% EV threshold, concluding the market deserves more weight than they do).

**So the baseline expectation for every hypothesis below is: no edge.** This registry exists to find an exception efficiently and to make the search honest, not to assume an exception is there.

The deliverable is the testing machine plus honest findings. A dozen well-killed hypotheses and a harness trustworthy enough that a surviving result would be believable is a success.

## The count

**Configurations evaluated to date: 48.**

**+1 on 2026-08-17 (third session): the H1 tier-stratified run.** One configuration — inherited rule, inherited model, one pre-specified contrast — scored on CLV and ROI. It **passed its pre-registered test**, which is the first time anything here has. `docs/H1_RESULT.md` carries the tables and the diagnostics, and the diagnostics matter more than the verdict.

**Eight other things ran that session and none of them counts.** Named individually rather than summarised, because "nothing else was evaluated" is indistinguishable from having skipped the step:

| what ran | why it does not count |
|---|---|
| `h1_coverage_probe.py` | counts price coverage and computes the bet floor's power arithmetic. Fits nothing, scores nothing |
| `h1_panel_check.py` | plumbing pre-flight driven by **random** probabilities. Prints no verdict; its numbers are noise by construction |
| `h1_diagnostics.py` — ordered logit arm | a deliberate null. The question is "would a dumb model do this too?", not "is this model better" |
| `h1_diagnostics.py` — anti-model arm | bets the **minimum**-EV outcome. Nobody hopes it wins; it exists to see the effect invert |
| `h1_diagnostics.py` — random-in-band arm | random selection. A control in the same sense as `random_bet_null` |
| `h1_holdout_coverage.py` | column presence only. No model |
| `h1_holdout_tiers.py` | **re-slices the settled Phase 6 run by tier.** Same configuration, same holdout, a question that run never asked — a new cut of an existing result, not a new configuration |
| `h1_odds_matched_null.py`, `h1_tier_nulls.py`, `clv_null_calibration.py` | prices only, no model fitted. They measure the **market**, like the exchange-vs-Pinnacle benchmark already ruled on above |

The line that separates them from the H1 run itself: **none searched for edge.** A control whose result nobody is hoping for cannot widen a search, and counting it would overstate the search exactly as surely as omitting a real evaluation would understate it.

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
| H1 tier-stratified CLV run | 1 | lower (tiers 3–5) vs upper (tiers 1–2), 2015-16 → 2024-25 |

**Reconciled 2026-08-17 (second session): the count stays at 47.** Nothing was evaluated in the registry's sense. Three things ran that could be mistaken for evaluations, so each is named rather than left to inference:

1. **A forward dry run** trained the pre-registered `NetConfig` over three seeds and predicted 127 fixtures. Same configuration, and the fixtures were unplayed — no RPS, no PnL, nothing scored against an outcome. This is the retrain ruled on below.
2. **A grading dry run** on 400 resolved matches produced CLV and ROI figures across four price columns. The "model" was the **de-vigged exchange close itself** — a deliberate control, not a candidate. It placed zero bets into its own prices, which is the closed-form check that the price join is right. Its CLV of 1.13 is the look-ahead in the test harness announcing itself, exactly as it should.
3. **A benchmark comparison** measured overround and de-vigged RPS for Pinnacle, exchange and Bet365 closing on 16,875 matches. That measures *markets*, not model configurations, and it was forced by a vendor removing a column rather than chosen by looking at results.

None of the three searched for edge, so counting them would overstate the search as surely as omitting a real one understates it.

**⚠️ This count was seeded at ~38 and corrected to 47 on its first reconciliation.** `docs/PREREGISTRATION.md` disclosed 7 tier-shift values where 11 were actually swept, and the tier-2 arms and feature-set variants were never counted at all. The pre-registration is a frozen record and is deliberately **not** edited retroactively — this registry carries the live number, and the discrepancy is recorded here rather than smoothed over.

That correction is the step working as designed. The failure mode is never a dishonest entry; it is a configuration tried casually, found uninteresting, and never written down.

This number goes up for **every** configuration scored. Distinguishing a 2% edge from zero needs ~45,000 bets; the count is what stops a widening search quietly manufacturing one.

**Ruling, 2026-08-17: a scheduled forward retrain does NOT increment the count.** `src/forward.py` retrains on every run, but it retrains the *same* configuration — the `NetConfig` defaults fixed by `docs/PREREGISTRATION.md`, three seeds, `ALL_FEATURES` — on data that has grown by a few hundred matches. Nothing is being chosen, and a count that ticked up twice a week would stop meaning anything within a month.

What *would* increment it: changing the seed count, the feature set, the architecture, the betting rule, or the price column, in the forward path or anywhere else. If a run has to be made cheaper on CI, **use a bigger runner rather than fewer seeds** — the runner is free of consequence, the seed count is a configuration change.

**The price-ladder change is also not a configuration.** Moving from Pinnacle close to exchange close is forced by the vendor removing a column, not chosen by looking at results, and the replacement was measured as no softer before being adopted. It is a documented benchmark change and belongs in the record rather than the count.

## Status board

| ID | hypothesis | status | cost to first answer | file | notes |
|---|---|---|---|---|---|
| H1 | Lower-division football is less efficiently priced | **`settled` — SUPPORTED** (2026-08-17) | run, ~50 min | `H1-lower-division-inefficiency.md` → `docs/H1_RESULT.md` | 🟢 lower stratum CLV **1.0083**, 52.53% shortened, 9,920 bets, p<0.0001. **But** tier 5 (the thinnest market of all) shows nothing, ROI is worse than random, and the data was already seen. Supported by its test, unexplained by its mechanism |
| H2 | Derived markets (**O/U 2.5 only** — see below) | `proposed` | zero new data for O/U; new ingest for the rest | `H2-derived-markets.md` | reuses the Poisson head, which has never been graded for betting. Blocked on the n-outcome work |
| H3 | Line movement is predictable (pre-close → close) | **`pre-registered`** (2026-08-17) | zero new data; holdout 2024/25 | `H3-line-movement.md` | ⚠️ **narrowed by H1's diagnostics**, which already showed a match model's disagreement predicts movement. H3 now tests whether fitting the label *directly* beats that incidental signal — a smaller and honest question |
| H4 | Betfair AU/NZ niche leagues (AFL, NRL, NBL, BBL) | `proposed` | new ingest, **source unverified** | `H4-exchange-niche-leagues.md` | best capacity-adjusted option — the exchange does not ban winners. The only one with a genuinely clean holdout |

**H2's scope was overstated on this board.** Only O/U 2.5 is in the football-data feed; BTTS and correct score are not there at all. **H3's cost was overstated too** — the pre-close and closing legs are both already in `matches.parquet`, so the free version needs no ingest.

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
