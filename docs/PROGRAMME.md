# The programme — hypothesis registry

**Live working doc.** Updated whenever a hypothesis changes status, and reconciled at every session wrap-up.

---

## Where we are — read this first

**Last session: 2026-08-17 (second session that day).** Forward validation is built and the four hypothesis files exist. 299 tests green at close.

**Done**
- Football v2 built, measured, and its betting question answered — see the graveyard below.
- `.claude/` setup: project CLAUDE.md, settings, `/wrap-up`. This repo had none before.
- This registry, seeded with an honest configuration count.
- **Forward validation, end to end.** `src/data/fixtures.py` (unplayed-fixture ingest), the `unplayed` contract in `src/features/horizon.py`, `src/forward.py` (predict and record), `src/grade.py` (grade and write the ledger), `src/refresh.py`, and `.github/workflows/forecast.yml` on a Tuesday/Friday cron. Dry-run measured at **4m05s locally** for the full path including three training seeds.
- **H1–H4 files written**, in `docs/hypotheses/`. All `proposed`; none pre-registered.

**The finding that changed the programme.** **football-data dropped Pinnacle entirely in 2026/27** — the columns are absent from the schema, not empty, and the last populated `psch` anywhere is 2026-01-14. The replacement is the **Betfair Exchange close**, and measuring the two on the 16,875 matches carrying both showed it is **not a downgrade**: equally accurate (de-vigged RPS 0.20404 vs 0.20408) on a quarter of the margin (overround 1.0089 vs 1.0389), with prices 3.9% longer. Beating it is at least as hard. Full detail and the commands in `docs/research/00-measured-facts.md`.

**The workflow is live and verified on a real runner.** Repo is `tracsj/match-predictor`, **private**, default branch `master`. Run `32068718466` went green end to end on 2026-08-17 and committed `docs/FORWARD_LEDGER.md` by itself. Measured, cold cache:

| step | cold | note |
|---|---|---|
| setup + `uv sync` | ~40s | |
| **refresh** | **271s** | 724 files, **0 errors** — football-data does not block runner IPs, which was a real open risk. Warm cache re-fetches only ~38 files |
| 302 tests | ~110s | green on a fresh runner |
| self-test guard | ~2s | the four checks genuinely ran on CI |
| predict | 3s | *no fixtures in the window* — this run did not exercise training |
| grade + commit | ~7s | pushed on its own |
| **total** | **8m14s** | |

The CI corpus reproduced local exactly: 296,218 matches, 2026-27 at 253 matches across 18 divisions. **Training on a runner is still unmeasured**, since the horizon was empty — 4m locally, so budget 12–20 min against the 120-minute timeout.

**Four bugs only a real runner would have found**, all of which failed silently or would have:
1. `refresh_current()` wrote `_missing.json` before the directory existed — `data/` is gitignored, so every checkout starts empty.
2. `--no-refresh` disabled the *fixtures* fetch as well as the corpus one, and `src.refresh` never fetches `fixtures.csv`.
3. `actions/checkout` defaults to a **depth-1 shallow clone**, on which `git log` finds nothing, so every prediction would have read as "uncommitted" and the ledger would have graded **nothing while exiting zero**. `fetch-depth: 0` is mandatory; the grader now refuses on a shallow clone.
4. `astral-sh/setup-uv` publishes major tags only to `v7` while its release is `v10.0.1`, so `@v10` does not resolve. Pinned exactly.

**Next, in order**

1. **Check Tuesday's scheduled run (2026-08-18, 13:15 UTC) — still outstanding.** The 2026-08-17 session could not do this: the cron had not fired yet, and the four runs `gh run list` showed were all that evening's manual dispatches. It is the first run with a non-empty horizon, so the first to exercise training on a runner and the first to write a real `predictions/YYYY-MM-DD.csv`. Concretely: `gh run list --workflow=forecast.yml --limit 3`, then `gh run view <id> --log`, and confirm three things — the predict step reports a fixture count rather than "no upcoming fixtures", training completed inside the 120-minute timeout (unmeasured on 2 vCPUs; 4m05s locally), and `predictions/` gained a file. If it timed out, raise the runner rather than cutting seeds — see the registry ruling below.

2. **Re-analyse Phase 6's CLV against a measured null.** Pre-registered, with its bet population re-derived rather than approximated by the unconditional drift. The highest-value item on this list, because it may overturn a *published* reading of the founding study — in the direction of the model being better than recorded, which is exactly why it needs a pre-registration rather than an afternoon.

3. **H3 in its free form, now much better equipped.** It targets line movement directly, which is what the H1 diagnostics say the model is actually doing. It inherits the drift measurement, the measured-null requirement, and the fact that no exchange pre-close exists historically.

4. **Read the ledger's "Schedule coverage" table after a few weeks.** There is a known structural gap and it is measured rather than assumed. The earliest observed Friday kickoff is **17:30 UK**, while the Friday run fires 18:15 UK under BST and takes ~20 minutes — so Friday early kickoffs can only ever be reached from *Tuesday's* snapshot, and whether that snapshot spans to Friday is not something one observation could settle. **No cron change fixes this**: the feed has exactly two states a week. If the table shows Friday-evening misses accumulating, the options are a cached model fast enough to fit between the 17:00 upload and a 17:30 kickoff, or accepting the gap and saying so.

5. **n-outcome harness generalisation** + move sport-specific code under `src/sports/football/`. ~23 lines across `metrics.py`, `net.py`, `baselines.py`, `betting.py`; `devig.py` and `split.py` already generalise. H2 forces this first, H4 needs the two-outcome case.

6. **H2 pre-registered, then run.**

7. **H1b, only if it earns a slot**: the 2012-15 seasons, never graded and never used for model selection. Reaching them requires relaxing `run_walk_forward`'s hardcoded `min_train_seasons=3`, which is a protocol change and belongs in a pre-registration rather than an improvisation after seeing a positive.

**Done 2026-08-17 (third session)** — H1 pre-registered and run. It **passed**, the first thing in this programme to do so, and the diagnostics then showed its test had been run against the wrong null. Write-up in `docs/H1_RESULT.md`; the correction is above.

**Open threads worth knowing**
- `uv run python -c` is denied in `.claude/settings.json` on purpose, to push analysis into re-runnable files. It cost three extra steps this session and was worth it each time — the scripts are re-runnable.
- The `no-script-file-mutation.py` hook was **not** adopted here. Its promotion review is dated on-or-after 2026-09-07 and requires re-running corpus validation against this repo's own transcripts. Evidence for that review: the 2026-08-17 sessions used script-based file edits heavily.
- **`HxG`/`AxG` arrived in 2026/27** and the loader parses them away. Per-division, not universal (present B1, N1; absent EC). A genuinely new feature source and the first new one in a while.
- **Eight 2026/27 divisions are still 404 upstream** (D1, E1, E2, F1, G1, I1, I2, T1) and are re-memoised in `_missing.json` on every refresh until they publish. `refresh_current()` purges and retries them each run, so this heals itself — but a division silently missing from the ledger for weeks would look exactly like this, so check the refresh line in the workflow log.
- `new_league_fixtures.csv` covers 14 of the 16 extra countries and is **not** read yet, so the forward path serves the 22 main divisions only.

**⚠️ The correction that matters most, 2026-08-17 (third session): every CLV number this project has ever reported was tested against the wrong null.**

`docs/PREREGISTRATION.md` and the H1 pre-registration both test closing-line value against a mean ratio of 1.0 and a shortening rate of 50%. That assumes the pre-close and the close are on average the same price. **Measured, they are not** — Pinnacle's overround tightens toward kickoff in every season from 2015-16, so prices lengthen by default and a randomly chosen band-eligible selection shortens only **45–48%** of the time. The odds-matched null puts 0.5 outside both strata's intervals, and matched and unmatched nulls agree to within 0.002, so it is not an odds-mix artifact. Instruments: `scripts/clv_null_calibration.py`, `scripts/h1_odds_matched_null.py`, `scripts/h1_tier_nulls.py`.

Consequences, in order of how much they change:

1. **Phase 6's headline reading is in doubt.** Its 42.4% shortened sits **+3.51pp above** the 38.89% drift on its own population, so its selections were on the *right* side of the market's movement, not the wrong one. `docs/PHASE6_RESULT.md` carries a dated addendum. **Its ROI tables and its "the rule lost money" conclusion are untouched.** Re-deriving its bet population against a measured null is a pre-registered re-analysis and was deliberately not done tonight.
2. **`CLAUDE.md`'s summary of the finished result needs one clause revisited** — *"closing-line value at 0.9952, meaning its selections sat on the wrong side of the market's own movement."* The number stands; the "meaning" clause is the one in doubt. The rest of that paragraph — the RPS figures, the losses in every price column, and "do not treat 0.2076 as a near-miss to be closed with better features" — is unaffected. Left for the wrap-up audit rather than edited mid-session.
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
| H3 | Line movement is predictable (pre-close → close) | `proposed` | **zero new data** for the free form | `H3-line-movement.md` | strongest published evidence; targets the market, not the match. CLV is the objective rather than a proxy, so it converges fast |
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
