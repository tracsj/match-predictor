# match-predictor

Pre-match football outcome prediction with a neural network, evaluated honestly against closing betting odds.

## Where things are

```
v1/                 the original build (Dec 2025), frozen. See "v1" below.
src/
  data/             ingest: football-data.co.uk CSVs, SportMonks API, team-id mapping
  features/         pi-ratings, Elo, opponent-adjusted rolling form, player encoders
  models/           baselines (Dixon-Coles, CatBoost), the network, calibration
  eval/             walk-forward splitter, Shin de-vig, RPS/log-loss/ECE, betting sim
docs/research/      what was measured and what the literature says. Read before modelling.
data/               gitignored. Raw cache + derived parquet.
models/             gitignored. Checkpoints.
```

## Results

Walk-forward, 45,629 out-of-sample matches across 22 divisions and 9 seasons:

| model | RPS | log loss |
|---|---|---|
| market (de-vigged Pinnacle close) | **0.20291** | 0.99815 |
| the network (GRU sequence branch) | 0.20765 | 1.01333 |
| ordered logit on 49 features | 0.20789 | 1.01440 |
| CatBoost / ordered logit on ratings only | 0.2090 | 1.0178 |
| base rate | 0.2292 | 1.0771 |
| uniform | 0.2340 | 1.0986 |

The network beats the best non-neural baseline (t = +2.50) and does not come
close to the market (t = +20). Three write-ups carry the detail:

- **`docs/PHASE6_RESULT.md`** — would it have made money? No. The
  pre-registered rule, run once on an untouched season, loses in every price
  column and produces closing-line value *below* 1.0.
- **`docs/TIER2_RESULT.md`** — what is a starting XI worth? No measurable
  effect, from an experiment that could only have detected a large one. Do not
  upgrade the SportMonks plan.
- **`docs/PREREGISTRATION.md`** — the betting rule, prices and holdout, fixed
  before any model PnL existed.

The single most useful finding for anyone repeating this: **the sequence
branch is the only neural component that beat the baseline.** Team embeddings,
league embeddings, extra trunk members and a wider hidden layer all either hurt
or did nothing. Rolling *means* over a team's last ten matches are order-blind;
a GRU over the same window is not, and that difference is worth more than every
other architectural choice combined.

## Read these first

- **`docs/research/00-measured-facts.md`** — what the SportMonks key actually reaches, what football-data.co.uk actually contains, and the exact commands that established each. Every claim here was probed, not assumed.
- **`docs/research/01-neural-nets-for-match-prediction.md`** — what wins on this task and what does not, with RPS numbers tagged to their datasets.
- **`docs/research/02-betting-evaluation-and-odds-data.md`** — how to build a backtest that would tell you the truth if the model were bad.

All three are dated 2026-08-15. API entitlements, vendor pricing and dataset column sets drift, so re-check before quoting.

## The shape of the thing

Two data tiers doing two different jobs, overlapping on the two leagues where we have both:

- **Tier 1, breadth** — football-data.co.uk, 22 divisions, ~7,800 matches a season, Pinnacle closing odds from 2012/13 (~109k matches). Team-level box-score features only. This is what the network trains on, because deep models only become competitive on this task at 100k+ matches across many leagues.
- **Tier 2, depth** — SportMonks free plan, Danish Superliga and Scottish Premiership, 2019/20 onward (~3,000 matches), with full starting XIs and 36–41 statistics per player. Both leagues also sit in Tier 1 with the same closing odds, so the same network can be trained on identical fixtures with and without a player encoder, and the difference measured.

Three things are always on the scoreboard: de-vigged Pinnacle closing odds (the ceiling), a tuned Dixon–Coles model (the baseline), and the network.

## Setup

```bash
uv sync
uv run python -c "import torch; print(torch.__version__)"
```

Python 3.12, managed by `uv`. `.env` holds `SPORTMONKS_API_TOKEN` and is gitignored.

## v1

The original build takes a player → team → Poisson route: ridge regression on `log1p(player goals)`, scaled by expected minutes, summed to a team λ, converted to W/D/L on a Poisson grid. It still runs — `v1/data` is a relative symlink to `data/`, so the scripts resolve their paths unchanged.

**Its betting numbers should not be trusted.** Five of its scripts assume the first-listed team is the home team; measured across 640 fixtures, that holds 63.6% of the time, so roughly a third of simulated bets were graded against the wrong side's price. `v1/scripts/build_web_bundle.py` is the exception — it resolves home and away correctly, which makes `v1/web/data.json` and the browser dashboard the only reliable betting output in v1. v2 gets a regression test for exactly this.
