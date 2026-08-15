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
