# Match Predictor Pipeline

This repo builds player-level expected goals, aggregates to team totals, and
uses Poisson outcome probabilities for match predictions.

## Data flow

1) Fetch fixtures (multi-league + multi-season supported)
```
python3 scripts/batch_fetch.py
```

2) Fetch odds (optional, for backtesting)
```
python3 scripts/fetch_fixture_odds.py
```

3) Build player-level dataset (rolling window)
```
python3 scripts/build_player_match_dataset.py
```

4) Train / evaluate
```
python3 scripts/train_player_goals.py
```

5) Backtest betting strategies
```
python3 scripts/backtest_betting.py
```

6) Predict a fixture
```
python3 scripts/predict_match.py --fixture-id <ID>
```

7) ROI summary by league/season (uses test split)
```
MIN_CONFIDENCE=0.45 SPORTMONKS_BOOKMAKER_ID=23 python3 scripts/roi_summary.py
```

## Defaults

- Rolling window: 5 matches (`SPORTMONKS_HISTORY_WINDOW`)
- Leagues: set `SPORTMONKS_LEAGUE_IDS` to pull multiple leagues
- Seasons: `SPORTMONKS_SEASONS_PER_LEAGUE=2` (latest two seasons per league)
- Odds: `SPORTMONKS_ODDS_FEED=pre-match`
- Bookmaker (optional): `SPORTMONKS_BOOKMAKER_ID=23` for Unibet
- Default confidence threshold: 0.45 (override with `--min-confidence`)

## Notes

- All rolling features are computed from matches that occurred before each fixture.
- Player-level model is log-linear ridge on goals (per player per match).
- Team totals are minutes-weighted; match outcome uses Poisson goal model.
