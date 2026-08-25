# Forward ledger

Predictions committed before kickoff, graded as results landed. Rewritten from
`predictions/*.csv` on every run, so nothing here is accumulated by hand.

**Read the CLV column first.** Distinguishing a 2% edge from zero needs roughly
43,632 bets at average odds 3.2; CLV converges about a hundred times
faster and is what correctly said stop in the backtest.

**The benchmark changed, and it is not softer.** The settled study graded against
Pinnacle closing, which left the feed in 2026/27. The Betfair Exchange close
replaces it, and on the 16,875 matches carrying both it is an equally accurate
estimate of the truth — de-vigged RPS 0.20404 against Pinnacle's 0.20408 — on a
quarter of the margin, 1.0089 against 1.0389. Its prices run 3.9% longer, so
beating it is if anything harder. CLV below is a like-for-like exchange ratio.

**Exchange ROI below is pre-commission.** 2–5% of net winnings is not deducted,
and that would absorb most of the price advantage. CLV is immune to it, since
both legs are exchange prices and the commission cancels in the ratio.

## Provenance

Each file's commit time against the earliest kickoff it predicts. A file
committed at or after any of its own kickoffs is not graded at all.

**The newest file normally shows `uncommitted` here, and that is correct.**
Grading runs before the commit step, so the file this run just wrote is still
untracked while this table is being built. It is committed moments later, in
the same workflow step that commits this ledger, and grades normally from the
next run onward. Nothing needs fixing.

```
          file  rows        committed_at       first_kickoff      status
2026-08-18.csv     3 2026-08-18 15:09:29 2026-08-19 20:00:00          ok
2026-08-21.csv   165 2026-08-21 18:56:49 2026-08-21 19:00:00          ok
2026-08-25.csv     5                 NaT 2026-08-25 20:00:00 uncommitted
```

## Coverage

- Predictions committed: **168**
- Results landed: **164**
- Awaiting result: **4**
- Divisions: **20**, kickoffs 2026-08-19 20:00:00 → 2026-08-24 20:30:00

## Schedule coverage

Every corpus fixture in the divisions and date span we have predicted, by
kickoff slot, and whether a prediction exists for it. **A miss here is not a
bad prediction — it is no prediction at all**, which is the failure mode that
does not announce itself.

- Fixtures in scope: **168** across 20 divisions
- Predicted: **164**
- Missed: **4**

Worst slots first. Friday early kickoffs are the known suspect.

```
weekday  hour  fixtures  predicted  missed
    Thu    20         2          0       2
    Sun    19         7          6       1
    Wed    20         1          0       1
    Mon    17         1          1       0
    Mon    19         3          3       0
    Mon    20         4          4       0
    Sat    12        21         21       0
    Mon    18         2          2       0
    Fri    19        10         10       0
    Fri    20         3          3       0
    Sat    16         3          3       0
    Sat    15        46         46       0
```

## Forecast quality

```
                         model   n    rps  log_loss  brier    ece  accuracy
             the net (forward) 164 0.2129    1.0144 0.6045 0.0586    0.4939
market, exchange close (n=164) 164 0.2133    1.0139 0.6041 0.0639    0.4634
          the net, same subset 164 0.2129    1.0144 0.6045 0.0586    0.4939
```

The market band to sanity-check against is RPS 0.19–0.21. Outside it, suspect
the pipeline before the model.

## Closing-line value

Bet at the pre-close exchange price recorded at prediction time; grade against
the exchange close of the same selection. A ratio at or below 1.0 means the
selections sat on the wrong side of the market's own movement.

```
    taken_at  n_bets  mean_ratio  pct_shortened  binom_p
exchange_pre      82      0.9937         0.4512   0.4397
```

## ROI, led by the sharpest price

```
       price_set  n_eligible  n_bets    roi  roi_lo  roi_hi  hit_rate  avg_odds                                  note
  exchange_close         164     100 0.1197 -0.1376  0.2569    0.3700    3.3093  the sharpest price still in the feed
      b365_close         164      65 0.3083  0.0409  0.4540    0.4154    3.3474 a book you could hold an account with
market_max_close         164      85 0.1373 -0.1413  0.2019    0.3647    3.3307                      optimistic bound
market_avg_close         164      66 0.2185 -0.2529  0.3334    0.3788    3.3814        softer benchmark, for coverage
```

Rule: pre-registered: ev>=0.05, odds 1.5-5.0: bet the max-EV outcome when EV >= +0.050 and price in [1.5, 5.0] — fixed by `docs/PREREGISTRATION.md`.

A result positive only in the market-maximum column is price shopping rather
than forecasting, and is the strategy that got Kaunitz et al. stake-limited
into uselessness.

