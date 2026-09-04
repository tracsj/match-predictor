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
2026-08-25.csv     5 2026-08-25 15:15:44 2026-08-25 20:00:00          ok
2026-08-29.csv   164 2026-08-29 02:12:11 2026-08-29 12:00:00          ok
2026-09-01.csv    48 2026-09-01 18:33:38 2026-09-01 19:45:00          ok
2026-09-04.csv   178                 NaT 2026-09-05 12:00:00 uncommitted
```

## Coverage

- Predictions committed: **385**
- Results landed: **381**
- Awaiting result: **4**
- Divisions: **22**, kickoffs 2026-08-19 20:00:00 → 2026-09-03 20:00:00

## Schedule coverage

Every corpus fixture in the divisions and date span we have predicted, by
kickoff slot, and whether a prediction exists for it. **A miss here is not a
bad prediction — it is no prediction at all**, which is the failure mode that
does not announce itself.

- Fixtures in scope: **429** across 22 divisions
- Predicted: **381**
- Missed: **48**

Worst slots first. Friday early kickoffs are the known suspect.

```
weekday  hour  fixtures  predicted  missed
    Fri    19        35         10      25
    Mon    15        11          0      11
    Fri    20         8          3       5
    Fri    17         2          0       2
    Thu    20         4          2       2
    Fri    18         1          0       1
    Sun    19        15         14       1
    Wed    20         5          4       1
    Mon    18         4          4       0
    Mon    17         3          3       0
    Sat    13         5          5       0
    Sat    14         5          5       0
```

## Forecast quality

```
                         model   n    rps  log_loss  brier    ece  accuracy
             the net (forward) 381 0.2075    0.9992 0.5958 0.0412    0.5066
market, exchange close (n=381) 381 0.2042    0.9884 0.5888 0.0504    0.5092
          the net, same subset 381 0.2075    0.9992 0.5958 0.0412    0.5066
```

The market band to sanity-check against is RPS 0.19–0.21. Outside it, suspect
the pipeline before the model.

## Closing-line value

Bet at the pre-close exchange price recorded at prediction time; grade against
the exchange close of the same selection.

**Read `pct_shortened` against `null_rate`, never against 50%.** The overround
tightens toward kickoff, so prices lengthen by default and a selection picked at
random inside the rule's odds band shortens less than half the time. `null_rate`
is that rate, measured on these same settled rows over every cell the rule could
legally have bet. Testing against 0.5 instead is what put a withdrawn reading
into `docs/PHASE6_RESULT.md`.

The null is measured here rather than imported. Pinnacle's pre-close was a
mature price; this one is a Tuesday/Friday snapshot taken a day out, and the
overround line below is what makes the difference legible rather than surprising.

```
    taken_at  n_bets  n_days  mean_ratio  pct_shortened  null_rate  excess_pp  two_prop_p  day_clustered_p
exchange_pre     204      11      0.9858         0.4216     0.3486     7.2973      0.0488              NaN
```

**The mechanism, on these rows.** The pre-close book sums to
1.0419 and the close to 1.0145, tightening in 87% of 375 rows.
That is where the default lengthening comes from, and it is measured rather
than assumed.

**Treat this as an early number, not a finding.** The null is itself an
estimate, from **961** eligible cells, and a binomial against
it would treat it as exact. `two_prop_p` does not, and accounts for that.

**`day_clustered_p` is the one to read, and it needs matchdays to read.**
Bets sharing a matchday share news and market-wide moves, so they are not
independent draws — block-bootstrapping days is what `bootstrap_ci` has
always done for ROI and what the shortening test did not do until
2026-08-27. On the two settled results that correction decided both:
Phase 6 fell from p 0.018 to 0.154, and H1's out-of-sample lower stratum
from 0.011 to 0.118. It is blank above until the forward record spans 20
matchdays, because a bootstrap over a handful of days estimates the error
downward and returns a p smaller than the uncorrected one — the correction
appearing to strengthen the result is the correction failing.

The graded bets are also a subset of the null's cells, which dilutes the
null toward the model and makes the comparison conservative. Nothing here
clears the p < 0.01 this project requires before claiming an edge.

## ROI, led by the sharpest price

```
       price_set  n_eligible  n_bets    roi  roi_lo  roi_hi  hit_rate  avg_odds                                  note
  exchange_close         381     247 0.1111 -0.0055  0.2785    0.3603    3.3887  the sharpest price still in the feed
      b365_close         381     168 0.0949 -0.0607  0.3538    0.3512    3.4049 a book you could hold an account with
market_max_close         381     215 0.0834 -0.0355  0.1876    0.3442    3.4007                      optimistic bound
market_avg_close         381     163 0.0582 -0.1297  0.2629    0.3374    3.4103        softer benchmark, for coverage
```

Rule: pre-registered: ev>=0.05, odds 1.5-5.0: bet the max-EV outcome when EV >= +0.050 and price in [1.5, 5.0] — fixed by `docs/PREREGISTRATION.md`.

A result positive only in the market-maximum column is price shopping rather
than forecasting, and is the strategy that got Kaunitz et al. stake-limited
into uselessness.

