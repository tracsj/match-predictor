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
2026-08-29.csv   164                 NaT 2026-08-29 12:00:00 uncommitted
```

## Coverage

- Predictions committed: **173**
- Results landed: **169**
- Awaiting result: **4**
- Divisions: **20**, kickoffs 2026-08-19 20:00:00 → 2026-08-27 20:00:00

## Schedule coverage

Every corpus fixture in the divisions and date span we have predicted, by
kickoff slot, and whether a prediction exists for it. **A miss here is not a
bad prediction — it is no prediction at all**, which is the failure mode that
does not announce itself.

- Fixtures in scope: **173** across 20 divisions
- Predicted: **169**
- Missed: **4**

Worst slots first. Friday early kickoffs are the known suspect.

```
weekday  hour  fixtures  predicted  missed
    Thu    20         3          1       2
    Sun    19         7          6       1
    Wed    20         2          1       1
    Fri    19        10         10       0
    Mon    19         3          3       0
    Mon    20         4          4       0
    Sat    12        21         21       0
    Sat    13         2          2       0
    Sat    15        46         46       0
    Fri    20         3          3       0
    Mon    17         1          1       0
    Mon    18         2          2       0
```

## Forecast quality

```
                         model   n    rps  log_loss  brier    ece  accuracy
             the net (forward) 169 0.2110    1.0039 0.5973 0.0632    0.5030
market, exchange close (n=169) 169 0.2121    1.0051 0.5978 0.0614    0.4675
          the net, same subset 169 0.2110    1.0039 0.5973 0.0632    0.5030
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
exchange_pre      84       5      0.9915         0.4405     0.3181    12.2404      0.0306              NaN
```

**The mechanism, on these rows.** The pre-close book sums to
1.0603 and the close to 1.0213, tightening in 88% of 167 rows.
That is where the default lengthening comes from, and it is measured rather
than assumed.

**Treat this as an early number, not a finding.** The null is itself an
estimate, from **415** eligible cells, and a binomial against
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
  exchange_close         169     102 0.1468 -0.1150  0.3376    0.3725    3.3395  the sharpest price still in the feed
      b365_close         169      66 0.3567  0.0232  0.6389    0.4242    3.3648 a book you could hold an account with
market_max_close         169      87 0.1686 -0.0387  0.3276    0.3678    3.3614                      optimistic bound
market_avg_close         169      67 0.2694 -0.1578  0.5186    0.3881    3.4000        softer benchmark, for coverage
```

Rule: pre-registered: ev>=0.05, odds 1.5-5.0: bet the max-EV outcome when EV >= +0.050 and price in [1.5, 5.0] — fixed by `docs/PREREGISTRATION.md`.

A result positive only in the market-maximum column is price shopping rather
than forecasting, and is the strategy that got Kaunitz et al. stake-limited
into uselessness.

