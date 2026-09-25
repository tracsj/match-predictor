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
2026-09-04.csv   178 2026-09-04 20:52:58 2026-09-05 12:00:00          ok
2026-09-11.csv   169 2026-09-11 20:57:57 2026-09-12 12:00:00          ok
2026-09-15.csv    29 2026-09-15 18:54:32 2026-09-15 19:00:00          ok
2026-09-18.csv   177 2026-09-18 20:52:36 2026-09-19 12:00:00          ok
2026-09-25.csv    36                 NaT 2026-09-26 13:00:00 uncommitted
```

## Coverage

- Predictions committed: **938**
- Results landed: **932**
- Awaiting result: **6**
- Divisions: **22**, kickoffs 2026-08-19 20:00:00 → 2026-09-20 20:30:00

## Schedule coverage

Every corpus fixture in the divisions and date span we have predicted, by
kickoff slot, and whether a prediction exists for it. **A miss here is not a
bad prediction — it is no prediction at all**, which is the failure mode that
does not announce itself.

- Fixtures in scope: **1,061** across 22 divisions
- Predicted: **932**
- Missed: **129**

Worst slots first. Friday early kickoffs are the known suspect.

```
weekday  hour  fixtures  predicted  missed
    Fri    19        75         10      65
    Fri    20        19          3      16
    Mon    15        11          0      11
    Fri    17         8          0       8
    Fri    18         6          0       6
    Tue    19        49         44       5
    Thu    20         8          4       4
    Wed    20         9          5       4
    Wed    19        18         15       3
    Tue    18         1          0       1
    Wed    17         1          0       1
    Tue    20         5          4       1
```

## Forecast quality

```
                         model   n    rps  log_loss  brier    ece  accuracy
             the net (forward) 932 0.2100    1.0194 0.6107 0.0221    0.4732
market, exchange close (n=932) 932 0.2049    1.0025 0.5997 0.0287    0.4946
          the net, same subset 932 0.2100    1.0194 0.6107 0.0221    0.4732
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
exchange_pre     499      22      0.9809         0.3928     0.3393     5.3500      0.0227           0.0008
```

**The mechanism, on these rows.** The pre-close book sums to
1.0469 and the close to 1.0116, tightening in 87% of 918 rows.
That is where the default lengthening comes from, and it is measured rather
than assumed.

**Treat this as an early number, not a finding.** The null is itself an
estimate, from **2352** eligible cells, and a binomial against
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
       price_set  n_eligible  n_bets     roi  roi_lo  roi_hi  hit_rate  avg_odds                                  note
  exchange_close         932     586 -0.0394 -0.1486  0.0873    0.3038    3.3982  the sharpest price still in the feed
      b365_close         932     399 -0.0706 -0.1949  0.1079    0.2932    3.3521 a book you could hold an account with
market_max_close         932     520 -0.1020 -0.2991  0.0253    0.2846    3.3814                      optimistic bound
market_avg_close         932     384 -0.1141 -0.2739  0.0741    0.2839    3.3385        softer benchmark, for coverage
```

Rule: pre-registered: ev>=0.05, odds 1.5-5.0: bet the max-EV outcome when EV >= +0.050 and price in [1.5, 5.0] — fixed by `docs/PREREGISTRATION.md`.

A result positive only in the market-maximum column is price shopping rather
than forecasting, and is the strategy that got Kaunitz et al. stake-limited
into uselessness.

