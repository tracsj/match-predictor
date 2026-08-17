# H3 — The direction of line movement is predictable

**Status:** `proposed`
**Opened:** 2026-08-17 · **Settled:** —

> Not a pre-registration. See the note at the top of `H1-lower-division-inefficiency.md`.

## The claim

Stop trying to predict the match and predict the *market* instead. If the
direction a price moves between the pre-close snapshot and the close can be
forecast from information available at the snapshot, then betting the side the
market is about to move toward captures closing-line value without ever holding
a better opinion about the football.

**Why this is the strongest of the four.** Every other hypothesis measures CLV as
a *proxy* for edge. Here CLV is the objective itself, which removes the whole
problem of an under-powered ROI sample: the settled study needed ~43,600 bets to
separate a 2% edge from zero, and CLV converges roughly a hundred times faster.
A hypothesis whose target metric is the fast-converging one can be settled with
the data already on disk.

**What would falsify it.** No better-than-chance prediction of movement
direction, or a directional accuracy that exists but is smaller than the spread
it would have to cross.

## The rule

Everything here is open — none of it inherits, because this is not a
match-outcome model and `BetRule` does not describe it.

1. **The target.** Sign of `log(close / pre_close)` for a chosen outcome, or a
   three-way categorical. The sign formulation is cleaner and should probably
   win.
2. **The features**, and the hard constraint on them: they must be knowable at
   the snapshot. The pre-close price itself is allowed; the close is the label
   and nothing derived from it may appear on the input side. This is the single
   place H3 can silently cheat, and it needs its own leakage assertion in the
   harness rather than care.
3. **What counts as a bet**, since "the price will move" is not a wager. Either
   bet the pre-close whenever predicted movement exceeds a threshold, or state
   plainly that H3 v1 is a *forecasting* result and not a strategy.
4. **Walk-forward split**, reusing `src/eval/split.py` unchanged.

## Data provenance

**The status board records this as "new ingest, free". That is wrong, and it is
the useful correction in this file:** football-data already carries both legs for
the same match, so a first version of H3 costs zero new data.

| | |
|---|---|
| source | football-data.co.uk, already parsed and in `matches.parquet` |
| coverage | Pinnacle `psh/psd/psa` → `psch/pscd/psca`, 2012/13 → Jan 2026, ~54% of the corpus carries the closing leg. Exchange `bfeh` → `bfech` from 2024/25 |
| odds timing | pre-close snapshot (Friday ≤17:00 UK / Tuesday ≤13:00 UK) and closing. Both dated, which is what a CLV claim requires |
| known gaps | Pinnacle gone from 2026/27 entirely; the exchange pair covers 2024/25 onward |

**What the pre-close leg is NOT.** It is not the opening line. football-data's
snapshot is taken one to three days out, so a large part of the move from open to
close has already happened by then. The published evidence that motivates H3 is
mostly about the *full* open-to-close path, so this version tests a shorter and
harder segment of it. A true-opening version would need
`sportsbookreviewsonline` (which 404s without a browser user-agent), and that is
the right second step **if and only if** the free version shows something.

Doing the free version first is the point: it is the cheapest possible answer to
the most promising question, and it cannot be justified to skip it in favour of
an ingest.

## Holdout

The corpus has been used extensively — but never for this label. No model in this
project has ever been fitted to price movement, so `close / pre_close` is an
untouched target even though the rows are not untouched rows. Record it that way,
and keep the final season aside properly this time: H3 is being designed before
any of its numbers exist, which is a luxury the football study did not have.

## Where it could be staked

The honest answer is that a positive H3 is the hardest of the four to convert
into money, and the easiest to convert into a *finding*.

- Capturing CLV requires getting on at the pre-close price, in size, repeatedly.
  A soft book will restrict that quickly; the exchange will not, but exchange
  liquidity one to three days out is much thinner than at the close.
- Steam-chasing is precisely the behaviour books limit fastest, because it is
  the signature of a sharp customer and costs them money on every bet.

So H3 should be framed from the start as: does the market's own movement contain
recoverable structure? Whether that structure is stakeable is a second question,
and pretending otherwise would repeat the mistake the market-maximum column was
invented to catch.

## Expected outcome

**No edge**, per the programme's prior — but with the lowest confidence of the
four, and that is worth writing down in advance so the result cannot be
re-narrated afterwards. Closing lines are efficient largely *because* money moves
them; whether the move is forecastable from public pre-match information is a
genuinely open question, and it is the one place where "the market is efficient"
does not immediately answer it.

## Result

*(filled after the run, whatever it says)*
