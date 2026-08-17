# H2 — Derived markets are priced less carefully than 1X2

**Status:** `proposed`
**Opened:** 2026-08-17 · **Settled:** —

> Not a pre-registration. See the note at the top of `H1-lower-division-inefficiency.md`.

## The claim

The match-odds market is where the money and the attention are. Over/under 2.5
goals, both-teams-to-score and correct score are derived from the same beliefs
but priced with less care, and the model already produces a full scoreline
distribution rather than three numbers — so it has something to say about them
that the 1X2 head cannot express.

**What would falsify it.** CLV at or below 1.0 in every derived market. A
sharper falsifier is available and worth pre-specifying: if the model's implied
1X2 has no edge and its implied O/U does, the difference has to come from the
*scoreline shape*, so the result should be checked against the possibility that
it is a de-vigging artifact of a two-outcome market rather than a forecast.

**Two-outcome markets are easier to look good in, and that is a trap.** O/U and
BTTS have one fewer degree of freedom and a lower overround, so a naive
comparison flatters them. The vig has to be removed consistently before any
comparison is made, not after.

## The rule

Threshold and staking inherit from `docs/PREREGISTRATION.md`. What is genuinely
new, and open:

1. **The probability source.** The Poisson goals head, which the settled study
   reported alongside the softmax head but explicitly did **not** use as the
   betting signal (`docs/PREREGISTRATION.md`). Using it now is a change of
   signal and must be declared as one — the head has never been graded for
   betting, so this is not inherited ground.
2. **Which markets, fixed in advance.** O/U 2.5 only, or O/U 2.5 plus BTTS.
   Correct score is a ~30-outcome market and needs its own thinking about de-vig
   and about minimum liquidity; it should probably be a separate hypothesis.
3. **De-vig method for a two-outcome market.** Shin is fitted for three
   outcomes here; the two-outcome case needs stating explicitly.

## Data provenance

| | |
|---|---|
| source | football-data.co.uk main files |
| coverage | O/U 2.5 pre-close and closing from roughly 2012/13; already parsed as `b365_o25`, `ps_o25`, `avg_o25`, `b365c_o25`, `psc_o25`, `avgc_o25`, `maxc_o25` in `MAIN_ODDS` |
| odds timing | both pre-close and closing exist, so CLV is well defined |
| known gaps | **BTTS is not in the feed at all** and **correct score is not either** — only O/U 2.5 and Asian handicap. Pinnacle's `psc_o25` dies with the rest of Pinnacle in 2026/27 |

**So H2 as currently written overstates what the data supports.** Of the three
markets named on the status board, exactly one — O/U 2.5 — is available from
this source. BTTS and correct score would need a new ingest, which changes H2's
cost from "zero new data" to something else, and that should be corrected on the
status board rather than discovered mid-run.

## Blocked on

**The n-outcome harness generalisation.** `metrics.py`, `net.py`, `baselines.py`
and `betting.py` assume three outcomes — roughly 23 lines across the four, per
`docs/PROGRAMME.md`. `devig.py` and `split.py` already generalise. H2 forces
that work, and it should be done as its own change with its own tests rather
than smuggled into H2's run, because a two-outcome bug would look exactly like
a two-outcome edge.

## Holdout

Same position as H1: no untouched data remains. The mitigating fact is stronger
here, though — the goals head has never been evaluated for betting in any market
at all, so O/U CLV is genuinely unasked. Record the 47-configuration disclosure
and move the count.

## Where it could be staked

Better than H1. O/U 2.5 on major leagues is a liquid market with real limits,
and it is available on the exchange, which does not ban winners. If an edge
existed here it would be stakeable, which is precisely why it is unlikely to
exist: liquid and heavily-traded is the condition under which prices are
efficient.

## Expected outcome

**No edge.** The specific reason: the Poisson head was measured as a *worse*
forecaster than the softmax head on 1X2 in the settled study. A head that is
worse at the market it was trained against is an odd candidate to be better at a
market derived from the same distribution.

## Result

*(filled after the run, whatever it says)*
