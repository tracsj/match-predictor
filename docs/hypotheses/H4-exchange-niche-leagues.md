# H4 — Niche Australasian leagues on the exchange are priced less efficiently

**Status:** `proposed`
**Opened:** 2026-08-17 · **Settled:** —

> Not a pre-registration. See the note at the top of `H1-lower-division-inefficiency.md`.

## The claim

AFL, NRL, NBL and BBL are major sports in one medium-sized country and minor
curiosities everywhere else. The modelling attention aimed at them is a small
fraction of what lands on European football, while the Betfair Exchange still
provides a liquid, low-margin market in them — so an inefficiency, if one exists
anywhere, is likelier here than in a market a thousand quants are staring at.

**What would falsify it.** CLV at or below 1.0 against the exchange close in each
competition.

**Why it ranks best on capacity, which is the criterion the others fail.** The
exchange does not ban winners. Kaunitz et al. made real money in soft books and
were limited to $1.25 stakes within months, and every football hypothesis here
runs into some version of that. An exchange edge is limited by liquidity, which
is a number you can measure and plan around, rather than by a risk desk's
opinion of you, which is not.

## What makes this different from the other three

H1, H2 and H3 all reuse a corpus that already exists. H4 requires a new sport, a
new ingest and a new outcome space, and the harness has to change before it can
run at all:

- **Two outcomes, not three.** NBL and BBL effectively cannot draw; AFL and NRL
  can but very rarely. That is a *simplification* of the n-outcome
  generalisation H2 forces, not an additional burden — but `metrics.py`,
  `net.py`, `baselines.py` and `betting.py` still assume three, and a
  two-outcome bug looks exactly like a two-outcome edge.
- **The features do not transfer.** Elo transfers. pi-ratings are defined on
  goal difference and would need re-thinking for a sport scoring 80–100 points a
  game, where margin distributions are nothing like football's.
- **Sample size is the real constraint.** An AFL season is 200-odd matches
  against football's 7,800. Even a decade across all four competitions is a few
  thousand rows, which is far below the 100k-plus at which deep models became
  competitive on the football task. **A neural approach is probably wrong here
  and a rating-plus-regression approach probably right** — worth deciding
  deliberately rather than by reflex, since reaching for the existing net is the
  path of least resistance and likely the wrong one.

## Data provenance

| | |
|---|---|
| source | **not yet established** |
| coverage | unknown |
| odds timing | must include a dated closing price, or H4 cannot support a CLV claim at all and should not be run |
| known gaps | unknown |

**Candidate to verify, not a finding:** `aussportsbetting.com` publishes free
spreadsheets of Australian sports odds that are widely cited as carrying opening
and closing prices. **This has not been checked** — not the coverage, not the
seasons, not whether the closing prices are dated, not the licence. Establishing
it is the first task if H4 is picked up, and the result belongs in
`docs/research/00-measured-facts.md` with the command that established it.

Betfair's own historical data is the other candidate and is the better one if
available, since the exchange close is the benchmark anyway.

**H4 must not proceed on an undated odds column.** That is the one disqualifying
condition, and it is worth stating before anyone has sunk a day into an ingest.

## Holdout

Genuinely clean, which is H4's real advantage and the reason it may be worth its
cost. No configuration in this project has touched Australasian sport. A proper
locked holdout is available here in a way it no longer is for H1, H2 or H3 — so
if H4 runs, the holdout should be locked at the start and the mistake recorded in
`docs/PREREGISTRATION.md` ("the plan said keep 2–3 seasons locked; that did not
survive") should not be repeated.

## Where it could be staked

The best of the four, and the reason H4 exists.

- Exchange, so no closure risk for winning.
- Liquidity is the binding constraint and it is measurable in advance: AFL and
  NRL match-odds markets are reasonably traded, NBL and BBL thinner. **Measure
  matched volume before modelling**, because a strategy that cannot be filled is
  a hobby, and the volume number is cheap to get relative to the ingest.
- Commission of 2–5% on net winnings applies and must be in the ROI, not a
  footnote to it.

## Expected outcome

**No edge.** The programme's prior, and the sweep that ruled out tennis and golf
found the same result reported independently for both — Wilkens (2021) beat no
odds-implied forecast across 15 architectures, Kovalchik (2016) found the
bookmaker consensus beat 11 published models, and Data Golf reports −0.92% ROI
for its own model. "Less studied" was the argument for those two as well.

The specific counter-argument to H4's own premise: Australian sports are heavily
bet *within Australia*, by a domestic industry with its own quants. "Niche" is
a description of attention from outside, and the price is set by whoever is
actually there.

## Result

*(filled after the run, whatever it says)*
