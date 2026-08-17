# What is a starting XI worth? — and should the SportMonks plan be upgraded?

**Run 2026-08-17.** Reproduce with `uv run python -m src.tier2`.

## The question

The two-tier design existed to answer one thing: does knowing the starting XI
predict a football match better than team-level form and ratings alone? If it
does, paying SportMonks for more leagues buys more of that signal. If it does
not, the free tier is enough and the money goes nowhere.

The literature could not answer it. The research sweep found **no published
pre-match football model that encodes a starting XI as a permutation-invariant
set of player vectors and predicts 1X2.** The nearest precedent is an NBA paper
(Hubáček et al. 2019) using a convolution over player statistics to consume a
variable roster. So this is a measurement, not a reproduction.

## The design

Same network, same fixtures, same walk-forward splits, trained twice. Both arms
warm-start from the **same** tier-1 model pretrained on all 296,208 matches, so
neither is handicapped by small-data optimisation. The only difference is a
Deep Sets encoder over each starting XI: a shared per-player MLP, then masked
mean and max pooling, which is permutation-invariant by construction.

Squad vectors carry 30 features per player — rolling per-90 form over their
last 10 appearances across 22 statistics, plus two rate statistics, appearance
count, average minutes, and a one-hot position line. Every value comes from
matches strictly before the one being predicted.

Panel: 2,946 matches, Danish Superliga and Scottish Premiership, 2019/20 →
2025/26, 98.2% of squad slots populated. Four walk-forward splits, 1,682
matches scored out of sample, three seeds averaged.

## The result

| model | RPS | log loss | ECE | accuracy |
|---|---|---|---|---|
| market (Pinnacle close) | 0.19595 | 0.96759 | 0.01364 | 52.8% |
| net **without** squad encoder | 0.20309 | 0.99038 | 0.01864 | 52.0% |
| net **with** squad encoder | 0.20424 | 0.99375 | 0.01584 | 51.2% |

Paired per-match RPS difference, bootstrapped over 571 matchdays:

```
delta  -0.00115     95% CI [-0.00301, +0.00070]     t = -1.17
```

**The interval spans zero, and the point estimate is negative.** On this
evidence the starting XI adds nothing measurable over team-level form and
ratings, and there is no hint of a positive effect being masked by noise.

## The caveat that decides the money question

This experiment could only have detected a large effect. Standard deviation of
the per-match RPS difference is 0.0403, so with n = 1,682 the **minimum
detectable effect at 80% power is +0.00275**.

For scale, against effects actually measured in this project:

| effect | size | n needed to detect it |
|---|---|---|
| GRU sequence branch vs none | +0.00019 | 352,906 |
| moved flags in the ratings | +0.00038 | 88,227 |
| net vs ordered logit, same data | +0.00053 | 45,354 |
| penaltyblog: tuning time decay | +0.00230 | 2,409 |

**This experiment had 1,682.** It could not have detected an effect the size of
the sequence branch's — the one component in this project that genuinely beat
the baseline — by a factor of roughly 200 in sample size.

So the honest statement is: *no evidence that the starting XI helps, from an
experiment that could only have found a large effect.* Not: *the starting XI
does not help.*

## Recommendation on the upgrade

**Do not upgrade to Starter (€29/mo).** Five leagues instead of two is roughly
2.5× the matches — about 4,200 test fixtures against the 45,354 needed to
resolve even the larger of the effect sizes above. It would buy a slightly
tighter interval around the same inconclusive answer.

Rough scaling, assuming ~420 test matches per league-season:

| plan | leagues | est. test matches over 7 seasons | resolves +0.00053? |
|---|---|---|---|
| Free (current) | 2 | ~1,700 | no |
| Starter €29 | 5 | ~4,200 | no |
| Growth €99 | 30 | ~25,000 | not quite |
| Pro €249 | 120 | ~100,000 | yes |

Only Pro reaches the sample where a subtle squad effect could be distinguished
from noise, and even Pro falls short of resolving an effect the size of the
sequence branch's.

⚠️ Two things unverified before any money moves: whether paid tiers carry the
same 22-season history the free tier grants for its leagues (the pricing page
does not say), and whether the rich per-player statistics extend before 2019/20
in other leagues. Both are checkable on the 14-day free trial.

## What this does not rule out

- A **larger squad model** — attention over players rather than mean/max
  pooling, or player embeddings learned across leagues — might extract more
  than Deep Sets does. Untested.
- **Injury and absence information**, which is the mechanism most people assume
  lineups carry, is only partly captured here: this encodes who *is* playing,
  not who is missing relative to the usual XI.
- The market beats both arms comfortably (0.19595), and bookmakers price
  lineups aggressively once teamsheets are published an hour before kickoff.
  That the market knows something here is not in doubt; that *this* encoder
  extracts it is what failed.
