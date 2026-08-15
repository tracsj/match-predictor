# Honest betting evaluation, and where the odds data comes from

**Research sweep run 2026-08-15. Re-check before quoting.**

## Bottom line

The honest answer to "would my model have made money?" is almost always **no, once you bet against a real price you could actually have taken.** The most informative published table found (Constantinou 2022, 13 EPL seasons) shows a well-built Bayesian rating model losing **−9.03% to −1.02% ROI at market-average 1X2 odds** across every low betting threshold, turning positive only at thresholds where the sample collapses to a few dozen bets.

That is the base rate. Everything below is how to build a backtest that reports that truth rather than hiding it.

---

## 1. Backtesting that survives scrutiny

**Random train/test splits are invalid here**, for three distinct reasons:

1. **Temporal leak** — a random split trains on May 2024 and tests on Nov 2023. Team strength is autocorrelated, so the model has seen the future state of the same teams.
2. **Feature leak (the one that actually kills projects)** — rolling-form features must be computed strictly from matches with kickoff timestamp **<** the target match's kickoff. The classic bug is computing form per season with a groupby and then shifting, which silently includes same-day fixtures. Concrete trap for this project: **football-data.co.uk only added a `Time` column from 2019/20** (verified). Before that you cannot order same-day fixtures, so same-round contamination is unavoidable unless the whole matchday is dropped from the feature window.
3. **Target/market leak** — including the odds (or any post-kickoff stat) as a feature and then betting against those odds. If the odds are a feature, the "edge" is a rearrangement of the bookmaker's own opinion.

**Use rolling-origin / walk-forward.** Train on everything up to *t*, predict *t*→*t+Δ*, roll, refit. Report bets only from prediction windows. Season-level holdout is the coarse version and is what the credible papers do. Constantinou evaluates 13 EPL seasons this way and warns that per-season optimisation is fantasy:

> "the high variance of θ suggests that is not reasonable to expect that we will be able to successfully predict the optimal value of θ before a season starts… the maximised profitability presented in Tables 10 and 11 is not a realistic expectation of real-world performance; only Table 9 is."
> — Constantinou (2022), §5.3

That is the most useful methodological sentence in the football-betting literature. Optimising the edge threshold per season raised his overall ROI from 5.7% to 6.33% at average odds, and **both numbers are artefacts.**

**Multiple-comparisons / strategy-selection bias.** Constantinou's Table 9 alone scans 21 thresholds × 2 odds regimes = 42 configurations. Report the best cell and the nominal p-value is meaningless. Formal fixes transfer directly from finance:

- **White's Reality Check** (White, 2000, *Econometrica*) — stationary bootstrap of the max mean out-performance across all candidate rules, testing whether the *best* rule beats the benchmark after accounting for the search.
- **Hansen's SPA test** (2005) — studentises and re-centres, excluding clearly-inferior rules; strictly more powerful.
- Practical minimum if neither is implemented: **pre-register the rule** (one threshold, one market, one staking scheme) before looking at PnL, hold out the final 2–3 seasons untouched, and **report how many configurations were tried** — the number itself is the disclosure.

**Realism constraints that belong in the backtest:** bets that could not have been placed (price gone, market suspended), minimum/maximum stakes, Betfair commission (2–5%), rounding, and the fact that an aggregator "best price" was frequently available for seconds and for £20.

---

## 2. The odds you must bet against

**Opening vs closing vs sharp.** Opening odds are the bookmaker's prior with the least information and the widest margin; beating them proves little. The **closing line** is the market's terminal consensus after all money and news, and under any efficient-markets framing it is the best available probability estimate. **Pinnacle** (and Betfair Exchange SP) are the reference because they run low margin, high limits, and do not ban winners — so their prices aggregate sharp money rather than recreational money.

### CLV is the standard proxy for edge, and there is a real measurement behind it

Joseph Buchdahl's analysis on football-data.co.uk used **87,960 pre-close/close odds pairs** (4 seasons of Pinnacle closing 1X2) and found the ratio (odds you bet ÷ closing odds) predicts realised level-stakes yield with a **slope of essentially 1.00**. Hence the rule of thumb:

```
expected yield  ≈  (your odds ÷ Pinnacle closing odds) ÷ (1 + margin)  −  1
```

Its diagnostic power is that it **converges fast**. Buchdahl detected a genuine ~6–7% long-run edge for one tipster from **26 tips over 2 months** (23 of 26 prices shortened; χ² p ≈ 1e−5) — a PnL-based test would have needed thousands of bets. Conversely, a tipster with a "25% yield from ~300 picks" (p ≈ 1/40,000 on PnL alone) had **zero CLV** and was almost certainly lucky.

**This is the strongest argument for making CLV the primary metric and ROI the secondary one.**

Caveat, stated plainly: CLV is a proxy, not a proof. Bet a bookmaker who copies Pinnacle with a lag and you can harvest CLV mechanically and still lose to variance; and in illiquid markets your own bet can move the line you are measuring against.

### Overround and de-vigging

With decimal odds `d_i`, raw implied `q_i = 1/d_i`, overround `= Σ q_i` (typically 1.02–1.05 Pinnacle 1X2, 1.06–1.12 soft books).

| Method | Form | Behaviour |
|---|---|---|
| Multiplicative / normalised | `p_i = q_i / Σq` | Removes margin proportionally. Simplest; assumes no favourite-longshot bias |
| Additive | `p_i = q_i − (Σq − 1)/n` | Equal margin per outcome; can go negative on longshots |
| Power | `p_i = q_i^k`, solve `k` s.t. `Σp = 1` | Always in [0,1]; over-corrects longshots relative to Shin |
| **Shin** | Solves for insider-trading proportion `z`: `p_i = (√(z² + 4(1−z)q_i²/Σq) − z) / (2(1−z))` | Endogenously models favourite-longshot bias. Reduces to additive for 2 outcomes |
| Worst-case | min across methods | Deliberately conservative; a pessimistic bound |

**Which is most accurate, peer-reviewed rather than blog-sourced:** Štrumbelj (2014), *On determining probability forecasts from betting odds*, International Journal of Forecasting 30(4), <https://www.sciencedirect.com/science/article/abs/pii/S0169207014000533> — **Shin probabilities were more accurate than basic normalisation and regression-based approaches for all bookmaker/sport pairs tested.** The paper also finds exchange odds are *not* always the best probability source, especially in smaller markets.

Everything on Outlier / OddsJam / BetHero comparing power vs Shin vs worst-case is downstream marketing content from +EV tools. Use them for formulas; cite Štrumbelj for the ranking.

### Bookmaker-average vs best price — where most backtests cheat

- `AvgH/AvgD/AvgA` (market average) includes high-margin books. Beating it is barely an achievement, and it is the number most amateur backtests quietly use.
- `MaxH/MaxD/MaxA` (best available) is what a real bettor takes — simultaneously the *most* achievable and *least* obtainable price, because it is the outlier book, often the one that limits you within a month.
- Constantinou quantifies the gap exactly: at θ=0%, 1X2 ROI is **−9.03% at average odds vs −1.20% at maximum odds**; at θ=8%, **+5.49% (814 bets) vs +7.40% (1,345 bets)**. **Nearly the entire apparent "edge" in the max-odds column is the vig you avoided by picking the outlier price, not model skill.**

**Honest protocol: report three columns** — Pinnacle closing (the truth test), a single named soft book you could actually hold an account with, and market max (the optimistic bound). If you are only profitable in the third column, you do not have a model, you have an odds-comparison screen.

---

## 3. Staking, and telling a real edge from noise

**Kelly.** Single binary outcome at decimal odds `d`, true probability `p`: `f* = (pd − 1)/(d − 1)`. For a **3-outcome 1X2 market** with possible simultaneous positions there is no closed form — maximise `Σ p_i log(1 − Σ_j f_j + f_i d_i)` subject to `f_j ≥ 0`, `Σ f_j < 1`, a concave program solved numerically.

- Full Kelly is optimal only if probabilities are *exactly* right. They are not, and estimation error makes full Kelly systematically over-bet. **Fractional Kelly (¼ to ½)** trades a modest fraction of growth rate for a large reduction in drawdown. Busseti, Ryu & Boyd, *Risk-Constrained Kelly Gambling*, <https://web.stanford.edu/~boyd/papers/pdf/kelly.pdf> formalises this.
- Hubáček et al., *Optimal Sports Betting Strategies in Practice*, <https://arxiv.org/pdf/2107.08827> — tests Kelly variants including drawdown-constrained Kelly on football data.
- **Flat staking is the right default for a backtest** — it makes ROI interpretable and stops one early lucky bet compounding the whole equity curve. Flat-stake ROI is the headline; Kelly is a supplementary bankroll simulation.
- Risk of ruin: with a 2% edge and per-bet σ ≈ 1.4, drawdowns of 30–40% of bankroll are routine over 10k bets **even when the edge is real.**

**Significance testing.** Per-bet return `R = d − 1` (win, prob `p`) or `−1` (lose). Mean `μ = pd − 1`; **`σ = d√(p(1−p))`**.

- A one-sample t-test on the per-bet series is the crude test. Better: **bootstrap over matches** (resample match-level PnL with replacement, 10k reps, BCa interval) — handles the skewed fat-tailed distribution correctly and preserves dependence between correlated bets on the same fixture. If bets cluster in time, use a **block / stationary bootstrap**.
- Always include a **random-bet null** at the same price distribution, as Kaunitz et al. did: their real strategy returned +3.5% against a random-strategy mean of −3.32%.

### How many bets to distinguish a 2% edge from zero

Using `n ≈ ((z_α + z_β)σ / μ)²`, 95% two-sided, 80% power (z sum 2.80), target μ = 0.02:

| Market | typical `d` | σ | bets needed |
|---|---|---|---|
| Asian handicap / O-U 2.5 | 1.95, p≈0.52 | ≈0.98 | **≈19,000** |
| Even-money-ish 1X2 favourite | 2.0, p≈0.51 | ≈1.00 | **≈19,600** |
| Mixed 1X2 portfolio (avg odds ≈3.2) | 3.2, p≈0.32 | ≈1.49 | **≈43,500** |

Bare significance without a power requirement still needs ≈6,800–15,000 bets. To detect a **5%** edge, divide by 6.25 (≈3,100–7,000 bets). At 5–10 qualifying bets per matchday, 20,000 bets is **a decade**.

This is why CLV exists as a metric, and why any backtest reporting a headline ROI on 300–2,000 bets has said nothing about whether the edge is real. **Report the required *n* beside any ROI figure.**

---

## 4. What the literature actually reports

### The one credible, fully-documented "we beat them"

Kaunitz, Zhong & Kreiner (2017), *Beating the bookies with their own numbers — and how the online sports betting market is rigged*, <https://arxiv.org/abs/1710.02824>

- 10-year simulation on **closing** odds: **56,435 bets, +3.5% return**, accuracy 44.4%. Random-bet null −3.32%; p < 1 in a billion.
- 6-month minute-by-minute simulation (bets 1–5h pre-kickoff): **6,994 bets, +9.9%**.
- Paper trading: 407 bets, +5.5%. **Real money: 265 bets over 5 months, +8.5%, $957.50 profit.**
- **Then it ended.** Bookmakers limited them: William Hill to ¥2,428.33, Interwetten to $11.11, Sportingbet to **$1.25**, Betway to $10.45.

Read the mechanism carefully before getting excited: their rule was **not a forecasting model.** They bet whenever one bookmaker's odds deviated above the market mean — a best-price-harvesting strategy, which dies exactly the way theirs died. The paper's own framing is that inefficiency exists but "the sports gambling industry compensates these market inefficiencies with discriminatory practices against successful clients."

### Other peer-reviewed results, with honest sample sizes

- **Constantinou (2022)**, *Investigating the efficiency of the Asian handicap football betting market with ratings and Bayesian networks*, J. Sports Analytics 8(3):171–193, <https://arxiv.org/abs/2003.09384> — the numbers to trust most, because he prints the full threshold sweep. 1X2, 13 EPL seasons, **average odds**: −9.03% (4,334 bets, θ=0) → −1.02% (1,339 bets, θ=6%) → +5.49% (814 bets, θ=8%) → +23.59% at θ=18% on **37 bets** (noise). **Maximum odds**: −1.20% at θ=0, +7.40% at θ=8%. **Asian handicap was worse**: −3.86% at θ=0 on average odds, negative at essentially every threshold below 9%.
- **Wheatcroft (2020)**, *A profitable model for predicting the over/under market in football*, IJF 36(3):916–932, <http://eprints.lse.ac.uk/103712/> — GAP ratings from **shots and corners rather than goals**, ten European leagues, twelve years: **≈+0.8% profit per bet.** That is what a genuine, carefully-validated, published football edge looks like. Sub-1%.
- **Berrar, Lopes & Dubitzky (2019) Soccer Prediction Challenge** — best in-competition result was k-NN on rating features: 50.49% accuracy, RPS 0.2149; best post-hoc XGBoost 51.94% / RPS 0.2054. Bookmaker odds are at or above this; nobody demonstrated a betting edge.
- **Wunderlich & Memmert (2020)**, *Are betting returns a useful measure of accuracy in (sports) forecasting?*, IJF 36(2), <https://www.sciencedirect.com/science/article/abs/pii/S016920701930233X> — accuracy and profitability are **not monotonically related**; betting returns are noisy, margin-contaminated and incomparable across studies. Prefer RPS/log-loss/Brier for model quality; treat ROI as a separate, high-variance question.
- The 2024 systematic review of ML in sports betting, <https://arxiv.org/abs/2410.21484>, catalogues dozens of papers reporting ROIs like 3.8% and 4.35% — with a grab-bag of metrics and inconsistent out-of-sample protocols. A map of the literature, not evidence the ROIs replicate.

**Documented cases of consistently beating the closing line** exist, but the documented ones are individuals in niche markets, not published models. Buchdahl's Jeremy Price case (~10% average beat of the close, predicted ~6–7% long-run yield) is the clearest — and the same article shows the price cut from 2.00 to 1.78 **in one move within a minute** of publication. The edge existed only for the first bet placed.

**On the base rate, be careful what you repeat.** The commonly-cited "only 3–5% of bettors are profitable" figure has **no primary source** that could be found — treat it as folklore. The one hard dataset is regulatory: UK Gambling Commission data (via Smart Betting Club, 2025) shows **643,779 of ~15 million accounts restricted (≈4.3%)**, 62% subject to stake limits — and notably **51.29% of restricted customers had actually lost money**, so restriction is a blunt profiling instrument rather than purely an anti-winner one.

Also treat as marketing, not evidence: "positive CLV bettors see 2–3× ROI" (BettorEdge), and the "$1,117.56 profit / 3.6% ROI over 31,247 bets" figure circulating on Pinnacle-adjacent affiliate sites. Neither traces to a primary study.

---

## 5. Odds data sources

### football-data.co.uk — free, and the correct starting point

See `00-measured-facts.md` for the column-level verification done for this project. Summary:

- Main leagues `/mmz4281/{season}/{div}.csv`: results + match stats + odds. Free, no key.
- **Pre-close odds collected Friday ≤17:00 BST for weekend fixtures, Tuesday ≤13:00 for midweek** (stated on [matches.php](https://www.football-data.co.uk/matches.php)). A genuine 1–3 day-out snapshot, not an opening price and not a last-second one.
- **Closing odds ("C" suffix) history:** none in 2011/12 and earlier; **Pinnacle closing 1X2 only from 2012/13**; full closing suite (per-book closing 1X2, `MaxC`/`AvgC`, closing O/U 2.5, closing Asian handicap) from **2019/20**. Betfair Exchange columns (`BFEH`, `BFECH`, …) in recent seasons.
- **Extra leagues** `/new/{country}.csv` (ARG, BRA, CHN, JPN, MEX, USA, DNK, NOR, SWE, FIN, IRL, POL, ROU, RUS, CHE, AUT): **closing odds only** — no pre-close, no O/U, no Asian handicap, no shots. If the thesis is "inefficiency lives in less-liquid leagues", this is the file — and it only gives closing prices, so you can measure calibration but cannot construct a realistic "bet at a pre-close price" simulation there.
- **Provenance** ([notes.txt](https://www.football-data.co.uk/notes.txt)): Betbrain, Oddsportal, individual bookmakers. So `Max`/`Avg` are aggregator-derived — "market max" is a price someone somewhere showed, not necessarily one takeable in size.
- The site is affiliate-funded and says so.

⚠️ A search summary claimed Pinnacle was dropped from Max/Avg from July 2025 due to feed unreliability. **Not confirmed on-site.** Pinnacle columns (`PSH`, `PSCH`) *are* present in 2025/26 (verified).

### OddsPortal — richest history, worst friction

Deep archives including opening/closing movement, but behind Cloudflare, and scraping is against its terms (the GitHub scrapers [oddsporter](https://github.com/gingeleski/oddsporter) and [scrapy-oddsportal](https://github.com/tvl/scrapy-oddsportal) carry that warning themselves). Practically: rate-limited, JS-rendered, brittle — and you would be re-deriving data football-data.co.uk already licenses from them. **Not worth it as a primary source.**

### The Odds API (the-odds-api.com)

Verified from [the primary page](https://the-odds-api.com/historical-odds-data/):

- Historical **featured markets** (h2h, spreads, totals) from **6 June 2020**, snapshots every **10 minutes**; **5-minute** intervals from **September 2022**.
- Historical **additional markets** (player props, period markets) from **3 May 2023**.
- **Historical data is paid-plans only.** Quota cost of `/v4/historical/.../odds` is **10 per region per market**; historical event-odds is 10 per region per market **per event**.
- Snapshots let you reconstruct a genuine closing price (walk `previous_timestamp` back from kickoff) — the main reason to pay.
- ⚠️ Tier pricing ($29 Professional / $99 Business, "historical free on Business") came from **a competitor's marketing blog**, not The Odds API's own pricing page. Verify before budgeting.

### API-Football / api-sports.io — do NOT plan on this for historical odds

Advertises "15+ years of historical data", but that refers to fixtures/stats. Odds are published roughly 1–14 days before a fixture and, per third-party developer documentation, the odds endpoint **retains only about the last 7 days** — you cannot retroactively pull odds for past seasons. ⚠️ That retention claim is from a competitor (oddspapi.io); verify with support before relying on it either way. **Treat API-Football as a live-capture source, not a historical archive.**

### Betfair Exchange historical data — the free tier is real

- <https://historicdata.betfair.com>, login with a normal Betfair account. Data **from April 2015**.
- **BASIC tier is free**: full market/runner metadata and settlement, **last traded price at 1-minute frequency**, no volume, no price ladder.
- ADVANCED / PRO are paid (higher frequency, ladder/volume detail).
- Files are compressed line-delimited JSON stream files (TAR of BZ2); `betfairlightweight` and the [Betfair Data Scientists tooling](https://betfair-datascientists.github.io/data/usingHistoricDataSite/) parse them.
- **Why it matters:** Betfair SP and last-traded price at kickoff give a near-zero-margin closing benchmark needing no de-vigging. But subtract commission (2–5%) and check the market had liquidity at your notional stake — for lower leagues matched volume is often tiny and the "price" is illusory.

### Free-vs-paid summary

| Source | Cost | Closing odds | Depth |
|---|---|---|---|
| football-data.co.uk main | Free | Pinnacle close 2012/13+; full close 2019/20+ | ~22 divisions, results to 1993/94 |
| football-data.co.uk extra | Free | **Closing only**, 1X2 | ~16 countries |
| Betfair historicdata BASIC | Free | Yes (1-min last traded → kickoff) | Apr 2015+ |
| The Odds API historical | Paid, 10× credits | Yes (reconstruct from snapshots) | Jun 2020+ |
| API-Football odds | Paid | **No historical** (≈7-day retention) | Live capture only |
| OddsPortal | Free but scraped | Yes | Deepest, highest friction/ToS risk |

---

## 6. Beyond 1X2 — where inefficiency actually lives

The honest evidence is thinner and more negative than the folklore suggests.

- **Asian handicap.** Constantinou (2022) is the only proper published efficiency study, and found AH **worse** than 1X2 for his model: −3.86% at θ=0 on average odds, negative at nearly every threshold below 9%; max odds barely positive. His summary: the AH market "shares the inefficiencies of the traditional market." It is **not** a softer market. AH does have the lowest margins in football (Pinnacle ~1.5–2.5%), which means less vig to overcome — and more sharp attention.
- **Over/under 2.5 goals.** The best evidence of a real edge: Wheatcroft (2020) got **≈+0.8% per bet over 12 years and 10 leagues** using GAP ratings built from **shots and corners, deliberately not goals**. The mechanism is the informative part — the edge came from using less-noisy underlying statistics than the goal counts the market anchors on.
- **Both teams to score.** Wheatcroft & Sienkiewicz (2022), IJF 38(3):895–909, <https://ideas.repec.org/a/eee/intfor/v38y2022i3p895-909.html> — an explicitly-targeted secondary market. Derived, wider margin at many books; the case for it is less attention, not structural softness.
- **Correct score.** High margin (often 15–25% overround), extreme longshot exposure, highest variance per bet. σ at odds 9.0 is ≈2.8, which **triples** required sample size. Avoid as a first market regardless of what the model says.
- **Favourite–longshot bias.** Genuinely documented in horse racing; in football weaker and partly absent at sharp books. Research on Pinnacle specifically found *no* clear longshot bias and only slight favourite bias — but exploitable FLB reappears when using **maximum market odds** across many books. Consistent with the theme throughout: what looks like "bias" is often "the outlier bookmaker's error", which is a price-shopping edge, not a modelling edge.
- **Lower leagues / less liquid markets.** The theoretical case is sound (lower limits, less sharp money, more relevant private information about rotation and travel) and the literature broadly finds top leagues more efficient. But note the trap: for lower/exotic leagues football-data.co.uk gives **closing odds only**, and Betfair volume is thin, so the price your backtest "takes" is often a price nobody could have got in size. Higher margins there also subtract directly from any edge.

**Where to look, on this evidence:** underlying-statistic-driven totals markets (replicate Wheatcroft's finding on our own data), and mid-tier European leagues where Pinnacle prices exist so CLV is measurable but attention is lower. **Not** correct score. **Not** "the model likes this longshot."

---

## 7. The protocol this project holds itself to

1. Walk-forward only; every feature from strictly-prior kickoffs; drop pre-2019/20 same-day fixtures or accept known contamination.
2. **Report calibration first** (log-loss, Brier, RPS) against a bookmaker-odds baseline. If we do not beat de-vigged Pinnacle closing on RPS, stop — there is no edge to stake.
3. De-vig with **Shin** (Štrumbelj 2014), report multiplicative alongside as sensitivity.
4. ROI in **three price columns**: Pinnacle close / one named soft book / market max. Lead with Pinnacle close.
5. Report **CLV distribution** (mean odds ratio, % of bets that shortened, χ² and t-test). It converges ~100× faster than ROI.
6. Flat 1-unit stakes for the headline; Kelly only as a supplementary bankroll sim, fractional at ≤½.
7. **Bootstrap BCa CI on ROI** (block-bootstrap over matchdays), plus a random-bet null with the same odds distribution.
8. Disclose the number of configurations searched; apply White/Hansen if more than a handful; keep a final untouched holdout.
9. State sample size honestly against the ~19,000–43,500 bets needed to resolve a 2% edge.
10. Model the real world: commission, stake limits, price availability — and the fact that if it works, you will be limited.

---

## Sources

- [Constantinou (2022), *Investigating the efficiency of the Asian handicap football betting market*, J. Sports Analytics 8(3)](https://arxiv.org/abs/2003.09384)
- [Kaunitz, Zhong & Kreiner (2017), *Beating the bookies with their own numbers*](https://arxiv.org/abs/1710.02824)
- [Štrumbelj (2014), *On determining probability forecasts from betting odds*, IJF 30(4)](https://www.sciencedirect.com/science/article/abs/pii/S0169207014000533)
- [Wunderlich & Memmert (2020), *Are betting returns a useful measure of accuracy?*, IJF 36(2)](https://www.sciencedirect.com/science/article/abs/pii/S016920701930233X)
- [Wheatcroft (2020), *A profitable model for predicting the over/under market in football*, IJF 36(3)](http://eprints.lse.ac.uk/103712/)
- [Wheatcroft & Sienkiewicz (2022), *…the case of 'both teams to score'*, IJF 38(3)](https://ideas.repec.org/a/eee/intfor/v38y2022i3p895-909.html)
- [Systematic review of ML in sports betting (2024)](https://arxiv.org/abs/2410.21484)
- [Busseti, Ryu & Boyd, *Risk-Constrained Kelly Gambling*](https://web.stanford.edu/~boyd/papers/pdf/kelly.pdf)
- [Hubáček et al., *Optimal Sports Betting Strategies in Practice*](https://arxiv.org/pdf/2107.08827)
- [MacLean, Thorp & Ziemba, *Good and bad properties of the Kelly criterion*](https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf)
- [Hsu & Kuan, *Re-Examining the Profitability of Technical Analysis with White's Reality Check and Hansen's SPA Test*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=685361)
- [Buchdahl, *Using Pinnacle.com's Closing Line to Predict Profits*](https://www.football-data.co.uk/blog/pinnacle_efficiency.php)
- [Buchdahl, *Using the Closing Betting Odds to test for a Tipster's Skill*](https://football-data.co.uk/blog/closing_odds.php)
- [football-data.co.uk notes.txt](https://www.football-data.co.uk/notes.txt) · [collection times](https://www.football-data.co.uk/matches.php)
- [The Odds API — historical odds data](https://the-odds-api.com/historical-odds-data/)
- [Betfair — what data the Historical Data service provides](https://support.developer.betfair.com/hc/en-us/articles/360002407732-What-data-is-provided-by-the-Historical-Data-service) · [portal](https://historicdata.betfair.com/) · [parsing guide](https://betfair-datascientists.github.io/data/usingHistoricDataSite/)
- [Smart Betting Club — UKGC account-restriction data](https://smartbettingclub.com/blog/gambling-commission-restrictions-data/)
- [Efficiency of online football betting markets, IJF (41 bookmakers, 11 leagues, 11 years)](https://www.sciencedirect.com/science/article/abs/pii/S0169207018301134)
- [Favorite-Longshot Bias and Market Efficiency in the Soccer Betting Market](https://www.researchgate.net/publication/351985837_Favorite-Longshot_Bias_and_Market_Efficiency_in_the_Soccer_Betting_Market)
- De-vig method comparisons (practitioner, **not** peer-reviewed): [Outlier](https://help.outlier.bet/en/articles/8208129-how-to-devig-odds-comparing-the-methods), [Bet Hero](https://betherosports.com/blog/devigging-methods-explained)
