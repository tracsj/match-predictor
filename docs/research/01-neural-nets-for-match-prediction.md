# Neural networks for pre-match football outcome prediction

**Research sweep run 2026-08-15. Re-check before quoting.**

Scope note that governs everything below: **pre-match** work is separated from **in-game / event-stream** work, because most of the impressive-looking deep learning in football is the latter and does not transfer. RPS values are not comparable across leagues or test sets; conflating them is the most common error in this literature, so every number here is tagged with its dataset.

---

## 1. What actually wins

### On goals-only tabular data, gradient-boosted trees are still the reference

The most authoritative recent survey states it directly:

> "gradient-boosted tree models such as CatBoost, applied to soccer-specific ratings such as pi-ratings, are currently the best-performing models on datasets containing only goals as the match features."
> — Bunker, Yeung & Fujii (2024), *Machine Learning for Soccer Match Result Prediction*, <https://arxiv.org/abs/2403.07669>

The same survey flags the honest caveat: there has **not** been a thorough comparison of deep learning against tree ensembles on datasets with *richer* feature types. So "GBTs win" is well-evidenced on goals/ratings features and under-tested elsewhere.

Corroborating from outside football: Grinsztajn, Oyallon & Varoquaux (2022), *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS Datasets & Benchmarks, <https://arxiv.org/abs/2207.08815> — tree ensembles remain state of the art on medium-sized (~10k sample) tabular problems, attributed to inductive bias: robustness to uninformative features, non-smooth target functions, rotation non-invariance. Twenty seasons of top-5 leagues sits squarely in that regime.

### The closest thing to a fair head-to-head

Yeung, Bunker, Umemoto & Fujii (2023/2024), *Evaluating soccer match prediction models: a deep learning approach and feature optimization for gradient-boosted trees* — <https://arxiv.org/abs/2309.14807> / *Machine Learning* 113 (2024), <https://link.springer.com/article/10.1007/s10994-024-06608-w>

- Data: >300,000 matches, 51 leagues, 2001 → April 2023. Prediction set 736 matches, 44 leagues.
- Architecture: inception block → Transformer Encoder → MLP, over recency features from the previous five matches (attacking strength, defensive strength, opposition strength, home advantage).
- Validation RPS:

| Model | RPS |
|---|---|
| **CatBoost + pi-ratings** | **0.2085** |
| Inception + Transformer Encoder + MLP | 0.2098 |
| LSTM + MLP | 0.2105 |
| Transformer Encoder + MLP | 0.2111 |
| XGBoost + Berrar ratings | 0.2141 |

The deep model's defence is **stability, not accuracy** — loss standard deviation 0.0051 vs CatBoost's 0.0083.

⚠️ Secondary sources summarising this paper claim the deep model "outperformed XGBoost and other remaining models." True against *XGBoost + Berrar*; false against *CatBoost + pi*. Read the table, not the summaries.

### The bar above every model: the closing line

Same paper, 2023 Soccer Prediction Challenge Task 2 (W/D/L):

- Bookmaker consensus: **RPS 0.2063**
- Their deep learning submission: **RPS 0.2195**
- The bookmaker consensus beat the best deep model by **6.42%**.

And the sharpest single datapoint found:

Pitcan (2026), *Does a Structural Model Add Anything to the Closing Price? Calibrated forecasting, incremental information, and match leverage in the Italian Serie A*, <https://arxiv.org/abs/2608.11505>

- Dixon–Coles with exponential decay, 19 Serie A seasons, 7,220 matches.
- Model accuracy 53.4%; model RPS **0.1972** vs market RPS **0.1905**.
- **The fitted pooling weight on the structural model against the market was 0.000.** The market had absorbed everything the model knew.
- Paired difference favours the market at +0.0067 (95% CI [0.0046, 0.0088]); **the market won in all seven test seasons.**
- A shots-on-target variant earned weight 0.35 against the goals model but **0.000 against the market** — shot-based signal is real, and already priced.

Also: Baboota & Kaur (2019), *Predictive analysis and modelling football results using machine learning approach for the English Premier League*, IJF, <https://www.sciencedirect.com/science/article/abs/pii/S0169207018300116> — a strong GBM with heavy feature engineering failed to outperform bookmaker odds.

### The contrary evidence, and how much to discount it

Wilkens (2026), *Can simple models predict football — and beat the odds? Lessons from the German Bundesliga*, SAGE, doi 10.1177/22150218261416681, <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5381388>

- Recent xG → Skellam distribution → W/D/L, isotonic calibration, 11 Bundesliga seasons (2014/15–2024/25).
- Claims ~10% ROI at average market odds, ~15% at best prices; profits come almost entirely from **home-win** bets.
- **Without isotonic calibration ROI collapses to ~1%.**
- The paper concedes bookmaker odds are better statistically calibrated.

Read: a single-league backtest whose entire claimed edge appears only after a calibration step, which raises the obvious question of where that isotonic map was fitted. Treat as hypothesis, not result. ⚠️ Full text was not accessible (SAGE 403); this rests on the abstract. It conflicts directly with Pitcan (2026) and Baboota & Kaur (2019), and that conflict was not resolved.

Hubáček, Šír & Železný (2022), *Forty years of score-based soccer match outcome prediction: an experimental review*, IMA J. Management Mathematics 33(1), <https://academic.oup.com/imaman/article/33/1/1/6342916> — reimplemented Poisson/Weibull models and rating systems (Elo, Steph, Gaussian-OD, Berrar, pi-ratings) on 218,916 matches from 52 leagues since 2000/01. Berrar ratings and Double-Weibull came out best. **No neural net won.**

### Numbers to calibrate expectations against

| Setting | Metric | Value | Source |
|---|---|---|---|
| Serie A, 19 seasons | Market RPS | **0.1905** | Pitcan 2026 |
| Serie A, 19 seasons | Dixon–Coles RPS / accuracy | 0.1972 / 53.4% | Pitcan 2026 |
| Eredivisie, tuned DC | RPS | ~0.1891 | penaltyblog 2025 |
| Eredivisie, 6 model families | RPS | 0.1914–0.1916 | penaltyblog 2025 |
| 2023 Challenge, 44 leagues | Bookmaker consensus RPS | **0.2063** | Yeung et al. 2024 |
| 2023 Challenge | Best deep model RPS | 0.2195 | Yeung et al. 2024 |
| 300k-match validation | CatBoost + pi RPS | 0.2085 | Yeung et al. 2024 |
| Kaggle FMPP 2022, 150k matches, no odds | 1st-place log loss | **0.99504** | [leaderboard](https://www.kaggle.com/competitions/football-match-probability-prediction/leaderboard) |
| Bookmaker closing odds, 1X2 | Accuracy | ~53–55% | [Pym](https://jep00.github.io/docs/work/bettingaccuracy.pdf) |

**Rules of thumb.** RPS 0.19–0.21 depending on league; accuracy 51–55%; log loss ~0.96–1.00. A uniform 3-class prior gives log loss 1.099 and the market reaches ~0.96–0.99, so **the entire addressable signal between "guess the base rate" and "be the market" is roughly 0.10–0.13 nats.**

Any paper reporting 70–90% accuracy on 1X2 is using in-game information, leaking, or evaluating on a non-representative subset. Concrete example: *SoccerNet: A Gated Recurrent Unit-based model to predict soccer match winners*, PLoS ONE 18(8):e0288933 (2023), <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0288933> claims >80% accuracy — its input is 22 features across six 15-minute **in-match** time slots. Not a pre-match model. Not a benchmark.

---

## 2. Architectures with published evidence

### Pre-match

**MLP on engineered features + ratings.** The default. Competitive, never beat CatBoost-on-pi-ratings in any fair comparison found.

**Transformer encoder over recent-match sequences.** Yeung et al. 2024 above — the best-documented pre-match deep architecture with published head-to-head numbers.

**GRU/LSTM over last-N matches.** The Kaggle *Football Match Probability Prediction* competition (2022) is the most useful evidence, because the task was designed for it: ~150,000 matches, 860+ leagues, 9,500 teams, 2019–2021, each row carrying each team's previous 10 matches as a sequence, evaluated on multi-class log loss, **odds excluded**. Winning score 0.99504 across 382 teams. Public top solutions were predominantly LSTM-based (e.g. a documented [12th-place LSTM](https://www.kaggle.com/code/lonnieqin/football-prob-prediction-lstm-12th-solution)).
⚠️ The 1st-place *architecture* is unverified — the discussion page did not fetch. Confirmed: sequence NNs were competitive at the top of a 382-team leaderboard on 150k matches with no odds.

**ML plus human/text signal.** Beal, Middleton, Norman & Ramchurn (2021), *Combining Machine Learning and Human Experts to Predict Match Outcomes in Football: A Baseline Model*, IAAI-21, <https://arxiv.org/abs/2012.04380> — NLP over *Guardian* match previews plus statistical data, 6 EPL seasons, 63.18% accuracy, +6.9% over statistical baselines. ⚠️ 63% is far above closing-odds accuracy (~53–55%) and the paper reports no bookmaker baseline, so the test period is probably not comparable. Interesting for the idea (pre-match text encodes team news); suspicious as a number.

### Graph and player-set architectures

Wang, Xu, Horton, Gudmundsson & Wang (2025), *Player-Team Heterogeneous Interaction Graph Transformer (HIGFormer)*, <https://arxiv.org/abs/2507.10626> — Player Interaction Network (heterogeneous graph convolutions + transformer) → Team Interaction Network → Match Comparison Transformer, on Wyscout open data. The most on-point published design for "encode a squad, predict a result." ⚠️ The abstract claims it "significantly outperforms existing methods" but **names no gradient-boosting baseline, no rating-model baseline, and no bookmaker odds baseline.** That is a finding, not an omission to gloss over.

Passing-network GNNs: *We know who wins: graph-oriented approaches of passing networks for predictive football match outcomes*, J. Big Data (2025), <https://link.springer.com/article/10.1186/s40537-025-01203-9>. Requires event data to build the graph, so post-hoc or in-game, not pre-match.

**Deep Sets / Set Transformer over rosters — searched for specifically, and not found.** There is no published pre-match football paper encoding a starting XI as a permutation-invariant set of player vectors predicting 1X2. Set Transformers appear in football only for in-game trajectory/positional modelling. The nearest architectural precedent is Hubáček, Šourek & Železný (2019), *Exploiting sports-betting market using machine learning*, IJF 35(2):783–796, <https://ida.fel.cvut.cz/papers/hubacek2019exploiting.html>, which used a **convolutional layer over player statistics** to consume a variable roster — **an NBA paper**, but the design pattern (conv/pool over players → team representation) and its second contribution (explicitly **decorrelating the model from the bookmaker's published odds**) are both directly portable.

### In-game — cite as design evidence only

- Horton & Lucey (2025), *Large-Scale In-Game Outcome Forecasting … using an Axial Transformer Neural Network*, Stats Perform, <https://arxiv.org/abs/2511.18730> — 62,610 games, 28 competitions; axial transformer over (players × time); jointly predicts 13 action types at player/team/game level. **The best worked example of a multi-task Poisson/Bernoulli-headed football network in the literature** — Poisson NLL for counts, BCE for binaries, summed. Steal the head design, not the task.
- Simpson, Beal, Locke & Norman (2022), *Seq2Event*, KDD 2022, <https://dl.acm.org/doi/10.1145/3534678.3539138>.

### Output heads

- **Softmax 1X2** — the default.
- **Poisson / bivariate Poisson score head** — Dixon & Coles (1997) introduced the ρ correction to independent Poisson for low-scoring results (0-0, 1-1, 1-0, 0-1); Karlis & Ntzoufras (2003) the bivariate Poisson. ⚠️ **No paper found puts a bivariate-Poisson likelihood head on a neural network for football and benchmarks it against softmax.** A real gap.
- **Ordinal / ordered-logit head** — well motivated because home-win is "closer" to draw than to away-win, the same argument that motivates RPS. Established in the rating literature: Arntzen & Hvattum (2021), *Predicting match outcomes in association football using team ratings and player ratings*, Statistical Modelling, <https://journals.sagepub.com/doi/abs/10.1177/1471082X20929881>. Recurring practitioner reports that NNs and XGBoost overfit on 1X2 because of the draw class make two learned cutpoints cheap insurance.
- **Multi-task heads** — only well-demonstrated in-game (Horton & Lucey). No pre-match football paper ablates auxiliary heads. Another gap.

---

## 3. Feature representation — "what are the pixels"

The consistent finding across surveys is blunt: **feature engineering matters more than model class.**

**Ratings are the single highest-value input.**

- **pi-ratings** — Constantinou & Fenton (2013), *Determining the level of ability of football teams by dynamic ratings based on the relative discrepancies in scores between adversaries*, JQAS 9(1):37–50, <https://www.eecs.qmul.ac.uk/~norman/papers/pi-ratings.pdf>. Separate home and away ratings, updated by goal-difference discrepancy with diminishing returns. Reported to beat Elo substantially. **This is the feature that makes CatBoost win.**
- **Berrar ratings** — Berrar, Lopes & Dubitzky (2019), *Incorporating domain knowledge in machine learning for soccer outcome prediction*, Machine Learning 108:97–126. Best-performing in the Hubáček 2022 review alongside Double-Weibull.
- **Elo** — the baseline; Club Elo is scrapeable via `soccerdata`.

**Rolling form.** Yeung et al. use last-5 recency features decomposed into attacking strength, defensive strength, opposition strength and home advantage — **opponent-adjusted**, not raw rolling averages. Kaggle FMPP used last-10 as a sequence. Both are validated shapes.

**xG features.** Wilkens 2026 builds an entire (claimed-profitable) system on recent xG alone. Pitcan 2026 finds shots-on-target adds real signal over goals (weight 0.35) but **zero over the market**. Interpretation: xG/shot features genuinely improve a model relative to goals-only; they do not give an edge over a market that already reads xG.

**Market odds as a feature — the tradeoff, precisely.** Including closing odds improves every metric and is correct *if the product is a forecast*. It is circular *if the product is a bet*: you are training toward the thing you want to beat, and by construction cannot exceed it except by noise. Bunker et al. note odds are used as features "sometimes as the sole model feature" and warn against it for betting applications. Two defensible designs: **(a)** odds excluded from features, used only as benchmark — what Kaggle FMPP enforced, and what this project does; **(b)** Hubáček's approach — include market information but add an explicit **penalty on correlation with bookmaker-implied probabilities**, rewarding the model for finding what the market missed. Also relevant: *A leakage-aware workflow for pre-match forecasting of secondary football markets: a LaLiga case study* (2026), <https://www.sciencedirect.com/science/article/pii/S2590005626003620>.

**Other supported inputs.** Rest days, travel, home advantage (~0.1–0.3 goals), player availability/lineups, and **predicted match statistics as an intermediate target** — Wheatcroft (2020/21), *Forecasting football matches by predicting match statistics*, <https://arxiv.org/abs/2001.09097>, uses GAP (Generalised Attacking Performance) ratings to forecast *statistics* pre-match then converts to outcome forecasts, claiming information beyond the odds. That two-stage "predict the box score, then the result" structure is a natural multi-task NN design and one of the few published claims of genuine incremental information over odds.

**Team identity embeddings** are ubiquitous in practitioner code and essentially absent as an ablated research contribution. With ~100–150 distinct teams per league across 20 seasons, a low-dimensional (8–16d) weight-decayed embedding is the obvious NN-native alternative to one-hot ids — but it competes directly with rating features, which already encode team strength more efficiently and with proper temporal decay.

---

## 4. Calibration and proper scoring

**Why RPS.** Constantinou & Fenton (2012), *Solving the Problem of Inadequate Scoring Rules for Assessing Probabilistic Football Forecast Models*, JQAS 8(1), <https://doi.org/10.1515/1559-0410.1418>. 1X2 outcomes are ordinal on a latent scale, so a scoring rule should penalise a confident home-win forecast less when the result is a draw than when it is an away win. RPS is proper *and* distance-sensitive. This is why RPS became the field standard.

**The dissent, worth taking seriously.** Wheatcroft (2019/2022), *Evaluating probabilistic forecasts of football matches: the case against the ranked probability score*, JQAS 17(4):273–287, <https://arxiv.org/abs/1908.08980>. Two simulation experiments find the **ignorance score (log score) outperforms both RPS and Brier**, casting doubt on the value of non-locality and distance-sensitivity here.

⇒ **Report both RPS and log loss.** They are cheap and the field is not settled. Log loss also has the property that matters for betting: it is the negative log-growth of a Kelly bankroll, so improvements translate directly into expected compounding. RPS does not.

**Why calibration beats accuracy.** A bet is +EV iff `p_model × odds > 1`. Accuracy (argmax correctness) is irrelevant to that inequality; only the *level* of `p` matters. A 55%-accurate but systematically overconfident model loses money faster than a 52%-accurate well-calibrated one. Wilkens 2026 is the cleanest demonstration: same model, same selection rule, ROI ~1% raw → ~10% after isotonic calibration. Which also means: **verify the calibration fit is out-of-sample, or the entire result is manufactured.**

**Methods.**

- **Temperature scaling** — Guo et al. (2017), *On Calibration of Modern Neural Networks*, <https://arxiv.org/abs/1706.04599>. Divide logits by one learned scalar fit on a held-out set. One parameter, preserves argmax, essentially cannot overfit. **The right default for a neural net.**
- **Vector / matrix scaling** — per-class; useful in football specifically because miscalibration concentrates in the **draw** class.
- **Isotonic regression** — non-parametric, monotone, more flexible; needs more held-out data and can overfit at a few thousand samples.
- **Diagnostics** — reliability diagrams per class, ECE, Brier decomposition (reliability / resolution / uncertainty). Always compare the reliability curve against the bookmaker's on the same matches; the market is the best-calibrated forecaster available and makes an excellent yardstick.

---

## 5. Datasets, benchmarks, repos

| Source | Contents | Coverage | Odds | Friction |
|---|---|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/data.php) | Results, HT scores, shots, SoT, corners, cards, referee | 2000/01 → 2025/26, ~22 European divisions + 16 extra worldwide | **Yes — pre-match and closing** | Free CSVs, no login. **The dataset for this project.** [notes.txt](https://www.football-data.co.uk/notes.txt) |
| [Open International Soccer Database](https://link.springer.com/article/10.1007/s10994-018-5726-0) | Date, league, teams, full-time goals — deliberately minimal | 216,743 matches, 52 leagues, 35 countries, from 2000 | ⚠️ believed none; not confirmed from the source paper | Free; the benchmark behind the 2017 & 2023 Soccer Prediction Challenges, so the one dataset where published RPS is comparable |
| [Understat](https://understat.com) | xG, npxG, xGA, xGChain, xGBuildup, shot-level with coordinates | 2014/15 → present, top 5 + RPL | No | No API, no explicit licence; data embedded as JSON in `<script>` tags. `soccerdata` parses it |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Full event data, lineups, some 360 freeze-frames | Selected competitions | No | Free but a bespoke user agreement (attribution + logo). Deep but narrow — good for player modelling, useless as a pre-match corpus |
| Wyscout public event data (Pappalardo et al., Scientific Data 2019) | Event data | 2017/18 big-5, Euro 2016, WC 2018 | No | Free; HIGFormer's dataset |
| [Kaggle European Soccer DB](https://www.kaggle.com/datasets/hugomathien/soccer) | 25k+ matches, 10k+ players, FIFA attributes, **lineups with formation coordinates** | 2008–2016, 11 countries | **Yes, up to 10 providers** | Free SQLite. Dated, but the easiest path to player-level pre-match features joined to odds |
| [Kaggle FMPP](https://www.kaggle.com/competitions/football-match-probability-prediction) | 150k+ matches, 860+ leagues, each row = both teams' previous 10 matches | 2019–2021 | No (excluded by design) | **The best public benchmark for sequence models.** 1st = 0.99504 log loss |
| [soccerdata](https://github.com/probberechts/soccerdata) | Scrapers for Club Elo, ESPN, FBref, football-data.co.uk, Sofascore, SoFIFA, Understat, WhoScored, FotMob → pandas, unified ids, cached | varies | via football-data | Solves ingestion and cross-source id matching, otherwise a week of work |
| [FBref](https://fbref.com) (via soccerdata) | Opta-derived team/player season & match stats incl. xG | ~2017/18→ for xG | No | Aggressive rate-limiting |

**Repos worth reading:**

- [penaltyblog](https://github.com/martineastwood/penaltyblog) — Cython-accelerated Poisson, Bivariate Poisson, Dixon–Coles, Zero-Inflated Poisson, Negative Binomial, Weibull-Count+Copula, plus ratings and odds utilities. **The baseline library for this project** (installed; v1.11.0 exposes `DixonColesGoalModel`, `BivariatePoissonGoalModel`, `WeibullCopulaGoalsModel` and others).
- [socceraction](https://github.com/ML-KULeuven/socceraction) — VAEP/xT action valuation, KU Leuven. Player-level, event data.
- [KU Leuven ECML 2024 dataset index](https://dtai.cs.kuleuven.be/tutorials/sports/ecml2024/notes/datasets/soccer/).

**The single most actionable practitioner finding:** [penaltyblog, *Football Prediction Models: Which Ones Work the Best?* (Mar 2025)](https://pena.lt/y/2025/03/10/which-model-should-you-use-to-predict-football-matches/) — Eredivisie, 10 seasons, rolling validation on 2023/24: Dixon–Coles 0.1914, Weibull-Count 0.1914, Poisson 0.1915, ZIP 0.1915, NegBin 0.1916, Bivariate Poisson 0.1916 — then **0.1891 after tuning the lookback (~4 seasons) and time-decay (ξ ≈ 0.001)**.

The spread across *model families* is 0.0002. The gain from tuning *temporal weighting* is 0.0023. **Ten times more of the available improvement lives in the time decay than in the choice of distribution.**

---

## 6. Scale reality check

Per season: EPL 380, La Liga 380, Serie A 380, Bundesliga 306, Ligue 1 306 (18 teams since 2023/24; 380 before) ≈ **1,752 matches/season now**, ~1,826 historically.

- 20 seasons of top-5 ≈ **35,000–36,500 matches** (~70,000 team-rows).
- Every European domestic league + cups ≈ 100–150k.
- Worldwide (Open International Soccer DB scope) ≈ 220k. Yeung et al. trained on >300k.

**Is 35k enough for a transformer? No.** That is firmly the small/medium tabular regime where Grinsztajn et al. show trees win. 35k rows × ~50–200 features, a 3-class target with irreducible entropy ~1.06 nats and best-achievable log loss ~0.96, means the total learnable signal is roughly 0.1 nats. Almost nothing for a high-capacity model to find, and enormous room to memorise noise.

Two things change the verdict:

1. **Train multi-league.** Every paper where deep learning is competitive trains on 100k–300k matches across 40+ leagues (Yeung: 300k/51; Kaggle FMPP: 150k/860; Horton & Lucey: 62k/28). **The top-5-only dataset is the wrong dataset.** Pull every league available and let league embeddings and shared rating dynamics do the transfer. **This is the highest-leverage decision in the build.**
2. **Sequence framing and auxiliary targets multiply effective supervision.** Predicting goals for/against, shots, xG, BTTS alongside 1X2 turns one 3-class label into a dozen supervision signals per match at zero data cost.

**Regularization at this scale** — nothing football-specific is published, so this is standard practice stated as such: hidden widths 64–256, 2–3 layers, dropout 0.1–0.3, weight decay, early stopping on a *temporally held-out* season, and above all **strictly forward-chaining time-series CV, never random k-fold**. Random splits leak future information through team form and are the commonest source of inflated published accuracy.

Also worth considering: **TabM** (Gorishniy et al., ICLR 2025, *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*, <https://arxiv.org/abs/2410.24210>) — a parameter-efficient implicit ensemble of MLPs, currently the strongest general-purpose tabular *neural* architecture, and a far more defensible "genuine neural net" than a bigger transformer at this data scale. **TabPFN v2** (Hollmann et al., *Nature*, 2025) is the other small-data option, though constrained on sample count and stronger in regression than classification.

**Hardware is not the constraint.** A 3-layer MLP over 35k–300k rows is seconds to minutes per epoch on CPU. A modest GRU over 10-match sequences for 300k matches is minutes per epoch on an M-series chip. At these model sizes **MPS is often slower than CPU** because kernel-launch overhead dominates — benchmark both. The bottleneck is data-pipeline correctness and the number of experiments runnable, not FLOPs. **Build the leak-free backtest harness first.**

---

## 7. Where the literature is genuinely empty

These are opportunities rather than oversights, and the Tier 2 branch of this project sits inside all of them:

- No pre-match football paper encoding a **starting XI as a permutation-invariant set** (Deep Sets / Set Transformer) with 1X2 output.
- No paper putting a **bivariate-Poisson likelihood head on a neural network** and ablating it against softmax.
- No **pre-match multi-task ablation** — does an xG/shots/possession auxiliary head improve 1X2 RPS? Horton & Lucey prove the head design works in-game; nobody has tested the transfer.
- No **NN vs GBT comparison on rich feature sets** — the Bunker survey names this explicitly as the field's open question.

---

## Could not verify

1. **Wilkens (2026) full text** — SAGE returned 403. The 10–15% ROI and the 1%→10% calibration effect come from the abstract only, and conflict with Pitcan (2026) and Baboota & Kaur (2019). Unresolved.
2. **Kaggle FMPP 1st-place architecture** — score confirmed from the leaderboard; method not.
3. **Whether the Open International Soccer Database contains odds** — inferred from the Bunker survey's framing, not confirmed from Dubitzky et al. (2019).
4. **Hubáček et al. (2019) challenge RPS figures** (0.2063 XGBoost-on-pi; 0.2054 Berrar post-competition) came from a search summary. ⚠️ Note the collision: 0.2063 also appears as the *bookmaker consensus* RPS in the **2023** challenge. **Different test sets. Do not treat as the same number.**
5. **HIGFormer's actual metrics** — the abstract reports none and names no GBM/odds baseline.
6. **Beal et al.'s 63.18%** — stated in the abstract, but with no bookmaker baseline it cannot be compared to the market numbers here.
