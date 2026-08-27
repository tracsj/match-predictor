# Measured facts: data availability

**Measured 2026-08-15. Re-check before quoting — API entitlements, vendor pricing and dataset column sets all drift.**

Everything here came from an actual probe, not from documentation. The command that produced each finding is recorded alongside it, so a later session can re-run the check rather than having to trust or discard the claim.

---

## SportMonks v3 — what the existing key actually reaches

Auth is `?api_token=` as a query parameter (not a header). The token lives in `.env` as `SPORTMONKS_API_TOKEN`.

### Plan tier

```bash
set -a; . ./.env; set +a
curl -s "https://api.sportmonks.com/v3/football/leagues?api_token=$SPORTMONKS_API_TOKEN&per_page=50"
```

Returns `HTTP 200` with a `subscription` block reading `{"plan": "Football Free Plan", "sport": "Football", "category": "Standard"}` and a `rate_limit` block showing **3,000 requests per hour**, resetting hourly, counted **per entity type** (the `requested_entity` field names which).

### League entitlement — four ids, two real leagues

| id | name | note |
|---|---|---|
| 271 | Danish Superliga | |
| 501 | Scottish Premiership | |
| 513 | Scottish Premiership Play-Offs | |
| 1659 | Danish Superliga Play-offs | |

`pagination.has_more` is `false`, so this is the complete list. **No top-5 European league is reachable on the free plan.**

### Season entitlement — 22 seasons per league, far more than v1 used

```bash
for p in 1 2; do curl -s "https://api.sportmonks.com/v3/football/seasons?api_token=$T&per_page=50&page=$p"; done
```

59 unique seasons: 22 for league 271, 22 for 501, 13 for 513, 2 for 1659. Names span **2005/2006 → 2026/2027**. Note the paging quirk: entitlement filtering appears to happen *after* pagination, so page 1 returns 50 and page 2 returns 9 even with `per_page=50`. Do not trust a single page.

**v1 pulled only 2 seasons per league** (`SPORTMONKS_SEASONS_PER_LEAGUE=2`), so the free plan was being used at roughly a tenth of its depth.

### Fetching fixtures for a historical season

The obvious endpoint does **not** exist:

```
GET /v3/football/fixtures/seasons/{id}   ->  {"message": "The requested endpoint does not exist"}
```

What works:

```bash
curl -s "https://api.sportmonks.com/v3/football/seasons/1937?api_token=$T&include=fixtures"
```

Returns the season object with a `fixtures` array (228 entries for Scottish Premiership 2015/16). `GET /v3/football/fixtures?filters=fixtureSeasons:{id}` also works but returns no `pagination.total`, which makes it harder to verify completeness.

### Data richness by era — the boundary that matters

Probed by taking the middle fixture of each season and requesting `include=participants;scores;lineups;statistics;events;odds`:

| Season | lineup entries | team stats | events | odds rows |
|---|---|---|---|---|
| SCO 2005/06 | 0 | 0 | 0 | 0 |
| SCO 2011/12 | 35 | 0 | 14 | 0 |
| SCO 2015/16 | 35 | 6 | 15 | 0 |
| SCO 2018/19 | 35 | 10 | 12 | 1,727 |
| SCO 2022/23 | 37 | 14 | 18 | 4,488 |
| DEN 2019/20 | 36 | 14 | 14 | 3,891 |

**Odds begin at 2018/19.** But team-level `statistics` counts hide the more important boundary — per-player detail. Probed separately with `include=lineups.details`, counting distinct `type_id` values across all lineup entries:

| Season | distinct player stat types | detail rows |
|---|---|---|
| SCO 2018/19 | **6** | 52 |
| SCO 2019/20 | 35 | 474 |
| SCO 2020/21 | 36 | 436 |
| SCO 2021/22 | 35 | 441 |
| SCO 2022/23 | 36 | 405 |
| DEN 2019/20 | 37 | 461 |
| DEN 2020/21 | 41 | 491 |
| DEN 2021/22 | 40 | 522 |
| DEN 2022/23 | 38 | 486 |

**Rich per-player statistics begin at 2019/20, not 2018/19.** 2018/19 has odds but almost no player detail, so it is not usable for the player-encoder branch.

⇒ **usable player-level window: 2019/20 → 2025/26, both leagues, ≈3,000 matches.**

### Odds market filtering — a 40× storage difference

```bash
# all markets
curl -s ".../odds/pre-match/fixtures/18535561?api_token=$T"              # 1,996,667 bytes
# one market
curl -s ".../odds/pre-match/fixtures/18535561/markets/1?api_token=$T"    #    50,300 bytes
```

Unfiltered: 4,488 odd rows spanning ~30 markets (Correct Score 618, Correct Score 1st Half 301, Goals O/U 281, Asian Handicap 236, …). Filtered to `markets/1` (Fulltime Result): **115 rows ≈ 38 bookmakers × 3 outcomes.**

v1 stored everything unfiltered and read only Match Winner, which is why `data/odds/` is 1.35 GB — 96% of the repo — for information that fits in roughly 35 MB.

### Upgrade pricing — read off the marketing page, NOT verified against an account

From <https://www.sportmonks.com/football-api/plans-pricing/> on 2026-08-15:

| Plan | Price | Leagues | Rate limit |
|---|---|---|---|
| Starter | €29/mo | any 5 worldwide | 2,000/hr per entity |
| Growth | €99/mo | any 30 worldwide | 2,500/hr per entity |
| Pro | €249/mo | any 120 worldwide | 3,000/hr per entity |
| Enterprise | custom | all 2,200+ | 5,000/hr per entity |

14-day free trial on the paid tiers; yearly billing advertised at 20% off.

⚠️ **Three things are unverified and all of them matter before money moves:** whether Starter includes the same 22-season history depth the free tier grants for its leagues (the page does not say); whether the 5 leagues can be changed after selection; and whether these prices are current. The free tier grants deep history for its two leagues, so depth *probably* follows the league grant — but that is an inference. **Use the free trial to check, don't pay first.**

---

## football-data.co.uk — free, and the backbone of the v2 corpus

No key, no login, no practical rate limit. Two URL shapes:

```
https://www.football-data.co.uk/mmz4281/{season}/{div}.csv   # e.g. 2324/E0.csv
https://www.football-data.co.uk/new/{country}.csv            # e.g. DNK.csv
```

Column key and odds provenance: <https://www.football-data.co.uk/notes.txt>

### Breadth — 22 main divisions, measured for 2023/24

Downloaded all 22 and counted rows with a `Date`:

| Div | Matches | | Div | Matches | | Div | Matches |
|---|---|---|---|---|---|---|---|
| E0 | 380 | | SC0 | 228 | | SP1 | 380 |
| E1 | 552 | | SC1 | 180 | | SP2 | 462 |
| E2 | 552 | | SC2 | 180 | | F1 | 306 |
| E3 | 552 | | SC3 | 180 | | F2 | 379 |
| EC | 552 | | D1 | 306 | | N1 | 306 |
| I1 | 380 | | D2 | 306 | | B1 | 312 |
| I2 | 380 | | | | | P1 | 306 |
| | | | | | | T1 | 380 |
| | | | | | | G1 | 240 |

**Total: 7,799 matches in one season**, every division carrying `PSCH` (Pinnacle closing).

### Depth — what exists when

Probed `E0` across seasons for column presence:

| Season | cols | shots | B365 open | Pinnacle open | Pinnacle CLOSE | full close suite | `Time` |
|---|---|---|---|---|---|---|---|
| 1993/94 | 8 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 2000/01 | 45 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 2005/06 | 68 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| 2010/11 | 71 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| 2011/12 | — | ✓ | ✓ | ✓ | **✗** | ✗ | ✗ |
| **2012/13** | — | ✓ | ✓ | ✓ | **✓** | ✗ | ✗ |
| 2015/16 | 65 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| **2019/20** | 106 | ✓ | ✓ | ✓ | ✓ | **✓** | **✓** |
| 2023/24 | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2025/26 | 132 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Two boundaries to design around:

1. **Pinnacle closing odds (`PSCH`/`PSCD`/`PSCA`) start at 2012/13.** Absent in 2011/12, present every season since. ⇒ **~109,000 matches with genuine closing odds** (14 seasons × 7,799).
2. **The `Time` column only exists from 2019/20.** Before that, same-day fixtures cannot be ordered, so any rolling feature risks same-matchday contamination. Drop the matchday from the feature window rather than pretend the ordering is known.

The full closing suite from 2019/20 adds per-book closing 1X2 (`B365C*`), `MaxC*`/`AvgC*`, closing O/U 2.5, closing Asian handicap, and Betfair Exchange closing (`BFEC*`).

**Collection timing caveat**: per <https://www.football-data.co.uk/matches.php>, "pre-close" odds are snapshotted Friday ≤17:00 BST for weekend fixtures and Tuesday ≤13:00 for midweek. So the pre-close price is a genuine 1–3 day-out price — not an opening price, and not a last-second one.

### ⚠️⚠️ Pinnacle is GONE from 2026/27 — the columns no longer exist

**Measured 2026-08-17. This supersedes the section below, which understates the
situation: the decay described there does not end in a gap, it ends in removal.**

```bash
curl -s https://www.football-data.co.uk/mmz4281/2627/B1.csv | tr -d '\r' \
  | sed '1s/^\xEF\xBB\xBF//' | head -1 | tr ',' '\n' | grep -cxE 'PSH|PSD|PSA|PSCH|PSCD|PSCA'
# 0
```

`PSH`, `PSD`, `PSA`, `PSCH`, `PSCD`, `PSCA`, `P>2.5`, `PAHH` are all absent from
the 2026/27 schema — not empty, absent. The `PP*` columns in that file are
**Paddy Power**, not Pinnacle (`notes.txt:86-91`). The last populated `PSCH`
anywhere in the corpus is **2026-01-14**.

Two other 2026/27 schema changes in the same diff:

- **`HxG`/`AxG` (expected goals) are NEW**, per-division rather than universal —
  present in B1 and N1, absent in EC. The loader does not map them, so they are
  currently parsed away.
- Bookmaker roster churned: Coral, Ladbrokes and BetMGM out, Skybet (`SKB*`) in.
  `Referee` was dropped from results files but survives in `fixtures.csv`.

**Not established:** whether the removal is deliberate or a feed outage. The site
still links a "Pinnacle Closing Odds Bet Tracker" (`matches.php:108`) and no
announcement was found. Re-check before quoting, and treat it as permanent for
planning purposes.

#### The replacement, and it is not a downgrade

The standing rule is to lead the sharpest price. The Betfair Exchange is now that
price, and it is in the feed on both legs — `BFEH/BFED/BFEA` pre-close,
`BFEC*` closing. Measured on the **16,875 matches carrying both closes**
(`scripts`-free reproduction: de-vig with `src.eval.devig.devig(method="shin")`,
score with `src.eval.metrics.rps`):

| book | mean overround | median overround | de-vigged RPS |
|---|---|---|---|
| Pinnacle close | 1.0389 | 1.0362 | 0.20408 |
| **exchange close** | **1.0089** | **1.0076** | **0.20404** |
| Bet365 close | 1.0698 | 1.0669 | 0.20327 |

Exchange prices run **3.9% longer** than Pinnacle's on the same matches (mean
ratio 1.0386; home 1.0313, draw 1.0311, away 1.0534).

**So the exchange close is an equally accurate estimate of the truth on a quarter
of the margin.** CLV against it is at least as demanding as against Pinnacle,
because there is less vig to hide in — the intuitive reading, that losing
Pinnacle means accepting a softer benchmark, is wrong. What *is* flattered is
exchange ROI, which is pre-commission; 2–5% of net winnings would absorb most of
that 3.9%. CLV is immune, since both legs are exchange prices.

**Exchange coverage begins in 2024/25**, so this substitution works forward and
not backward:

| season | matches | `bfec*` | `bfe*` pre | `psc*` |
|---|---|---|---|---|
| 2021-22 → 2023-24 | ~7,800/yr | 0 | 0 | ~complete |
| 2024-25 | 7,681 | 7,680 | 7,632 | 7,681 |
| 2025-26 | 7,646 | 7,205 | 7,131 | 2,964 |
| 2026-27 | 99 | 99 | 98 | 0 |

Anything testing the 2012/13–2024/25 panel still needs Pinnacle. Anything forward
uses the exchange.

#### Pinnacle closing coverage is UNIFORM across division tiers

Measured 2026-08-17 from the raw per-season files. Relevant because H1 assumes
lower divisions are answerable, and the intuitive guess — that a sharp book
covers League Two worse than the Premier League — is wrong.

`PSCH` is ~100% populated in **every** tier from 2012/13 through 2024/25: E0, E1,
E2, E3, EC, SC0, SC1, D1, D2, I1, I2, SP1, SP2, F1, F2, N1, B1, P1, T1, G1 all
show `rows == PSCH` for 2023/24 and 2024/25.

Two exceptions worth carrying:

- **SC2 and SC3 begin at 2016/17**, not 2012/13. Not previously recorded.
- 2025/26 pooled coverage is **2,964/7,647 ≈ 38.8%** and zero after 2026-01-14.

So H1's "zero new data" claim holds, capped at 2024/25.

### ⚠️ Pinnacle closing odds stop partway through 2025/26

*(Superseded by the section above — kept because the monthly decay profile is
still the record of how it happened. The conclusion "evaluate against Bet365/Avg
and label it a softer benchmark" is no longer the best available answer: use the
exchange close, which is not softer.)*

Measured from the built corpus (233,687 main-division matches), counting non-null `PSCH` by month for season 2025-26:

| Month | matches | Pinnacle close | Bet365 close | Avg close |
|---|---|---|---|---|
| 2025-07 | 8 | 8 | 8 | 8 |
| 2025-08 | 744 | 744 | 744 | 744 |
| 2025-09 | 713 | 713 | 713 | 713 |
| 2025-10 | 709 | **668** | 709 | 709 |
| 2025-11 | 793 | **307** | 793 | 793 |
| 2025-12 | 762 | **365** | 762 | 762 |
| 2026-01 | 822 | **159** | 822 | 822 |
| 2026-02 | 880 | **0** | 880 | 880 |
| 2026-03 | 820 | **0** | 820 | 820 |
| 2026-04 | 868 | **0** | 868 | 868 |
| 2026-05 | 527 | **0** | 527 | 527 |

**Pinnacle closing coverage degrades from October 2025 and stops entirely from February 2026.** Bet365 closing and market-average closing remain complete throughout.

A search summary had claimed the Pinnacle feed became unreliable "since 2025-07-23" and was excluded from Max/Avg. The direction is right and the date is wrong — August and September 2025 are complete, and the decay starts in October.

**Consequence for the backtest, and it is load-bearing:** the truth test is "de-vigged Pinnacle closing", and **2025/26 cannot supply it.** Either end the locked holdout at 2024/25, or evaluate 2025/26 against Bet365/Avg closing and label that column honestly as a softer benchmark. Do not quietly fall back to `AvgC` and keep calling it the closing line.

Across the whole corpus (main + extra) **160,868 matches carry Pinnacle closing odds** — more than the 109k estimated from main divisions alone, because the extra-country files carry `PSCH` too.

### What `fixtures.csv` actually contains

Measured 2026-08-17. This is the feed the forward path reads, and four of its
properties are load-bearing.

```bash
curl -s https://www.football-data.co.uk/fixtures.csv | tr -d '\r' | head -1        # 94 cols
curl -s https://www.football-data.co.uk/fixtures.csv | tr -d '\r' | awk 'NR>1 && $0!~/^,*$/' | wc -l   # 127
```

1. **It is a rolling ~4-day window, not a season fixture list.** The snapshot held
   127 rows across 14 of the 22 main divisions, dated 14–17 August. **So a
   schedule reading it must fire at least every four days or fixtures are never
   predicted at all.** A weekly cron silently drops them.
2. **No Pinnacle.** `PSH/PSD/PSA` absent entirely, consistent with the removal
   above. Populated: `B365H/D/A` 127/127, `MaxH/D/A` 127/127, `AvgH/D/A` 127/127,
   **`BFEH/BFED/BFEA` 117/127**. Every `*C*` closing column is present in the
   header and **0/127** populated, as it must be before kickoff.
3. **It retains already-played fixtures**, so filtering on kickoff is mandatory
   rather than defensive. `Time` is 127/127 populated and UK local.
4. **93 of its 94 columns exist in the 2026/27 results schema** (only `Referee`
   differs), so it is the current-season results file minus post-match columns.
   The mapping problem is availability, not naming.

Bare `curl` works — no browser user-agent needed. UTF-8 BOM on column 1.

**A second, separate feed covers the extra-country leagues:**
`new_league_fixtures.csv`, 116 rows, 14 of the 16 loader countries (no RUS, no
SWZ), schema `Country,League,Date,Time,Home,Away,PSH,...`. Its `PSH/PSD/PSA` are
**declared but 0/116 populated**. Not read by `src/data/fixtures.py`.

### The exchange pre-close is a thin snapshot, and its drift is nothing like Pinnacle's

Measured 2026-08-27, over the first 167 settled forward predictions. This is the
fact that makes the forward ledger's CLV null believable rather than surprising.

`BFEH/BFED/BFEA` reach the corpus only through `fixtures.csv`, collected Tuesday
≤13:00 and Friday ≤17:00 UK — **a median 21 hours before kickoff**, against a
Betfair market that is still thin. The book shows it:

| | overround |
|---|---|
| exchange pre-close (`bfe*`, from `fixtures.csv`) | **1.0603** |
| exchange close (`bfec*`) | **1.0213** |

It tightens in **88% of rows**. So prices lengthen by default on this ladder,
and only **31.81%** of band-eligible selections shortened — far below Pinnacle's
45–48%, whose pre-close is a mature price rather than a day-out snapshot.

**Do not import Pinnacle's null here.** Reading this ladder against 45–48%, or
against 50%, misstates it by 13 to 18 points, which is larger than any margin
this project has ever claimed.

Odds-matching does not move it: matched 0.3169 against unmatched 0.3181, a gap
of 0.0012 on this ladder's own cells. Re-derivable, and it should be re-run as
the forward corpus grows:

```bash
PYTHONPATH=. uv run python scripts/forward_matched_null.py
uv run python -m src.grade        # the overround line prints in docs/FORWARD_LEDGER.md
```

The second command is the durable one — `src/grade.py` now measures the
overround both ways on every run, so this fact maintains itself rather than
going stale in a document.

### `download_all` cannot refresh, and `_missing.json` is the worse half

Measured 2026-08-17, and it is the failure that would have quietly hollowed out
the forward ledger.

`download_all` skips any file already on disk *and* any key memoised in
`_missing.json` — no mtime check, no force flag. So on a warm cache the
current-season file is fetched exactly once and never updated.

The second cache is the trap. Eight of the 2026/27 divisions were already
memoised missing, because their files 404 upstream before the season starts:

```
main/2627/D1  main/2627/E1  main/2627/E2  main/2627/F1
main/2627/G1  main/2627/I1  main/2627/I2  main/2627/T1
```

Meanwhile `fixtures.csv` was already carrying E1 and E2 fixtures. Deleting the
files without purging those keys leaves those eight divisions permanently
unfetched: predictions get made, nothing errors, and the ledger simply reports
nothing for a third of the corpus.

`footballdata.refresh_current()` does both, and `src/refresh.py` fails loudly if
the current-season file count falls. Verified: the refresh retried all eight
(they 404 again — genuinely not published yet) and the corpus grew 243 → 253
matches, 17 → 18 divisions, where `download_all` alone gained nothing.

**Results lag the fixtures feed.** On 2026-08-17 the latest result in the corpus
was 2026-08-10 while the feed already listed 14–17 August. Grading is therefore
always a few days behind prediction, by design rather than by fault.

### football-data serves a SUBSTITUTE file for a division-season that does not exist

Not documented anywhere, and it silently triples match counts if you trust filenames.

Requesting `mmz4281/9394/P1.csv`, `.../SC1.csv`, `.../SP2.csv` and `.../SP1.csv` returns **four byte-identical files**, all containing Spanish La Liga 1993/94 with `Div=SP1`. Same for 2026/27, where `E0.csv`, `E3.csv` and `EC.csv` all return the same National League rows with `Div=EC`. This is the same server behaviour that produces the HTTP 300 "Multiple Choices" responses.

**Two defences, both required:** trust the file's own `Div` column over the filename, and deduplicate on `(div, date, home, away)`. Together these collapsed 3,278 rows into 1,253 real matches. Verified that nothing real is lost — the substituted division-seasons genuinely do not exist upstream.

**One check that looked like a bug and was not:** 554 matches carry a season label disagreeing with a July–June season boundary. All 554 are the COVID-delayed 2019/20 season running into July and August 2020. The data is right; the heuristic was wrong.

**Provenance caveat**: `notes.txt` names Betbrain, Oddsportal and individual bookmakers as sources. `Max`/`Avg` are aggregator-derived, so "market max" is a price *someone somewhere showed*, not necessarily one that could have been taken in size.

### The overlap that makes the two-tier design work

Both SportMonks free-plan leagues are also here, with Pinnacle closing odds:

- **Scotland** — `SC0` in the main files.
- **Denmark** — `new/DNK.csv`: 2,968 rows, seasons 2012/13 → 2026/27, columns `Country, League, Season, Date, Time, Home, Away, HG, AG, Res, PSCH/D/A, MaxCH/D/A, AvgCH/D/A, BFECH/D/A, B365CH/D/A`.

Note what the extra-league files do **not** have: no pre-close odds, no shots/corners/cards, no O/U, no Asian handicap. **Closing odds only.** So for Denmark you can measure calibration against the close, but you cannot simulate "bet at a pre-close price" the way you can for Scotland.

---

## Hardware

Apple M5, 10 cores, 16 GB RAM, macOS 26.2. `uv` 0.11.6 present. System Python is 3.9.6; the project uses its own 3.12 venv.

Compute is not a constraint for anything in this plan. A 3-layer network over 100k rows is seconds per epoch on CPU. Note that at this model size **MPS is frequently slower than CPU** because kernel-launch overhead dominates — benchmark both rather than assuming the GPU path is faster.
