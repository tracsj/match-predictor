const state = {
  data: null,
  leagueId: null,
  seasonId: null,
  roundId: null,
  minConfidence: 0.45,
  oddsMin: 1.01,
  oddsMax: 100.0,
  sparklineMode: "profit",
  showLineups: false,
};

const el = (id) => document.getElementById(id);
const STORAGE_KEY = "matchPredictorSettings";

function formatOdds(value) {
  if (value === undefined || value === null) return "--";
  return Number(value).toFixed(2);
}

function formatScore(score) {
  if (!score || score.length !== 2) return "--";
  return `${score[0]}-${score[1]}`;
}

function normalizeOddsWindow(min, max) {
  const minVal = Number.isFinite(min) ? min : 1.01;
  const maxVal = Number.isFinite(max) ? max : 100.0;
  if (minVal > maxVal) return [maxVal, minVal];
  return [minVal, maxVal];
}

function withinOdds(oddsValue, min, max) {
  if (oddsValue === undefined || oddsValue === null) return false;
  const value = Number(oddsValue);
  if (!Number.isFinite(value)) return false;
  return value >= min && value <= max;
}

function computeRoi(fixtures, minConfidence, oddsMin, oddsMax) {
  let bets = 0;
  let wins = 0;
  let profit = 0;
  const [minOdds, maxOdds] = normalizeOddsWindow(oddsMin, oddsMax);
  fixtures.forEach((fixture) => {
    if (!fixture.probs || !fixture.odds || !fixture.scores) return;
    const outcome = fixture.probs.outcome;
    const confidence = fixture.probs.confidence;
    if (confidence < minConfidence) return;
    const odds = fixture.odds[outcome];
    if (!withinOdds(odds, minOdds, maxOdds)) return;
    const actual = fixture.scores;
    const actualLabel = actual[0] > actual[1] ? "W" : actual[0] < actual[1] ? "L" : "D";
    bets += 1;
    if (actualLabel === outcome) {
      wins += 1;
      profit += odds - 1;
    } else {
      profit -= 1;
    }
  });
  if (!bets) {
    return { bets: 0, hitRate: 0, profit: 0, roi: 0 };
  }
  return {
    bets,
    hitRate: wins / bets,
    profit,
    roi: profit / bets,
  };
}

function computeSeasonTimeline(leagueId, seasonId, minConfidence, oddsMin, oddsMax) {
  const league = state.data.leagues[String(leagueId)];
  const season = league?.seasons[String(seasonId)];
  if (!season) return [];
  const rounds = Object.values(season.rounds || {}).sort((a, b) => Number(a.id) - Number(b.id));
  let cumulativeProfit = 0;
  let cumulativeBets = 0;
  let cumulativeWins = 0;
  const [minOdds, maxOdds] = normalizeOddsWindow(oddsMin, oddsMax);
  return rounds.map((round) => {
    const fixtures = (round.fixtures || []).map((fid) => state.data.fixtures[String(fid)]).filter(Boolean);
    let bets = 0;
    let wins = 0;
    let profit = 0;
    fixtures.forEach((fixture) => {
      if (!fixture.probs || !fixture.odds || !fixture.scores) return;
      const outcome = fixture.probs.outcome;
      const confidence = fixture.probs.confidence;
      if (confidence < minConfidence) return;
      const odds = fixture.odds[outcome];
      if (!withinOdds(odds, minOdds, maxOdds)) return;
      const actual = fixture.scores;
      const actualLabel = actual[0] > actual[1] ? "W" : actual[0] < actual[1] ? "L" : "D";
      bets += 1;
      if (actualLabel === outcome) {
        wins += 1;
        profit += odds - 1;
      } else {
        profit -= 1;
      }
    });
    cumulativeProfit += profit;
    cumulativeBets += bets;
    cumulativeWins += wins;
    return {
      id: round.id,
      label: round.min_date ? `${round.min_date} → ${round.max_date}` : `Round ${round.id}`,
      bets,
      wins,
      profit,
      hitRate: bets ? wins / bets : 0,
      cumulativeProfit,
      cumulativeRoi: cumulativeBets ? cumulativeProfit / cumulativeBets : 0,
      cumulativeHit: cumulativeBets ? cumulativeWins / cumulativeBets : 0,
    };
  });
}

function getLeagueOptions() {
  const leagues = Object.values(state.data.leagues || {});
  return leagues
    .map((league) => ({ id: String(league.id), label: league.name || `League ${league.id}` }))
    .sort((a, b) => Number(a.id) - Number(b.id));
}

function getSeasonOptions(leagueId) {
  const league = state.data.leagues[String(leagueId)];
  if (!league) return [];
  return Object.values(league.seasons || {})
    .map((season) => ({ id: String(season.id), label: `Season ${season.id}` }))
    .sort((a, b) => Number(a.id) - Number(b.id));
}

function getRoundOptions(leagueId, seasonId) {
  const league = state.data.leagues[String(leagueId)];
  if (!league) return [];
  const season = league.seasons[String(seasonId)];
  if (!season) return [];
  const rounds = Object.values(season.rounds || {});
  return rounds
    .map((round) => {
      const label = round.min_date ? `Round ${round.id} (${round.min_date} → ${round.max_date})` : `Round ${round.id}`;
      const fixtures = round.fixtures || [];
      const predictedCount = fixtures.filter((fid) => {
        const fixture = state.data.fixtures[String(fid)];
        return fixture && fixture.prediction;
      }).length;
      return {
        id: String(round.id),
        label: `${label} · ${predictedCount} predicted`,
        fixtures,
        predictedCount,
      };
    })
    .filter((round) => round.predictedCount > 0)
    .sort((a, b) => Number(a.id) - Number(b.id));
}

function inferCurrentRound(rounds) {
  if (!rounds.length) return null;
  const today = new Date();
  let candidate = rounds[0];
  rounds.forEach((round) => {
    const fixtureId = round.fixtures && round.fixtures.length ? round.fixtures[0] : null;
    if (!fixtureId) return;
    const fixture = state.data.fixtures[String(fixtureId)];
    if (!fixture || !fixture.starting_at) return;
    const start = new Date(fixture.starting_at.replace(" ", "T") + "Z");
    if (start <= today) {
      candidate = round;
    }
  });
  return candidate.id;
}

function updateSelect(select, options, selectedId) {
  select.innerHTML = "";
  options.forEach((option) => {
    const opt = document.createElement("option");
    opt.value = option.id;
    opt.textContent = option.label || `ID ${option.id}`;
    select.appendChild(opt);
  });
  if (selectedId) select.value = selectedId;
}

function renderFixtureList() {
  const fixturesEl = el("fixtures");
  fixturesEl.innerHTML = "";

  const league = state.data.leagues[String(state.leagueId)];
  const season = league?.seasons[state.seasonId];
  const round = season?.rounds[state.roundId];

  if (!round) {
    fixturesEl.textContent = "No fixtures found for this selection.";
    return;
  }

  const fixtureCards = round.fixtures.map((fixtureId) => {
    const fixture = state.data.fixtures[String(fixtureId)];
    if (!fixture) return null;

    const prediction = fixture.prediction;
    const probs = fixture.probs;
    const odds = fixture.odds;
    const confidence = probs?.confidence ?? 0;
    let suggestion = "PASS";
    if (confidence >= state.minConfidence && probs?.outcome && odds) {
      const oddsValue = odds[probs.outcome];
      if (withinOdds(oddsValue, ...normalizeOddsWindow(state.oddsMin, state.oddsMax))) {
        suggestion = probs.outcome;
      }
    }
    let oddsFavorite = null;
    if (odds) {
      const entries = ["W", "D", "L"]
        .map((key) => ({ key, value: odds[key] }))
        .filter((entry) => entry.value !== undefined && entry.value !== null);
      if (entries.length) {
        entries.sort((a, b) => Number(a.value) - Number(b.value));
        oddsFavorite = entries[0].key;
      }
    }
    const isContrarian =
      suggestion !== "PASS" &&
      oddsFavorite &&
      oddsFavorite !== suggestion &&
      confidence >= state.minConfidence;

    const card = document.createElement("article");
    card.className = `fixture-card ${isContrarian ? "contrarian" : ""}`;
    card.innerHTML = `
      <div class="fixture-header">
        <div class="fixture-title">${fixture.name || "Fixture"}</div>
        <span class="tag">${fixture.starting_at || "--"}</span>
      </div>
      <div class="fixture-grid">
        <div class="metric">
          Predicted goals
          <strong>${prediction ? `${prediction.home.toFixed(2)} - ${prediction.away.toFixed(2)}` : "--"}</strong>
        </div>
        <div class="metric">
          W/D/L probs
          <strong>${probs ? `${probs.W.toFixed(2)} / ${probs.D.toFixed(2)} / ${probs.L.toFixed(2)}` : "--"}</strong>
        </div>
        <div class="metric">
          Odds (W/D/L)
          <strong>${odds ? `${formatOdds(odds.W)} / ${formatOdds(odds.D)} / ${formatOdds(odds.L)}` : "--"}</strong>
        </div>
        <div class="metric">
          Actual
          <strong>${fixture.scores ? formatScore(fixture.scores) : "--"}</strong>
        </div>
      </div>
      <div class="bet ${suggestion === "PASS" ? "pass" : ""}">
        <strong>${suggestion}</strong>
        <span>conf ${confidence.toFixed(2)}</span>
      </div>
    `;

    if (isContrarian) {
      const badge = document.createElement("div");
      badge.className = "contrarian-badge";
      badge.textContent = `Against odds favorite (${oddsFavorite})`;
      card.appendChild(badge);
    }

    if (state.showLineups && fixture.lineups) {
      const lines = document.createElement("div");
      lines.className = "lineups";
      const homeLineup = fixture.lineups[fixture.home_id] || [];
      const awayLineup = fixture.lineups[fixture.away_id] || [];
      lines.innerHTML = `
        <div><strong>${fixture.home_name || "Home"}:</strong> ${homeLineup.join(", ") || "--"}</div>
        <div><strong>${fixture.away_name || "Away"}:</strong> ${awayLineup.join(", ") || "--"}</div>
      `;
      card.appendChild(lines);
    }
    return card;
  });

  fixtureCards.filter(Boolean).forEach((card) => fixturesEl.appendChild(card));

  const roundFixtures = fixtureCards.filter(Boolean).map((card, idx) => {
    const fixtureId = round.fixtures[idx];
    return state.data.fixtures[String(fixtureId)];
  });

  const roi = computeRoi(roundFixtures, state.minConfidence, state.oddsMin, state.oddsMax);
  el("roiSummary").textContent = roi.bets
    ? `$${roi.profit.toFixed(2)} (${(roi.roi * 100).toFixed(1)}%)`
    : "No bets";
  el("betCount").textContent = `${roi.bets}`;
  el("hitRate").textContent = roi.bets ? `${(roi.hitRate * 100).toFixed(1)}%` : "--";
  el("oddsWindow").textContent = `${state.oddsMin.toFixed(2)} - ${state.oddsMax.toFixed(2)}`;

  const timeline = computeSeasonTimeline(
    state.leagueId,
    state.seasonId,
    state.minConfidence,
    state.oddsMin,
    state.oddsMax
  );
  const seasonProfit = timeline.length ? timeline[timeline.length - 1].cumulativeProfit : 0;
  const seasonRoi = timeline.length ? timeline[timeline.length - 1].cumulativeRoi : 0;
  el("seasonProfit").textContent = `$${seasonProfit.toFixed(2)}`;
  el("seasonRoi").textContent = `${(seasonRoi * 100).toFixed(1)}%`;
  const drawdown = computeMaxDrawdown(timeline.map((row) => row.cumulativeProfit));
  el("seasonDrawdown").textContent = `$${drawdown.toFixed(2)}`;
  renderSeasonSparkline(timeline, state.sparklineMode);
  renderSeasonTable(timeline);
}

function computeMaxDrawdown(values) {
  let peak = values[0] || 0;
  let maxDrawdown = 0;
  values.forEach((value) => {
    if (value > peak) peak = value;
    const drawdown = peak - value;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;
  });
  return maxDrawdown;
}

function loadSettings() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    if (stored.minConfidence !== undefined) state.minConfidence = stored.minConfidence;
    if (stored.oddsMin !== undefined) state.oddsMin = stored.oddsMin;
    if (stored.oddsMax !== undefined) state.oddsMax = stored.oddsMax;
    if (stored.sparklineMode) state.sparklineMode = stored.sparklineMode;
    if (stored.showLineups !== undefined) state.showLineups = stored.showLineups;
  } catch (err) {
    return;
  }
}

function saveSettings() {
  const payload = {
    minConfidence: state.minConfidence,
    oddsMin: state.oddsMin,
    oddsMax: state.oddsMax,
    sparklineMode: state.sparklineMode,
    showLineups: state.showLineups,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function renderSeasonSparkline(timeline, mode) {
  const container = el("seasonSparkline");
  container.innerHTML = "";
  if (!timeline.length) {
    container.textContent = "No season data available for this selection.";
    return;
  }
  const values =
    mode === "roi"
      ? timeline.map((row) => row.cumulativeRoi * 100)
      : timeline.map((row) => row.cumulativeProfit);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const width = 600;
  const height = 120;
  const points = values
    .map((value, idx) => {
      const x = (idx / (values.length - 1 || 1)) * (width - 20) + 10;
      const y = height - ((value - min) / range) * (height - 20) - 10;
      return `${x},${y}`;
    })
    .join(" ");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  poly.setAttribute("fill", "none");
  poly.setAttribute("stroke", mode === "roi" ? "#1a6b42" : "#d1582a");
  poly.setAttribute("stroke-width", "2");
  poly.setAttribute("points", points);
  svg.appendChild(poly);
  container.appendChild(svg);
}

function renderSeasonTable(timeline) {
  const container = el("seasonTable");
  container.innerHTML = "";
  if (!timeline.length) return;
  const header = document.createElement("div");
  header.className = "season-row header";
  header.innerHTML = `
    <span>Round</span>
    <span>Bets</span>
    <span>Hit %</span>
    <span>Profit</span>
    <span>Cum Profit</span>
    <span>Cum ROI</span>
  `;
  container.appendChild(header);

  timeline.forEach((row) => {
    const elRow = document.createElement("div");
    elRow.className = "season-row";
    const profitClass = row.profit >= 0 ? "positive" : "negative";
    const cumClass = row.cumulativeProfit >= 0 ? "positive" : "negative";
    elRow.innerHTML = `
      <span>${row.label}</span>
      <span>${row.bets}</span>
      <span>${row.bets ? (row.hitRate * 100).toFixed(1) + "%" : "--"}</span>
      <span class="${profitClass}">$${row.profit.toFixed(2)}</span>
      <span class="${cumClass}">$${row.cumulativeProfit.toFixed(2)}</span>
      <span>${(row.cumulativeRoi * 100).toFixed(1)}%</span>
    `;
    container.appendChild(elRow);
  });
}

function initializeSelectors() {
  const leagueSelect = el("leagueSelect");
  const seasonSelect = el("seasonSelect");
  const roundSelect = el("roundSelect");

  const leagues = getLeagueOptions();
  state.leagueId = leagues[0]?.id;
  updateSelect(leagueSelect, leagues, state.leagueId);

  const seasons = getSeasonOptions(state.leagueId);
  state.seasonId = seasons[seasons.length - 1]?.id;
  updateSelect(seasonSelect, seasons, state.seasonId);

  const rounds = getRoundOptions(state.leagueId, state.seasonId);
  state.roundId = inferCurrentRound(rounds) || rounds[0]?.id;
  updateSelect(roundSelect, rounds, state.roundId);

  leagueSelect.addEventListener("change", (event) => {
    state.leagueId = String(event.target.value);
    const newSeasons = getSeasonOptions(state.leagueId);
    state.seasonId = newSeasons[newSeasons.length - 1]?.id;
    updateSelect(seasonSelect, newSeasons, state.seasonId);
    const newRounds = getRoundOptions(state.leagueId, state.seasonId);
    state.roundId = inferCurrentRound(newRounds) || newRounds[0]?.id;
    updateSelect(roundSelect, newRounds, state.roundId);
    renderFixtureList();
  });

  seasonSelect.addEventListener("change", (event) => {
    state.seasonId = String(event.target.value);
    const newRounds = getRoundOptions(state.leagueId, state.seasonId);
    state.roundId = inferCurrentRound(newRounds) || newRounds[0]?.id;
    updateSelect(roundSelect, newRounds, state.roundId);
    renderFixtureList();
  });

  roundSelect.addEventListener("change", (event) => {
    state.roundId = String(event.target.value);
    renderFixtureList();
  });

  el("confidenceInput").addEventListener("input", (event) => {
    state.minConfidence = Number(event.target.value);
    saveSettings();
    renderFixtureList();
  });

  el("oddsMinInput").addEventListener("input", (event) => {
    state.oddsMin = Number(event.target.value);
    saveSettings();
    renderFixtureList();
  });

  el("oddsMaxInput").addEventListener("input", (event) => {
    state.oddsMax = Number(event.target.value);
    saveSettings();
    renderFixtureList();
  });

  el("sparklineToggle").addEventListener("change", (event) => {
    state.sparklineMode = event.target.checked ? "roi" : "profit";
    saveSettings();
    renderFixtureList();
  });

  el("lineupToggle").addEventListener("change", (event) => {
    state.showLineups = event.target.checked;
    saveSettings();
    renderFixtureList();
  });

  el("resetFilters").addEventListener("click", () => {
    state.minConfidence = 0.45;
    state.oddsMin = 1.01;
    state.oddsMax = 100.0;
    state.sparklineMode = "profit";
    state.showLineups = false;
    el("confidenceInput").value = state.minConfidence;
    el("oddsMinInput").value = state.oddsMin;
    el("oddsMaxInput").value = state.oddsMax;
    el("sparklineToggle").checked = false;
    el("lineupToggle").checked = false;
    saveSettings();
    renderFixtureList();
  });
}

async function init() {
  const response = await fetch("data.json");
  state.data = await response.json();

  el("generatedAt").textContent = state.data.generated_at || "--";
  el("bookmakerId").textContent = state.data.bookmaker_id ?? "best available";

  const bundleNotice = el("bundleNotice");
  const bundleScope = el("bundleScope");
  if (state.data.quick_limit || state.data.filters?.league_id || state.data.filters?.season_id) {
    const parts = [];
    if (state.data.quick_limit) {
      parts.push(`Last ${state.data.quick_limit} fixture dates`);
    }
    if (state.data.filters?.league_id) {
      parts.push(`League ${state.data.filters.league_id}`);
    }
    if (state.data.filters?.season_id) {
      parts.push(`Season ${state.data.filters.season_id}`);
    }
    if (state.data.filters?.window_days) {
      parts.push(`Window ${state.data.filters.window_days}d`);
    }
    if (state.data.filters?.retrain_days) {
      parts.push(`Retrain ${state.data.filters.retrain_days}d`);
    }
    bundleScope.textContent = parts.join(" · ");
    bundleNotice.hidden = false;
  }

  loadSettings();
  el("confidenceInput").value = state.minConfidence;
  el("oddsMinInput").value = state.oddsMin;
  el("oddsMaxInput").value = state.oddsMax;
  el("sparklineToggle").checked = state.sparklineMode === "roi";
  el("lineupToggle").checked = state.showLineups;

  initializeSelectors();
  renderFixtureList();
}

init();
