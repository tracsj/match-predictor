import csv
import json
import os
import sys
import time
import datetime as dt

import requests


def load_dotenv(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def parse_formation_field(value):
    if not value or ":" not in value:
        return None
    left, right = value.split(":", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def build_adjacency_edges(lineups):
    edges = []
    team_index = {}
    for entry in lineups:
        team_id = entry.get("team_id")
        if not team_id:
            continue
        team_index.setdefault(team_id, []).append(entry)

    for team_id, entries in team_index.items():
        parsed = []
        for entry in entries:
            coords = parse_formation_field(entry.get("formation_field"))
            if coords is None:
                continue
            parsed.append((entry, coords))

        for i, (entry_a, (row_a, col_a)) in enumerate(parsed):
            for entry_b, (row_b, col_b) in parsed[i + 1:]:
                if abs(row_a - row_b) + abs(col_a - col_b) == 1:
                    edges.append({
                        "fixture_id": entry_a.get("fixture_id"),
                        "team_id": team_id,
                        "player_id_a": entry_a.get("player_id"),
                        "player_id_b": entry_b.get("player_id"),
                        "method": "formation_adjacent",
                        "weight": 1.0,
                    })
    return edges


def extract_event_minute(event):
    base = event.get("minute")
    extra = event.get("extra_minute")
    minute = None
    if isinstance(base, int):
        minute = base
    elif isinstance(base, str) and base.isdigit():
        minute = int(base)
    if minute is None:
        return None
    if isinstance(extra, int):
        minute += extra
    elif isinstance(extra, str) and extra.isdigit():
        minute += int(extra)
    return minute


def build_minutes_map(lineups, events, match_minutes):
    minutes = {}
    player_types = {}
    for entry in lineups:
        player_id = entry.get("player_id")
        if not player_id:
            continue
        player_types[player_id] = entry.get("type_id")
        if entry.get("type_id") == 11:
            minutes[player_id] = match_minutes
        else:
            minutes[player_id] = 0

    for event in events or []:
        is_sub = False
        event_type = event.get("type") or {}
        name = (event_type.get("name") or event_type.get("short_name") or "").lower()
        if "substitution" in name or event.get("type_id") == 18:
            is_sub = True
        if not is_sub:
            continue

        minute = extract_event_minute(event)
        if minute is None:
            continue

        player_id = event.get("player_id")
        related_id = event.get("related_player_id")

        off_id = player_id
        on_id = related_id

        if isinstance(player_id, int) and isinstance(related_id, int):
            player_type = player_types.get(player_id)
            related_type = player_types.get(related_id)
            if player_type == 12 and related_type == 11:
                on_id = player_id
                off_id = related_id
            elif player_type == 11 and related_type == 12:
                on_id = related_id
                off_id = player_id

        if isinstance(off_id, int) and off_id in minutes:
            minutes[off_id] = max(0, min(match_minutes, minute))
        if isinstance(on_id, int) and on_id in minutes:
            minutes[on_id] = max(0, match_minutes - minute)

    return minutes


def build_shared_minutes_edges(lineups, match_minutes, minutes_map):
    edges = []
    team_index = {}
    for entry in lineups:
        team_id = entry.get("team_id")
        if not team_id:
            continue
        team_index.setdefault(team_id, []).append(entry)

    for team_id, entries in team_index.items():
        players = []
        for entry in entries:
            minutes = minutes_map.get(entry.get("player_id"))
            if minutes is None:
                if entry.get("type_id") == 11:
                    minutes = match_minutes
                else:
                    minutes = 0
            if minutes <= 0:
                continue
            players.append((entry, minutes))

        for i, (entry_a, minutes_a) in enumerate(players):
            for entry_b, minutes_b in players[i + 1:]:
                overlap = min(minutes_a, minutes_b)
                weight = overlap / match_minutes if match_minutes else 0
                edges.append({
                    "fixture_id": entry_a.get("fixture_id"),
                    "team_id": team_id,
                    "player_id_a": entry_a.get("player_id"),
                    "player_id_b": entry_b.get("player_id"),
                    "method": "shared_minutes",
                    "weight": round(weight, 4),
                    "minutes_overlap": overlap,
                })
    return edges


def compute_synergy(interactions):
    team_stats = {}
    for edge in interactions:
        team_id = edge.get("team_id")
        if team_id is None:
            continue
        stats = team_stats.setdefault(team_id, {
            "edges_total": 0,
            "edges_adjacency": 0,
            "edges_shared_minutes": 0,
            "adjacency_weight_sum": 0.0,
            "shared_minutes_weight_sum": 0.0,
            "shared_minutes_overlap_sum": 0,
        })

        stats["edges_total"] += 1
        method = edge.get("method")
        weight = edge.get("weight") or 0.0
        if method == "formation_adjacent":
            stats["edges_adjacency"] += 1
            stats["adjacency_weight_sum"] += float(weight)
        elif method == "shared_minutes":
            stats["edges_shared_minutes"] += 1
            stats["shared_minutes_weight_sum"] += float(weight)
            stats["shared_minutes_overlap_sum"] += int(edge.get("minutes_overlap") or 0)

    for stats in team_stats.values():
        shared_count = stats["edges_shared_minutes"] or 1
        stats["shared_minutes_weight_avg"] = round(
            stats["shared_minutes_weight_sum"] / shared_count, 4
        )
        stats["synergy_score"] = round(
            stats["shared_minutes_weight_sum"] + stats["adjacency_weight_sum"], 4
        )
    return team_stats


def extract_participants(data):
    participants = data.get("participants", []) or []
    by_id = {}
    for team in participants:
        meta = team.get("meta") or {}
        by_id[team.get("id")] = {
            "team_id": team.get("id"),
            "name": team.get("name"),
            "location": meta.get("location"),
            "winner": meta.get("winner"),
            "position": meta.get("position"),
        }
    return by_id


def extract_goals_by_team(data):
    goals = {}
    for score in data.get("scores", []) or []:
        if score.get("description") != "CURRENT":
            continue
        participant_id = score.get("participant_id")
        value = score.get("score") or {}
        goals_val = value.get("goals")
        if participant_id is not None and isinstance(goals_val, int):
            goals[participant_id] = goals_val
    return goals


def parse_datetime(value):
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_formation(formation):
    if not formation or "-" not in formation:
        return None
    parts = []
    for part in formation.split("-"):
        try:
            parts.append(int(part))
        except ValueError:
            return None
    return parts


def extract_formations(data):
    formations = {}
    for entry in data.get("formations", []) or []:
        team_id = entry.get("participant_id")
        if team_id:
            formations[team_id] = entry.get("formation")
    return formations


def extract_weather(data):
    weather = data.get("weatherreport") or {}
    current = weather.get("current") or {}
    def parse_pct(value):
        if isinstance(value, str) and value.endswith("%"):
            value = value[:-1]
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return {
        "weather_temp": current.get("temp") or weather.get("temperature", {}).get("day"),
        "weather_humidity": parse_pct(current.get("humidity") or weather.get("humidity")),
        "weather_wind": (current.get("wind") or weather.get("wind", {})).get("speed"),
        "weather_pressure": current.get("pressure") or weather.get("pressure"),
        "weather_clouds": parse_pct(current.get("clouds") or weather.get("clouds")),
    }


def build_lineup_context_features(
    starters,
    team_id,
    player_history,
    pair_history,
    min_history=1,
    history_window=None,
):
    base_for = {}
    base_against = {}
    for player_id in starters:
        history = player_history.get(team_id, {}).get(player_id)
        if not history or history["matches"] < min_history:
            continue
        goals_for_list = history.get("goals_for_list") or []
        goals_against_list = history.get("goals_against_list") or []
        if history_window:
            goals_for_list = goals_for_list[-history_window:]
            goals_against_list = goals_against_list[-history_window:]
        if not goals_for_list or not goals_against_list:
            continue
        base_for[player_id] = sum(goals_for_list) / len(goals_for_list)
        base_against[player_id] = sum(goals_against_list) / len(goals_against_list)

    base_for_values = list(base_for.values())
    base_against_values = list(base_against.values())
    avg_player_quality_for = (
        round(sum(base_for_values) / len(base_for_values), 4) if base_for_values else None
    )
    avg_player_quality_against = (
        round(sum(base_against_values) / len(base_against_values), 4) if base_against_values else None
    )

    lift_for_sum = 0.0
    lift_against_sum = 0.0
    lift_pairs = 0
    per_player_lift_for = {pid: [] for pid in starters}
    per_player_lift_against = {pid: [] for pid in starters}

    for i, player_a in enumerate(starters):
        for player_b in starters[i + 1:]:
            pair_key = (player_a, player_b) if player_a < player_b else (player_b, player_a)
            pair_stats = pair_history.get(team_id, {}).get(pair_key)
            if not pair_stats or pair_stats["matches"] < min_history:
                continue
            pair_for_list = pair_stats.get("goals_for_list") or []
            pair_against_list = pair_stats.get("goals_against_list") or []
            if history_window:
                pair_for_list = pair_for_list[-history_window:]
                pair_against_list = pair_against_list[-history_window:]
            if not pair_for_list or not pair_against_list:
                continue
            pair_avg_for = sum(pair_for_list) / len(pair_for_list)
            pair_avg_against = sum(pair_against_list) / len(pair_against_list)

            if player_a not in base_for or player_b not in base_for:
                continue

            expected_for = (base_for[player_a] + base_for[player_b]) / 2
            expected_against = (base_against[player_a] + base_against[player_b]) / 2
            lift_for = pair_avg_for - expected_for
            lift_against = pair_avg_against - expected_against

            lift_for_sum += lift_for
            lift_against_sum += lift_against
            lift_pairs += 1
            per_player_lift_for[player_a].append(lift_for)
            per_player_lift_for[player_b].append(lift_for)
            per_player_lift_against[player_a].append(lift_against)
            per_player_lift_against[player_b].append(lift_against)

    lineup_chemistry_for_avg = round(lift_for_sum / lift_pairs, 4) if lift_pairs else None
    lineup_chemistry_against_avg = round(lift_against_sum / lift_pairs, 4) if lift_pairs else None

    weighted_for_values = []
    weighted_against_values = []
    for player_id in starters:
        if player_id not in base_for:
            continue
        lift_for_list = per_player_lift_for[player_id]
        lift_against_list = per_player_lift_against[player_id]
        lift_for_avg = sum(lift_for_list) / len(lift_for_list) if lift_for_list else 0.0
        lift_against_avg = sum(lift_against_list) / len(lift_against_list) if lift_against_list else 0.0
        weighted_for_values.append(base_for[player_id] * (1 + lift_for_avg))
        weighted_against_values.append(base_against[player_id] * (1 + lift_against_avg))

    context_weighted_quality_for = (
        round(sum(weighted_for_values) / len(weighted_for_values), 4) if weighted_for_values else None
    )
    context_weighted_quality_against = (
        round(sum(weighted_against_values) / len(weighted_against_values), 4) if weighted_against_values else None
    )

    return {
        "avg_player_quality_for": avg_player_quality_for,
        "avg_player_quality_against": avg_player_quality_against,
        "lineup_chemistry_for_sum": round(lift_for_sum, 4) if lift_pairs else None,
        "lineup_chemistry_for_avg": lineup_chemistry_for_avg,
        "lineup_chemistry_against_sum": round(lift_against_sum, 4) if lift_pairs else None,
        "lineup_chemistry_against_avg": lineup_chemistry_against_avg,
        "lineup_chemistry_pair_count": lift_pairs,
        "context_weighted_quality_for": context_weighted_quality_for,
        "context_weighted_quality_against": context_weighted_quality_against,
    }


def load_player_stats(season_id):
    stats_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", f"player_stats_season_{season_id}.json")
    )
    if not os.path.exists(stats_path):
        return None
    with open(stats_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_types():
    types_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "types.json")
    )
    if not os.path.exists(types_path):
        return {}
    with open(types_path, "r", encoding="utf-8") as handle:
        types = json.load(handle)
    mapping = {}
    for item in types:
        type_id = item.get("id")
        if type_id is None:
            continue
        mapping[type_id] = {
            "name": item.get("name"),
            "code": item.get("code"),
            "developer_name": item.get("developer_name"),
            "model_type": item.get("model_type"),
        }
    return mapping


STAT_CODE_MAP = {
    "goals": "goals",
    "assists": "assists",
    "shots-total": "shots_total",
    "shots-on-target": "shots_on_target",
    "key-passes": "key_passes",
    "passes": "passes_total",
    "passes-total": "passes_total",
    "pass-accuracy": "pass_accuracy",
    "passes-accuracy": "pass_accuracy",
    "tackles": "tackles",
    "successful-interceptions": "interceptions",
    "clearances": "clearances",
    "minutes": "minutes_played",
    "minutes-played": "minutes_played",
    "saves": "saves",
    "yellowcards": "yellow_cards",
    "redcards": "red_cards",
    "cleansheets": "clean_sheets",
    "appearances": "appearances",
    "lineups": "lineups",
    "own-goals": "own_goals",
}


def normalize_stat_key(type_info):
    if not type_info:
        return None
    if type_info.get("model_type") != "statistic":
        return None
    candidates = [
        type_info.get("code"),
        type_info.get("developer_name"),
        type_info.get("name"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        value = str(candidate).lower().replace("-", "_").replace(" ", "_")
        original = str(candidate).lower()
        if original in STAT_CODE_MAP:
            return STAT_CODE_MAP[original]
    return None


def extract_player_stat_values(player_stats, type_map):
    totals = {}
    for stat in player_stats or []:
        for detail in stat.get("details", []) or []:
            value = detail.get("value")
            stat_key = normalize_stat_key(type_map.get(detail.get("type_id")))
            if not stat_key:
                continue
            if isinstance(value, dict):
                value = value.get("total")
            try:
                value_num = float(value)
            except (TypeError, ValueError):
                continue
            totals[stat_key] = totals.get(stat_key, 0.0) + value_num
    return totals


def build_lineup_quality_features(lineups, team_id, player_stats_index, type_map):
    starters = [l for l in lineups if l.get("team_id") == team_id and l.get("type_id") == 11]
    if not starters or not player_stats_index:
        return {
            "avg_player_goals": None,
            "sum_player_goals": None,
            "avg_player_assists": None,
            "sum_player_assists": None,
            "avg_player_minutes": None,
            "sum_player_minutes": None,
            "avg_player_shots": None,
            "sum_player_shots": None,
            "avg_player_shots_on_target": None,
            "sum_player_shots_on_target": None,
            "avg_player_key_passes": None,
            "sum_player_key_passes": None,
            "avg_player_passes": None,
            "sum_player_passes": None,
            "avg_player_pass_accuracy": None,
            "avg_player_tackles": None,
            "sum_player_tackles": None,
            "avg_player_interceptions": None,
            "sum_player_interceptions": None,
            "avg_player_clearances": None,
            "sum_player_clearances": None,
            "avg_player_saves": None,
            "sum_player_saves": None,
            "avg_player_yellow_cards": None,
            "sum_player_yellow_cards": None,
            "avg_player_red_cards": None,
            "sum_player_red_cards": None,
            "avg_player_clean_sheets": None,
            "sum_player_clean_sheets": None,
            "avg_player_appearances": None,
            "sum_player_appearances": None,
            "avg_player_lineups": None,
            "sum_player_lineups": None,
            "avg_player_own_goals": None,
            "sum_player_own_goals": None,
        }

    goals = []
    assists = []
    minutes = []
    shots = []
    shots_on_target = []
    key_passes = []
    passes = []
    pass_accuracy = []
    tackles = []
    interceptions = []
    clearances = []
    saves = []
    yellow_cards = []
    red_cards = []
    clean_sheets = []
    appearances = []
    lineups = []
    own_goals = []

    for entry in starters:
        player_id = entry.get("player_id")
        if player_id is None:
            continue
        player = player_stats_index.get(str(player_id)) or player_stats_index.get(player_id)
        if not player:
            continue
        totals = extract_player_stat_values(player.get("statistics"), type_map)
        if "goals" in totals:
            goals.append(totals["goals"])
        if "assists" in totals:
            assists.append(totals["assists"])
        if "minutes_played" in totals:
            minutes.append(totals["minutes_played"])
        if "shots_total" in totals:
            shots.append(totals["shots_total"])
        if "shots_on_target" in totals:
            shots_on_target.append(totals["shots_on_target"])
        if "key_passes" in totals:
            key_passes.append(totals["key_passes"])
        if "passes_total" in totals:
            passes.append(totals["passes_total"])
        if "pass_accuracy" in totals:
            pass_accuracy.append(totals["pass_accuracy"])
        if "tackles" in totals:
            tackles.append(totals["tackles"])
        if "interceptions" in totals:
            interceptions.append(totals["interceptions"])
        if "clearances" in totals:
            clearances.append(totals["clearances"])
        if "saves" in totals:
            saves.append(totals["saves"])
        if "yellow_cards" in totals:
            yellow_cards.append(totals["yellow_cards"])
        if "red_cards" in totals:
            red_cards.append(totals["red_cards"])
        if "clean_sheets" in totals:
            clean_sheets.append(totals["clean_sheets"])
        if "appearances" in totals:
            appearances.append(totals["appearances"])
        if "lineups" in totals:
            lineups.append(totals["lineups"])
        if "own_goals" in totals:
            own_goals.append(totals["own_goals"])

    def avg(values):
        return round(sum(values) / len(values), 4) if values else None

    return {
        "avg_player_goals": avg(goals),
        "sum_player_goals": round(sum(goals), 4) if goals else None,
        "avg_player_assists": avg(assists),
        "sum_player_assists": round(sum(assists), 4) if assists else None,
        "avg_player_minutes": avg(minutes),
        "sum_player_minutes": round(sum(minutes), 4) if minutes else None,
        "avg_player_shots": avg(shots),
        "sum_player_shots": round(sum(shots), 4) if shots else None,
        "avg_player_shots_on_target": avg(shots_on_target),
        "sum_player_shots_on_target": round(sum(shots_on_target), 4) if shots_on_target else None,
        "avg_player_key_passes": avg(key_passes),
        "sum_player_key_passes": round(sum(key_passes), 4) if key_passes else None,
        "avg_player_passes": avg(passes),
        "sum_player_passes": round(sum(passes), 4) if passes else None,
        "avg_player_pass_accuracy": avg(pass_accuracy),
        "avg_player_tackles": avg(tackles),
        "sum_player_tackles": round(sum(tackles), 4) if tackles else None,
        "avg_player_interceptions": avg(interceptions),
        "sum_player_interceptions": round(sum(interceptions), 4) if interceptions else None,
        "avg_player_clearances": avg(clearances),
        "sum_player_clearances": round(sum(clearances), 4) if clearances else None,
        "avg_player_saves": avg(saves),
        "sum_player_saves": round(sum(saves), 4) if saves else None,
        "avg_player_yellow_cards": avg(yellow_cards),
        "sum_player_yellow_cards": round(sum(yellow_cards), 4) if yellow_cards else None,
        "avg_player_red_cards": avg(red_cards),
        "sum_player_red_cards": round(sum(red_cards), 4) if red_cards else None,
        "avg_player_clean_sheets": avg(clean_sheets),
        "sum_player_clean_sheets": round(sum(clean_sheets), 4) if clean_sheets else None,
        "avg_player_appearances": avg(appearances),
        "sum_player_appearances": round(sum(appearances), 4) if appearances else None,
        "avg_player_lineups": avg(lineups),
        "sum_player_lineups": round(sum(lineups), 4) if lineups else None,
        "avg_player_own_goals": avg(own_goals),
        "sum_player_own_goals": round(sum(own_goals), 4) if own_goals else None,
    }


def fetch_fixture_detail(base_url, token, fixture_id, include):
    params = {"api_token": token, "include": include}
    response = requests.get(f"{base_url}/{fixture_id}", params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Fixture {fixture_id} error: {response.status_code}")
    return response.json().get("data", {})


def fetch_fixtures_list(base_url, token, league_id, season_id):
    fixture_ids = []
    target_league = int(league_id)
    list_url = f"https://api.sportmonks.com/v3/football/seasons/{season_id}"
    params = {
        "api_token": token,
        "include": "fixtures",
    }
    response = requests.get(list_url, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"Fixtures list error: {response.status_code} {response.text}"
        )
    payload = response.json()
    data = payload.get("data") or {}
    fixtures = data.get("fixtures") or []

    for item in fixtures:
        if item.get("league_id") == target_league:
            fixture_id = item.get("id")
            if fixture_id:
                fixture_ids.append(fixture_id)
    return fixture_ids


def fetch_league_seasons(token, league_id):
    url = f"https://api.sportmonks.com/v3/football/leagues/{league_id}"
    params = {"api_token": token, "include": "seasons"}
    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"League seasons error: {response.status_code} {response.text}")
    data = response.json().get("data") or {}
    return data.get("seasons") or []


def season_sort_key(season):
    for field in ("start_date", "starting_at", "start"):
        value = season.get(field)
        if value:
            try:
                return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
    year = season.get("year")
    if isinstance(year, int):
        return dt.datetime(year, 1, 1)
    name = season.get("name")
    if isinstance(name, str):
        digits = "".join(ch for ch in name if ch.isdigit())
        if len(digits) >= 4:
            return dt.datetime(int(digits[:4]), 1, 1)
    return dt.datetime.min


def main():
    root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv(root_env)

    token = os.getenv("SPORTMONKS_API_TOKEN", "").strip()
    league_id = os.getenv("SPORTMONKS_LEAGUE_ID", "").strip()
    season_id = os.getenv("SPORTMONKS_SEASON_ID", "").strip()
    league_ids = os.getenv("SPORTMONKS_LEAGUE_IDS", "").strip()
    seasons_per_league = os.getenv("SPORTMONKS_SEASONS_PER_LEAGUE", "").strip()
    seasons_per_league = int(seasons_per_league) if seasons_per_league.isdigit() else 2
    max_fixtures = os.getenv("SPORTMONKS_MAX_FIXTURES", "").strip()
    max_fixtures = int(max_fixtures) if max_fixtures.isdigit() else None
    history_window = os.getenv("SPORTMONKS_HISTORY_WINDOW", "").strip()
    history_window = int(history_window) if history_window.isdigit() else 5
    write_features = os.getenv("SPORTMONKS_WRITE_FEATURES", "0").strip().lower() in {"1", "true", "yes"}

    if not token:
        print("Missing SPORTMONKS_API_TOKEN.")
        sys.exit(1)

    base_url = "https://api.sportmonks.com/v3/football/fixtures"
    include = ";".join([
        "participants",
        "lineups.player",
        "lineups.details",
        "events",
        "scores",
    ])

    fixtures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "fixtures"))
    os.makedirs(fixtures_dir, exist_ok=True)
    features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "features.csv"))

    if league_ids:
        league_list = [item.strip() for item in league_ids.split(",") if item.strip()]
    else:
        league_list = [league_id] if league_id else []

    if not league_list:
        print("Missing SPORTMONKS_LEAGUE_ID or SPORTMONKS_LEAGUE_IDS.")
        sys.exit(1)

    season_plan = []
    if season_id and league_id and not league_ids:
        season_plan.append((league_id, season_id))
    else:
        for lid in league_list:
            seasons = fetch_league_seasons(token, lid)
            seasons.sort(key=season_sort_key, reverse=True)
            for season in seasons[:seasons_per_league]:
                season_plan.append((lid, season.get("id")))

    if not season_plan:
        print("No seasons found for requested leagues.")
        sys.exit(1)

    fixtures = []
    multi_season = len(season_plan) > 1
    for league_id, season_id in season_plan:
        fixture_ids = fetch_fixtures_list(base_url, token, league_id, season_id)
        if max_fixtures:
            fixture_ids = fixture_ids[:max_fixtures]
        if not fixture_ids:
            print(f"No fixtures found for league {league_id} season {season_id}.")
            continue

        for index, fixture_id in enumerate(fixture_ids, start=1):
            fixture_path = os.path.join(fixtures_dir, f"fixture_{fixture_id}.json")
            if os.path.exists(fixture_path):
                print(f"Skipping fixture {fixture_id} (cached)")
                if write_features and not multi_season:
                    with open(fixture_path, "r", encoding="utf-8") as handle:
                        cached = json.load(handle).get("data")
                    if cached:
                        fixtures.append(cached)
                continue
            print(f"Fetching fixture {fixture_id} ({index}/{len(fixture_ids)})")
            data = fetch_fixture_detail(base_url, token, fixture_id, include)
            if data:
                with open(fixture_path, "w", encoding="utf-8") as fixture_handle:
                    json.dump({"data": data}, fixture_handle, indent=2)
                fixtures.append(data)
            time.sleep(0.2)

    if not write_features:
        print("Fixture fetch complete. Skipping features.csv generation.")
        return

    if multi_season:
        print("Multiple seasons requested: skip features.csv generation.")
        return

    player_stats_index = load_player_stats(season_plan[0][1])
    type_map = load_types()

    fixtures.sort(key=lambda item: parse_datetime(item.get("starting_at")) or dt.datetime.max)

    with open(features_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "fixture_id",
            "team_id",
            "opponent_id",
            "is_home",
            "starting_at",
            "team_position",
            "opponent_position",
            "avg_player_quality_for",
            "avg_player_quality_against",
            "lineup_chemistry_for_sum",
            "lineup_chemistry_for_avg",
            "lineup_chemistry_against_sum",
            "lineup_chemistry_against_avg",
            "lineup_chemistry_pair_count",
            "opp_avg_scored",
            "opp_avg_conceded",
            "adj_player_quality_for",
            "adj_player_quality_against",
            "team_form_points",
            "team_form_goals_for",
            "team_form_goals_against",
            "team_form_goal_diff",
            "goals_for",
            "goals_against",
            "result",
        ])
        writer.writeheader()

        total_fixtures = len(fixtures)
        for index, data in enumerate(fixtures, start=1):
            fixture_id = data.get("id")
            print(f"Processing fixture {fixture_id} ({index}/{total_fixtures})")

            fixture_path = os.path.join(fixtures_dir, f"fixture_{fixture_id}.json")
            if not os.path.exists(fixture_path):
                with open(fixture_path, "w", encoding="utf-8") as fixture_handle:
                    json.dump({"data": data}, fixture_handle, indent=2)

            lineups = data.get("lineups", []) or []
            match_minutes = data.get("length") if isinstance(data.get("length"), int) else 90
            minutes_map = build_minutes_map(lineups, data.get("events", []), match_minutes)
            adjacency_edges = build_adjacency_edges([l for l in lineups if l.get("type_id") == 11])
            shared_edges = build_shared_minutes_edges(lineups, match_minutes, minutes_map)
            interactions = adjacency_edges + shared_edges
            team_stats = compute_synergy(interactions)

            participants = extract_participants(data)
            goals_by_team = extract_goals_by_team(data)
            formations = extract_formations(data)
            weather = extract_weather(data)
            team_lineups = {}
            for entry in lineups:
                if entry.get("type_id") != 11:
                    continue
                team_id = entry.get("team_id")
                player_id = entry.get("player_id")
                if team_id and player_id:
                    team_lineups.setdefault(team_id, []).append(player_id)

            team_ids = list(team_stats.keys())
            if len(team_ids) != 2:
                continue

            team_a, team_b = team_ids[0], team_ids[1]
            for team_id, opponent_id in [(team_a, team_b), (team_b, team_a)]:
                stats = team_stats[team_id]
                is_home = participants.get(team_id, {}).get("location") == "home"
                goals_for = goals_by_team.get(team_id)
                goals_against = goals_by_team.get(opponent_id)
                starters = team_lineups.get(team_id, [])
                context_features = build_lineup_context_features(
                    starters,
                    team_id,
                    player_history,
                    pair_history,
                    min_history=min_history,
                    history_window=history_window,
                )

                opp_history = team_history.get(opponent_id, {"scored": [], "conceded": []})
                opp_scored = opp_history.get("scored", [])
                opp_conceded = opp_history.get("conceded", [])
                opp_avg_scored = round(sum(opp_scored) / len(opp_scored), 4) if opp_scored else None
                opp_avg_conceded = round(sum(opp_conceded) / len(opp_conceded), 4) if opp_conceded else None

                team_hist = team_history.get(team_id, {"scored": [], "conceded": []})
                team_scored = team_hist.get("scored", [])
                team_conceded = team_hist.get("conceded", [])
                if team_scored and team_conceded:
                    team_form_goals_for = round(sum(team_scored) / len(team_scored), 4)
                    team_form_goals_against = round(sum(team_conceded) / len(team_conceded), 4)
                    team_form_goal_diff = round(team_form_goals_for - team_form_goals_against, 4)
                    team_form_points = 0
                    for gf, ga in zip(team_scored, team_conceded):
                        if gf > ga:
                            team_form_points += 3
                        elif gf == ga:
                            team_form_points += 1
                    team_form_points = round(team_form_points / len(team_scored), 4)
                else:
                    team_form_goals_for = None
                    team_form_goals_against = None
                    team_form_goal_diff = None
                    team_form_points = None

                adj_player_quality_for = None
                adj_player_quality_against = None
                if context_features["avg_player_quality_for"] is not None and opp_avg_conceded is not None:
                    adj_player_quality_for = round(
                        context_features["avg_player_quality_for"] + opp_avg_conceded, 4
                    )
                if context_features["avg_player_quality_against"] is not None and opp_avg_scored is not None:
                    adj_player_quality_against = round(
                        context_features["avg_player_quality_against"] + opp_avg_scored, 4
                    )
                result = None
                if goals_for is not None and goals_against is not None:
                    if goals_for > goals_against:
                        result = "W"
                    elif goals_for < goals_against:
                        result = "L"
                    else:
                        result = "D"

                writer.writerow({
                    "fixture_id": fixture_id,
                    "team_id": team_id,
                    "opponent_id": opponent_id,
                    "is_home": 1 if is_home else 0,
                    "starting_at": data.get("starting_at"),
                    "team_position": participants.get(team_id, {}).get("position"),
                    "opponent_position": participants.get(opponent_id, {}).get("position"),
                    "avg_player_quality_for": context_features["avg_player_quality_for"],
                    "avg_player_quality_against": context_features["avg_player_quality_against"],
                    "lineup_chemistry_for_sum": context_features["lineup_chemistry_for_sum"],
                    "lineup_chemistry_for_avg": context_features["lineup_chemistry_for_avg"],
                    "lineup_chemistry_against_sum": context_features["lineup_chemistry_against_sum"],
                    "lineup_chemistry_against_avg": context_features["lineup_chemistry_against_avg"],
                    "lineup_chemistry_pair_count": context_features["lineup_chemistry_pair_count"],
                    "opp_avg_scored": opp_avg_scored,
                    "opp_avg_conceded": opp_avg_conceded,
                    "adj_player_quality_for": adj_player_quality_for,
                    "adj_player_quality_against": adj_player_quality_against,
                    "team_form_points": team_form_points,
                    "team_form_goals_for": team_form_goals_for,
                    "team_form_goals_against": team_form_goals_against,
                    "team_form_goal_diff": team_form_goal_diff,
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "result": result,
                })

            if goals_by_team:
                for team_id, player_ids in team_lineups.items():
                    goals_for = goals_by_team.get(team_id)
                    goals_against = None
                    for other_id in team_lineups.keys():
                        if other_id != team_id:
                            goals_against = goals_by_team.get(other_id)
                            break
                    if goals_for is None or goals_against is None:
                        continue
                    team_stats_history = team_history.setdefault(
                        team_id, {"scored": [], "conceded": []}
                    )
                    team_stats_history["scored"].append(goals_for)
                    team_stats_history["conceded"].append(goals_against)
                    if history_window:
                        team_stats_history["scored"] = team_stats_history["scored"][-history_window:]
                        team_stats_history["conceded"] = team_stats_history["conceded"][-history_window:]

                    for player_id in player_ids:
                        history = player_history.setdefault(team_id, {}).setdefault(
                            player_id, {
                                "matches": 0,
                                "goals_for_list": [],
                                "goals_against_list": [],
                            }
                        )
                        history["matches"] += 1
                        history["goals_for_list"].append(goals_for)
                        history["goals_against_list"].append(goals_against)
                        if history_window:
                            history["goals_for_list"] = history["goals_for_list"][-history_window:]
                            history["goals_against_list"] = history["goals_against_list"][-history_window:]

                    for i, player_a in enumerate(player_ids):
                        for player_b in player_ids[i + 1:]:
                            pair_key = (player_a, player_b) if player_a < player_b else (player_b, player_a)
                            pair_stats = pair_history.setdefault(team_id, {}).setdefault(
                                pair_key, {
                                    "matches": 0,
                                    "goals_for_list": [],
                                    "goals_against_list": [],
                                }
                            )
                            pair_stats["matches"] += 1
                            pair_stats["goals_for_list"].append(goals_for)
                            pair_stats["goals_against_list"].append(goals_against)
                            if history_window:
                                pair_stats["goals_for_list"] = pair_stats["goals_for_list"][-history_window:]
                                pair_stats["goals_against_list"] = pair_stats["goals_against_list"][-history_window:]

    print(f"Wrote: {features_path}")


if __name__ == "__main__":
    main()
