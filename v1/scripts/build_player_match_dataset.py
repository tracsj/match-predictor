import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROLLING_WINDOW = 5

STAT_DEFS = [
    {"name": "goals", "type_id": 52, "kind": "count"},
    {"name": "assists", "type_id": 79, "kind": "count"},
    {"name": "shots_total", "type_id": 42, "kind": "count"},
    {"name": "shots_on_target", "type_id": 86, "kind": "count"},
    {"name": "shots_off_target", "type_id": 41, "kind": "count"},
    {"name": "key_passes", "type_id": 117, "kind": "count"},
    {"name": "passes", "type_id": 80, "kind": "count"},
    {"name": "accurate_passes", "type_id": 116, "kind": "count"},
    {"name": "accurate_passes_pct", "type_id": 1584, "kind": "rate"},
    {"name": "touches", "type_id": 120, "kind": "count"},
    {"name": "total_crosses", "type_id": 98, "kind": "count"},
    {"name": "long_balls", "type_id": 122, "kind": "count"},
    {"name": "long_balls_won", "type_id": 123, "kind": "count"},
    {"name": "long_balls_won_pct", "type_id": 27270, "kind": "rate"},
    {"name": "total_duels", "type_id": 105, "kind": "count"},
    {"name": "duels_won", "type_id": 106, "kind": "count"},
    {"name": "duels_lost", "type_id": 1491, "kind": "count"},
    {"name": "duels_won_pct", "type_id": 27276, "kind": "rate"},
    {"name": "aerials", "type_id": 27274, "kind": "count"},
    {"name": "aerials_won", "type_id": 107, "kind": "count"},
    {"name": "aerials_lost", "type_id": 27266, "kind": "count"},
    {"name": "aerials_won_pct", "type_id": 27275, "kind": "rate"},
    {"name": "tackles", "type_id": 78, "kind": "count"},
    {"name": "tackles_won", "type_id": 27267, "kind": "count"},
    {"name": "tackles_won_pct", "type_id": 27268, "kind": "rate"},
    {"name": "interceptions", "type_id": 100, "kind": "count"},
    {"name": "clearances", "type_id": 101, "kind": "count"},
    {"name": "fouls", "type_id": 56, "kind": "count"},
    {"name": "fouls_drawn", "type_id": 96, "kind": "count"},
    {"name": "dribble_attempts", "type_id": 108, "kind": "count"},
    {"name": "successful_dribbles", "type_id": 109, "kind": "count"},
    {"name": "dribbled_past", "type_id": 110, "kind": "count"},
    {"name": "dispossessed", "type_id": 94, "kind": "count"},
    {"name": "possession_lost", "type_id": 27273, "kind": "count"},
    {"name": "ball_recovery", "type_id": 27271, "kind": "count"},
    {"name": "rating", "type_id": 118, "kind": "rate"},
    {"name": "goals_conceded", "type_id": 88, "kind": "count"},
]

COUNT_STATS = {s["name"] for s in STAT_DEFS if s["kind"] == "count"}
RATE_STATS = {s["name"] for s in STAT_DEFS if s["kind"] == "rate"}
STAT_BY_ID = {s["type_id"]: s for s in STAT_DEFS}


def load_dotenv(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def parse_dt(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def load_fixtures(fixtures_dir):
    fixtures = []
    for path in sorted(fixtures_dir.glob("fixture_*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        data = payload.get("data") or {}
        fixtures.append(data)
    fixtures.sort(
        key=lambda f: (
            parse_dt(f.get("starting_at")) or datetime.min,
            f.get("starting_at_timestamp") or 0,
            f.get("id") or 0,
        )
    )
    return fixtures


def extract_team_info(participants):
    info = {}
    for participant in participants or []:
        team_id = participant.get("id")
        if not team_id:
            continue
        meta = participant.get("meta") or {}
        info[team_id] = {
            "team_id": team_id,
            "location": meta.get("location"),
            "position": meta.get("position"),
        }
    return info


def extract_scores(scores):
    goals = {}
    for score in scores or []:
        if score.get("type_id") != 1525:
            continue
        team_id = score.get("participant_id")
        if team_id is None:
            continue
        goals[team_id] = (score.get("score") or {}).get("goals") or 0
    return goals


def lineup_stats(lineups):
    by_team = defaultdict(dict)
    for lineup in lineups or []:
        team_id = lineup.get("team_id")
        player_id = lineup.get("player_id")
        if not team_id or not player_id:
            continue
        detail_map = {}
        for detail in lineup.get("details") or []:
            type_id = detail.get("type_id")
            if type_id is None:
                continue
            value = (detail.get("data") or {}).get("value")
            detail_map[type_id] = value
        minutes = detail_map.get(119) or 0
        goals = detail_map.get(52) or 0
        position_group = 0
        formation_field = lineup.get("formation_field")
        if isinstance(formation_field, str) and ":" in formation_field:
            row = formation_field.split(":", 1)[0]
            try:
                line = int(row)
            except ValueError:
                line = 0
            if line == 1:
                position_group = 1  # GK
            elif line == 2:
                position_group = 2  # DEF
            elif line == 3:
                position_group = 3  # MID
            elif line >= 4:
                position_group = 4  # ATT
        by_team[team_id][player_id] = {
            "player_id": player_id,
            "team_id": team_id,
            "lineup_type": lineup.get("type_id"),
            "minutes": minutes,
            "goals": goals,
            "position_group": position_group,
            "stats": detail_map,
        }
    return by_team


def rolling_player_features(history, window):
    history = [entry for entry in history if (entry.get("minutes") or 0) > 0]
    if window:
        history = history[-window:]
    matches = len(history)
    minutes_total = sum(entry.get("minutes") or 0 for entry in history)
    features = {
        "player_hist_matches": matches,
        "player_hist_minutes": minutes_total,
        "player_hist_avg_minutes": round(minutes_total / matches, 2) if matches else 0,
    }

    if not matches or minutes_total <= 0:
        for stat in COUNT_STATS:
            features[f"roll_{stat}_per90"] = ""
        for stat in RATE_STATS:
            features[f"roll_{stat}_avg"] = ""
        return features

    for stat in COUNT_STATS:
        total = 0.0
        for entry in history:
            value = entry.get("stats", {}).get(stat)
            if value is None:
                continue
            total += value
        features[f"roll_{stat}_per90"] = round(total / minutes_total * 90, 4)

    for stat in RATE_STATS:
        values = []
        for entry in history:
            value = entry.get("stats", {}).get(stat)
            if value is None:
                continue
            values.append(value)
        features[f"roll_{stat}_avg"] = round(sum(values) / len(values), 4) if values else ""

    return features


def rolling_team_form(history, window):
    if window:
        history = history[-window:]
    if not history:
        return {
            "team_form_points": 0,
            "team_form_goals_for": 0,
            "team_form_goals_against": 0,
            "team_form_goal_diff": 0,
        }
    points = 0
    goals_for = 0
    goals_against = 0
    for entry in history:
        gf = entry.get("goals_for", 0)
        ga = entry.get("goals_against", 0)
        goals_for += gf
        goals_against += ga
        if gf > ga:
            points += 3
        elif gf == ga:
            points += 1
    matches = len(history)
    return {
        "team_form_points": round(points / matches, 3),
        "team_form_goals_for": round(goals_for / matches, 3),
        "team_form_goals_against": round(goals_against / matches, 3),
        "team_form_goal_diff": round((goals_for - goals_against) / matches, 3),
    }


def cumulative_team_strength(history):
    if not history:
        return 0.0, 0.0
    matches = len(history)
    goals_for = sum(entry.get("goals_for", 0) for entry in history)
    goals_against = sum(entry.get("goals_against", 0) for entry in history)
    return goals_for / matches, goals_against / matches


def main():
    root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv(root_env)
    window = int(os.getenv("SPORTMONKS_HISTORY_WINDOW", ROLLING_WINDOW))

    fixtures_dir = Path(__file__).resolve().parent.parent / "data" / "fixtures"
    if not fixtures_dir.exists():
        print(f"Missing fixtures directory: {fixtures_dir}")
        sys.exit(1)

    fixtures = load_fixtures(fixtures_dir)
    if not fixtures:
        print("No fixtures found in data/fixtures.")
        sys.exit(1)

    player_history = defaultdict(list)
    team_history = defaultdict(list)
    league_history = []

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "player_match_features.csv")

    base_columns = [
        "fixture_id",
        "starting_at",
        "team_id",
        "opponent_id",
        "player_id",
        "is_home",
        "team_position",
        "opponent_position",
        "lineup_type",
        "position_group",
        "minutes_played",
        "goals",
    ]
    feature_columns = [
        "player_hist_matches",
        "player_hist_minutes",
        "player_hist_avg_minutes",
    ]
    for stat in COUNT_STATS:
        feature_columns.append(f"roll_{stat}_per90")
    for stat in RATE_STATS:
        feature_columns.append(f"roll_{stat}_avg")

    team_form_columns = [
        "team_form_points",
        "team_form_goals_for",
        "team_form_goals_against",
        "team_form_goal_diff",
        "opp_form_points",
        "opp_form_goals_for",
        "opp_form_goals_against",
        "opp_form_goal_diff",
        "league_avg_goals_per_team",
        "team_attack_index",
        "team_defense_index",
        "opp_attack_index",
        "opp_defense_index",
    ]

    rows = []

    for fixture in fixtures:
        fixture_id = fixture.get("id")
        fixture_start = parse_dt(fixture.get("starting_at"))
        starting_at = fixture.get("starting_at")
        participants = fixture.get("participants") or []
        team_info = extract_team_info(participants)
        if len(team_info) < 2:
            continue
        scores = extract_scores(fixture.get("scores") or [])
        lineups_by_team = lineup_stats(fixture.get("lineups") or [])

        team_ids = list(team_info.keys())
        for team_id in team_ids:
            opponent_id = next((tid for tid in team_ids if tid != team_id), None)
            if opponent_id is None:
                continue

            team_meta = team_info.get(team_id) or {}
            opp_meta = team_info.get(opponent_id) or {}
            is_home = 1 if team_meta.get("location") == "home" else 0
            team_position = team_meta.get("position")
            opponent_position = opp_meta.get("position")

            team_form = rolling_team_form(team_history[team_id], window)
            opp_form = rolling_team_form(team_history[opponent_id], window)
            team_avg_for, team_avg_against = cumulative_team_strength(team_history[team_id])
            opp_avg_for, opp_avg_against = cumulative_team_strength(team_history[opponent_id])
            if league_history:
                league_avg_goals = sum(league_history) / len(league_history)
            else:
                league_avg_goals = 0.0
            denom = league_avg_goals if league_avg_goals > 0 else 1.0
            team_attack_index = team_avg_for / denom
            team_defense_index = team_avg_against / denom
            opp_attack_index = opp_avg_for / denom
            opp_defense_index = opp_avg_against / denom

            lineup_players = lineups_by_team.get(team_id, {})
            for player_id, entry in lineup_players.items():
                history_features = rolling_player_features(player_history[player_id], window)

                row = {
                    "fixture_id": fixture_id,
                    "starting_at": starting_at,
                    "team_id": team_id,
                    "opponent_id": opponent_id,
                    "player_id": player_id,
                    "is_home": is_home,
                    "team_position": team_position if team_position is not None else "",
                    "opponent_position": opponent_position if opponent_position is not None else "",
                    "lineup_type": entry.get("lineup_type") or "",
                    "position_group": entry.get("position_group") or 0,
                    "minutes_played": entry.get("minutes") or 0,
                    "goals": entry.get("goals") or 0,
                }
                row.update(history_features)
                row.update(team_form)
                row.update(
                    {
                        "opp_form_points": opp_form["team_form_points"],
                        "opp_form_goals_for": opp_form["team_form_goals_for"],
                        "opp_form_goals_against": opp_form["team_form_goals_against"],
                        "opp_form_goal_diff": opp_form["team_form_goal_diff"],
                        "league_avg_goals_per_team": round(league_avg_goals, 4),
                        "team_attack_index": round(team_attack_index, 4),
                        "team_defense_index": round(team_defense_index, 4),
                        "opp_attack_index": round(opp_attack_index, 4),
                        "opp_defense_index": round(opp_defense_index, 4),
                    }
                )
                rows.append(row)

        # update histories after using features
        for team_id in team_ids:
            opponent_id = next((tid for tid in team_ids if tid != team_id), None)
            goals_for = scores.get(team_id, 0)
            goals_against = scores.get(opponent_id, 0) if opponent_id else 0
            team_history[team_id].append(
                {"goals_for": goals_for, "goals_against": goals_against}
            )
            league_history.append(goals_for)

            lineup_players = lineups_by_team.get(team_id, {})
            for player_id, entry in lineup_players.items():
                stat_values = {}
                for type_id, value in (entry.get("stats") or {}).items():
                    stat_def = STAT_BY_ID.get(type_id)
                    if not stat_def:
                        continue
                    stat_values[stat_def["name"]] = value
                player_history[player_id].append(
                    {
                        "minutes": entry.get("minutes") or 0,
                        "stats": stat_values,
                    }
                )

    all_columns = base_columns + feature_columns + team_form_columns
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in all_columns})

    print(f"Wrote: {output_path}")
    print(f"Rows: {len(rows)}")
    print(f"Rolling window: {window} matches")


if __name__ == "__main__":
    main()
