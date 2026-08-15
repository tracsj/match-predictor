import json
import os
import sys
import time

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


def fetch_season_fixtures(token, season_id):
    url = f"https://api.sportmonks.com/v3/football/seasons/{season_id}"
    params = {"api_token": token, "include": "fixtures.participants"}
    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Season error: {response.status_code} {response.text}")
    data = response.json().get("data") or {}
    return data.get("fixtures") or []


def collect_team_ids(fixtures):
    team_ids = set()
    for fixture in fixtures:
        for team in fixture.get("participants", []) or []:
            team_id = team.get("id")
            if team_id:
                team_ids.add(team_id)
    return sorted(team_ids)


def fetch_team_squad_stats(token, season_id, team_id):
    url = f"https://api.sportmonks.com/v3/football/squads/seasons/{season_id}/teams/{team_id}"
    params = {
        "api_token": token,
        "include": "player;player.statistics;player.statistics.details",
        "filters": f"playerStatisticSeasons:{season_id}",
    }
    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Squad error: {response.status_code} {response.text}")
    return response.json().get("data") or []


def build_player_stats_index(squad_rows, season_id):
    players = {}
    for row in squad_rows:
        player = row.get("player") or {}
        player_id = player.get("id")
        if not player_id:
            continue
        stats = []
        for stat in player.get("statistics", []) or []:
            season = stat.get("season_id") or stat.get("season", {}).get("id")
            if season and int(season) != int(season_id):
                continue
            stats.append(stat)
        players[player_id] = {
            "player_id": player_id,
            "name": player.get("name"),
            "position_id": player.get("position_id"),
            "detailed_position_id": player.get("detailed_position_id"),
            "statistics": stats,
        }
    return players


def main():
    root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv(root_env)

    token = os.getenv("SPORTMONKS_API_TOKEN", "").strip()
    season_id = os.getenv("SPORTMONKS_SEASON_ID", "").strip()
    league_id = os.getenv("SPORTMONKS_LEAGUE_ID", "").strip()
    if not token or not season_id:
        print("Missing SPORTMONKS_API_TOKEN or SPORTMONKS_SEASON_ID.")
        sys.exit(1)

    fixtures = fetch_season_fixtures(token, season_id)
    team_ids = collect_team_ids(fixtures)
    if not team_ids:
        print("No teams found for this season.")
        sys.exit(1)

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(output_dir, exist_ok=True)
    players_out = {}

    for index, team_id in enumerate(team_ids, start=1):
        print(f"Fetching squad stats for team {team_id} ({index}/{len(team_ids)})")
        squad_rows = fetch_team_squad_stats(token, season_id, team_id)
        players = build_player_stats_index(squad_rows, season_id)
        players_out[str(team_id)] = {
            "team_id": team_id,
            "league_id": int(league_id) if league_id else None,
            "season_id": int(season_id),
            "players": players,
        }
        time.sleep(0.2)

    output_path = os.path.join(output_dir, f"player_stats_season_{season_id}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(players_out, handle, indent=2)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
