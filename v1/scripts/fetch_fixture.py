import os
import sys
import json
import re
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


ROOT_ENV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(ROOT_ENV)

API_TOKEN = os.getenv("SPORTMONKS_API_TOKEN", "").strip()
FIXTURE_ID = os.getenv("SPORTMONKS_FIXTURE_ID", "").strip()

if not API_TOKEN or not FIXTURE_ID:
    print("Missing SPORTMONKS_API_TOKEN or SPORTMONKS_FIXTURE_ID environment variable.")
    sys.exit(1)

BASE_URL = f"https://api.sportmonks.com/v3/football/fixtures/{FIXTURE_ID}"
include_parts = [
    "lineups.player",
    "participants",
    "lineups.details",
    "events.type",
    "events.player",
    "events.relatedplayer",
]

params = {
    "api_token": API_TOKEN,
    "include": ";".join(include_parts),
}

response = requests.get(BASE_URL, params=params, timeout=30)

if response.status_code != 200:
    print(f"Error: {response.status_code} {response.text}")
    sys.exit(1)

data = response.json().get("data", {})
print(f"Match: {data.get('name')}")

starting_lineups = []

for lineup in data.get("lineups", []):
    if lineup.get("type_id") == 11:  # Starting XI
        player_name = lineup.get("player_name")
        position = lineup.get("formation_field")
        print(f"Player: {player_name} | Pitch Position: {position}")
        starting_lineups.append(lineup)


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
        event_type = event.get("type") or {}
        name = (event_type.get("name") or event_type.get("short_name") or "").lower()
        is_sub = "substitution" in name
        if not is_sub:
            type_id = event.get("type_id")
            if type_id == 18:
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


def extract_minutes_played(entry, minutes_map):
    player_id = entry.get("player_id")
    if player_id in minutes_map:
        return minutes_map[player_id]

    value = entry.get("minutes_played")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)

    for detail in entry.get("details", []) or []:
        detail_type = detail.get("type") or {}
        name = detail_type.get("name") or detail_type.get("short_name") or ""
        if "minute" in name.lower():
            raw = detail.get("value")
            if isinstance(raw, int):
                return raw
            if isinstance(raw, str):
                match = re.search(r"\d+", raw)
                if match:
                    return int(match.group(0))
    return None


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
            minutes = extract_minutes_played(entry, minutes_map)
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


edges = build_adjacency_edges(starting_lineups)
print(f"Adjacency edges: {len(edges)}")

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"fixture_{FIXTURE_ID}_adjacency.json")
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(edges, handle, indent=2)
print(f"Wrote: {output_path}")

match_minutes = 90
if isinstance(data.get("length"), int):
    match_minutes = data.get("length")
env_minutes = os.getenv("SPORTMONKS_MATCH_MINUTES")
if env_minutes and env_minutes.isdigit():
    match_minutes = int(env_minutes)

minutes_map = build_minutes_map(data.get("lineups", []), data.get("events", []), match_minutes)
shared_edges = build_shared_minutes_edges(data.get("lineups", []), match_minutes, minutes_map)
print(f"Shared-minutes edges: {len(shared_edges)}")
shared_path = os.path.join(output_dir, f"fixture_{FIXTURE_ID}_shared_minutes.json")
with open(shared_path, "w", encoding="utf-8") as handle:
    json.dump(shared_edges, handle, indent=2)
print(f"Wrote: {shared_path}")

minutes_path = os.path.join(output_dir, f"fixture_{FIXTURE_ID}_minutes_map.json")
with open(minutes_path, "w", encoding="utf-8") as handle:
    json.dump(minutes_map, handle, indent=2, sort_keys=True)
print(f"Wrote: {minutes_path}")

interactions = edges + shared_edges
interactions_path = os.path.join(output_dir, f"fixture_{FIXTURE_ID}_interactions.json")
with open(interactions_path, "w", encoding="utf-8") as handle:
    json.dump(interactions, handle, indent=2)
print(f"Wrote: {interactions_path}")

events = data.get("events", []) or []
type_counts = {}
for event in events:
    event_type = event.get("type") or {}
    name = event_type.get("name") or event_type.get("short_name") or "unknown"
    type_counts[name] = type_counts.get(name, 0) + 1

summary_path = os.path.join(output_dir, f"fixture_{FIXTURE_ID}_events_summary.json")
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(type_counts, handle, indent=2)
print(f"Events: {len(events)} (summary written to {summary_path})")
if not events:
    print("No events returned; pass-link edges may require a different endpoint.")
