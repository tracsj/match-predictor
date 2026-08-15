import json
import os
import sys


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

FIXTURE_ID = os.getenv("SPORTMONKS_FIXTURE_ID", "").strip()
if not FIXTURE_ID:
    print("Missing SPORTMONKS_FIXTURE_ID environment variable.")
    sys.exit(1)

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
interactions_path = os.path.join(output_dir, f"fixture_{FIXTURE_ID}_interactions.json")

if not os.path.exists(interactions_path):
    print(f"Missing interactions file: {interactions_path}")
    sys.exit(1)

with open(interactions_path, "r", encoding="utf-8") as handle:
    interactions = json.load(handle)

team_stats = {}

for edge in interactions:
    team_id = edge.get("team_id")
    if team_id is None:
        continue
    stats = team_stats.setdefault(team_id, {
        "team_id": team_id,
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

synergy_path = os.path.join(output_dir, f"fixture_{FIXTURE_ID}_synergy.json")
with open(synergy_path, "w", encoding="utf-8") as handle:
    json.dump(sorted(team_stats.values(), key=lambda s: s["team_id"]), handle, indent=2)

print(f"Wrote: {synergy_path}")
