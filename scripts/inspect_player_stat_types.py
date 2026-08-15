import json
import os
import sys
from collections import Counter


def load_types():
    types_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "types.json")
    )
    if not os.path.exists(types_path):
        print(f"Missing types file: {types_path}")
        sys.exit(1)
    with open(types_path, "r", encoding="utf-8") as handle:
        types = json.load(handle)
    mapping = {}
    for item in types:
        type_id = item.get("id")
        if type_id is None:
            continue
        mapping[type_id] = item
    return mapping


def main():
    stats_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "player_stats_season_25598.json")
    )
    if not os.path.exists(stats_path):
        print(f"Missing player stats file: {stats_path}")
        sys.exit(1)

    with open(stats_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    type_map = load_types()
    counts = Counter()

    for team in data.values():
        for player in (team.get("players") or {}).values():
            for stat in player.get("statistics", []) or []:
                for detail in stat.get("details", []) or []:
                    type_id = detail.get("type_id")
                    if type_id is not None:
                        counts[type_id] += 1

    print("Top stat type_ids (count, id, name, code, developer_name, model_type):")
    for type_id, count in counts.most_common(40):
        info = type_map.get(type_id, {})
        print(
            f"{count:5d} {type_id:6d} {info.get('name')} | {info.get('code')} | "
            f"{info.get('developer_name')} | {info.get('model_type')}"
        )


if __name__ == "__main__":
    main()
