import json
import os
from datetime import datetime
from pathlib import Path

import train_player_goals as tpg
from backtest_betting import pick_best_market, pick_market_for_bookmaker, build_feature_cols


LEAGUE_NAMES = {
    501: "Scottish Premiership",
    271: "Danish Superliga",
}


def load_fixture_meta(fixtures_dir):
    meta = {}
    for path in fixtures_dir.glob("fixture_*.json"):
        fixture_id = path.stem.split("_")[-1]
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        data = payload.get("data") or {}
        meta[fixture_id] = {
            "league_id": data.get("league_id"),
            "season_id": data.get("season_id"),
            "round_id": data.get("round_id"),
            "starting_at": data.get("starting_at"),
            "name": data.get("name"),
        }
    return meta


def load_fixture_payload(fixtures_dir, fixture_id):
    path = fixtures_dir / f"fixture_{fixture_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("data") or {}
    except json.JSONDecodeError:
        return None


def build_league_index(fixture_meta):
    league_ids = sorted({meta.get("league_id") for meta in fixture_meta.values() if meta.get("league_id")})
    items = []
    for lid in league_ids:
        label = LEAGUE_NAMES.get(lid, f"League {lid}")
        items.append((lid, label))
    return items


def season_labels(fixture_meta, league_id):
    seasons = {}
    for meta in fixture_meta.values():
        if meta.get("league_id") != league_id:
            continue
        season_id = meta.get("season_id")
        start = tpg.parse_dt(meta.get("starting_at")) if meta.get("starting_at") else None
        if not season_id or not start:
            continue
        season_info = seasons.setdefault(season_id, {"min": start, "max": start})
        if start < season_info["min"]:
            season_info["min"] = start
        if start > season_info["max"]:
            season_info["max"] = start
    items = []
    for season_id, info in seasons.items():
        label = f"{season_id} ({info['min'].year}-{info['max'].year})"
        items.append((season_id, label, info["min"]))
    items.sort(key=lambda item: item[2])
    return [(sid, label) for sid, label, _ in items]


def build_rounds(fixture_meta, league_id, season_id):
    rounds = {}
    for fixture_id, meta in fixture_meta.items():
        if meta.get("league_id") != league_id or meta.get("season_id") != season_id:
            continue
        round_id = meta.get("round_id")
        start = tpg.parse_dt(meta.get("starting_at")) if meta.get("starting_at") else None
        if not round_id or not start:
            continue
        info = rounds.setdefault(round_id, {"min": start, "fixtures": []})
        if start < info["min"]:
            info["min"] = start
        info["fixtures"].append(fixture_id)
    ordered = sorted(rounds.items(), key=lambda item: item[1]["min"])
    now = datetime.utcnow()
    current_index = 0
    for idx, (_, info) in enumerate(ordered):
        if info["min"] <= now:
            current_index = idx
    return ordered, current_index


def prompt_choice(title, options, default_index=0):
    print(title)
    for idx, (_, label) in enumerate(options, start=1):
        marker = "*" if idx - 1 == default_index else " "
        print(f"  {idx:2d}. {label} {marker}")
    while True:
        raw = input(f"Select [default {default_index + 1}]: ").strip()
        if not raw:
            return options[default_index][0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        print("Invalid selection. Try again.")


def outcome_from_scores(fixture_data):
    if not fixture_data:
        return None
    participants = fixture_data.get("participants") or []
    location = {p.get("id"): p.get("meta", {}).get("location") for p in participants}
    scores = fixture_data.get("scores") or []
    home_goals = None
    away_goals = None
    for score in scores:
        if score.get("type_id") != 1525 and score.get("description") != "CURRENT":
            continue
        team_id = score.get("participant_id")
        goals = (score.get("score") or {}).get("goals")
        if team_id is None or goals is None:
            continue
        if location.get(team_id) == "home":
            home_goals = goals
        elif location.get(team_id) == "away":
            away_goals = goals
    if home_goals is None or away_goals is None:
        return None
    return int(home_goals), int(away_goals)


def extract_lineups(fixture_data):
    lineups = fixture_data.get("lineups") or []
    starters = {}
    for entry in lineups:
        if entry.get("type_id") != 11:
            continue
        team_id = entry.get("team_id")
        starters.setdefault(team_id, []).append(entry)
    for team_id in starters:
        starters[team_id].sort(key=lambda e: e.get("formation_position") or 0)
    return starters


def main():
    fixtures_dir = Path(__file__).resolve().parent.parent / "data" / "fixtures"
    odds_dir = Path(__file__).resolve().parent.parent / "data" / "odds"
    data_path = Path(__file__).resolve().parent.parent / "data" / "player_match_features.csv"
    if not data_path.exists():
        print(f"Missing dataset: {data_path}")
        return

    bookmaker_id = os.getenv("SPORTMONKS_BOOKMAKER_ID", "").strip()
    bookmaker_id = int(bookmaker_id) if bookmaker_id.isdigit() else None

    fixture_meta = load_fixture_meta(fixtures_dir)
    leagues = build_league_index(fixture_meta)
    if not leagues:
        print("No fixtures found.")
        return
    league_id = prompt_choice("Select league:", leagues)

    seasons = season_labels(fixture_meta, league_id)
    if not seasons:
        print("No seasons found for that league.")
        return
    season_id = prompt_choice("Select season:", seasons, default_index=len(seasons) - 1)

    rounds, current_idx = build_rounds(fixture_meta, league_id, season_id)
    if not rounds:
        print("No rounds found for that league/season.")
        return
    round_options = []
    for idx, (round_id, info) in enumerate(rounds, start=1):
        label = f"Round {idx} (id {round_id}) - {info['min'].date()} - fixtures {len(info['fixtures'])}"
        round_options.append((round_id, label))
    round_id = prompt_choice("Select gameweek/round:", round_options, default_index=current_idx)

    threshold_raw = input("Min confidence [default 0.45]: ").strip()
    min_confidence = float(threshold_raw) if threshold_raw else 0.45
    show_lineups = input("Show lineups? [y/N]: ").strip().lower().startswith("y")

    rows = tpg.load_rows(data_path)
    rows = [r for r in rows if (tpg.safe_float(r.get("player_hist_matches")) or 0) > 0]
    rows = [r for r in rows if (tpg.safe_float(r.get("minutes_played")) or 0) > 0]
    rows.sort(
        key=lambda r: (
            tpg.parse_dt(r.get("starting_at")) or tpg.datetime.min,
            int(r.get("fixture_id") or 0),
            int(r.get("player_id") or 0),
        )
    )

    feature_cols = build_feature_cols(rows)
    X, y = tpg.build_feature_matrix(rows, feature_cols)
    X, means, stds = tpg.standardize_impute(X)
    coef = tpg.ridge_fit_log1p(X, y, alpha=1.0)

    odds_by_fixture = {}
    for path in odds_dir.glob("fixture_*.json"):
        fixture_id = path.stem.split("_")[-1]
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if bookmaker_id is not None:
            best = pick_market_for_bookmaker(payload, bookmaker_id)
        else:
            best = pick_best_market(payload)
        if best:
            odds_by_fixture[fixture_id] = best

    rows_by_fixture = {}
    for row in rows:
        fid = str(row.get("fixture_id"))
        meta = fixture_meta.get(fid)
        if not meta or meta.get("league_id") != league_id or meta.get("season_id") != season_id:
            continue
        if meta.get("round_id") != round_id:
            continue
        rows_by_fixture.setdefault(fid, []).append(row)

    fixtures_in_round = None
    for rid, info in rounds:
        if rid == round_id:
            fixtures_in_round = info["fixtures"]
            break
    if not fixtures_in_round:
        print("No fixtures found for selected round.")
        return

    print("")
    print(f"League {league_id} | Season {season_id} | Round {round_id}")
    print(f"Min confidence: {min_confidence}")
    if bookmaker_id:
        print(f"Bookmaker: {bookmaker_id}")
    print("")

    bets = 0
    wins = 0
    profit = 0.0

    for fixture_id in fixtures_in_round:
        meta = fixture_meta.get(str(fixture_id), {})
        fixture_rows = rows_by_fixture.get(str(fixture_id))
        if not fixture_rows:
            print(f"{fixture_id} | {meta.get('name','Fixture')} (no lineup features)")
            continue
        X_fixture, _ = tpg.build_feature_matrix(fixture_rows, feature_cols)
        X_fixture, _, _ = tpg.standardize_impute(X_fixture, means, stds)
        preds = tpg.predict_log1p(X_fixture, coef)
        team_pred = {}
        for row, pred in zip(fixture_rows, preds):
            team_id = row.get("team_id")
            team_pred[team_id] = team_pred.get(team_id, 0.0) + pred * (tpg.expected_minutes(row) / 90.0)
        if len(team_pred) != 2:
            continue

        team_ids = list(team_pred.keys())
        t1, t2 = team_ids[0], team_ids[1]
        pred_1 = team_pred[t1]
        pred_2 = team_pred[t2]
        p_home, p_draw, p_away = tpg.poisson_outcome_probs(pred_1, pred_2)
        probs = {"W": p_home, "D": p_draw, "L": p_away}
        outcome = max(probs, key=probs.get)
        confidence = probs[outcome]
        odds = odds_by_fixture.get(str(fixture_id), {}).get("odds", {})

        fixture_data = load_fixture_payload(fixtures_dir, fixture_id)
        actual = outcome_from_scores(fixture_data)
        actual_label = None
        if actual:
            actual_label = tpg.outcome_label(actual[0], actual[1])

        name = meta.get("name", "Fixture")
        print(f"{fixture_id} | {name} | {meta.get('starting_at')}")
        print(f"  pred goals: {pred_1:.2f} - {pred_2:.2f}")
        print(f"  probs: W={p_home:.3f} D={p_draw:.3f} L={p_away:.3f}")
        if odds:
            print(f"  odds: W={odds.get('W')} D={odds.get('D')} L={odds.get('L')}")
        if confidence >= min_confidence:
            print(f"  suggestion: BET {outcome} (conf {confidence:.2f})")
        else:
            print(f"  suggestion: PASS (conf {confidence:.2f})")
        if actual:
            print(f"  actual: {actual[0]}-{actual[1]} ({actual_label})")

        if confidence >= min_confidence and odds and outcome in odds and actual_label:
            bets += 1
            if actual_label == outcome:
                wins += 1
                profit += float(odds[outcome]) - 1.0
            else:
                profit -= 1.0

        if show_lineups and fixture_data:
            starters = extract_lineups(fixture_data)
            participants = fixture_data.get("participants") or []
            team_names = {p.get("id"): p.get("name") for p in participants}
            for team_id, players in starters.items():
                names = [p.get("player_name") for p in players][:11]
                print(f"  lineup {team_names.get(team_id, team_id)}: {', '.join(names)}")
        print("")

    if bets:
        roi = profit / bets
        hit = wins / bets
        print(f"Round ROI (if followed bets): bets {bets} | hit {hit:.3f} | profit ${profit:.2f} | roi {roi:.3f}")
    else:
        print("Round ROI: no bets placed at current threshold.")


if __name__ == "__main__":
    main()
