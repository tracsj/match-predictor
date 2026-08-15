import argparse
import json
import math
import os
from datetime import datetime, timedelta
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
        meta[fixture_id] = data
    return meta


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
    return {tid: [e.get("player_name") for e in entries][:11] for tid, entries in starters.items()}


def outcome_from_scores(fixture_data):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-id", type=int, default=None)
    parser.add_argument("--season-id", type=int, default=None)
    parser.add_argument("--quick", type=int, default=None, help="Only include last N fixture dates.")
    parser.add_argument("--window-days", type=int, default=120, help="Rolling training window in days.")
    parser.add_argument("--retrain-days", type=int, default=7, help="Refit model every N days.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    fixtures_dir = root / "data" / "fixtures"
    odds_dir = root / "data" / "odds"
    data_path = root / "data" / "player_match_features.csv"
    out_dir = root / "web"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Missing dataset: {data_path}")
        return

    bookmaker_id = os.getenv("SPORTMONKS_BOOKMAKER_ID", "").strip()
    bookmaker_id = int(bookmaker_id) if bookmaker_id.isdigit() else None

    fixture_meta = load_fixture_meta(fixtures_dir)
    rows = tpg.load_rows(data_path)
    rows = [r for r in rows if (tpg.safe_float(r.get("player_hist_matches")) or 0) > 0]
    rows = [r for r in rows if (tpg.safe_float(r.get("minutes_played")) or 0) > 0]
    if args.league_id is not None or args.season_id is not None:
        filtered = []
        for row in rows:
            fixture_id = str(row.get("fixture_id"))
            meta = fixture_meta.get(fixture_id)
            if not meta:
                continue
            if args.league_id is not None and meta.get("league_id") != args.league_id:
                continue
            if args.season_id is not None and meta.get("season_id") != args.season_id:
                continue
            filtered.append(row)
        rows = filtered
    rows.sort(
        key=lambda r: (
            tpg.parse_dt(r.get("starting_at")) or tpg.datetime.min,
            int(r.get("fixture_id") or 0),
            int(r.get("player_id") or 0),
        )
    )

    feature_cols = build_feature_cols(rows)
    X_all, y_all = tpg.build_feature_matrix(rows, feature_cols)
    dates = []
    for row in rows:
        start = tpg.parse_dt(row.get("starting_at")) or tpg.datetime.min
        dates.append(start)

    unique_dates = sorted({d for d in dates})
    if args.quick:
        unique_dates = unique_dates[-args.quick :]
        sample_rows = [row for row, d in zip(rows, dates) if d in unique_dates]
        league_season = {}
        for row in sample_rows:
            fixture_id = str(row.get("fixture_id"))
            meta = fixture_meta.get(fixture_id, {})
            key = (meta.get("league_id"), meta.get("season_id"))
            league_season[key] = league_season.get(key, 0) + 1
        print("Quick bundle includes fixture dates for:")
        for (lid, sid), count in sorted(league_season.items(), key=lambda item: item[1], reverse=True):
            print(f"  league {lid} season {sid}: {count} rows")

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

    def aggregate_team_preds(row_indices, preds):
        team_preds = {}
        for row_index, pred in zip(row_indices, preds):
            row = rows[row_index]
            fixture_id = str(row.get("fixture_id"))
            team_id = str(row.get("team_id"))
            weight = tpg.expected_minutes(row) / 90.0
            entry = team_preds.setdefault(fixture_id, {})
            entry[team_id] = entry.get(team_id, 0.0) + pred * weight
        return team_preds

    def fixture_outcomes(fixture_ids):
        outcomes = {}
        for fixture_id in fixture_ids:
            data = fixture_meta.get(str(fixture_id))
            if not data:
                continue
            score = outcome_from_scores(data)
            if not score:
                continue
            outcomes[str(fixture_id)] = tpg.outcome_label(score[0], score[1])
        return outcomes

    def temperature_scale(probs, temperature):
        if temperature <= 0:
            return probs
        scaled = {}
        denom = 0.0
        for key, value in probs.items():
            value = max(value, 1e-9)
            scaled[key] = value ** (1.0 / temperature)
            denom += scaled[key]
        return {k: v / denom for k, v in scaled.items()}

    def fit_temperature(fixture_probs, fixture_actuals):
        if not fixture_probs:
            return 1.0
        best_t = 1.0
        best_loss = float("inf")
        for t in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
            loss = 0.0
            count = 0
            for fixture_id, probs in fixture_probs.items():
                actual = fixture_actuals.get(fixture_id)
                if not actual:
                    continue
                scaled = temperature_scale(probs, t)
                loss -= math.log(max(scaled.get(actual, 1e-9), 1e-9))
                count += 1
            if count and loss < best_loss:
                best_loss = loss
                best_t = t
        return best_t

    predictions = {}
    fixture_temp = {}
    last_fit_date = None
    cached = None
    total_dates = len(unique_dates)
    for idx, date in enumerate(unique_dates, start=1):
        window_start = date - timedelta(days=args.window_days)
        train_idx = [i for i, d in enumerate(dates) if window_start <= d < date]
        test_idx = [i for i, d in enumerate(dates) if d == date]
        if not train_idx or not test_idx:
            continue

        refit = last_fit_date is None or (date - last_fit_date).days >= args.retrain_days
        if refit:
            print(f"[{idx}/{total_dates}] Fitting model @ {date.date().isoformat()} (window {args.window_days}d)")
            X_train = [X_all[i] for i in train_idx]
            y_train = [y_all[i] for i in train_idx]
            X_train, means, stds = tpg.standardize_impute(X_train)
            coef = tpg.ridge_fit_log1p(X_train, y_train, alpha=1.0)

            train_preds = tpg.predict_log1p(X_train, coef)
            train_team_preds = aggregate_team_preds(train_idx, train_preds)
            train_fixture_ids = list(train_team_preds.keys())
            train_actuals = fixture_outcomes(train_fixture_ids)
            train_fixture_probs = {}
            for fixture_id, teams in train_team_preds.items():
                if len(teams) != 2:
                    continue
                team_ids = list(teams.keys())
                pred_1 = teams[team_ids[0]]
                pred_2 = teams[team_ids[1]]
                p_home, p_draw, p_away = tpg.poisson_outcome_probs(pred_1, pred_2)
                train_fixture_probs[fixture_id] = {"W": p_home, "D": p_draw, "L": p_away}

            temperature = fit_temperature(train_fixture_probs, train_actuals)
            cached = (coef, means, stds, temperature)
            last_fit_date = date

        if cached is None:
            continue
        coef, means, stds, _ = cached
        X_test = [X_all[i] for i in test_idx]
        X_test, _, _ = tpg.standardize_impute(X_test, means, stds)
        preds = tpg.predict_log1p(X_test, coef)
        team_preds = aggregate_team_preds(test_idx, preds)

        for fixture_id, teams in team_preds.items():
            entry = predictions.setdefault(fixture_id, {})
            entry.update(teams)
            fixture_temp[fixture_id] = cached[3] if cached else 1.0

    fixtures = {}
    for fixture_id, data in fixture_meta.items():
        if args.league_id is not None and data.get("league_id") != args.league_id:
            continue
        if args.season_id is not None and data.get("season_id") != args.season_id:
            continue
        participants = data.get("participants") or []
        home_id = None
        away_id = None
        home_name = None
        away_name = None
        for participant in participants:
            location = (participant.get("meta") or {}).get("location")
            if location == "home":
                home_id = participant.get("id")
                home_name = participant.get("name")
            elif location == "away":
                away_id = participant.get("id")
                away_name = participant.get("name")

        pred = None
        pred_probs = None
        key_home = str(home_id) if home_id is not None else None
        key_away = str(away_id) if away_id is not None else None
        if fixture_id in predictions and key_home in predictions[fixture_id] and key_away in predictions[fixture_id]:
            pred_home = predictions[fixture_id][key_home]
            pred_away = predictions[fixture_id][key_away]
            p_home, p_draw, p_away = tpg.poisson_outcome_probs(pred_home, pred_away)
            pred = {"home": pred_home, "away": pred_away}
            probs = {"W": p_home, "D": p_draw, "L": p_away}
            temperature = fixture_temp.get(fixture_id, 1.0)
            calibrated = temperature_scale(probs, temperature)
            outcome = max(calibrated, key=calibrated.get)
            pred_probs = {
                "W": calibrated["W"],
                "D": calibrated["D"],
                "L": calibrated["L"],
                "outcome": outcome,
                "confidence": max(calibrated.values()),
            }

        fixtures[fixture_id] = {
            "id": data.get("id"),
            "name": data.get("name"),
            "starting_at": data.get("starting_at"),
            "league_id": data.get("league_id"),
            "season_id": data.get("season_id"),
            "round_id": data.get("round_id"),
            "home_id": home_id,
            "away_id": away_id,
            "home_name": home_name,
            "away_name": away_name,
            "scores": outcome_from_scores(data),
            "odds": (odds_by_fixture.get(fixture_id) or {}).get("odds"),
            "prediction": pred,
            "probs": pred_probs,
            "lineups": extract_lineups(data),
        }

    leagues = {}
    for fixture in fixtures.values():
        league_id = fixture.get("league_id")
        if not league_id:
            continue
        leagues.setdefault(
            league_id,
            {"id": league_id, "name": LEAGUE_NAMES.get(league_id, f"League {league_id}"), "seasons": {}},
        )
        seasons = leagues[league_id]["seasons"]
        season_id = fixture.get("season_id")
        if not season_id:
            continue
        seasons.setdefault(season_id, {"id": season_id, "rounds": {}})
        rounds = seasons[season_id]["rounds"]
        round_id = fixture.get("round_id")
        if round_id:
            rounds.setdefault(round_id, {"id": round_id, "fixtures": [], "min_date": None, "max_date": None})
            rounds[round_id]["fixtures"].append(fixture.get("id"))
            start = tpg.parse_dt(fixture.get("starting_at") or "")
            if start:
                if rounds[round_id]["min_date"] is None or start < rounds[round_id]["min_date"]:
                    rounds[round_id]["min_date"] = start
                if rounds[round_id]["max_date"] is None or start > rounds[round_id]["max_date"]:
                    rounds[round_id]["max_date"] = start

    # sort fixtures by date within rounds
    for league in leagues.values():
        for season in league["seasons"].values():
            for round_info in season["rounds"].values():
                round_info["fixtures"] = sorted(
                    round_info["fixtures"],
                    key=lambda fid: tpg.parse_dt(fixtures[str(fid)]["starting_at"] or "") or tpg.datetime.min,
                )
                if round_info["min_date"]:
                    round_info["min_date"] = round_info["min_date"].date().isoformat()
                if round_info["max_date"]:
                    round_info["max_date"] = round_info["max_date"].date().isoformat()

    bundle = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "bookmaker_id": bookmaker_id,
        "feature_cols": feature_cols,
        "quick_limit": args.quick,
        "filters": {
            "league_id": args.league_id,
            "season_id": args.season_id,
            "window_days": args.window_days,
            "retrain_days": args.retrain_days,
        },
        "leagues": leagues,
        "fixtures": fixtures,
    }

    out_path = out_dir / "data.json"
    out_path.write_text(json.dumps(bundle, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
