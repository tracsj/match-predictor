import os
import json
from pathlib import Path

import train_player_goals as tpg
from backtest_betting import pick_best_market, pick_market_for_bookmaker


def build_feature_cols(rows):
    feature_cols = [
        "is_home",
        "team_position",
        "opponent_position",
        "lineup_type",
        "position_group",
        "player_hist_matches",
        "player_hist_minutes",
        "player_hist_avg_minutes",
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
    for col in rows[0].keys():
        if col.startswith("roll_"):
            feature_cols.append(col)
    return feature_cols


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
            "starting_at": data.get("starting_at"),
        }
    return meta


def main():
    data_path = Path(__file__).resolve().parent.parent / "data" / "player_match_features.csv"
    if not data_path.exists():
        print(f"Missing dataset: {data_path}")
        return

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

    split = int(len(rows) * 0.8)
    train_rows = rows[:split]
    test_rows = rows[split:]
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]

    coef = tpg.ridge_fit_log1p(X_train, y_train, alpha=1.0)
    preds = tpg.predict_log1p(X_test, coef)

    # odds data
    odds_dir = Path(__file__).resolve().parent.parent / "data" / "odds"
    if not odds_dir.exists():
        print(f"Missing odds directory: {odds_dir}")
        return
    bookmaker_filter = os.getenv("SPORTMONKS_BOOKMAKER_ID", "").strip()
    bookmaker_filter = int(bookmaker_filter) if bookmaker_filter.isdigit() else None

    odds_by_fixture = {}
    for path in odds_dir.glob("fixture_*.json"):
        fixture_id = path.stem.split("_")[-1]
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if bookmaker_filter is not None:
            best = pick_market_for_bookmaker(payload, bookmaker_filter)
        else:
            best = pick_best_market(payload)
        if best:
            odds_by_fixture[fixture_id] = best

    fixture_meta = load_fixture_meta(Path(__file__).resolve().parent.parent / "data" / "fixtures")

    threshold = float(os.getenv("MIN_CONFIDENCE", "0.45"))

    team_pred = {}
    team_actual = {}
    for row, pred in zip(test_rows, preds):
        key = (row.get("fixture_id"), row.get("team_id"))
        team_actual[key] = team_actual.get(key, 0.0) + float(row.get("goals") or 0)
        team_pred[key] = team_pred.get(key, 0.0) + pred * (tpg.expected_minutes(row) / 90.0)

    fixtures = {}
    for (fixture_id, team_id), actual in team_actual.items():
        fixtures.setdefault(fixture_id, {})[team_id] = actual

    summary = {}
    overall = {"bets": 0, "wins": 0, "profit": 0.0}
    for fixture_id, teams in fixtures.items():
        if fixture_id not in odds_by_fixture:
            continue
        if len(teams) != 2:
            continue
        t1, t2 = list(teams.keys())
        pred_1 = team_pred.get((fixture_id, t1), 0.0)
        pred_2 = team_pred.get((fixture_id, t2), 0.0)
        p_home, p_draw, p_away = tpg.poisson_outcome_probs(pred_1, pred_2)
        probs = {"W": p_home, "D": p_draw, "L": p_away}
        outcome = max(probs, key=probs.get)
        confidence = probs[outcome]
        if confidence < threshold:
            continue
        odds = odds_by_fixture[fixture_id]["odds"]
        if outcome not in odds:
            continue
        actual = tpg.outcome_label(teams[t1], teams[t2])
        meta = fixture_meta.get(str(fixture_id)) or {}
        key = (meta.get("league_id"), meta.get("season_id"))
        bucket = summary.setdefault(key, {"bets": 0, "wins": 0, "profit": 0.0})
        bucket["bets"] += 1
        overall["bets"] += 1
        if actual == outcome:
            bucket["wins"] += 1
            overall["wins"] += 1
            bucket["profit"] += float(odds[outcome]) - 1.0
            overall["profit"] += float(odds[outcome]) - 1.0
        else:
            bucket["profit"] -= 1.0
            overall["profit"] -= 1.0

    print(f"ROI summary (threshold {threshold:.2f}, bookmaker {bookmaker_filter or 'best'})")
    for (league_id, season_id), stats in sorted(summary.items()):
        bets = stats["bets"]
        if bets == 0:
            continue
        roi = stats["profit"] / bets
        hit = stats["wins"] / bets
        print(
            f"league {league_id} season {season_id} | bets {bets:3d} "
            f"| hit {hit:.3f} | profit ${stats['profit']:.2f} | roi {roi:.3f}"
        )
    if overall["bets"]:
        roi = overall["profit"] / overall["bets"]
        hit = overall["wins"] / overall["bets"]
        print(
            f"overall | bets {overall['bets']:3d} | hit {hit:.3f} "
            f"| profit ${overall['profit']:.2f} | roi {roi:.3f}"
        )


if __name__ == "__main__":
    main()
