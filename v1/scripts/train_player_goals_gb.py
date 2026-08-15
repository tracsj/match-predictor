import math
import os
from pathlib import Path

import train_player_goals as tpg


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


def main():
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except Exception:
        print("Missing scikit-learn. Install with: python3 -m pip install scikit-learn")
        return

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
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    test_rows = rows[split:]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_pred = [max(0.0, p) for p in test_pred]

    print(f"Rows total: {len(rows)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Train MAE: {tpg.mae(y_train, train_pred):.3f}")
    print(f"Train RMSE: {tpg.rmse(y_train, train_pred):.3f}")
    print(f"Test MAE: {tpg.mae(y_test, test_pred):.3f}")
    print(f"Test RMSE: {tpg.rmse(y_test, test_pred):.3f}")

    # baseline
    baseline_test = [tpg.baseline_player_pred(row) for row in test_rows]
    print(f"Baseline Test MAE: {tpg.mae(y_test, baseline_test):.3f}")
    print(f"Baseline Test RMSE: {tpg.rmse(y_test, baseline_test):.3f}")

    # team aggregation with minutes weighting
    team_actual = {}
    team_pred = {}
    for row, pred in zip(test_rows, test_pred):
        key = (row.get("fixture_id"), row.get("team_id"))
        team_actual[key] = team_actual.get(key, 0.0) + float(row.get("goals") or 0)
        team_pred[key] = team_pred.get(key, 0.0) + pred * (tpg.expected_minutes(row) / 90.0)

    actual_vals = [team_actual[k] for k in team_actual]
    pred_vals = [team_pred.get(k, 0.0) for k in team_actual]
    print("")
    print("Team-level aggregation:")
    print(f"Team MAE: {tpg.mae(actual_vals, pred_vals):.3f}")
    print(f"Team RMSE: {tpg.rmse(actual_vals, pred_vals):.3f}")

    # match outcome with Poisson
    fixtures = {}
    for (fixture_id, team_id), goals in team_actual.items():
        fixtures.setdefault(fixture_id, {})[team_id] = goals
    outcome_rows = []
    for fixture_id, teams in fixtures.items():
        if len(teams) != 2:
            continue
        team_ids = list(teams.keys())
        t1, t2 = team_ids[0], team_ids[1]
        actual_outcome = tpg.outcome_label(teams[t1], teams[t2])
        pred_1 = team_pred.get((fixture_id, t1), 0.0)
        pred_2 = team_pred.get((fixture_id, t2), 0.0)
        p_home, p_draw, p_away = tpg.poisson_outcome_probs(pred_1, pred_2)
        probs = {"W": p_home, "D": p_draw, "L": p_away}
        pred_outcome = max(probs, key=probs.get)
        outcome_rows.append((actual_outcome, pred_outcome))

    print("")
    print(f"Outcome accuracy (poisson): {tpg.outcome_accuracy(outcome_rows):.3f}")


if __name__ == "__main__":
    main()
