import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-id", required=True, help="Fixture ID to predict")
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

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
    coef = tpg.ridge_fit_log1p(X, y, alpha=1.0)

    fixture_rows = [r for r in rows if str(r.get("fixture_id")) == str(args.fixture_id)]
    if not fixture_rows:
        print(f"No rows found for fixture {args.fixture_id}.")
        return

    X_fixture, _ = tpg.build_feature_matrix(fixture_rows, feature_cols)
    X_fixture, _, _ = tpg.standardize_impute(X_fixture, means, stds)
    preds = tpg.predict_log1p(X_fixture, coef)

    team_pred = {}
    for row, pred in zip(fixture_rows, preds):
        key = (row.get("team_id"))
        team_pred[key] = team_pred.get(key, 0.0) + pred * (tpg.expected_minutes(row) / 90.0)

    if len(team_pred) != 2:
        print(f"Expected 2 teams, found {len(team_pred)}.")
        return

    team_ids = list(team_pred.keys())
    t1, t2 = team_ids[0], team_ids[1]
    pred_1 = team_pred[t1]
    pred_2 = team_pred[t2]
    p_home, p_draw, p_away = tpg.poisson_outcome_probs(pred_1, pred_2)
    probs = {"W": p_home, "D": p_draw, "L": p_away}
    outcome = max(probs, key=probs.get)
    confidence = probs[outcome]

    if confidence < args.min_confidence:
        if args.json:
            import json
            print(json.dumps({
                "fixture_id": args.fixture_id,
                "team_ids": [t1, t2],
                "predicted_goals": {str(t1): round(pred_1, 3), str(t2): round(pred_2, 3)},
                "probs": {"W": round(p_home, 3), "D": round(p_draw, 3), "L": round(p_away, 3)},
                "predicted_outcome": outcome,
                "confidence": round(confidence, 3),
                "skipped": True,
                "min_confidence": args.min_confidence,
            }))
        else:
            print(f"Confidence {confidence:.3f} below threshold {args.min_confidence:.3f}. Skipping.")
        return

    if args.json:
        import json
        print(json.dumps({
            "fixture_id": args.fixture_id,
            "team_ids": [t1, t2],
            "predicted_goals": {str(t1): round(pred_1, 3), str(t2): round(pred_2, 3)},
            "probs": {"W": round(p_home, 3), "D": round(p_draw, 3), "L": round(p_away, 3)},
            "predicted_outcome": outcome,
            "confidence": round(confidence, 3),
            "bookmaker_id": bookmaker_id or None,
            "skipped": False,
            "min_confidence": args.min_confidence,
        }))
        return

    bookmaker_id = os.getenv("SPORTMONKS_BOOKMAKER_ID", "").strip()
    bookmaker_label = f"bookmaker {bookmaker_id}" if bookmaker_id else "bookmaker"
    print(f"Fixture {args.fixture_id}")
    print(f"Team {t1}: {pred_1:.2f} goals")
    print(f"Team {t2}: {pred_2:.2f} goals")
    print(f"Outcome probs: W={p_home:.3f} D={p_draw:.3f} L={p_away:.3f}")
    print(f"Predicted outcome: {outcome} (confidence {confidence:.3f})")
    print(f"Recommended bet: {outcome} @ {bookmaker_label}")


if __name__ == "__main__":
    main()
