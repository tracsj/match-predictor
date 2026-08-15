import csv
import math
import os
from datetime import datetime
from pathlib import Path


def parse_dt(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def build_feature_matrix(rows, feature_cols):
    X = []
    y = []
    for row in rows:
        feats = []
        for col in feature_cols:
            value = safe_float(row.get(col))
            feats.append(value)
        X.append(feats)
        y.append(int(float(row.get("goals", 0) or 0)))
    return X, y


def standardize_impute(X, means=None, stds=None):
    if not X:
        return X, [], []
    n = len(X)
    m = len(X[0])
    if means is None:
        means = [0.0] * m
        stds = [1.0] * m
        for j in range(m):
            values = [X[i][j] for i in range(n) if X[i][j] is not None]
            if not values:
                means[j] = 0.0
                stds[j] = 1.0
                continue
            mean = sum(values) / len(values)
            means[j] = mean
            var = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(var)
            stds[j] = std if std > 1e-9 else 1.0
    Xn = []
    for i in range(n):
        row = []
        for j in range(m):
            value = X[i][j]
            if value is None:
                value = means[j]
            row.append((value - means[j]) / stds[j])
        Xn.append(row)
    return Xn, means, stds


def add_bias(X):
    return [[1.0] + row for row in X]


def ridge_fit_log1p(X, y, alpha=1.0):
    Xb = add_bias(X)
    n = len(Xb)
    m = len(Xb[0])
    # normal equations with ridge
    xtx = [[0.0 for _ in range(m)] for _ in range(m)]
    xty = [0.0 for _ in range(m)]
    for i in range(n):
        row = Xb[i]
        yi = math.log1p(y[i])
        for a in range(m):
            xty[a] += row[a] * yi
            for b in range(m):
                xtx[a][b] += row[a] * row[b]
    for d in range(m):
        xtx[d][d] += alpha
    # solve via Gauss-Jordan
    aug = [xtx[i] + [xty[i]] for i in range(m)]
    for i in range(m):
        pivot = aug[i][i]
        if abs(pivot) < 1e-9:
            continue
        inv = 1.0 / pivot
        for j in range(i, m + 1):
            aug[i][j] *= inv
        for k in range(m):
            if k == i:
                continue
            factor = aug[k][i]
            if abs(factor) < 1e-12:
                continue
            for j in range(i, m + 1):
                aug[k][j] -= factor * aug[i][j]
    coef = [aug[i][m] for i in range(m)]
    return coef


def predict_log1p(X, coef):
    Xb = add_bias(X)
    preds = []
    for row in Xb:
        eta = sum(row[j] * coef[j] for j in range(len(coef)))
        pred = math.expm1(eta)
        preds.append(max(pred, 0.0))
    return preds


def baseline_player_pred(row):
    roll_goals = safe_float(row.get("roll_goals_per90"))
    avg_minutes = safe_float(row.get("player_hist_avg_minutes"))
    if roll_goals is None or avg_minutes is None:
        return 0.0
    return max(roll_goals * (avg_minutes / 90.0), 0.0)


def expected_minutes(row):
    avg_minutes = safe_float(row.get("player_hist_avg_minutes"))
    if avg_minutes is not None and avg_minutes > 0:
        return avg_minutes
    lineup_type = int(float(row.get("lineup_type") or 0))
    if lineup_type == 11:
        return 90.0
    if lineup_type == 12:
        return 20.0
    return 0.0


def mae(y, yhat):
    return sum(abs(a - b) for a, b in zip(y, yhat)) / len(y)


def rmse(y, yhat):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y, yhat)) / len(y))


def fit_calibration(x_vals, y_vals):
    if not x_vals:
        return 0.0, 1.0
    x_mean = sum(x_vals) / len(x_vals)
    y_mean = sum(y_vals) / len(y_vals)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    den = sum((x - x_mean) ** 2 for x in x_vals)
    if abs(den) < 1e-9:
        return y_mean, 0.0
    b = num / den
    a = y_mean - b * x_mean
    return a, b


def apply_calibration(a, b, x_vals):
    return [a + b * x for x in x_vals]


def outcome_label(goals_for, goals_against):
    if goals_for > goals_against:
        return "W"
    if goals_for < goals_against:
        return "L"
    return "D"


def outcome_accuracy(rows):
    correct = 0
    for actual, pred in rows:
        if actual == pred:
            correct += 1
    return correct / len(rows) if rows else 0.0


def outcome_confusion(rows):
    labels = ["W", "D", "L"]
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for actual, pred in rows:
        matrix[actual][pred] += 1
    return matrix


def poisson_outcome_probs(lam_home, lam_away, max_goals=6):
    lam_home = max(lam_home, 1e-6)
    lam_away = max(lam_away, 1e-6)
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    home_probs = [math.exp(-lam_home) * lam_home**k / math.factorial(k) for k in range(max_goals + 1)]
    away_probs = [math.exp(-lam_away) * lam_away**k / math.factorial(k) for k in range(max_goals + 1)]
    for i, ph in enumerate(home_probs):
        for j, pa in enumerate(away_probs):
            p = ph * pa
            if i > j:
                p_home += p
            elif i < j:
                p_away += p
            else:
                p_draw += p
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total
    return p_home, p_draw, p_away


def main():
    data_path = Path(__file__).resolve().parent.parent / "data" / "player_match_features.csv"
    if not data_path.exists():
        print(f"Missing dataset: {data_path}")
        return

    rows = load_rows(data_path)
    if not rows:
        print("No rows found in dataset.")
        return

    # filter out players with no prior history
    filtered = []
    for row in rows:
        matches = safe_float(row.get("player_hist_matches")) or 0.0
        if matches <= 0:
            continue
        filtered.append(row)

    if not filtered:
        print("No rows with player history > 0.")
        return

    filtered.sort(
        key=lambda r: (
            parse_dt(r.get("starting_at")) or datetime.min,
            int(r.get("fixture_id") or 0),
            int(r.get("player_id") or 0),
        )
    )

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

    split = int(len(filtered) * 0.8)
    train_rows = filtered[:split]
    test_rows = filtered[split:]

    # train only on players who actually played minutes
    train_rows = [row for row in train_rows if (safe_float(row.get("minutes_played")) or 0) > 0]
    test_rows = [row for row in test_rows if (safe_float(row.get("minutes_played")) or 0) > 0]

    X_train, y_train = build_feature_matrix(train_rows, feature_cols)
    X_test, y_test = build_feature_matrix(test_rows, feature_cols)

    X_train, means, stds = standardize_impute(X_train)
    X_test, _, _ = standardize_impute(X_test, means, stds)

    coef = ridge_fit_log1p(X_train, y_train, alpha=1.0)
    train_pred = predict_log1p(X_train, coef)
    test_pred = predict_log1p(X_test, coef)

    print(f"Rows total: {len(rows)}")
    print(f"Rows with history >0: {len(filtered)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Train MAE: {mae(y_train, train_pred):.3f}")
    print(f"Train RMSE: {rmse(y_train, train_pred):.3f}")
    print(f"Test MAE: {mae(y_test, test_pred):.3f}")
    print(f"Test RMSE: {rmse(y_test, test_pred):.3f}")
    baseline_test = [baseline_player_pred(row) for row in test_rows]
    print(f"Baseline Test MAE: {mae(y_test, baseline_test):.3f}")
    print(f"Baseline Test RMSE: {rmse(y_test, baseline_test):.3f}")

    print("")
    print("Sample predictions (first 10 test rows):")
    for row, pred in list(zip(test_rows, test_pred))[:10]:
        print(
            f"fixture {row.get('fixture_id')} player {row.get('player_id')} "
            f"pred_goals={pred:.2f} actual={row.get('goals')}"
        )

    # Team-level aggregation on full lineup rows (history rows may be missing for some players)
    print("")
    print("Team-level aggregation (history > 0):")
    all_rows = [row for row in rows if (safe_float(row.get("player_hist_matches")) or 0) > 0]
    all_rows = sorted(
        all_rows,
        key=lambda r: (
            parse_dt(r.get("starting_at")) or datetime.min,
            int(r.get("fixture_id") or 0),
            int(r.get("player_id") or 0),
        ),
    )
    X_all, _ = build_feature_matrix(all_rows, feature_cols)
    X_all, _, _ = standardize_impute(X_all, means, stds)
    all_pred = predict_log1p(X_all, coef)

    team_actual = {}
    team_pred = {}
    team_baseline = {}
    team_attack = {}
    team_defense = {}
    opp_attack = {}
    opp_defense = {}
    league_avg = {}
    for row, pred in zip(all_rows, all_pred):
        key = (row.get("fixture_id"), row.get("team_id"))
        minutes_scale = expected_minutes(row) / 90.0
        pred_scaled = pred * minutes_scale
        team_actual[key] = team_actual.get(key, 0.0) + float(row.get("goals") or 0)
        team_pred[key] = team_pred.get(key, 0.0) + pred_scaled
        team_baseline[key] = team_baseline.get(key, 0.0) + baseline_player_pred(row)
        # keep indices for potential diagnostics; not used in default pipeline
        team_attack[key] = float(row.get("team_attack_index") or 0.0)
        team_defense[key] = float(row.get("team_defense_index") or 0.0)
        opp_attack[key] = float(row.get("opp_attack_index") or 0.0)
        opp_defense[key] = float(row.get("opp_defense_index") or 0.0)
        league_avg[key] = float(row.get("league_avg_goals_per_team") or 0.0)

    actual_vals = [team_actual[k] for k in team_actual]
    pred_vals = [team_pred.get(k, 0.0) for k in team_actual]
    base_vals = [team_baseline.get(k, 0.0) for k in team_actual]
    print(f"Team MAE: {mae(actual_vals, pred_vals):.3f}")
    print(f"Team RMSE: {rmse(actual_vals, pred_vals):.3f}")
    print(f"Team Baseline MAE: {mae(actual_vals, base_vals):.3f}")
    print(f"Team Baseline RMSE: {rmse(actual_vals, base_vals):.3f}")

    sample_keys = list(team_actual.keys())[:10]
    print("")
    print("Sample team predictions (first 10):")
    for key in sample_keys:
        fixture_id, team_id = key
        print(
            f"fixture {fixture_id} team {team_id} pred={team_pred[key]:.2f} "
            f"baseline={team_baseline[key]:.2f} actual={team_actual[key]:.0f}"
        )

    # Match-level scorelines & W/D/L accuracy (history > 0 only)
    print("")
    print("Match-level scorelines (single-stage):")
    fixtures = {}
    for (fixture_id, team_id), actual in team_actual.items():
        if fixture_id not in fixtures:
            fixtures[fixture_id] = {}
        fixtures[fixture_id][team_id] = {
            "actual": actual,
            "pred": team_pred.get((fixture_id, team_id), 0.0),
            "baseline": team_baseline.get((fixture_id, team_id), 0.0),
        }

    outcome_rows = []
    baseline_outcome_rows = []
    poisson_outcome_rows = []
    calibrated_outcome_rows = []
    match_rows = []
    # calibration fit on training fixtures only
    train_cutoff = parse_dt(test_rows[0].get("starting_at")) if test_rows else None
    train_actual = []
    train_pred = []
    for fixture_id, teams in fixtures.items():
        if train_cutoff is not None:
            sample_row = next((r for r in all_rows if r.get("fixture_id") == fixture_id), None)
            if sample_row and parse_dt(sample_row.get("starting_at")) >= train_cutoff:
                continue
        if len(teams) != 2:
            continue
        for team_id in teams.keys():
            train_actual.append(teams[team_id]["actual"])
            train_pred.append(teams[team_id]["pred"])
    a_cal, b_cal = fit_calibration(train_pred, train_actual)
    for fixture_id, teams in fixtures.items():
        if len(teams) != 2:
            continue
        team_ids = list(teams.keys())
        t1, t2 = team_ids[0], team_ids[1]
        actual_1 = teams[t1]["actual"]
        actual_2 = teams[t2]["actual"]
        pred_1 = teams[t1]["pred"]
        pred_2 = teams[t2]["pred"]
        pred_1_cal = max(a_cal + b_cal * pred_1, 0.0)
        pred_2_cal = max(a_cal + b_cal * pred_2, 0.0)
        base_1 = teams[t1]["baseline"]
        base_2 = teams[t2]["baseline"]

        actual_outcome = outcome_label(actual_1, actual_2)
        pred_outcome = outcome_label(pred_1, pred_2)
        pred_cal_outcome = outcome_label(pred_1_cal, pred_2_cal)
        base_outcome = outcome_label(base_1, base_2)
        outcome_rows.append((actual_outcome, pred_outcome))
        calibrated_outcome_rows.append((actual_outcome, pred_cal_outcome))
        baseline_outcome_rows.append((actual_outcome, base_outcome))

        p_home, p_draw, p_away = poisson_outcome_probs(pred_1, pred_2)
        poisson_outcome = "W" if p_home >= max(p_draw, p_away) else ("D" if p_draw >= p_away else "L")
        poisson_outcome_rows.append((actual_outcome, poisson_outcome))

        match_rows.append(
            (
                fixture_id,
                t1,
                t2,
                actual_1,
                actual_2,
                pred_1,
                pred_2,
                pred_1_cal,
                pred_2_cal,
                base_1,
                base_2,
            )
        )

    print(f"Outcome accuracy (single-stage): {outcome_accuracy(outcome_rows):.3f}")
    print(f"Outcome accuracy (calibrated): {outcome_accuracy(calibrated_outcome_rows):.3f}")
    print(f"Outcome accuracy (baseline): {outcome_accuracy(baseline_outcome_rows):.3f}")
    print(f"Outcome accuracy (poisson): {outcome_accuracy(poisson_outcome_rows):.3f}")
    print("Outcome confusion (single-stage):")
    matrix = outcome_confusion(outcome_rows)
    for actual in ["W", "D", "L"]:
        print(
            f"{actual} "
            f"W:{matrix[actual]['W']:3d} "
            f"D:{matrix[actual]['D']:3d} "
            f"L:{matrix[actual]['L']:3d}"
        )
    print("Outcome confusion (poisson):")
    matrix = outcome_confusion(poisson_outcome_rows)
    for actual in ["W", "D", "L"]:
        print(
            f"{actual} "
            f"W:{matrix[actual]['W']:3d} "
            f"D:{matrix[actual]['D']:3d} "
            f"L:{matrix[actual]['L']:3d}"
        )
    print("Outcome confusion (calibrated):")
    matrix = outcome_confusion(calibrated_outcome_rows)
    for actual in ["W", "D", "L"]:
        print(
            f"{actual} "
            f"W:{matrix[actual]['W']:3d} "
            f"D:{matrix[actual]['D']:3d} "
            f"L:{matrix[actual]['L']:3d}"
        )

    # Confidence filtering on Poisson probabilities
    print("")
    print("Poisson accuracy vs confidence threshold:")
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
    # reuse poisson outcomes by recomputing probabilities
    confidence_rows = []
    for fixture_id, teams in fixtures.items():
        if len(teams) != 2:
            continue
        team_ids = list(teams.keys())
        t1, t2 = team_ids[0], team_ids[1]
        actual_1 = teams[t1]["actual"]
        actual_2 = teams[t2]["actual"]
        pred_1 = teams[t1]["pred"]
        pred_2 = teams[t2]["pred"]
        actual_outcome = outcome_label(actual_1, actual_2)
        p_home, p_draw, p_away = poisson_outcome_probs(pred_1, pred_2)
        probs = {"W": p_home, "D": p_draw, "L": p_away}
        pred_outcome = max(probs, key=probs.get)
        confidence_rows.append((actual_outcome, pred_outcome, probs[pred_outcome]))

    total = len(confidence_rows)
    for thresh in thresholds:
        filtered = [(a, p) for a, p, c in confidence_rows if c >= thresh]
        coverage = len(filtered) / total if total else 0.0
        acc = outcome_accuracy(filtered)
        print(f"threshold {thresh:.2f} | coverage {coverage:.2f} | accuracy {acc:.3f}")

    print("")
    print("Sample match predictions (first 10):")
    for row in match_rows[:10]:
        fixture_id, t1, t2, a1, a2, p1, p2, pc1, pc2, b1, b2 = row
        print(
            f"fixture {fixture_id} teams {t1}-{t2} "
            f"pred={p1:.2f}-{p2:.2f} cal={pc1:.2f}-{pc2:.2f} "
            f"actual={a1:.0f}-{a2:.0f} baseline={b1:.2f}-{b2:.2f}"
        )


if __name__ == "__main__":
    main()
