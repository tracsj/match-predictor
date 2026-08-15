import csv
import datetime as dt
import math
import os
import sys

import numpy as np


def parse_date(value):
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        fieldnames = reader.fieldnames or []
        for row in reader:
            row["fixture_id"] = int(row["fixture_id"])
            row["team_id"] = int(row["team_id"])
            row["opponent_id"] = int(row["opponent_id"])
            row["is_home"] = int(row["is_home"])
            row["starting_at_dt"] = parse_date(row.get("starting_at"))
            row["goals_for"] = int(row["goals_for"]) if row["goals_for"] not in (None, "", "None") else None
            row["goals_against"] = int(row["goals_against"]) if row["goals_against"] not in (None, "", "None") else None
            rows.append(row)
    return rows, fieldnames


def drop_sparse_columns(rows, fieldnames, threshold=0.9):
    if not rows:
        return fieldnames, []
    drop = []
    keep = []
    for name in fieldnames:
        missing = 0
        for row in rows:
            value = row.get(name)
            if value in (None, "", "None"):
                missing += 1
        if missing / len(rows) > threshold:
            drop.append(name)
        else:
            keep.append(name)
    return keep, drop


def build_feature_matrix(rows, feature_names):
    matrix = []
    for row in rows:
        values = []
        for name in feature_names:
            raw = row.get(name)
            if raw in (None, "", "None"):
                values.append(float("nan"))
            else:
                try:
                    values.append(float(raw))
                except ValueError:
                    values.append(float("nan"))
        matrix.append(values)
    return np.array(matrix, dtype=float)


def filter_rows_with_required_features(rows, required_features):
    filtered = []
    for row in rows:
        missing = False
        for name in required_features:
            value = row.get(name)
            if value in (None, "", "None"):
                missing = True
                break
        if not missing:
            filtered.append(row)
    return filtered


def infer_numeric_features(rows, fieldnames):
    excluded = {
        "fixture_id",
        "team_id",
        "opponent_id",
        "starting_at",
        "result",
        "goals_for",
        "goals_against",
    }
    candidates = [name for name in fieldnames if name not in excluded]
    numeric = []
    for name in candidates:
        any_value = False
        all_missing = True
        for row in rows:
            raw = row.get(name)
            if raw in (None, "", "None"):
                continue
            all_missing = False
            try:
                float(raw)
                any_value = True
            except ValueError:
                any_value = False
                break
        if any_value and not all_missing:
            numeric.append(name)
    return numeric


FEATURE_SETS = {
    "base": {
        "is_home",
        "team_position",
        "opponent_position",
        "avg_player_quality_for",
        "avg_player_quality_against",
        "lineup_chemistry_for_avg",
        "lineup_chemistry_against_avg",
        "lineup_chemistry_pair_count",
    },
    "extended": {
        "is_home",
        "team_position",
        "opponent_position",
        "avg_player_quality_for",
        "avg_player_quality_against",
        "lineup_chemistry_for_avg",
        "lineup_chemistry_against_avg",
        "lineup_chemistry_pair_count",
        "opp_avg_scored",
        "opp_avg_conceded",
        "adj_player_quality_for",
        "adj_player_quality_against",
        "team_form_points",
        "team_form_goals_for",
        "team_form_goals_against",
        "team_form_goal_diff",
    },
}


def apply_feature_whitelist(features, feature_set):
    allowlist = FEATURE_SETS[feature_set]
    return [name for name in features if name in allowlist]


def impute_mean(train_X, test_X):
    means = np.nanmean(train_X, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    train_X = np.where(np.isnan(train_X), means, train_X)
    test_X = np.where(np.isnan(test_X), means, test_X)
    return train_X, test_X, means


def standardize(train_X, test_X):
    mean = np.mean(train_X, axis=0)
    std = np.std(train_X, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (train_X - mean) / std, (test_X - mean) / std, mean, std


def fit_linear_regression(X, y):
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    return coef


def fit_ridge_regression(X, y, alpha=1.0):
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    n_features = X_aug.shape[1]
    identity = np.eye(n_features)
    identity[0, 0] = 0  # don't regularize intercept
    # Solve augmented system to avoid explicit inverse
    A = np.vstack([X_aug, math.sqrt(alpha) * identity])
    b = np.concatenate([y, np.zeros(n_features)])
    coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return coef


def fit_log_linear_ridge(X, y, alpha=1.0, clip_coef=5.0):
    y = np.clip(y, 0.0, None)
    y_log = np.log1p(y)
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    n_features = X_aug.shape[1]
    identity = np.eye(n_features)
    identity[0, 0] = 0
    A = np.vstack([X_aug, math.sqrt(alpha) * identity])
    b = np.concatenate([y_log, np.zeros(n_features)])
    coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    coef = np.clip(coef, -clip_coef, clip_coef)
    return coef


def predict_linear(X, coef):
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_clean = np.clip(X_clean, -10.0, 10.0)
    X_aug = np.hstack([np.ones((X_clean.shape[0], 1)), X_clean])
    coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
    return X_aug @ coef


def split_rows(rows, test_ratio=0.2):
    rows_sorted = sorted(
        rows,
        key=lambda r: (r["starting_at_dt"] is None, r["starting_at_dt"], r["fixture_id"]),
    )
    split_index = int(len(rows_sorted) * (1 - test_ratio))
    return rows_sorted[:split_index], rows_sorted[split_index:]


def train_baseline(train_rows):
    home_goals = []
    away_goals = []
    for row in train_rows:
        if row["goals_for"] is None or row["goals_against"] is None:
            continue
        if row["is_home"] == 1:
            home_goals.append(row["goals_for"])
        else:
            away_goals.append(row["goals_for"])

    avg_home = sum(home_goals) / len(home_goals) if home_goals else 1.0
    avg_away = sum(away_goals) / len(away_goals) if away_goals else 1.0
    return avg_home, avg_away


def predict_goals(row, avg_home, avg_away):
    if row["is_home"] == 1:
        return avg_home, avg_away
    return avg_away, avg_home


def outcome_label(goals_for, goals_against):
    if goals_for > goals_against:
        return "W"
    if goals_for < goals_against:
        return "L"
    return "D"


def outcome_label_pred(goals_for, goals_against, draw_threshold=0.3):
    if abs(goals_for - goals_against) <= draw_threshold:
        return "D"
    return outcome_label(goals_for, goals_against)


def evaluate(rows, avg_home, avg_away, draw_threshold=0.3):
    total = 0
    correct = 0
    mae_for = 0.0
    mae_against = 0.0
    conf = {"W": {"W": 0, "D": 0, "L": 0}, "D": {"W": 0, "D": 0, "L": 0}, "L": {"W": 0, "D": 0, "L": 0}}

    for row in rows:
        if row["goals_for"] is None or row["goals_against"] is None:
            continue
        pred_for, pred_against = predict_goals(row, avg_home, avg_away)
        pred_label = outcome_label_pred(pred_for, pred_against, draw_threshold=draw_threshold)
        true_label = outcome_label(row["goals_for"], row["goals_against"])
        conf[true_label][pred_label] += 1

        mae_for += abs(row["goals_for"] - pred_for)
        mae_against += abs(row["goals_against"] - pred_against)
        total += 1
        if pred_label == true_label:
            correct += 1

    if total == 0:
        return None
    return {
        "accuracy": correct / total,
        "mae_for": mae_for / total,
        "mae_against": mae_against / total,
        "confusion": conf,
        "count": total,
    }


def print_confusion(conf):
    print("Confusion matrix (rows=true, cols=pred):")
    header = "      W     D     L"
    print(header)
    for label in ("W", "D", "L"):
        row = conf[label]
        print(f"{label}  {row['W']:5d} {row['D']:5d} {row['L']:5d}")

def main():
    features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "features.csv"))
    if not os.path.exists(features_path):
        print(f"Missing features file: {features_path}")
        sys.exit(1)

    rows, fieldnames = load_rows(features_path)
    keep, drop = drop_sparse_columns(rows, fieldnames)
    if drop:
        print("Dropping sparse columns (>90% missing):")
        for name in drop:
            print(f"- {name}")

    train_rows, test_rows = split_rows(rows)
    avg_home, avg_away = train_baseline(train_rows)
    numeric_features_all = infer_numeric_features(train_rows, keep)

    print(f"Train size: {len(train_rows)}")
    print(f"Test size: {len(test_rows)}")
    print(f"Avg home goals: {avg_home:.3f}")
    print(f"Avg away goals: {avg_away:.3f}")

    train_eval = evaluate(train_rows, avg_home, avg_away, draw_threshold=0.0)
    test_eval = evaluate(test_rows, avg_home, avg_away, draw_threshold=0.0)

    if train_eval:
        print(f"Train accuracy: {train_eval['accuracy']:.3f}")
        print(f"Train MAE goals_for: {train_eval['mae_for']:.3f}")
        print(f"Train MAE goals_against: {train_eval['mae_against']:.3f}")
    if test_eval:
        print(f"Test accuracy: {test_eval['accuracy']:.3f}")
        print(f"Test MAE goals_for: {test_eval['mae_for']:.3f}")
        print(f"Test MAE goals_against: {test_eval['mae_against']:.3f}")
        print_confusion(test_eval["confusion"])

    def run_feature_set(feature_set_name, required_features):
        numeric_features = apply_feature_whitelist(numeric_features_all, feature_set_name)
        if not numeric_features:
            print(f"[{feature_set_name}] No usable numeric features.")
            return

        train_rows_filtered = filter_rows_with_required_features(train_rows, required_features)
        test_rows_filtered = filter_rows_with_required_features(test_rows, required_features)

        if len(train_rows_filtered) != len(train_rows):
            print(f"[{feature_set_name}] Filtered train rows: {len(train_rows_filtered)}")
        if len(test_rows_filtered) != len(test_rows):
            print(f"[{feature_set_name}] Filtered test rows: {len(test_rows_filtered)}")

        train_X = build_feature_matrix(train_rows_filtered, numeric_features)
        test_X = build_feature_matrix(test_rows_filtered, numeric_features)
        train_X, test_X, _ = impute_mean(train_X, test_X)
        train_X, test_X, _, _ = standardize(train_X, test_X)
        train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
        test_X = np.nan_to_num(test_X, nan=0.0, posinf=0.0, neginf=0.0)
        train_X = np.clip(train_X, -10.0, 10.0)
        test_X = np.clip(test_X, -10.0, 10.0)

        def filter_finite(X, rows_set):
            mask = np.all(np.isfinite(X), axis=1)
            return X[mask], [row for row, keep in zip(rows_set, mask) if keep]

        mask = np.array(
            [row["goals_for"] is not None and row["goals_against"] is not None for row in train_rows_filtered],
            dtype=bool,
        )
        train_rows_labeled = [
            row for row in train_rows_filtered if row["goals_for"] is not None and row["goals_against"] is not None
        ]
        train_X_filtered = train_X[mask]
        train_X_filtered, train_rows_labeled = filter_finite(train_X_filtered, train_rows_labeled)
        train_y_for = np.array([row["goals_for"] for row in train_rows_labeled], dtype=float)
        train_y_against = np.array([row["goals_against"] for row in train_rows_labeled], dtype=float)

        coef_for = None
        coef_against = None
        if train_X_filtered.size and train_y_for.size and train_y_against.size:
            coef_for = fit_log_linear_ridge(train_X_filtered, train_y_for, alpha=10.0, clip_coef=3.0)
            coef_against = fit_log_linear_ridge(train_X_filtered, train_y_against, alpha=10.0, clip_coef=3.0)
            coef_for = np.nan_to_num(coef_for, nan=0.0, posinf=0.0, neginf=0.0)
            coef_against = np.nan_to_num(coef_against, nan=0.0, posinf=0.0, neginf=0.0)

        if coef_for is None or coef_against is None:
            print(f"[{feature_set_name}] Skipping model (insufficient data).")
            return

        print(f"\n[{feature_set_name}] Feature-based model")
        print(f"Using {len(numeric_features)} numeric features.")
        print("Features:")
        for name in numeric_features:
            print(f"- {name}")

        def predict_log_linear(X_set, coef):
            X_clean = np.nan_to_num(X_set, nan=0.0, posinf=0.0, neginf=0.0)
            X_clean = np.clip(X_clean, -10.0, 10.0)
            X_aug = np.hstack([np.ones((X_clean.shape[0], 1)), X_clean])
            coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
            coef = np.clip(coef, -3.0, 3.0)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                eta = X_aug @ coef
            eta = np.nan_to_num(eta, nan=0.0, posinf=6.0, neginf=-6.0)
            eta = np.clip(eta, -6.0, 6.0)
            return np.expm1(eta)

        def eval_feature_model(rows_set, X_set):
            total = 0
            correct = 0
            mae_for = 0.0
            mae_against = 0.0
            conf = {"W": {"W": 0, "D": 0, "L": 0}, "D": {"W": 0, "D": 0, "L": 0}, "L": {"W": 0, "D": 0, "L": 0}}
            pred_for = np.clip(predict_log_linear(X_set, coef_for), 0.0, None)
            pred_against = np.clip(predict_log_linear(X_set, coef_against), 0.0, None)
            for i, row in enumerate(rows_set):
                if row["goals_for"] is None or row["goals_against"] is None:
                    continue
                pf = pred_for[i]
                pa = pred_against[i]
                pred_label = outcome_label_pred(pf, pa)
                true_label = outcome_label(row["goals_for"], row["goals_against"])
                conf[true_label][pred_label] += 1
                mae_for += abs(row["goals_for"] - pf)
                mae_against += abs(row["goals_against"] - pa)
                total += 1
                if pred_label == true_label:
                    correct += 1
            if total == 0:
                return None
            return {
                "accuracy": correct / total,
                "mae_for": mae_for / total,
                "mae_against": mae_against / total,
                "confusion": conf,
                "count": total,
            }

        train_feat = eval_feature_model(train_rows_filtered, train_X)
        test_feat = eval_feature_model(test_rows_filtered, test_X)
        if train_feat:
            print(f"Train accuracy: {train_feat['accuracy']:.3f}")
            print(f"Train MAE goals_for: {train_feat['mae_for']:.3f}")
            print(f"Train MAE goals_against: {train_feat['mae_against']:.3f}")
        if test_feat:
            print(f"Test accuracy: {test_feat['accuracy']:.3f}")
            print(f"Test MAE goals_for: {test_feat['mae_for']:.3f}")
            print(f"Test MAE goals_against: {test_feat['mae_against']:.3f}")
            print_confusion(test_feat["confusion"])

        if test_rows_filtered:
            sample_count = min(5, len(test_rows_filtered))
            print("\nSample predictions (first 5 test rows):")
            pred_for = np.clip(predict_log_linear(test_X, coef_for), 0.0, None)
            pred_against = np.clip(predict_log_linear(test_X, coef_against), 0.0, None)
            for i in range(sample_count):
                row = test_rows_filtered[i]
                print(
                    f"fixture {row['fixture_id']} team {row['team_id']} "
                    f"pred_for={pred_for[i]:.2f} pred_against={pred_against[i]:.2f} "
                    f"actual={row['goals_for']}-{row['goals_against']}"
                )

        def train_softmax(X, y_labels, alpha=1.0, lr=0.02, epochs=600, clip_weights=3.0):
            classes = ["W", "D", "L"]
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y = np.array([class_to_idx[label] for label in y_labels], dtype=int)
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
            weights = np.zeros((X_aug.shape[1], len(classes)))
            for _ in range(epochs):
                weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                weights = np.clip(weights, -clip_weights, clip_weights)
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    logits = X_aug @ weights
                logits = logits - np.max(logits, axis=1, keepdims=True)
                exp_scores = np.exp(logits)
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                y_onehot = np.zeros_like(probs)
                y_onehot[np.arange(len(y)), y] = 1.0
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    grad = X_aug.T @ (probs - y_onehot) / len(y)
                grad += alpha * np.vstack([np.zeros((1, len(classes))), weights[1:]])
                grad = np.clip(grad, -1.0, 1.0)
                weights -= lr * grad
            return weights, classes

        def predict_softmax(X, weights, classes):
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            weights = np.clip(weights, -3.0, 3.0)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                logits = X_aug @ weights
            preds = np.argmax(logits, axis=1)
            return [classes[i] for i in preds]

        # Direct W/D/L classifier
        train_labels = []
        test_labels = []
        for row in train_rows_filtered:
            if row["goals_for"] is None or row["goals_against"] is None:
                continue
            train_labels.append(outcome_label(row["goals_for"], row["goals_against"]))
        for row in test_rows_filtered:
            if row["goals_for"] is None or row["goals_against"] is None:
                continue
            test_labels.append(outcome_label(row["goals_for"], row["goals_against"]))

        if train_labels and test_labels:
            weights, classes = train_softmax(train_X, train_labels, alpha=0.2, lr=0.02, epochs=600, clip_weights=3.0)
            train_pred = predict_softmax(train_X, weights, classes)
            test_pred = predict_softmax(test_X, weights, classes)

            def eval_classifier(true_labels, pred_labels):
                total = len(true_labels)
                correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
                conf = {"W": {"W": 0, "D": 0, "L": 0}, "D": {"W": 0, "D": 0, "L": 0}, "L": {"W": 0, "D": 0, "L": 0}}
                for t, p in zip(true_labels, pred_labels):
                    conf[t][p] += 1
                return correct / total if total else None, conf

            train_acc, train_conf = eval_classifier(train_labels, train_pred)
            test_acc, test_conf = eval_classifier(test_labels, test_pred)
            print("\nDirect W/D/L classifier")
            if train_acc is not None:
                print(f"Train accuracy: {train_acc:.3f}")
                print_confusion(train_conf)
            if test_acc is not None:
                print(f"Test accuracy: {test_acc:.3f}")
                print_confusion(test_conf)

        def top_coeffs(coef, names, top_n=10):
            pairs = list(zip(names, coef[1:]))
            pairs.sort(key=lambda item: abs(item[1]), reverse=True)
            return pairs[:top_n]

        print("\nTop coefficients (goals_for):")
        for name, value in top_coeffs(coef_for, numeric_features):
            print(f"- {name}: {value:.4f}")
        print("\nTop coefficients (goals_against):")
        for name, value in top_coeffs(coef_against, numeric_features):
            print(f"- {name}: {value:.4f}")

    run_feature_set(
        "base",
        [
            "avg_player_quality_for",
            "avg_player_quality_against",
            "lineup_chemistry_for_avg",
            "lineup_chemistry_against_avg",
            "lineup_chemistry_pair_count",
        ],
    )
    run_feature_set(
        "extended",
        [
            "avg_player_quality_for",
            "avg_player_quality_against",
            "lineup_chemistry_for_avg",
            "lineup_chemistry_against_avg",
            "lineup_chemistry_pair_count",
            "adj_player_quality_for",
            "adj_player_quality_against",
            "team_form_points",
            "team_form_goals_for",
            "team_form_goals_against",
        ],
    )


if __name__ == "__main__":
    main()
