import csv
import json
import os
import sys
from pathlib import Path

import train_player_goals as tpg


def load_match_actuals(rows):
    team_actual = {}
    for row in rows:
        key = (row.get("fixture_id"), row.get("team_id"))
        team_actual[key] = team_actual.get(key, 0.0) + float(row.get("goals") or 0)
    fixtures = {}
    for (fixture_id, team_id), goals in team_actual.items():
        fixtures.setdefault(fixture_id, {})[team_id] = goals
    return fixtures


def load_fixture_seasons(fixtures_dir):
    mapping = {}
    season_dates = {}
    for path in fixtures_dir.glob("fixture_*.json"):
        fixture_id = path.stem.split("_")[-1]
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        data = payload.get("data") or {}
        season_id = data.get("season_id")
        starting_at = data.get("starting_at")
        mapping[fixture_id] = {"season_id": season_id, "starting_at": starting_at}
        if season_id and starting_at:
            if season_id not in season_dates:
                season_dates[season_id] = starting_at
            else:
                if starting_at < season_dates[season_id]:
                    season_dates[season_id] = starting_at
    season_order = sorted(season_dates.keys(), key=lambda sid: season_dates[sid])
    return mapping, season_order


def pick_best_market(odds_data):
    # best available bookmaker for match winner 3-way
    entries = odds_data.get("data") or []
    candidates = {}
    for entry in entries:
        if (entry.get("market_description") or "").lower() not in {"match winner", "match winner 3-way"}:
            continue
        bookmaker_id = entry.get("bookmaker_id")
        label = (entry.get("label") or "").lower()
        value = entry.get("value")
        if bookmaker_id is None or value is None:
            continue
        outcome = None
        if label in {"home", "1"}:
            outcome = "W"
        elif label in {"draw", "x"}:
            outcome = "D"
        elif label in {"away", "2"}:
            outcome = "L"
        if outcome is None:
            continue
        candidates.setdefault(bookmaker_id, {})[outcome] = float(value)

    best = None
    for bookmaker_id, odds in candidates.items():
        if {"W", "D", "L"}.issubset(odds.keys()):
            best = {"bookmaker_id": bookmaker_id, "odds": odds}
            break
    return best


def coverage_by_bookmaker(odds_dir):
    counts = {}
    total = 0
    for path in odds_dir.glob("fixture_*.json"):
        total += 1
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        entries = payload.get("data") or []
        by_book = {}
        for entry in entries:
            if (entry.get("market_description") or "").lower() not in {"match winner", "match winner 3-way"}:
                continue
            bid = entry.get("bookmaker_id")
            label = (entry.get("label") or "").lower()
            if bid is None:
                continue
            if label in {"home", "1"}:
                outcome = "W"
            elif label in {"draw", "x"}:
                outcome = "D"
            elif label in {"away", "2"}:
                outcome = "L"
            else:
                continue
            by_book.setdefault(bid, set()).add(outcome)
        for bid, outcomes in by_book.items():
            if {"W", "D", "L"}.issubset(outcomes):
                counts[bid] = counts.get(bid, 0) + 1
    return total, counts


def pick_market_for_bookmaker(odds_data, bookmaker_id):
    entries = odds_data.get("data") or []
    odds = {}
    for entry in entries:
        if (entry.get("market_description") or "").lower() not in {"match winner", "match winner 3-way"}:
            continue
        if entry.get("bookmaker_id") != bookmaker_id:
            continue
        label = (entry.get("label") or "").lower()
        value = entry.get("value")
        if value is None:
            continue
        if label in {"home", "1"}:
            odds["W"] = float(value)
        elif label in {"draw", "x"}:
            odds["D"] = float(value)
        elif label in {"away", "2"}:
            odds["L"] = float(value)
    if {"W", "D", "L"}.issubset(odds.keys()):
        return {"bookmaker_id": bookmaker_id, "odds": odds}
    return None


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


def backtest_on_rows(train_rows, test_rows, odds_by_fixture, thresholds, odds_min, odds_max, ev_min):
    feature_cols = build_feature_cols(train_rows)
    X_train, y_train = tpg.build_feature_matrix(train_rows, feature_cols)
    X_test, _ = tpg.build_feature_matrix(test_rows, feature_cols)
    X_train, means, stds = tpg.standardize_impute(X_train)
    X_test, _, _ = tpg.standardize_impute(X_test, means, stds)
    coef = tpg.ridge_fit_log1p(X_train, y_train, alpha=1.0)
    pred = tpg.predict_log1p(X_test, coef)

    team_pred = {}
    team_actual = {}
    for row, pr in zip(test_rows, pred):
        key = (row.get("fixture_id"), row.get("team_id"))
        team_actual[key] = team_actual.get(key, 0.0) + float(row.get("goals") or 0)
        team_pred[key] = team_pred.get(key, 0.0) + pr * (tpg.expected_minutes(row) / 90.0)

    fixtures_actual = load_match_actuals(test_rows)

    results = {"thresholds": [], "ev": None}
    for thresh in thresholds:
        bets = 0
        wins = 0
        profit = 0.0
        for fixture_id, teams in fixtures_actual.items():
            if fixture_id not in odds_by_fixture:
                continue
            team_ids = list(teams.keys())
            if len(team_ids) != 2:
                continue
            t1, t2 = team_ids[0], team_ids[1]
            pred_1 = team_pred.get((fixture_id, t1), 0.0)
            pred_2 = team_pred.get((fixture_id, t2), 0.0)
            p_home, p_draw, p_away = tpg.poisson_outcome_probs(pred_1, pred_2)
            probs = {"W": p_home, "D": p_draw, "L": p_away}
            outcome = max(probs, key=probs.get)
            confidence = probs[outcome]
            if confidence < thresh:
                continue
            odds = odds_by_fixture[fixture_id]["odds"]
            if outcome not in odds:
                continue
            bets += 1
            actual = tpg.outcome_label(teams[t1], teams[t2])
            if actual == outcome:
                wins += 1
                profit += float(odds[outcome]) - 1.0
            else:
                profit -= 1.0
        roi = profit / bets if bets else 0.0
        hit_rate = wins / bets if bets else 0.0
        results["thresholds"].append((thresh, bets, hit_rate, profit, roi))

    bets = 0
    wins = 0
    profit = 0.0
    avg_odds = 0.0
    avg_prob = 0.0
    for fixture_id, teams in fixtures_actual.items():
        if fixture_id not in odds_by_fixture:
            continue
        team_ids = list(teams.keys())
        if len(team_ids) != 2:
            continue
        t1, t2 = team_ids[0], team_ids[1]
        pred_1 = team_pred.get((fixture_id, t1), 0.0)
        pred_2 = team_pred.get((fixture_id, t2), 0.0)
        p_home, p_draw, p_away = tpg.poisson_outcome_probs(pred_1, pred_2)
        probs = {"W": p_home, "D": p_draw, "L": p_away}
        odds = odds_by_fixture[fixture_id]["odds"]
        best_ev = None
        best_outcome = None
        best_odds = None
        best_prob = None
        for outcome, prob in probs.items():
            if outcome not in odds:
                continue
            odd = float(odds[outcome])
            if odd < odds_min or odd > odds_max:
                continue
            ev = prob * odd - 1.0
            if best_ev is None or ev > best_ev:
                best_ev = ev
                best_outcome = outcome
                best_odds = odd
                best_prob = prob
        if best_outcome is None or best_ev is None or best_ev < ev_min:
            continue
        bets += 1
        actual = tpg.outcome_label(teams[t1], teams[t2])
        if actual == best_outcome:
            wins += 1
            profit += best_odds - 1.0
        else:
            profit -= 1.0
        avg_odds += best_odds
        avg_prob += best_prob
    roi = profit / bets if bets else 0.0
    hit_rate = wins / bets if bets else 0.0
    if bets:
        avg_odds /= bets
        avg_prob /= bets
    results["ev"] = (bets, hit_rate, profit, roi, avg_odds, avg_prob)
    return results


def main():
    data_path = Path(__file__).resolve().parent.parent / "data" / "player_match_features.csv"
    if not data_path.exists():
        print(f"Missing dataset: {data_path}")
        sys.exit(1)

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

    odds_dir = Path(__file__).resolve().parent.parent / "data" / "odds"
    if not odds_dir.exists():
        print(f"Missing odds directory: {odds_dir}")
        sys.exit(1)

    bookmaker_filter = os.getenv("SPORTMONKS_BOOKMAKER_ID", "").strip()
    bookmaker_filter = int(bookmaker_filter) if bookmaker_filter.isdigit() else None

    odds_by_fixture = {}
    odds_any = 0
    odds_full = 0
    for path in odds_dir.glob("fixture_*.json"):
        fixture_id = path.stem.split("_")[-1]
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        data_entries = payload.get("data") or []
        if data_entries:
            odds_any += 1
        if bookmaker_filter is not None:
            best = pick_market_for_bookmaker(payload, bookmaker_filter)
        else:
            best = pick_best_market(payload)
        if best:
            odds_by_fixture[fixture_id] = best
            odds_full += 1

    thresholds = [0.40, 0.45, 0.50, 0.55]
    odds_min = float(os.getenv("EV_ODDS_MIN", "1.5"))
    odds_max = float(os.getenv("EV_ODDS_MAX", "5.0"))
    ev_min = float(os.getenv("EV_MIN", "0.05"))

    print("Backtest (flat $1 per bet, match winner 3-way):")
    print(
        f"Fixtures with any odds data: {odds_any} / {len(list(odds_dir.glob('fixture_*.json')))}"
    )
    print(f"Fixtures with full 1X2 odds: {odds_full}")
    if bookmaker_filter is None:
        total, counts = coverage_by_bookmaker(odds_dir)
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        names = {}
        bookmaker_path = Path(__file__).resolve().parent.parent / "data" / "bookmakers.json"
        if bookmaker_path.exists():
            try:
                data = json.loads(bookmaker_path.read_text())
                for item in data:
                    bid = item.get("id")
                    if bid is not None:
                        names[bid] = item.get("name")
            except json.JSONDecodeError:
                pass
        print("Top bookmaker coverage (by fixture count):")
        for bid, count in top:
            label = f"{bid}"
            if bid in names and names[bid]:
                label = f"{bid} ({names[bid]})"
            print(f"  bookmaker_id {label}: {count}/{total}")
    else:
        print(f"Bookmaker filter: {bookmaker_filter}")

    split = int(len(rows) * 0.8)
    train_rows = rows[:split]
    test_rows = rows[split:]
    results = backtest_on_rows(train_rows, test_rows, odds_by_fixture, thresholds, odds_min, odds_max, ev_min)
    for thresh, bets, hit_rate, profit, roi in results["thresholds"]:
        print(
            f"threshold {thresh:.2f} | bets {bets:3d} | hit {hit_rate:.3f} | "
            f"profit ${profit:.2f} | roi {roi:.3f}"
        )
    bets, hit_rate, profit, roi, avg_odds, avg_prob = results["ev"]
    print("")
    print("EV-based backtest (bet when expected value > 0):")
    print(
        f"bets {bets:3d} | hit {hit_rate:.3f} | profit ${profit:.2f} | roi {roi:.3f} "
        f"| avg_odds {avg_odds:.2f} | avg_prob {avg_prob:.3f}"
    )

    # Walk-forward by season
    fixtures_dir = Path(__file__).resolve().parent.parent / "data" / "fixtures"
    mapping, season_order = load_fixture_seasons(fixtures_dir)
    if season_order:
        print("")
        print("Walk-forward backtest by season:")
        for idx in range(1, len(season_order)):
            test_season = season_order[idx]
            train_seasons = set(season_order[:idx])
            train_rows = [r for r in rows if mapping.get(r.get("fixture_id"), {}).get("season_id") in train_seasons]
            test_rows = [r for r in rows if mapping.get(r.get("fixture_id"), {}).get("season_id") == test_season]
            if not train_rows or not test_rows:
                continue
            results = backtest_on_rows(train_rows, test_rows, odds_by_fixture, thresholds, odds_min, odds_max, ev_min)
            print(f"Season {test_season}: rows {len(test_rows)}")
            for thresh, bets, hit_rate, profit, roi in results["thresholds"]:
                print(
                    f"  threshold {thresh:.2f} | bets {bets:3d} | hit {hit_rate:.3f} | "
                    f"profit ${profit:.2f} | roi {roi:.3f}"
                )
            bets, hit_rate, profit, roi, avg_odds, avg_prob = results["ev"]
            print(
                f"  EV | bets {bets:3d} | hit {hit_rate:.3f} | profit ${profit:.2f} | "
                f"roi {roi:.3f} | avg_odds {avg_odds:.2f} | avg_prob {avg_prob:.3f}"
            )


if __name__ == "__main__":
    main()
