import argparse
import json
import os
from pathlib import Path

import train_player_goals as tpg
from backtest_betting import pick_best_market, pick_market_for_bookmaker
from roi_summary import build_feature_cols
from roi_summary import load_fixture_meta as load_roi_fixture_meta


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
            "name": data.get("name"),
        }
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-id", required=True)
    parser.add_argument("--season-id", required=False)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--upcoming-only", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--use-last-lineup",
        action="store_true",
        help="Use latest available lineup fixture as a proxy for upcoming fixtures.",
    )
    args = parser.parse_args()

    bookmaker_id = os.getenv("SPORTMONKS_BOOKMAKER_ID", "").strip()
    bookmaker_id = int(bookmaker_id) if bookmaker_id.isdigit() else None

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

    fixture_meta = load_fixture_meta(Path(__file__).resolve().parent.parent / "data" / "fixtures")
    league_id = int(args.league_id)
    season_id = int(args.season_id) if args.season_id else None
    now = tpg.datetime.utcnow()

    if season_id is None:
        upcoming_counts = {}
        latest_by_season = {}
        for meta in fixture_meta.values():
            if meta.get("league_id") != league_id:
                continue
            start = tpg.parse_dt(meta.get("starting_at")) if meta.get("starting_at") else None
            if start:
                if start >= now:
                    sid = meta.get("season_id")
                    if sid is not None:
                        upcoming_counts[sid] = upcoming_counts.get(sid, 0) + 1
                sid = meta.get("season_id")
                if sid is not None:
                    prev = latest_by_season.get(sid)
                    if prev is None or start > prev:
                        latest_by_season[sid] = start
        if upcoming_counts:
            season_id = max(
                upcoming_counts.items(),
                key=lambda item: (item[1], latest_by_season.get(item[0]) or tpg.datetime.min),
            )[0]
            print(f"Auto-selected season with upcoming fixtures: {season_id}")
        elif latest_by_season:
            season_id = max(latest_by_season.items(), key=lambda item: item[1])[0]
            print(f"Auto-selected latest season: {season_id}")

    feature_cols = build_feature_cols(rows)
    X, y = tpg.build_feature_matrix(rows, feature_cols)
    X, means, stds = tpg.standardize_impute(X)
    coef = tpg.ridge_fit_log1p(X, y, alpha=1.0)

    odds_dir = Path(__file__).resolve().parent.parent / "data" / "odds"
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

    upcoming = {}
    for row in rows:
        fid = str(row.get("fixture_id"))
        meta = fixture_meta.get(fid)
        if not meta or meta.get("league_id") != league_id:
            continue
        if season_id and meta.get("season_id") != season_id:
            continue
        upcoming.setdefault(fid, []).append(row)

    if not upcoming:
        print("No fixtures found for that league/season in dataset.")
        return

    print(f"League {league_id} | Season {season_id or 'all'}")
    print(f"Min confidence: {args.min_confidence}")
    if bookmaker_id:
        print(f"Bookmaker: {bookmaker_id}")
    print("")
    print("ROI summary (test split):")
    os.environ["MIN_CONFIDENCE"] = str(args.min_confidence)
    if bookmaker_id:
        os.environ["SPORTMONKS_BOOKMAKER_ID"] = str(bookmaker_id)
    from roi_summary import main as roi_main
    roi_main()
    print("")

    printed = 0
    upcoming_fixtures = []
    if args.upcoming_only:
        for fid, meta in fixture_meta.items():
            if meta.get("league_id") != league_id:
                continue
            if season_id and meta.get("season_id") != season_id:
                continue
            start = tpg.parse_dt(meta.get("starting_at")) if meta.get("starting_at") else None
            if start and start >= now:
                upcoming_fixtures.append(fid)

    for fixture_id, fixture_rows in upcoming.items():
        meta = fixture_meta.get(str(fixture_id), {})
        if season_id and meta.get("season_id") != season_id:
            continue
        if args.upcoming_only and meta.get("starting_at"):
            start = tpg.parse_dt(meta.get("starting_at"))
            if start and start < now:
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
        odds = odds_by_fixture.get(str(fixture_id), {}).get("odds")

        if confidence < args.min_confidence:
            continue

        name = fixture_meta.get(str(fixture_id), {}).get("name", "Fixture")
        print(f"{fixture_id} | {name}")
        print(f"  pred goals: {pred_1:.2f} - {pred_2:.2f}")
        print(f"  probs: W={p_home:.3f} D={p_draw:.3f} L={p_away:.3f}")
        if odds and outcome in odds:
            print(f"  bet: {outcome} @ {odds[outcome]}")
        else:
            print(f"  bet: {outcome} (odds unavailable)")
        print("")
        printed += 1
        if printed >= args.limit:
            break

    if printed == 0 and args.upcoming_only:
        if not upcoming_fixtures:
            print("No upcoming fixtures found for this league/season.")
        else:
            print(
                "Upcoming fixtures exist, but no lineup feature rows are available. "
                "Pull latest fixtures/lineups and rebuild features."
            )
            if args.use_last_lineup:
                latest_fixture_id = None
                latest_start = None
                for fixture_id, fixture_rows in upcoming.items():
                    meta = fixture_meta.get(str(fixture_id), {})
                    if season_id and meta.get("season_id") != season_id:
                        continue
                    start = tpg.parse_dt(meta.get("starting_at")) if meta.get("starting_at") else None
                    if start and (latest_start is None or start > latest_start):
                        latest_start = start
                        latest_fixture_id = fixture_id
                if latest_fixture_id:
                    fixture_rows = upcoming.get(latest_fixture_id, [])
                    X_fixture, _ = tpg.build_feature_matrix(fixture_rows, feature_cols)
                    X_fixture, _, _ = tpg.standardize_impute(X_fixture, means, stds)
                    preds = tpg.predict_log1p(X_fixture, coef)
                    team_pred = {}
                    for row, pred in zip(fixture_rows, preds):
                        team_id = row.get("team_id")
                        team_pred[team_id] = team_pred.get(team_id, 0.0) + pred * (
                            tpg.expected_minutes(row) / 90.0
                        )
                    if len(team_pred) == 2:
                        team_ids = list(team_pred.keys())
                        t1, t2 = team_ids[0], team_ids[1]
                        pred_1 = team_pred[t1]
                        pred_2 = team_pred[t2]
                        p_home, p_draw, p_away = tpg.poisson_outcome_probs(pred_1, pred_2)
                        probs = {"W": p_home, "D": p_draw, "L": p_away}
                        outcome = max(probs, key=probs.get)
                        odds = odds_by_fixture.get(str(latest_fixture_id), {}).get("odds")
                        name = fixture_meta.get(str(latest_fixture_id), {}).get("name", "Fixture")
                        print("")
                        print("Proxy using latest available lineup:")
                        print(f"{latest_fixture_id} | {name}")
                        print(f"  pred goals: {pred_1:.2f} - {pred_2:.2f}")
                        print(f"  probs: W={p_home:.3f} D={p_draw:.3f} L={p_away:.3f}")
                        if odds and outcome in odds:
                            print(f"  bet: {outcome} @ {odds[outcome]}")
                        else:
                            print(f"  bet: {outcome} (odds unavailable)")


if __name__ == "__main__":
    main()
