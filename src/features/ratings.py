"""Team strength ratings: Elo and pi-ratings.

These are the highest-value inputs on this task. Yeung et al. (2024) found
CatBoost on pi-ratings the best model on goals-only football data, beating
every neural architecture they tried -- so a rating is not a baseline feature,
it is the feature the network has to justify itself against.

**Why ratings rather than a fitted model for the baseline.** Both are computed
online: each match is scored using the ratings as they stood BEFORE kickoff,
and only then do the ratings absorb the result. That makes them leak-free by
construction rather than by discipline, and it means a newly promoted team
simply arrives at the default rating instead of raising `ValueError: Both
teams must have been in the training data`, which is what penaltyblog's
Dixon-Coles does (verified 2026-08-15).

**On same-day ordering.** football-data has no kickoff time before 2019/20, so
same-day fixtures cannot be ordered. That does not affect ratings: a team's
rating updates only from its own matches, and no team plays twice in a day, so
any within-day permutation produces identical pre-match ratings. It *does*
affect opponent-adjusted rolling features, which is a separate problem handled
in the feature builder.

**Rating pool scope.** One pool per country, spanning every division. A team
promoted from the Championship carries its rating up with it, which is both
true and the only way the walk-forward loop can price a promoted side at all.
Teams do not cross countries in this dataset.

pi-ratings follow Constantinou & Fenton (2013), "Determining the level of
ability of football teams by dynamic ratings based on the relative
discrepancies in scores between adversaries", JQAS 9(1):37-50.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.horizon import unplayed_flags

__all__ = ["elo_features", "pi_rating_features", "add_ratings", "PiParams", "EloParams"]


# Division tiers within a country, so a promotion can be recognised as one.
# Extra-country files are single-division and simply have no tier.
TIER: dict[str, int] = {
    "E0": 1, "E1": 2, "E2": 3, "E3": 4, "EC": 5,
    "SC0": 1, "SC1": 2, "SC2": 3, "SC3": 4,
    "D1": 1, "D2": 2, "I1": 1, "I2": 2, "SP1": 1, "SP2": 2,
    "F1": 1, "F2": 2,
}


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home_advantage: float = 65.0     # in Elo points, ~0.25 goals
    start: float = 1500.0
    # Ratings regress toward the pool mean between seasons, because squads turn
    # over. 0.0 keeps everything, 1.0 resets. Club Elo uses a partial carry.
    season_regression: float = 0.15
    # Elo points to deduct per tier climbed (add per tier dropped) for a team
    # in the season after it changes division. DEFAULT ZERO -- deliberately,
    # after measuring it three ways. Kept as a parameter so the finding is
    # reproducible rather than folklore.
    #
    # THE REAL PROBLEM IS REAL. Divisions connect only through promotion and
    # relegation (there are no cup fixtures in this data), so rating
    # information crosses between them slowly and the tiers come out
    # compressed: mean Elo runs E0 1642 > E1 1524 > E2 1507 > E3 1462 > EC
    # 1454. A promoted team carries a rating earned against weaker opposition
    # into a stronger division. Uncorrected, Elo expected promoted teams to
    # take 0.5301 points in their first match up and they took 0.4277 (bias
    # -10.2pp, n=588); relegated teams were underrated by +11.1pp (n=597).
    # Teams that stayed put were unbiased to four decimals over 323,189
    # team-matches, so the pool is sound and only movers are mispriced.
    #
    # ATTEMPT 1, MUTATING THE STORED RATING at the season boundary: zeroed the
    # bias at 130 points and improved out-of-sample RPS by 0.00046 (t = +7.0).
    # It also wrecked the pool. Because the correction travels with the team,
    # every division mean drifts inward -- E0 fell 1642 -> 1568 and EC ROSE
    # 1454 -> 1502, putting the National League above the Championship. The
    # damage reached settled teams, not just movers, so it was structural
    # rather than book-keeping. Two metrics agreed with the change and a third
    # exposed the mechanism as wrong.
    #
    # ATTEMPT 2, ADJUSTING ONLY THE OUTPUT COLUMNS and never writing back:
    # keeps every division ordering correct. 76 points zeroes the bias
    # (promoted +0.0003, relegated -0.0036). But out-of-sample RPS gets
    # monotonically WORSE as the shift grows -- 0.20901 at zero, 0.20910 at 76,
    # 0.20913 at 90. Bias correction and predictive accuracy point in opposite
    # directions.
    #
    # WHY: the model already receives `elo_home_moved` / `elo_away_moved` and
    # learns the correction itself, conditioned on everything else, which a
    # single hand-set constant cannot do. Dropping those two flags costs
    # 0.00038 RPS (t = +5.53, n = 45,629); adding a shift on top of them buys
    # nothing. So the flags stay and the shift stays at zero.
    #
    # Note that attempt 1's apparent +7.0 t-statistic was a comparison between
    # two flawed variants, not evidence the correction helped.
    tier_shift: float = 0.0


@dataclass(frozen=True)
class PiParams:
    # Learning rate for the team's own venue rating, and the rate at which the
    # other venue rating follows it. The paper's tuned values.
    lam: float = 0.035
    gamma: float = 0.7
    b: float = 10.0
    c: float = 3.0


def _expected_gd(rating: float, p: PiParams) -> float:
    """Map a pi-rating to an expected goal difference.

    ghat = (b^(|R|/c) - 1) * sign(R). The exponential shape is the paper's:
    the same rating gap means more goals between strong sides than weak ones.
    """
    if rating == 0.0:
        return 0.0
    return (p.b ** (abs(rating) / p.c) - 1.0) * np.sign(rating)


def pi_rating_features(df: pd.DataFrame, params: PiParams = PiParams()) -> pd.DataFrame:
    """Pre-match pi-ratings and the expected goal difference they imply.

    Returns one row per match, aligned to `df`, carrying the ratings as they
    stood BEFORE that match.
    """
    p = params
    home_r: dict[tuple, float] = {}   # rating when playing at home
    away_r: dict[tuple, float] = {}   # rating when playing away

    n = len(df)
    out = {k: np.empty(n) for k in
           ("pi_home_h", "pi_home_a", "pi_away_h", "pi_away_a", "pi_exp_gd")}

    country = df["country"].to_numpy()
    hk, ak = df["home_key"].to_numpy(), df["away_key"].to_numpy()
    gh, ga = df["fthg"].to_numpy(), df["ftag"].to_numpy()
    unplayed = unplayed_flags(df)

    for i in range(n):
        H, A = (country[i], hk[i]), (country[i], ak[i])
        rHh, rHa = home_r.get(H, 0.0), away_r.get(H, 0.0)
        rAh, rAa = home_r.get(A, 0.0), away_r.get(A, 0.0)

        # Expected goal difference: home team's HOME rating against the away
        # team's AWAY rating. Using the wrong pair here is the whole point of
        # keeping two ratings per team, so it is spelled out.
        ghat_h = _expected_gd(rHh, p)
        ghat_a = _expected_gd(rAa, p)
        exp_gd = ghat_h - ghat_a

        out["pi_home_h"][i], out["pi_home_a"][i] = rHh, rHa
        out["pi_away_h"][i], out["pi_away_a"][i] = rAh, rAa
        out["pi_exp_gd"][i] = exp_gd

        # --- update, using the result we just scored against ---
        # An unplayed fixture has no result to absorb. Writing one here would
        # put NaN into the stored rating and poison every later row for both
        # teams -- see src/features/horizon.py.
        if unplayed[i]:
            continue
        obs_gd = float(gh[i] - ga[i])
        err = abs(obs_gd - exp_gd)
        psi = p.c * np.log10(1.0 + err)
        direction = np.sign(obs_gd - exp_gd)

        # Home team: its home rating moves; its away rating follows partly.
        new_rHh = rHh + psi * p.lam * direction
        home_r[H] = new_rHh
        away_r[H] = rHa + (new_rHh - rHh) * p.gamma

        # Away team: mirrored, so an over-performing away side gains.
        new_rAa = rAa - psi * p.lam * direction
        away_r[A] = new_rAa
        home_r[A] = rAh + (new_rAa - rAa) * p.gamma

    return pd.DataFrame(out, index=df.index)


def elo_features(df: pd.DataFrame, params: EloParams = EloParams()) -> pd.DataFrame:
    """Pre-match Elo ratings, the difference, and the implied home win expectancy.

    Standard Elo with a home-advantage offset and a goal-difference multiplier
    on K, so a 4-0 moves the rating more than a 1-0. Draws are handled by
    scoring 0.5, which is Elo's usual fudge for a three-outcome sport -- it is
    why Elo alone underrates draws, and why the mapping to 1X2 probabilities is
    fitted separately rather than read off the expectancy.
    """
    p = params
    rating: dict[tuple, float] = {}
    last_season: dict[tuple, str] = {}
    last_tier: dict[tuple, int] = {}
    move_state: dict[tuple, int] = {}   # +1 promoted / -1 relegated, this season

    n = len(df)
    out = {k: np.empty(n) for k in
           ("elo_home", "elo_away", "elo_diff", "elo_exp_home",
            "elo_home_moved", "elo_away_moved")}

    country = df["country"].to_numpy()
    season = df["season"].to_numpy()
    div = df["div"].to_numpy()
    hk, ak = df["home_key"].to_numpy(), df["away_key"].to_numpy()
    gh, ga = df["fthg"].to_numpy(), df["ftag"].to_numpy()
    unplayed = unplayed_flags(df)

    for i in range(n):
        H, A = (country[i], hk[i]), (country[i], ak[i])
        tier = TIER.get(div[i])
        moves = {}
        for slot, key in (("home", H), ("away", A)):
            if key not in rating:
                rating[key] = p.start
            elif last_season.get(key) != season[i]:
                if p.season_regression:
                    # Squad turnover between seasons: pull partway to the mean.
                    rating[key] += (p.start - rating[key]) * p.season_regression
                prev = last_tier.get(key)
                if tier is not None and prev is not None and prev != tier:
                    # +1 promoted, -1 relegated. Held for the whole season, so
                    # the correction applies to every match at the new level,
                    # not only the first one.
                    move_state[key] = prev - tier
                else:
                    move_state[key] = 0
            last_season[key] = season[i]
            if tier is not None:
                last_tier[key] = tier
            moves[slot] = move_state.get(key, 0)
        out["elo_home_moved"][i] = moves["home"]
        out["elo_away_moved"][i] = moves["away"]

        # The adjustment lives here, on the way out, and is never written back
        # into `rating` -- that is what keeps the pool's division structure
        # intact while still correcting the team that moved.
        rh = rating[H] - p.tier_shift * moves["home"]
        ra = rating[A] - p.tier_shift * moves["away"]
        exp_home = 1.0 / (1.0 + 10 ** (-((rh + p.home_advantage) - ra) / 400.0))

        out["elo_home"][i], out["elo_away"][i] = rh, ra
        out["elo_diff"][i] = (rh + p.home_advantage) - ra
        out["elo_exp_home"][i] = exp_home

        # An unplayed fixture is scored above and absorbed nowhere. Without this
        # the next line raises on `int(nan)`, which is the loud half of the
        # problem; the quiet half is in the other three builders.
        #
        # The season and tier bookkeeping above deliberately still runs: which
        # division a team is in and whether it has changed are facts about the
        # calendar, not about a result, and a promoted side's first fixture of
        # the season should carry its `moved` flag whether or not it has kicked
        # off yet.
        if unplayed[i]:
            continue

        gd = int(gh[i]) - int(ga[i])
        score = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        # Goal-difference multiplier (the FiveThirtyEight form): a rout is more
        # informative than a squeaker, with diminishing returns.
        mult = 1.0 if abs(gd) <= 1 else (1.5 if abs(gd) == 2 else (1.75 + (abs(gd) - 3) / 8))
        # Update the STORED rating, not the adjusted one, or the shift would be
        # baked in permanently and we would be back to corrupting the pool.
        # The adjusted expectancy is still the right thing to update against:
        # a promoted side that performs as a promoted side should keeps its
        # rating, which is how the stored value converges to true strength on
        # one common scale.
        delta = p.k * mult * (score - exp_home)
        rating[H] += delta
        rating[A] -= delta

    return pd.DataFrame(out, index=df.index)


def add_ratings(df: pd.DataFrame, pi: PiParams = PiParams(),
                elo: EloParams = EloParams()) -> pd.DataFrame:
    """Attach every rating feature. `df` must be sorted by kickoff.

    Ratings are built over the WHOLE frame in chronological order, including
    matches that later land in a test split. That is not leakage: each row's
    features come only from matches strictly before it, which is exactly what
    would have been knowable at kickoff. Rebuilding them per split would give
    identical values at far greater cost.
    """
    if not df["kickoff"].is_monotonic_increasing:
        raise ValueError("df must be sorted by kickoff before building ratings")
    return pd.concat([df, pi_rating_features(df, pi), elo_features(df, elo)], axis=1)
