"""Baselines the network has to beat.

Three families, chosen because the literature says each is genuinely hard to
improve on rather than because they are easy:

**Ordered logit on ratings.** 1X2 is ordinal -- away < draw < home on a latent
scale of home strength -- and a two-cutpoint ordered model exploits that with
a handful of parameters. It is the structure behind Arntzen & Hvattum (2021),
and it is cheap insurance against the draw class, which is where flexible
classifiers reliably overfit.

**CatBoost on ratings.** The reference. Yeung et al. (2024) found CatBoost on
pi-ratings the best model on goals-only data at RPS 0.2085 across 300k
matches, beating their transformer encoder (0.2098) and LSTM (0.2105). If the
network cannot pass this, it has not earned its place.

**Dixon-Coles.** The classical goal model, via penaltyblog. Note that it
raises `ValueError: Both teams must have been in the training data` for a
promoted side, so it needs an explicit policy rather than a try/except --
see `DixonColesBaseline`.

Everything here returns probabilities in (H, D, A) column order, matching
src.eval.metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from src.eval.metrics import OUTCOMES

__all__ = ["OrderedLogit", "CatBoostBaseline", "DixonColesBaseline",
           "RATING_FEATURES", "FORM_FEATURES", "ALL_FEATURES"]

# The feature set the literature says carries the signal. Deliberately small:
# at ~0.1 nats of total learnable signal, more columns mostly buy overfitting.
RATING_FEATURES = [
    "elo_diff", "elo_exp_home",
    "pi_exp_gd", "pi_home_h", "pi_away_a",
    "elo_home_moved", "elo_away_moved",
]

# Rolling form. Both sides, two windows. The opponent-Elo columns are the
# opposition-strength component: without them "scored a lot" and "played weak
# defences" are the same number.
FORM_FEATURES = [
    f"{side}_{stat}_{w}"
    for side in ("h", "a")
    for w in (5, 10)
    for stat in ("pts", "gf", "ga", "gd", "sot_f", "sot_a", "corners_f",
                 "opp_elo", "home_share")
] + ["h_rest_days", "a_rest_days", "h_played", "a_played",
     "league_goals_avg", "h_days_since_season_start"]

# What the network gets. Deliberately wider than the baseline's, because a net
# with exactly the baseline's inputs has nothing to find that the baseline has
# not already found -- which is the first thing this project measured.
ALL_FEATURES = RATING_FEATURES + FORM_FEATURES


def _standardize(fit_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(fit_x, axis=0)
    sd = np.nanstd(fit_x, axis=0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


class OrderedLogit:
    """Proportional-odds ordered logit over (away < draw < home).

        z      = w . x                       latent home strength
        P(A)   = sigmoid(c1 - z)
        P(D)   = sigmoid(c2 - z) - sigmoid(c1 - z)
        P(H)   = 1 - sigmoid(c2 - z)

    with c1 < c2 enforced by optimising c1 and log(c2 - c1). Two cutpoints and
    one weight vector, so the draw is a *band* on the latent scale rather than
    a class the model has to characterise directly. That is the whole argument
    for using it here.
    """

    def __init__(self, l2: float = 1e-4):
        self.l2 = l2
        self.w: np.ndarray | None = None
        self.c1: float = -0.5
        self.gap: float = 1.0
        self.mu = self.sd = None

    def _probs(self, z: np.ndarray, c1: float, c2: float) -> np.ndarray:
        p_a = expit(c1 - z)
        p_ad = expit(c2 - z)
        p_d = np.clip(p_ad - p_a, 1e-12, None)
        p_h = np.clip(1.0 - p_ad, 1e-12, None)
        p = np.column_stack([p_h, p_d, np.clip(p_a, 1e-12, None)])
        return p / p.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y) -> "OrderedLogit":
        X = np.asarray(X, dtype=float)
        self.mu, self.sd = _standardize(X)
        Xs = np.nan_to_num((X - self.mu) / self.sd)
        a = np.asarray([OUTCOMES.index(str(v).strip().upper()) for v in y])

        def nll(theta):
            w, c1, loggap = theta[:-2], theta[-2], theta[-1]
            c2 = c1 + np.exp(loggap)
            p = self._probs(Xs @ w, c1, c2)
            return -np.mean(np.log(p[np.arange(len(a)), a])) + self.l2 * np.sum(w ** 2)

        theta0 = np.concatenate([np.zeros(Xs.shape[1]), [-0.4, np.log(0.8)]])
        res = minimize(nll, theta0, method="L-BFGS-B")
        self.w, self.c1, self.gap = res.x[:-2], float(res.x[-2]), float(np.exp(res.x[-1]))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("fit() before predict_proba()")
        Xs = np.nan_to_num((np.asarray(X, dtype=float) - self.mu) / self.sd)
        return self._probs(Xs @ self.w, self.c1, self.c1 + self.gap)


class CatBoostBaseline:
    """CatBoost multiclass on rating features. The number to beat."""

    def __init__(self, iterations: int = 400, depth: int = 4,
                 learning_rate: float = 0.05, seed: int = 0):
        self.kwargs = dict(iterations=iterations, depth=depth,
                           learning_rate=learning_rate, random_seed=seed,
                           loss_function="MultiClass", verbose=False,
                           allow_writing_files=False)
        self.model = None
        self._classes: list[str] = []

    def fit(self, X, y) -> "CatBoostBaseline":
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier(**self.kwargs)
        y = np.asarray([str(v).strip().upper() for v in y])
        self.model.fit(np.nan_to_num(np.asarray(X, dtype=float)), y)
        self._classes = [str(c) for c in self.model.classes_]
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = self.model.predict_proba(np.nan_to_num(np.asarray(X, dtype=float)))
        # CatBoost orders its classes alphabetically (A, D, H); everything
        # downstream expects H, D, A. Getting this wrong would scramble every
        # metric silently rather than loudly, so it is done by name.
        #
        # A class can be missing entirely if a training slice never saw it.
        # That is degenerate but must not crash a walk-forward run halfway
        # through, so the absent outcome gets probability zero -- an honest
        # statement about a model with no basis to predict it -- and the
        # metrics module clips the log.
        out = np.zeros((len(p), 3))
        for j, outcome in enumerate(OUTCOMES):
            if outcome in self._classes:
                out[:, j] = p[:, self._classes.index(outcome)]
        total = out.sum(axis=1, keepdims=True)
        total[total <= 0] = 1.0
        return out / total


@dataclass
class DixonColesBaseline:
    """Dixon-Coles via penaltyblog, with an explicit unseen-team policy.

    Two decisions that a try/except would hide:

    **Pooling.** One model per country across all its divisions, not per
    division. A promoted team then arrives with real history attached, and the
    cross-division information (a Championship side genuinely is weaker) is
    carried rather than discarded. Fitting per division would leave every
    promoted team unseen every season.

    **Fallback.** A team still unseen after pooling -- newly entering the data
    entirely -- cannot be predicted at all: penaltyblog raises. Rather than
    dropping those matches (which would quietly shrink the test set and
    flatter the model), they fall back to the training-set outcome base rate,
    and the count is recorded in `n_fallback` so it is visible on the
    scoreboard instead of silent.

    **Time decay.** xi is the Dixon-Coles decay parameter. Tuning lookback and
    decay was worth 0.0023 RPS on penaltyblog's own Eredivisie comparison,
    against 0.0002 across six model families -- roughly ten times more of the
    available improvement lives here than in the choice of distribution.
    """

    xi: float = 0.0018
    max_goals: int = 10
    n_fallback: int = field(default=0, init=False)
    n_total: int = field(default=0, init=False)

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        import penaltyblog as pb

        base = np.array([(train["result"] == o).mean() for o in OUTCOMES])
        base = base / base.sum()

        out = np.tile(base, (len(test), 1))
        self.n_fallback = 0
        self.n_total = len(test)

        for country, te in test.groupby("country", sort=False):
            tr = train[train["country"] == country]
            if len(tr) < 200:
                self.n_fallback += len(te)
                continue

            weights = pb.models.dixon_coles_weights(tr["kickoff"], xi=self.xi)
            model = pb.models.DixonColesGoalModel(
                tr["fthg"].to_numpy(), tr["ftag"].to_numpy(),
                tr["home_key"].to_numpy(), tr["away_key"].to_numpy(),
                weights=weights,
            )
            model.fit()
            seen = set(tr["home_key"]) | set(tr["away_key"])

            for pos, row in zip(te.index, te.itertuples()):
                i = test.index.get_loc(pos)
                if row.home_key not in seen or row.away_key not in seen:
                    self.n_fallback += 1
                    continue
                grid = model.predict(row.home_key, row.away_key, max_goals=self.max_goals)
                hda = np.asarray(grid.home_draw_away, dtype=float)
                out[i] = hda / hda.sum()

        return out
