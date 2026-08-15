"""Tests for the baseline models the network has to beat."""

import numpy as np
import pandas as pd
import pytest

from src.data.footballdata import OUT_DIR
from src.eval.betting import PINNACLE_CLOSE
from src.eval.metrics import OUTCOMES, rps
from src.eval.split import season_walk_forward
from src.features.ratings import add_ratings
from src.models.baselines import (
    CatBoostBaseline, DixonColesBaseline, OrderedLogit, RATING_FEATURES,
)

PARQUET = OUT_DIR / "matches.parquet"
needs_data = pytest.mark.skipif(not PARQUET.exists(), reason="matches.parquet not built")


@pytest.fixture(scope="module")
def panel():
    df = pd.read_parquet(PARQUET).sort_values("kickoff").reset_index(drop=True)
    df = add_ratings(df)
    df = df[(df["source"] == "main")
            & df["season"].between("2016-17", "2024-25")
            & df[PINNACLE_CLOSE.cols].notna().all(axis=1)]
    return df.sort_values("kickoff").reset_index(drop=True)


# --------------------------------------------------------------------------
# Ordered logit
# --------------------------------------------------------------------------

def test_ordered_logit_outputs_valid_probabilities():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = np.where(X[:, 0] > 0.5, "H", np.where(X[:, 0] < -0.5, "A", "D"))
    p = OrderedLogit().fit(X, y).predict_proba(X)
    assert p.shape == (400, 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all()


def test_ordered_logit_respects_the_ordinal_scale():
    """The structural claim: one latent score, two cutpoints, so the draw sits
    BETWEEN away and home. As the latent score rises, P(H) must rise
    monotonically and P(A) must fall monotonically."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 1))
    y = np.where(X[:, 0] > 0.6, "H", np.where(X[:, 0] < -0.6, "A", "D"))
    m = OrderedLogit().fit(X, y)
    grid = np.linspace(-3, 3, 40).reshape(-1, 1)
    p = m.predict_proba(grid)
    if m.w[0] < 0:                       # sign of the latent axis is arbitrary
        p = p[::-1]
    assert np.all(np.diff(p[:, 0]) > -1e-9), "P(home) must be monotone in the latent score"
    assert np.all(np.diff(p[:, 2]) < 1e-9), "P(away) must be monotone in the latent score"
    # ...and the draw is a band, so it peaks in the middle rather than at an end
    assert p[:, 1].argmax() not in (0, len(p) - 1)


def test_ordered_logit_cutpoints_stay_ordered():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 3))
    y = rng.choice(list(OUTCOMES), size=300)
    m = OrderedLogit().fit(X, y)
    assert m.gap > 0, "c2 must exceed c1 or the draw band is negative"


def test_ordered_logit_beats_a_constant_when_signal_exists():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3000, 1))
    latent = X[:, 0] + rng.normal(scale=0.5, size=3000)
    y = np.where(latent > 0.5, "H", np.where(latent < -0.5, "A", "D"))
    m = OrderedLogit().fit(X, y)
    base = np.tile([(y == o).mean() for o in OUTCOMES], (len(y), 1))
    assert rps(m.predict_proba(X), y) < rps(base, y) - 0.01


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        OrderedLogit().predict_proba(np.zeros((2, 3)))


# --------------------------------------------------------------------------
# CatBoost
# --------------------------------------------------------------------------

def test_catboost_returns_columns_in_hda_order():
    """CatBoost sorts its classes alphabetically (A, D, H). If the reorder into
    H/D/A were wrong every downstream metric would be silently scrambled, so
    this checks it against a case with an unmistakable answer."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(600, 1))
    y = np.where(X[:, 0] > 0, "H", "A")          # no draws, cleanly separated
    m = CatBoostBaseline(iterations=60, depth=3).fit(X, y)
    p = m.predict_proba(np.array([[3.0], [-3.0]]))
    assert p[0, 0] > 0.8, "strong positive input should be a home win"
    assert p[1, 2] > 0.8, "strong negative input should be an away win"
    assert np.allclose(p.sum(axis=1), 1.0)


# --------------------------------------------------------------------------
# Dixon-Coles
# --------------------------------------------------------------------------

@needs_data
def test_dixon_coles_falls_back_visibly_for_unseen_teams(panel):
    """penaltyblog raises ValueError for a team absent from training. The
    fallback must be counted, not swallowed -- silently dropping those matches
    would shrink the test set and flatter the model."""
    splits = list(season_walk_forward(panel, min_train_seasons=3))
    s = splits[0]
    train = panel.iloc[s.train_idx]
    test = panel.iloc[s.test_idx].head(400)

    dc = DixonColesBaseline()
    p = dc.fit_predict(train, test)

    assert p.shape == (len(test), 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert dc.n_total == len(test)
    assert dc.n_fallback >= 0
    # Promotion guarantees some unseen teams in any real season.
    assert dc.n_fallback < len(test), "everything fell back -- pooling is broken"


@needs_data
def test_dixon_coles_is_a_credible_model(panel):
    """It must comfortably beat the base rate, and land in the published band."""
    splits = list(season_walk_forward(panel, min_train_seasons=3))
    s = splits[-1]
    train = panel.iloc[s.train_idx]
    test = panel.iloc[s.test_idx].head(1500)

    p = DixonColesBaseline().fit_predict(train, test)
    y = test["result"].to_numpy()
    base = np.tile([(train["result"] == o).mean() for o in OUTCOMES], (len(y), 1))
    base = base / base.sum(axis=1, keepdims=True)

    assert rps(p, y) < rps(base, y) - 0.01
    assert 0.17 < rps(p, y) < 0.23, f"Dixon-Coles RPS {rps(p, y):.4f} outside the plausible band"


# --------------------------------------------------------------------------
# Walk-forward, against the real corpus
# --------------------------------------------------------------------------

@needs_data
def test_rating_baselines_reproduce_the_published_band(panel):
    """The Phase 3 gate.

    Yeung et al. (2024) report CatBoost on pi-ratings at RPS 0.2085 over 300k
    matches, against a bookmaker consensus of 0.2063 and their best deep model
    at 0.2195. Our ratings baselines must land in that neighbourhood, and must
    sit between the base rate and the market. If they do not, the pipeline is
    wrong, not the model.
    """
    X = panel[RATING_FEATURES].to_numpy(float)
    y = panel["result"].to_numpy()
    P, Y = [], []
    for s in season_walk_forward(panel, min_train_seasons=3):
        P.append(OrderedLogit().fit(X[s.train_idx], y[s.train_idx]).predict_proba(X[s.test_idx]))
        Y.append(y[s.test_idx])
    P, Y = np.vstack(P), np.concatenate(Y)

    from src.eval.devig import devig
    market = devig(panel[PINNACLE_CLOSE.cols].to_numpy(float), method="shin")
    market_rps = rps(market[-len(Y):] if len(market) != len(Y) else market, Y) \
        if len(market) == len(Y) else None

    model_rps = rps(P, Y)
    base = np.tile([(Y == o).mean() for o in OUTCOMES], (len(Y), 1))
    assert model_rps < rps(base, Y), "ratings model must beat the base rate"
    assert 0.200 < model_rps < 0.215, f"RPS {model_rps:.4f} outside the published band"


@needs_data
def test_the_moved_flags_earn_their_place(panel):
    """Regression test for the tier-shift investigation.

    The promotion/relegation correction lives in these two features rather
    than in a hand-set Elo adjustment, because the model learns it better.
    Dropping them must cost measurable accuracy -- if it ever stops costing
    anything, the EloParams.tier_shift note needs revisiting.
    """
    from src.eval.metrics import rps_per_match

    y = panel["result"].to_numpy()
    core = ["elo_diff", "elo_exp_home", "pi_exp_gd", "pi_home_h", "pi_away_a"]

    def run(feats):
        X = panel[feats].to_numpy(float)
        P, Y = [], []
        for s in season_walk_forward(panel, min_train_seasons=3):
            P.append(OrderedLogit().fit(X[s.train_idx], y[s.train_idx])
                     .predict_proba(X[s.test_idx]))
            Y.append(y[s.test_idx])
        return np.vstack(P), np.concatenate(Y)

    without = run(core)
    with_flags = run(core + ["elo_home_moved", "elo_away_moved"])
    d = rps_per_match(*without) - rps_per_match(*with_flags)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert d.mean() > 0, "the moved flags should help"
    assert t > 3.0, f"expected a clear effect, got t={t:.2f}"
