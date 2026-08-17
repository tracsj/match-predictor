"""Feature scaling: that it happens, that it is fitted correctly, and that it
is the right kind.

The raw features span a 2,020x range in standard deviation -- elo_diff has
sigma about 123, h_home_share_10 about 0.06. Unscaled, gradient updates would
be dominated by whichever column happens to carry the largest units, and the
form features would barely register.

Two choices are asserted here rather than left as comments:

**Standardization, not min-max.** After standardizing, the real corpus spans
-21.6 to +15.3 sigma. Those are genuine outliers (a freak run of results, a
300-day gap between fixtures). Min-max scaling would anchor the 0-1 interval
to them and compress the other 99% of matches into a sliver, which is worse
than not scaling. Standardization leaves an outlier as an outlier.

**Fitted on train only.** Computing the mean and standard deviation over the
whole dataset leaks the test period's distribution backwards. It inflates
results and never looks like a bug.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.footballdata import OUT_DIR
from src.models.baselines import ALL_FEATURES, CatBoostBaseline, OrderedLogit
from src.models.net import NetConfig, build_vocab, predict, train_net

FEATURES = OUT_DIR / "features.parquet"
needs_data = pytest.mark.skipif(not FEATURES.exists(), reason="features.parquet not built")


def toy(n=900, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    y = np.where(latent > 0.5, "H", np.where(latent < -0.5, "A", "D"))
    return pd.DataFrame({
        "home_key": rng.choice([f"t{i}" for i in range(10)], n),
        "away_key": rng.choice([f"u{i}" for i in range(10)], n),
        "fthg": rng.poisson(1.4, n), "ftag": rng.poisson(1.1, n),
        "result": y, "country": "X", "div": "E0", "season": "2020-21",
        "kickoff": pd.date_range("2020-08-01", periods=n, freq="6h"),
    }), np.column_stack([latent, rng.normal(size=n)])


# --------------------------------------------------------------------------
# The property that proves scaling works
# --------------------------------------------------------------------------

def test_ordered_logit_is_invariant_to_rescaling_a_feature():
    """Multiply one input by 1000. A model that standardizes internally must
    give the same answer; a model that does not would be dominated by it."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1200, 3))
    y = np.where(X[:, 0] > 0.4, "H", np.where(X[:, 0] < -0.4, "A", "D"))

    a = OrderedLogit().fit(X, y).predict_proba(X)
    Xb = X.copy()
    Xb[:, 0] *= 1000.0
    b = OrderedLogit().fit(Xb, y).predict_proba(Xb)

    assert np.allclose(a, b, atol=1e-3), "rescaling one column changed the forecast"


def test_ordered_logit_is_invariant_to_shifting_a_feature():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1200, 3))
    y = np.where(X[:, 1] > 0.4, "H", np.where(X[:, 1] < -0.4, "A", "D"))
    a = OrderedLogit().fit(X, y).predict_proba(X)
    Xb = X.copy()
    Xb[:, 1] += 5000.0
    b = OrderedLogit().fit(Xb, y).predict_proba(Xb)
    assert np.allclose(a, b, atol=1e-3)


def test_net_is_broadly_invariant_to_rescaling_a_feature():
    """Same property for the net. Tolerance is looser than the logit's because
    SGD on a finite budget is not exactly scale-free, but a model that failed
    to standardize would diverge far beyond this."""
    df, X = toy(1200)
    v = build_vocab(df)
    cfg = NetConfig(seq_hidden=0, members=2, hidden=24, max_epochs=12, patience=6, seed=0)

    m1, meta1 = train_net(df, X, v, cfg)
    p1 = predict(m1, df, X, v, meta1)["hda"]

    X2 = X.copy()
    X2[:, 0] *= 500.0
    m2, meta2 = train_net(df, X2, v, cfg)
    p2 = predict(m2, df, X2, v, meta2)["hda"]

    assert np.abs(p1 - p2).mean() < 0.02, (
        f"rescaling moved the mean forecast by {np.abs(p1 - p2).mean():.4f}")


def test_the_standardizer_actually_normalises():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 4)) * np.array([1.0, 100.0, 0.01, 5.0])
    y = rng.choice(["H", "D", "A"], 500)
    m = OrderedLogit().fit(X, y)
    Xs = (X - m.mu) / m.sd
    assert np.allclose(Xs.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Xs.std(axis=0), 1.0, atol=1e-9)


def test_a_constant_column_does_not_divide_by_zero():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(size=400), np.full(400, 7.0)])
    y = rng.choice(["H", "D", "A"], 400)
    p = OrderedLogit().fit(X, y).predict_proba(X)
    assert np.isfinite(p).all()


# --------------------------------------------------------------------------
# Fitted on train only
# --------------------------------------------------------------------------

def test_scaler_statistics_come_from_training_data_only():
    """Fit on a slice whose distribution differs sharply from the rest. The
    stored mean and sd must match the TRAINING slice, not the full array."""
    rng = np.random.default_rng(4)
    train_X = rng.normal(loc=0.0, scale=1.0, size=(600, 2))
    test_X = rng.normal(loc=50.0, scale=9.0, size=(600, 2))
    y = rng.choice(["H", "D", "A"], 600)

    m = OrderedLogit().fit(train_X, y)
    assert np.allclose(m.mu, train_X.mean(axis=0), atol=1e-9)
    assert np.allclose(m.sd, train_X.std(axis=0), atol=1e-9)

    full = np.vstack([train_X, test_X])
    assert not np.allclose(m.mu, full.mean(axis=0), atol=1.0), (
        "the scaler appears to have seen the test distribution")


def test_net_scaler_excludes_the_validation_tail():
    """train_net holds out the last 15% for early stopping. The standardizer
    must be fitted on the first 85% only -- otherwise the statistics of the
    stopping set bleed into the fitted model."""
    df, X = toy(1000)
    X = X.copy()
    cut = int(len(df) * 0.85)
    X[cut:, 0] += 400.0             # make the tail wildly different

    v = build_vocab(df)
    _, meta = train_net(df, X, v, NetConfig(seq_hidden=0, members=2, hidden=16,
                                            max_epochs=2, patience=2))
    assert meta["mu"][0] == pytest.approx(X[:cut, 0].mean(), abs=1e-6)
    assert abs(meta["mu"][0] - X[:, 0].mean()) > 1.0


# --------------------------------------------------------------------------
# Trees do not need it
# --------------------------------------------------------------------------

def test_catboost_is_scale_invariant_without_any_scaling():
    """Trees split on thresholds, so scaling buys nothing -- which is part of
    why tree ensembles are so robust on tabular data. Asserted so nobody later
    'fixes' the baseline by adding a scaler it does not need."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(800, 2))
    y = np.where(X[:, 0] > 0, "H", "A")
    a = CatBoostBaseline(iterations=60, depth=3).fit(X, y).predict_proba(X)
    Xb = X.copy()
    Xb[:, 0] *= 10_000.0
    b = CatBoostBaseline(iterations=60, depth=3).fit(Xb, y).predict_proba(Xb)
    assert np.allclose(a, b, atol=1e-9)


# --------------------------------------------------------------------------
# On the real corpus
# --------------------------------------------------------------------------

@needs_data
def test_raw_feature_scales_really_do_span_orders_of_magnitude():
    """The motivation, kept as a live measurement rather than a comment. If
    this ever stops being true the scaling argument needs revisiting."""
    df = pd.read_parquet(FEATURES, columns=ALL_FEATURES)
    sd = df.std(numeric_only=True).to_numpy()
    sd = sd[np.isfinite(sd) & (sd > 0)]
    assert sd.max() / sd.min() > 100, "expected wildly different raw scales"


@needs_data
def test_standardized_corpus_has_outliers_that_would_break_min_max():
    """The reason for z-scores rather than 0-1. Real matches sit many sigma
    from the mean; min-max would compress everything else to nothing."""
    df = pd.read_parquet(FEATURES, columns=ALL_FEATURES)
    X = df.to_numpy(float)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd[sd < 1e-9] = 1.0
    Z = np.nan_to_num((X - mu) / sd)
    assert Z.max() > 8 or Z.min() < -8, "expected genuine outliers in the corpus"

    # What min-max would do to the bulk of the data, for the record.
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    mm = (X - lo) / span
    worst = np.nanmin(np.nanpercentile(mm, 99, axis=0) - np.nanpercentile(mm, 1, axis=0))
    assert worst < 0.5, (
        "expected at least one feature where min-max squashes the middle 98% "
        "into less than half the range")
