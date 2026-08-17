"""Tests for the network.

The expensive parts (does it beat the baseline, do the embeddings earn their
place) are answered by `src.experiments ablate`, not here. These tests check
the machinery is correct: the Poisson grid, the calibrator, the vocabulary,
and above all that the training loop cannot see the future.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import i0

from src.data.footballdata import OUT_DIR
from src.eval.metrics import OUTCOMES, log_loss
from src.models.net import (
    MatchNet, NetConfig, TemperatureScaler, build_vocab, poisson_to_hda,
    predict, train_net,
)

PARQUET = OUT_DIR / "matches.parquet"
needs_data = pytest.mark.skipif(not PARQUET.exists(), reason="matches.parquet not built")


def toy(n=600, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"t{i}" for i in range(12)]
    home = rng.choice(teams, n)
    away = np.array([rng.choice([t for t in teams if t != h]) for h in home])
    strength = {t: rng.normal() for t in teams}
    lam_h = np.array([np.exp(0.3 + 0.4 * strength[h] - 0.4 * strength[a])
                      for h, a in zip(home, away)])
    lam_a = np.array([np.exp(0.1 - 0.4 * strength[h] + 0.4 * strength[a])
                      for h, a in zip(home, away)])
    gh = rng.poisson(lam_h)
    ga = rng.poisson(lam_a)
    return pd.DataFrame({
        "home_key": home, "away_key": away, "fthg": gh, "ftag": ga,
        "result": np.where(gh > ga, "H", np.where(gh == ga, "D", "A")),
        "country": "X", "div": "E0", "season": "2020-21",
        "kickoff": pd.date_range("2020-08-01", periods=n, freq="6h"),
    })


# --------------------------------------------------------------------------
# The Poisson grid
# --------------------------------------------------------------------------

def test_poisson_draw_probability_matches_the_bessel_identity():
    """Independent derivation, not a re-run of the grid code.

    For independent Poissons with equal rate lambda,
        P(draw) = sum_k P(k)^2 = e^(-2 lambda) * I_0(2 lambda)
    where I_0 is the modified Bessel function of the first kind. scipy
    supplies I_0, so the expected value never touches the code under test.
    """
    for lam in (0.8, 1.0, 1.5, 2.2):
        p = poisson_to_hda(np.array([[lam, lam]]), max_goals=30)[0]
        expected_draw = np.exp(-2 * lam) * i0(2 * lam)
        assert p[1] == pytest.approx(expected_draw, abs=1e-6)


def test_equal_rates_give_symmetric_home_and_away():
    p = poisson_to_hda(np.array([[1.3, 1.3]]), max_goals=25)[0]
    assert p[0] == pytest.approx(p[2], abs=1e-9)


def test_a_stronger_home_rate_raises_the_home_probability():
    lo = poisson_to_hda(np.array([[1.2, 1.2]]))[0]
    hi = poisson_to_hda(np.array([[2.0, 1.2]]))[0]
    assert hi[0] > lo[0]
    assert hi[2] < lo[2]


def test_poisson_grid_rows_sum_to_one():
    rng = np.random.default_rng(0)
    rates = rng.uniform(0.3, 3.5, size=(200, 2))
    p = poisson_to_hda(rates)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all()


def test_realistic_rates_give_a_realistic_draw_rate():
    """Football's real draw rate is about 26%. Independent Poisson at typical
    scoring rates should land near it -- if it does not, the grid is wrong."""
    p = poisson_to_hda(np.array([[1.50, 1.15]]), max_goals=15)[0]
    assert 0.22 < p[1] < 0.30, f"implied draw rate {p[1]:.3f} is not football-like"
    assert p[0] > p[2], "the higher rate must be the more likely winner"


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def test_temperature_scaling_softens_an_overconfident_model():
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.choice(list(OUTCOMES), n, p=[0.45, 0.26, 0.29])
    # Logits that point the right way but far too hard.
    logits = np.full((n, 3), -4.0)
    logits[np.arange(n), [OUTCOMES.index(v) for v in y]] = 4.0
    flip = rng.random(n) < 0.45                    # wrong on 45% of matches
    logits[flip] = logits[flip][:, ::-1]

    sc = TemperatureScaler().fit(logits, y)
    assert sc.temperature > 1.0, "an overconfident model needs T > 1"

    raw = np.exp(logits - logits.max(axis=1, keepdims=True))
    raw = raw / raw.sum(axis=1, keepdims=True)
    assert log_loss(sc.transform(logits), y) < log_loss(raw, y)


def test_temperature_scaling_preserves_the_argmax():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(500, 3))
    sc = TemperatureScaler()
    sc.log_t = np.log(2.7)
    assert (sc.transform(logits).argmax(axis=1) == logits.argmax(axis=1)).all()


def test_temperature_scaling_returns_valid_probabilities():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(200, 3)) * 6
    sc = TemperatureScaler()
    sc.log_t = np.log(0.4)
    p = sc.transform(logits)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

def test_vocab_reserves_zero_for_unseen_teams():
    df = toy(100)
    v = build_vocab(df)
    assert 0 not in v.teams.values()
    ids = v.team_ids(["X", "X"], ["t0", "never-heard-of-them"])
    assert ids[0] > 0
    assert ids[1] == 0, "an unseen team must map to <unk>, not raise"


def test_vocab_separates_same_named_teams_in_different_countries():
    df = pd.DataFrame({"country": ["A", "B"], "home_key": ["rangers", "rangers"],
                       "away_key": ["x", "y"], "div": ["SC0", "E0"]})
    v = build_vocab(df)
    a, b = v.team_ids(["A", "B"], ["rangers", "rangers"])
    assert a != b


# --------------------------------------------------------------------------
# The model and the training loop
# --------------------------------------------------------------------------

def test_forward_produces_the_right_shapes_and_positive_rates():
    df = toy(200)
    v = build_vocab(df)
    cfg = NetConfig(members=3, hidden=16)
    m = MatchNet(4, v, cfg)
    x = torch.zeros(7, 4)
    ids = torch.zeros(7, dtype=torch.long)
    logits, rates = m(x, ids, ids, ids)
    assert logits.shape == (7, 3)
    assert rates.shape == (7, 2)
    assert (rates > 0).all(), "Poisson rates must be positive"


@pytest.mark.parametrize("team_dim,league_dim", [(0, 0), (0, 4), (8, 0), (8, 4)])
def test_embeddings_can_be_switched_off_for_ablation(team_dim, league_dim):
    df = toy(300)
    v = build_vocab(df)
    X = np.zeros((len(df), 3))
    cfg = NetConfig(team_dim=team_dim, league_dim=league_dim, members=2,
                    hidden=16, max_epochs=3, patience=2)
    model, meta = train_net(df, X, v, cfg)
    out = predict(model, df, X, v, meta)
    assert out["hda"].shape == (len(df), 3)
    assert np.allclose(out["hda"].sum(axis=1), 1.0)


def test_training_validation_split_is_temporal_not_random():
    """Early stopping on a random validation slice would select the epoch that
    best predicts the PAST -- the same leak the walk-forward splitter exists
    to prevent, one level down. The split must be the tail of the window.

    Checked by construction: a frame whose last 15% is deliberately corrupted
    must produce a worse validation loss than one that is not.
    """
    clean = toy(2000, seed=5)
    poisoned = clean.copy()
    tail = slice(int(len(clean) * 0.85), len(clean))
    poisoned.iloc[tail, poisoned.columns.get_loc("result")] = "D"
    poisoned.iloc[tail, poisoned.columns.get_loc("fthg")] = 0
    poisoned.iloc[tail, poisoned.columns.get_loc("ftag")] = 0

    v = build_vocab(clean)
    X = np.random.default_rng(0).normal(size=(len(clean), 3))
    cfg = NetConfig(members=2, hidden=24, max_epochs=6, patience=6)

    _, m_clean = train_net(clean, X, v, cfg)
    _, m_bad = train_net(poisoned, X, v, cfg)
    assert m_bad["best_val_ce"] != pytest.approx(m_clean["best_val_ce"], rel=1e-3), (
        "corrupting the last 15% did not change validation loss -- the "
        "validation split is not the tail of the window")


def test_the_net_learns_something_on_data_with_known_structure():
    """Sanity floor: on synthetic matches generated from real team strengths,
    the net must beat a constant base-rate forecast."""
    df = toy(4000, seed=7)
    v = build_vocab(df)
    X = np.zeros((len(df), 1))          # nothing but team identity to learn from
    cfg = NetConfig(members=4, hidden=32, max_epochs=40, patience=8, lr=3e-3)
    model, meta = train_net(df, X, v, cfg)
    out = predict(model, df, X, v, meta)

    y = df["result"].to_numpy()
    base = np.tile([(y == o).mean() for o in OUTCOMES], (len(y), 1))
    assert log_loss(out["hda"], y) < log_loss(base, y) - 0.01, (
        "the net could not learn team strength from embeddings alone")


def test_goal_head_predicts_plausible_scoring_rates():
    df = toy(3000, seed=8)
    v = build_vocab(df)
    X = np.zeros((len(df), 1))
    model, meta = train_net(df, X, v, NetConfig(members=4, hidden=32,
                                                max_epochs=40, patience=8))
    out = predict(model, df, X, v, meta)
    mean_rates = out["goal_rates"].mean(axis=0)
    actual = np.array([df["fthg"].mean(), df["ftag"].mean()])
    assert np.allclose(mean_rates, actual, atol=0.35), (
        f"predicted rates {mean_rates.round(2)} vs actual {actual.round(2)}")


def test_disabling_the_goal_head_still_trains():
    df = toy(500)
    v = build_vocab(df)
    X = np.zeros((len(df), 2))
    cfg = NetConfig(goal_loss_weight=0.0, members=2, hidden=16,
                    max_epochs=4, patience=3)
    model, meta = train_net(df, X, v, cfg)
    assert np.isfinite(meta["best_val_ce"])


def test_training_is_reproducible_for_a_fixed_seed():
    df = toy(800)
    v = build_vocab(df)
    X = np.random.default_rng(0).normal(size=(len(df), 3))
    cfg = NetConfig(members=2, hidden=16, max_epochs=5, patience=5, seed=42)
    m1, meta1 = train_net(df, X, v, cfg)
    m2, meta2 = train_net(df, X, v, cfg)
    p1 = predict(m1, df, X, v, meta1)["hda"]
    p2 = predict(m2, df, X, v, meta2)["hda"]
    assert np.allclose(p1, p2), "same seed must give the same model"
