"""Scoring rules and calibration diagnostics for 1X2 forecasts.

Outcome order is fixed as (H, D, A) everywhere in this project. RPS depends on
that ordering being the real ordinal scale -- home win, draw, away win -- so
any code that reorders the columns silently changes what RPS means.

Report RPS *and* log loss, always. RPS is the field standard (Constantinou &
Fenton 2012) because 1X2 is ordinal and a distance-sensitive rule should
penalise a confident home pick less on a draw than on an away win. But
Wheatcroft (2019) argues from simulation that the ignorance/log score
outperforms it, and the field is not settled. Both are cheap. Log loss also
has the property that matters for staking: it is the negative log-growth of a
Kelly bankroll, so an improvement in log loss translates directly into
expected compounding. RPS does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")
_IDX = {o: i for i, o in enumerate(OUTCOMES)}

__all__ = [
    "OUTCOMES", "to_onehot", "rps", "log_loss", "brier", "accuracy",
    "ece", "reliability_table", "summary",
]


def to_onehot(y) -> np.ndarray:
    """('H','D','A', ...) -> (n, 3) indicator matrix in H/D/A order."""
    y = np.asarray(y, dtype=object)
    out = np.zeros((len(y), 3), dtype=float)
    for i, v in enumerate(y):
        try:
            out[i, _IDX[str(v).strip().upper()]] = 1.0
        except KeyError:
            raise ValueError(
                f"outcome {v!r} at row {i} is not one of {OUTCOMES}"
            ) from None
    return out


def _check(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"probabilities must be (n, 3) in H/D/A order, got {p.shape}")
    s = p.sum(axis=1)
    bad = ~np.isclose(s, 1.0, atol=1e-6)
    if bad.any():
        raise ValueError(f"{bad.sum()} rows do not sum to 1 (e.g. {s[bad][:3]})")
    return p


def rps(p, y) -> float:
    """Mean ranked probability score. Lower is better.

        RPS = 1/(r-1) * sum_{i=1..r-1} ( cumsum(p)_i - cumsum(a)_i )^2

    For reference: bookmaker closing odds score roughly 0.19-0.21 depending on
    league. A uniform (1/3, 1/3, 1/3) forecast scores exactly 2/9 = 0.2222 when
    the three outcomes are equally likely, and about 0.234 at realistic 1X2
    base rates (H .45 / D .26 / A .29) -- the draw is cheap for a uniform
    forecast under RPS, which is precisely the distance-sensitivity the rule
    was chosen for.
    """
    p = _check(p)
    a = to_onehot(y)
    cp = np.cumsum(p, axis=1)[:, :-1]
    ca = np.cumsum(a, axis=1)[:, :-1]
    return float(np.mean(np.sum((cp - ca) ** 2, axis=1) / (p.shape[1] - 1)))


def rps_per_match(p, y) -> np.ndarray:
    """Per-match RPS. Needed for paired bootstrap comparisons between models."""
    p = _check(p)
    a = to_onehot(y)
    cp = np.cumsum(p, axis=1)[:, :-1]
    ca = np.cumsum(a, axis=1)[:, :-1]
    return np.sum((cp - ca) ** 2, axis=1) / (p.shape[1] - 1)


def log_loss(p, y, eps: float = 1e-15) -> float:
    """Mean negative log likelihood in nats. A uniform forecast scores 1.0986."""
    p = _check(p)
    a = to_onehot(y)
    return float(-np.mean(np.sum(a * np.log(np.clip(p, eps, 1.0)), axis=1)))


def log_loss_per_match(p, y, eps: float = 1e-15) -> np.ndarray:
    p = _check(p)
    a = to_onehot(y)
    return -np.sum(a * np.log(np.clip(p, eps, 1.0)), axis=1)


def brier(p, y) -> float:
    """Multi-class Brier score (sum of squared errors across the 3 classes)."""
    p = _check(p)
    a = to_onehot(y)
    return float(np.mean(np.sum((p - a) ** 2, axis=1)))


def accuracy(p, y) -> float:
    """Argmax hit rate. Reported for orientation only.

    Accuracy is not what decides a bet: a wager is +EV iff p*odds > 1, which
    depends on the level of p, not on whether it is the largest. A more
    accurate but overconfident model loses money faster than a less accurate
    calibrated one.
    """
    p = _check(p)
    pred = np.array(OUTCOMES)[np.argmax(p, axis=1)]
    return float(np.mean(pred == np.asarray([str(v).strip().upper() for v in y])))


def reliability_table(p, y, bins: int = 10) -> pd.DataFrame:
    """Per-class reliability, the input to a reliability diagram.

    Each class's forecast probabilities are binned; within a bin, mean
    forecast is compared to observed frequency. A perfectly calibrated model
    sits on the diagonal. Plot the bookmaker's curve on the same axes -- the
    market is the best-calibrated forecaster available and makes the yardstick.
    """
    p = _check(p)
    a = to_onehot(y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for k, name in enumerate(OUTCOMES):
        idx = np.clip(np.digitize(p[:, k], edges, right=True) - 1, 0, bins - 1)
        for b in range(bins):
            m = idx == b
            if not m.any():
                continue
            rows.append({
                "outcome": name,
                "bin": b,
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "n": int(m.sum()),
                "mean_forecast": float(p[m, k].mean()),
                "observed_freq": float(a[m, k].mean()),
            })
    return pd.DataFrame(rows)


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error, averaged over the three classes.

    Weighted mean |mean_forecast - observed_freq| across bins. Lower is better.
    """
    tab = reliability_table(p, y, bins=bins)
    if tab.empty:
        return float("nan")
    per_class = []
    for _, g in tab.groupby("outcome"):
        w = g["n"] / g["n"].sum()
        per_class.append(float((w * (g["mean_forecast"] - g["observed_freq"]).abs()).sum()))
    return float(np.mean(per_class))


def summary(p, y, label: str = "") -> dict:
    """Every headline metric in one row, for the scoreboard."""
    return {
        "model": label,
        "n": int(len(y)),
        "rps": rps(p, y),
        "log_loss": log_loss(p, y),
        "brier": brier(p, y),
        "ece": ece(p, y),
        "accuracy": accuracy(p, y),
    }
