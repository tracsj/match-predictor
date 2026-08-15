"""Convert bookmaker odds into probabilities by removing the overround.

Four methods, because the choice is not neutral and the differences show up
exactly where betting decisions get made -- at the extremes of the price range.

Shin is the default. Strumbelj (2014), "On determining probability forecasts
from betting odds", International Journal of Forecasting 30(4), found Shin
probabilities more accurate than basic normalisation and regression-based
approaches for *all* bookmaker/sport pairs tested. Everything comparing these
methods on +EV-tool marketing sites is downstream of that paper.

Report multiplicative alongside Shin as a sensitivity check. If a betting
conclusion flips between the two, it was never robust.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = ["overround", "multiplicative", "additive", "power", "shin", "devig"]


def _as_2d(odds: np.ndarray) -> tuple[np.ndarray, bool]:
    a = np.asarray(odds, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, -1), True
    return a, False


def overround(odds) -> np.ndarray:
    """Sum of raw implied probabilities. 1.0 = no margin; 1.05 = 5% overround."""
    a, single = _as_2d(odds)
    out = np.sum(1.0 / a, axis=1)
    return out[0] if single else out


def multiplicative(odds) -> np.ndarray:
    """p_i = q_i / sum(q). Removes margin proportionally to probability."""
    a, single = _as_2d(odds)
    q = 1.0 / a
    p = q / q.sum(axis=1, keepdims=True)
    return p[0] if single else p


def additive(odds) -> np.ndarray:
    """p_i = q_i - (sum(q) - 1)/n. Equal margin per outcome.

    Can produce negatives on longshots; those are clipped to zero and the
    remainder renormalised, which is the standard practical fix.
    """
    a, single = _as_2d(odds)
    q = 1.0 / a
    n = a.shape[1]
    p = q - (q.sum(axis=1, keepdims=True) - 1.0) / n
    p = np.clip(p, 1e-12, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p[0] if single else p


def power(odds) -> np.ndarray:
    """p_i = q_i**k with k solved so the row sums to 1.

    Always in [0,1]. Over-corrects longshots relative to Shin.
    """
    a, single = _as_2d(odds)
    q = 1.0 / a
    out = np.empty_like(q)
    for i, row in enumerate(q):
        if not np.all(np.isfinite(row)) or np.any(row <= 0):
            out[i] = np.nan
            continue
        if abs(row.sum() - 1.0) < 1e-12:
            out[i] = row
            continue

        def f(k, r=row):
            return np.sum(r ** k) - 1.0

        try:
            k = brentq(f, 1.0, 50.0, xtol=1e-12)
        except ValueError:
            out[i] = row / row.sum()
            continue
        p = row ** k
        out[i] = p / p.sum()
    return out[0] if single else out


def shin(odds, return_z: bool = False):
    """Shin's method: solve for the implied proportion of insider money z.

        p_i = (sqrt(z^2 + 4(1-z) q_i^2 / Q) - z) / (2(1-z)),   Q = sum(q)

    z is chosen so the row sums to 1. Larger z means more of the margin is
    attributed to informed traders, which shifts correction toward longshots
    -- that is what makes it model favourite-longshot bias endogenously,
    rather than assuming the margin is spread proportionally.

    For a 2-outcome market Shin reduces to the additive method.
    """
    a, single = _as_2d(odds)
    q = 1.0 / a
    out = np.empty_like(q)
    zs = np.full(q.shape[0], np.nan)

    for i, row in enumerate(q):
        if not np.all(np.isfinite(row)) or np.any(row <= 0):
            out[i] = np.nan
            continue
        Q = row.sum()
        if Q <= 1.0 + 1e-12:          # no margin to remove
            out[i] = row / Q
            zs[i] = 0.0
            continue

        def total(z, r=row, Qs=Q):
            p = (np.sqrt(z * z + 4.0 * (1.0 - z) * r * r / Qs) - z) / (2.0 * (1.0 - z))
            return p.sum() - 1.0

        # total(0) = sqrt(Q) - 1 > 0 when there is margin; total -> < 0 as z
        # approaches 1, so a root is bracketed on (0, 1).
        try:
            z = brentq(total, 0.0, 1.0 - 1e-9, xtol=1e-12)
        except ValueError:
            out[i] = row / Q
            zs[i] = 0.0
            continue
        p = (np.sqrt(z * z + 4.0 * (1.0 - z) * row * row / Q) - z) / (2.0 * (1.0 - z))
        out[i] = p / p.sum()          # guard against float drift
        zs[i] = z

    if single:
        return (out[0], zs[0]) if return_z else out[0]
    return (out, zs) if return_z else out


_METHODS = {
    "shin": shin,
    "multiplicative": multiplicative,
    "additive": additive,
    "power": power,
}


def devig(odds, method: str = "shin") -> np.ndarray:
    """Dispatch by name. Default is Shin -- see module docstring."""
    try:
        fn = _METHODS[method]
    except KeyError:
        raise ValueError(
            f"unknown de-vig method {method!r}; choose from {sorted(_METHODS)}"
        ) from None
    return fn(odds)
