"""Betting simulation, closing-line value, and the statistics that stop a
backtest lying to you.

Three design decisions are load-bearing and none of them are arbitrary:

**Flat stakes for the headline.** One unit per bet. It keeps ROI interpretable
and stops a single early lucky result compounding through the whole equity
curve. Kelly is available as a supplementary bankroll simulation, fractional,
because full Kelly systematically over-bets whenever probabilities are
estimated rather than known -- which is always.

**Three price columns, always, led by the sharpest.** Constantinou (2022)
measured the gap: at a zero edge threshold, 1X2 ROI over 13 EPL seasons was
-9.03% at market-average odds and -1.20% at market-maximum. Nearly all of that
difference is vig avoided by taking the outlier bookmaker, not model skill. A
strategy profitable only in the max column is an odds-comparison screen.

**CLV is the headline, ROI is secondary.** Distinguishing a 2% ROI edge from
zero needs roughly 19,000-43,500 bets depending on the odds -- a decade at
club-level volume. Closing-line value converges perhaps a hundred times
faster: Buchdahl detected a genuine ~6-7% edge from 26 tips. Any ROI figure
reported here carries the sample size it would actually need beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.devig import devig
from src.eval.metrics import OUTCOMES

__all__ = [
    "PriceSet", "BetRule", "simulate", "summarize", "clv_report",
    "bootstrap_ci", "random_bet_null", "required_sample_size",
    "PINNACLE_CLOSE", "B365_CLOSE", "MARKET_MAX_CLOSE", "DEFAULT_PRICES",
]


@dataclass(frozen=True)
class PriceSet:
    """One set of H/D/A odds columns, with a label for the scoreboard.

    `sharp=True` marks the benchmark we lead with. Exactly one price set in a
    run should be sharp, and it should be the closing line of a low-margin
    book.
    """
    label: str
    home: str
    draw: str
    away: str
    sharp: bool = False

    @property
    def cols(self) -> list[str]:
        return [self.home, self.draw, self.away]


# The standard three for this project. Pinnacle closing is the truth test;
# Bet365 closing is a book you could actually hold an account with; market
# maximum is the optimistic bound.
PINNACLE_CLOSE = PriceSet("pinnacle_close", "psch", "pscd", "psca", sharp=True)
B365_CLOSE = PriceSet("b365_close", "b365ch", "b365cd", "b365ca")
MARKET_MAX_CLOSE = PriceSet("market_max_close", "maxch", "maxcd", "maxca")
DEFAULT_PRICES = (PINNACLE_CLOSE, B365_CLOSE, MARKET_MAX_CLOSE)


@dataclass(frozen=True)
class BetRule:
    """When to bet and on what.

    Pre-register this before looking at PnL. Sweeping thresholds and reporting
    the best cell is the commonest way a backtest manufactures an edge --
    Constantinou's own threshold sweep moves 1X2 ROI from -9% to +23% purely
    by shrinking the sample to 37 bets.
    """
    min_ev: float = 0.05
    min_odds: float = 1.5
    max_odds: float = 5.0
    stake: float = 1.0
    name: str = "ev>=0.05, odds 1.5-5.0"

    def describe(self) -> str:
        return (f"{self.name}: bet the max-EV outcome when EV >= {self.min_ev:+.3f} "
                f"and price in [{self.min_odds}, {self.max_odds}]")


def simulate(
    df: pd.DataFrame,
    probs: np.ndarray,
    prices: PriceSet,
    rule: BetRule = BetRule(),
    result_col: str = "result",
    date_col: str = "kickoff",
) -> pd.DataFrame:
    """Run one price set through one rule. Returns one row per bet placed.

    Rows whose price set is incomplete are skipped rather than filled -- an
    invented price becomes an invented profit.
    """
    if len(df) != len(probs):
        raise ValueError(f"df has {len(df)} rows but probs has {len(probs)}")
    missing = [c for c in prices.cols if c not in df.columns]
    if missing:
        raise KeyError(f"price set {prices.label!r} needs missing columns {missing}")

    odds = df[prices.cols].to_numpy(dtype=float)
    usable = np.isfinite(odds).all(axis=1) & np.isfinite(probs).all(axis=1)

    ev = probs * odds - 1.0
    pick = np.argmax(np.where(np.isfinite(ev), ev, -np.inf), axis=1)
    rows = np.arange(len(df))
    best_ev = ev[rows, pick]
    best_odds = odds[rows, pick]
    best_p = probs[rows, pick]

    take = (
        usable
        & (best_ev >= rule.min_ev)
        & (best_odds >= rule.min_odds)
        & (best_odds <= rule.max_odds)
    )
    if not take.any():
        return pd.DataFrame(columns=[
            "match_id", "date", "selection", "prob", "odds", "ev",
            "result", "won", "stake", "pnl", "price_set", "rule",
        ])

    sel = np.array(OUTCOMES)[pick[take]]
    actual = df[result_col].to_numpy()[take]
    won = sel == actual
    stake = rule.stake
    pnl = np.where(won, (best_odds[take] - 1.0) * stake, -stake)

    out = pd.DataFrame({
        "match_id": df["match_id"].to_numpy()[take] if "match_id" in df else rows[take],
        "date": pd.to_datetime(df[date_col]).to_numpy()[take],
        "selection": sel,
        "prob": best_p[take],
        "odds": best_odds[take],
        "ev": best_ev[take],
        "result": actual,
        "won": won,
        "stake": stake,
        "pnl": pnl,
        "price_set": prices.label,
        "rule": rule.name,
    })
    out.attrs["n_candidates"] = int(usable.sum())
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def _blocks(bets: pd.DataFrame) -> np.ndarray:
    """Group bets by matchday.

    Bets on the same day are not independent -- they share weather, news
    cycles, and often the same model quirk -- so the bootstrap resamples days,
    not bets. Resampling individual bets would understate the interval.
    """
    return pd.to_datetime(bets["date"]).dt.floor("D").to_numpy()


def bootstrap_ci(
    bets: pd.DataFrame,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """BCa bootstrap confidence interval on ROI, resampling whole matchdays.

    Bias-corrected and accelerated rather than plain percentile, because
    per-bet returns are skewed and fat-tailed: a win at odds 5.0 pays +4 and a
    loss pays -1, so the distribution is nothing like normal and the percentile
    interval is visibly off-centre at realistic sample sizes.
    """
    if bets.empty:
        return {"roi": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_bets": 0}

    blocks = _blocks(bets)
    uniq = np.unique(blocks)
    by_block = {b: bets.loc[blocks == b, ["pnl", "stake"]].to_numpy() for b in uniq}

    def roi_of(chosen) -> float:
        arr = np.concatenate([by_block[b] for b in chosen])
        staked = arr[:, 1].sum()
        return float(arr[:, 0].sum() / staked) if staked else float("nan")

    theta = roi_of(uniq)

    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for i in range(n_boot):
        reps[i] = roi_of(rng.choice(uniq, size=len(uniq), replace=True))

    # Bias correction: where the observed value sits in the bootstrap cloud.
    prop = float(np.mean(reps < theta))
    prop = min(max(prop, 1.0 / (n_boot + 1)), 1.0 - 1.0 / (n_boot + 1))
    z0 = stats.norm.ppf(prop)

    # Acceleration: jackknife over blocks.
    jack = np.array([roi_of(np.delete(uniq, i)) for i in range(len(uniq))])
    jbar = jack.mean()
    num = float(np.sum((jbar - jack) ** 3))
    den = 6.0 * float(np.sum((jbar - jack) ** 2)) ** 1.5
    a = num / den if den else 0.0

    def adjust(q):
        z = stats.norm.ppf(q)
        return float(stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))

    lo_q, hi_q = adjust(alpha / 2), adjust(1 - alpha / 2)
    return {
        "roi": theta,
        "lo": float(np.quantile(reps, lo_q)),
        "hi": float(np.quantile(reps, hi_q)),
        "n_bets": int(len(bets)),
        "n_blocks": int(len(uniq)),
        "z0": float(z0),
        "accel": float(a),
    }


def random_bet_null(
    df: pd.DataFrame,
    prices: PriceSet,
    n_bets: int,
    rule: BetRule = BetRule(),
    n_sims: int = 1000,
    seed: int = 0,
    result_col: str = "result",
) -> dict:
    """What ROI would the same number of bets earn, chosen at random?

    Kaunitz et al. ran exactly this control: their real strategy returned
    +3.5% against a random-strategy mean of -3.32%. Without it, a positive ROI
    is uninterpretable -- you cannot tell skill from a favourable price mix.
    """
    odds = df[prices.cols].to_numpy(dtype=float)
    ok = np.isfinite(odds).all(axis=1)
    odds, actual = odds[ok], df[result_col].to_numpy()[ok]
    if len(odds) == 0 or n_bets == 0:
        return {"mean_roi": float("nan"), "lo": float("nan"), "hi": float("nan")}

    rng = np.random.default_rng(seed)
    rois = np.empty(n_sims)
    for s in range(n_sims):
        idx = rng.integers(0, len(odds), size=n_bets)
        pick = rng.integers(0, 3, size=n_bets)
        o = odds[idx, pick]
        won = np.array(OUTCOMES)[pick] == actual[idx]
        rois[s] = np.where(won, o - 1.0, -1.0).mean()

    return {
        "mean_roi": float(rois.mean()),
        "lo": float(np.quantile(rois, 0.025)),
        "hi": float(np.quantile(rois, 0.975)),
        "n_sims": n_sims,
        "n_bets": n_bets,
    }


def required_sample_size(mean_odds: float, edge: float = 0.02,
                         power: float = 0.80, alpha: float = 0.05) -> int:
    """Bets needed to distinguish `edge` from zero at the given power.

    n = ((z_alpha + z_beta) * sigma / mu)^2, with sigma = d*sqrt(p(1-p)) and p
    implied by the target edge at those odds. Printed beside every ROI so the
    number is read with its uncertainty attached: a headline ROI over 300 bets
    is not evidence about anything.
    """
    p = (1.0 + edge) / mean_odds
    p = min(max(p, 1e-6), 1 - 1e-6)
    sigma = mean_odds * np.sqrt(p * (1 - p))
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil((z * sigma / edge) ** 2))


def clv_report(bets: pd.DataFrame, closing_odds: pd.Series) -> dict:
    """Closing-line value: did the price you took shorten by kickoff?

    The ratio (taken odds / closing odds) predicts realised level-stakes yield
    with a slope of about 1.00, measured by Buchdahl over 87,960 pre-close /
    close pairs. Its virtue is speed of convergence, not novelty.

    `closing_odds` must be the closing price for the *same selection* the bet
    was placed on, aligned to the bets frame.
    """
    if bets.empty:
        return {"n": 0, "mean_ratio": float("nan"), "pct_shortened": float("nan")}

    taken = bets["odds"].to_numpy(dtype=float)
    close = np.asarray(closing_odds, dtype=float)
    ok = np.isfinite(taken) & np.isfinite(close) & (close > 0)
    if not ok.any():
        return {"n": 0, "mean_ratio": float("nan"), "pct_shortened": float("nan")}

    ratio = taken[ok] / close[ok]
    shortened = ratio > 1.0            # got a bigger price than the close
    t = stats.ttest_1samp(ratio, 1.0) if ok.sum() > 1 else None
    # Binomial test against a coin flip: is the shortening rate better than chance?
    binom = stats.binomtest(int(shortened.sum()), int(ok.sum()), 0.5)

    return {
        "n": int(ok.sum()),
        "mean_ratio": float(ratio.mean()),
        "median_ratio": float(np.median(ratio)),
        "pct_shortened": float(shortened.mean()),
        "t_stat": float(t.statistic) if t is not None else float("nan"),
        "t_pvalue": float(t.pvalue) if t is not None else float("nan"),
        "binom_pvalue": float(binom.pvalue),
    }


def summarize(bets: pd.DataFrame, with_ci: bool = True, seed: int = 0) -> dict:
    """One scoreboard row for a (model, price set, rule) combination."""
    if bets.empty:
        # Keep the same keys as a populated row, so a zero-bet strategy lines
        # up in the scoreboard instead of printing a row of NaN labels.
        return {"price_set": "", "rule": "", "n_bets": 0, "profit": 0.0,
                "roi": float("nan"), "hit_rate": float("nan"),
                "avg_odds": float("nan"), "avg_prob": float("nan"),
                "n_needed_for_2pct": 0}

    staked = bets["stake"].sum()
    out = {
        "price_set": bets["price_set"].iloc[0],
        "rule": bets["rule"].iloc[0],
        "n_bets": int(len(bets)),
        "profit": float(bets["pnl"].sum()),
        "roi": float(bets["pnl"].sum() / staked),
        "hit_rate": float(bets["won"].mean()),
        "avg_odds": float(bets["odds"].mean()),
        "avg_prob": float(bets["prob"].mean()),
    }
    out["n_needed_for_2pct"] = required_sample_size(out["avg_odds"], edge=0.02)
    if with_ci:
        out.update({f"roi_{k}": v for k, v in bootstrap_ci(bets, seed=seed).items()
                    if k in ("lo", "hi")})
    return out
