"""Evaluation in price space (what a user cares about) and return space (where skill lives)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def evaluate(close_t: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, band: float = 0.025) -> dict:
    """Metrics for next-day forecasts. ``close_t`` is today's price, ``y_*`` are log returns."""
    price_true = close_t * np.exp(y_true)
    price_pred = close_t * np.exp(y_pred)
    err = price_pred - price_true
    rel = np.abs(err) / price_true
    nz = (y_true != 0) & (y_pred != 0)  # a zero forecast has no direction
    constant = np.ptp(y_pred) == 0
    return {
        "price_MAE": float(np.abs(err).mean()),
        "price_RMSE": float(np.sqrt((err**2).mean())),
        "price_MAPE_%": float(100 * rel.mean()),
        f"within_{band:.1%}_of_actual_%": float(100 * (rel <= band).mean()),
        "directional_accuracy_%": float(100 * (np.sign(y_pred[nz]) == np.sign(y_true[nz])).mean()) if nz.any() else float("nan"),
        "return_RMSE": float(np.sqrt(((y_pred - y_true) ** 2).mean())),
        "return_IC_spearman": float("nan") if constant else float(spearmanr(y_pred, y_true).statistic),
        "n_test": int(len(y_true)),
    }


def skill_vs_naive(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Out-of-sample R^2 against the zero-return (naive) forecast: >0 means better than naive."""
    return float(1 - ((y_true - y_pred) ** 2).sum() / (y_true**2).sum())


def bootstrap_skill_ci(y_true, y_pred, n_boot: int = 1000, block: int = 20, seed: int = 0) -> tuple[float, float]:
    """Moving-block bootstrap 95% CI for `skill_vs_naive` (blocks keep volatility clustering)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, k)])[:n]
        vals.append(skill_vs_naive(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def table(rows: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).T
