"""Synthetic price series with a *known* answer, used to validate the evaluation harness."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_prices(kind: str = "random_walk", n: int = 3000, seed: int = 0, phi: float = 0.15) -> pd.DataFrame:
    """``random_walk``: returns are iid, so NO model can beat the naive forecast.
    ``ar1``: returns follow an AR(1) with coefficient ``phi``, so a lag model has real, bounded skill."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0004, 0.018, n)
    if kind == "random_walk":
        r = eps
    elif kind == "ar1":
        r = np.zeros(n)
        for t in range(1, n):
            r[t] = phi * r[t - 1] + eps[t]
    else:
        raise ValueError(kind)
    close = 50 * np.exp(np.cumsum(r))
    vol = rng.lognormal(15, 0.3, n)
    idx = pd.bdate_range("2012-01-02", periods=n, name="Date")
    return pd.DataFrame({"Close": close, "Volume": vol}, index=idx)
