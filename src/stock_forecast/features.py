"""Features and targets. Every feature at row t uses data up to and including day t only."""

from __future__ import annotations

import numpy as np
import pandas as pd

N_LAGS = 10


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` where y[t] is the log return from day t to t+1.

    The moving averages the original project plotted are now *inputs* (as ratios to price, so
    they are scale-free and comparable across a 100x price range).
    """
    close = df["Close"]
    ret = np.log(close).diff()
    X = pd.DataFrame(index=df.index)
    for k in range(N_LAGS):
        X[f"ret_lag{k}"] = ret.shift(k)
    for w in (20, 50, 200):
        X[f"log_close_over_ma{w}"] = np.log(close / close.rolling(w).mean())
    X["vol20"] = ret.rolling(20).std()
    X["log_volume_over_ma20"] = np.log(df["Volume"].replace(0, np.nan) / df["Volume"].rolling(20).mean())
    y = ret.shift(-1).rename("target")  # NaN on the last row: nothing to predict
    ok = X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok]


def time_split(index: pd.DatetimeIndex, train: float = 0.70, val: float = 0.15, embargo: int = 1):
    """Chronological masks. ``embargo`` rows are dropped between blocks so a label (next-day return)
    never straddles a block boundary."""
    n = len(index)
    a, b = int(n * train), int(n * (train + val))
    pos = np.arange(n)
    return pos < a, (pos >= a + embargo) & (pos < b), pos >= b + embargo
