"""Price loading: Yahoo Finance download with a CSV cache, plus validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parents[2] / "data"
REQUIRED = ["Close", "Volume"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("index must be strictly increasing dates")
    if (df["Close"] <= 0).any() or df["Close"].isna().any():
        raise ValueError("Close must be positive and complete")
    return df


def load_prices(ticker: str = "NVDA", years: int = 20, refresh: bool = False) -> pd.DataFrame:
    """Daily OHLCV with split/dividend-adjusted `Close`. Cached to data/<ticker>.csv."""
    path = CACHE / f"{ticker.upper()}.csv"
    if path.exists() and not refresh:
        return validate(pd.read_csv(path, index_col=0, parse_dates=True))
    import yfinance as yf  # optional dependency, only needed to download

    end = pd.Timestamp.today().normalize()
    raw = yf.download(ticker, start=end - pd.DateOffset(years=years), end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"no data returned for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):  # yfinance >= 0.2.40 returns (field, ticker)
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw.index.name = "Date"
    path.parent.mkdir(exist_ok=True)
    raw.to_csv(path)
    return validate(raw)
