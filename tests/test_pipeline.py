import numpy as np
import pandas as pd
import pytest

from stock_forecast import features, metrics, models
from stock_forecast.run import evaluate_all
from stock_forecast.synthetic import make_prices


def test_features_use_only_past_information():
    """Changing any FUTURE price must not change today's features."""
    df = make_prices("random_walk", n=600, seed=1)
    X, _ = features.build(df)
    t = X.index[300]
    df2 = df.copy()
    df2.loc[df2.index > t, ["Close", "Volume"]] *= 7.0  # corrupt everything after t
    X2, _ = features.build(df2)
    pd.testing.assert_series_equal(X.loc[t], X2.loc[t])


def test_target_is_next_day_log_return():
    df = make_prices("random_walk", n=400, seed=2)
    X, y = features.build(df)
    t = X.index[100]
    i = df.index.get_loc(t)
    assert y.loc[t] == pytest.approx(np.log(df["Close"].iloc[i + 1] / df["Close"].iloc[i]))


def test_split_is_chronological_with_embargo():
    idx = pd.bdate_range("2020-01-01", periods=1000)
    tr, va, te = features.time_split(idx, embargo=1)
    assert not (tr & va).any() and not (va & te).any()
    assert idx[tr].max() < idx[va].min() < idx[va].max() < idx[te].min()
    assert idx[va].min() - idx[tr].max() > pd.Timedelta(days=1)  # at least one row dropped


def test_scaler_is_fit_on_train_only():
    df = make_prices("random_walk", n=1500, seed=3)
    X, y = features.build(df)
    tr, _, te = features.time_split(X.index)
    m = models.ridge().fit(X[tr], y[tr])
    scaler = m[0]
    np.testing.assert_allclose(scaler.mean_, X[tr].mean().to_numpy(), rtol=1e-9)


def test_naive_is_hard_to_beat_on_random_walk():
    table, meta = evaluate_all(make_prices("random_walk", n=6000, seed=4))
    lo, hi = meta["skill"]["ridge"]["ci95"]
    assert lo <= 0.0 <= hi or meta["skill"]["ridge"]["skill"] < 0.01  # no real skill to find


def test_ridge_finds_planted_signal():
    table, meta = evaluate_all(make_prices("ar1", n=20000, seed=5, phi=0.15))
    s = meta["skill"]["ridge"]
    assert s["ci95"][0] > 0, s  # significantly better than naive
    assert s["skill"] < 0.05  # but bounded: a 0.15 AR(1) cannot explain much


def test_within_band_metric_flatters_naive():
    """The reason ±2.5% 'accuracy' is a poor headline: a no-change forecast already scores high."""
    df = make_prices("random_walk", n=6000, seed=6)
    table, _ = evaluate_all(df)
    assert table.loc["naive (price unchanged)", "within_2.5%_of_actual_%"] > 80


def test_metrics_basic_arithmetic():
    close = np.array([100.0, 100.0])
    y_true = np.array([0.01, -0.02])
    out = metrics.evaluate(close, y_true, y_true)  # perfect forecast
    assert out["price_MAE"] == pytest.approx(0) and out["directional_accuracy_%"] == 100
    assert metrics.skill_vs_naive(y_true, y_true) == 1.0
    assert metrics.skill_vs_naive(y_true, np.zeros(2)) == 0.0
