"""Forecasters. All predict the next-day log return; `naive` predicts zero (price unchanged)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Naive:
    """Tomorrow's price = today's price. The bar every model has to clear."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class Drift:
    """Constant: average training return."""

    def fit(self, X, y):
        self.mu_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu_)


def ridge():
    """Linear model on lagged returns, moving-average ratios, volatility, volume. Scaler is fit on train only
    because it lives inside the pipeline."""
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20)))


class LSTMForecaster:
    """Two-layer LSTM over a window of feature rows (optional: `pip install .[lstm]`).

    Window features are standardised with *training-set* statistics, early stopping watches a
    chronologically later validation block, and the original project's 2-epoch/batch-1 run is gone.
    """

    def __init__(self, window: int = 30, epochs: int = 40, batch_size: int = 64, seed: int = 0):
        self.window, self.epochs, self.batch_size, self.seed = window, epochs, batch_size, seed

    def _windows(self, X: pd.DataFrame) -> np.ndarray:
        Z = ((X.to_numpy() - self.mu_) / self.sd_).astype("float32")
        pad = np.vstack([np.repeat(Z[:1], self.window - 1, axis=0), Z])
        return np.stack([pad[i : i + self.window] for i in range(len(Z))])

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        import keras  # lazy: heavy optional dependency

        keras.utils.set_random_seed(self.seed)
        self.mu_ = X.to_numpy().mean(0)
        self.sd_ = X.to_numpy().std(0) + 1e-9
        self.scale_ = float(np.std(y))  # targets are tiny; learn them in unit variance
        m = keras.Sequential(
            [
                keras.layers.Input((self.window, X.shape[1])),
                keras.layers.LSTM(32, return_sequences=True),
                keras.layers.LSTM(16),
                keras.layers.Dense(1),
            ]
        )
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        m.fit(
            self._windows(X), (y.to_numpy() / self.scale_).astype("float32"),
            validation_data=(self._windows(X_val), (y_val.to_numpy() / self.scale_).astype("float32")),
            epochs=self.epochs, batch_size=self.batch_size, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        )
        self.model_ = m
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._windows(X), verbose=0).ravel() * self.scale_
