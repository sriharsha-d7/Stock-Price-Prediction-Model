"""CLI: ``python -m stock_forecast.run --ticker NVDA`` or ``--synthetic ar1``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import features, metrics, models
from .data import load_prices
from .synthetic import make_prices


def evaluate_all(df: pd.DataFrame, with_lstm: bool = False) -> tuple[pd.DataFrame, dict]:
    X, y = features.build(df)
    tr, va, te = features.time_split(X.index)
    close_t = df["Close"].reindex(X.index).to_numpy()
    fitted = {"naive (price unchanged)": models.Naive().fit(X[tr], y[tr]),
              "drift (mean train return)": models.Drift().fit(X[tr], y[tr]),
              "ridge": models.ridge().fit(X[tr], y[tr])}
    if with_lstm:
        fitted["lstm"] = models.LSTMForecaster().fit(X[tr], y[tr], X[va], y[va])
    rows, extra, preds = {}, {}, {}
    for name, m in fitted.items():
        p = m.predict(X[te])
        preds[name] = p
        rows[name] = metrics.evaluate(close_t[te], y[te].to_numpy(), p)
        if not name.startswith("naive"):
            skill = metrics.skill_vs_naive(y[te].to_numpy(), p)
            lo, hi = metrics.bootstrap_skill_ci(y[te].to_numpy(), p)
            rows[name]["skill_vs_naive"] = skill
            extra[name] = {"skill": skill, "ci95": [lo, hi]}
    meta = {"train": [str(X.index[tr][0].date()), str(X.index[tr][-1].date())],
            "test": [str(X.index[te][0].date()), str(X.index[te][-1].date())], "skill": extra}
    return metrics.table(rows).round(4), meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--synthetic", choices=["random_walk", "ar1"])
    ap.add_argument("--lstm", action="store_true", help="also train the LSTM (needs keras)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    df = make_prices(a.synthetic) if a.synthetic else load_prices(a.ticker)
    name = a.synthetic or a.ticker.upper()
    table, meta = evaluate_all(df, a.lstm)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    table.to_csv(out / f"{name}_metrics.csv")
    (out / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{name}: train {meta['train']} test {meta['test']}\n")
    print(table.to_string())
    print("\nskill vs naive (out-of-sample R^2 on returns, 95% block-bootstrap CI):")
    for k, v in meta["skill"].items():
        print(f"  {k}: {v['skill']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
