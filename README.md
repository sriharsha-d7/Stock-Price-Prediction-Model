# Stock Price Prediction: a leak-free benchmark

[![CI](https://github.com/sriharsha-d7/stock-price-prediction-model/actions/workflows/ci.yml/badge.svg)](https://github.com/sriharsha-d7/stock-price-prediction-model/actions/workflows/ci.yml)

Next-day stock forecasting built so that the answer to "does the model actually beat doing nothing?" is
measured, not assumed. The pipeline predicts tomorrow's log return from lagged returns, moving-average ratios,
volatility and volume, and scores every model against a **naive "price unchanged"** baseline with a block-bootstrap
confidence interval.

```bash
pip install -e ".[dev]"            # add ".[lstm]" for the Keras model
python -m stock_forecast.run --ticker NVDA          # downloads 20y of prices, writes results/NVDA_metrics.csv
python -m stock_forecast.run --ticker NVDA --lstm   # also trains the LSTM
python -m stock_forecast.run --synthetic ar1        # offline demo on a planted-signal series
pytest                                              # 8 tests
```

## Why this was rebuilt

The first version of this project (kept in [`archive/`](archive/original_lstm_experiment.ipynb)) fit an LSTM to NVDA
and reported RMSE $3.41 on the test period. Reviewing it turned up problems that make that number uninformative:

| Issue in the original | What this version does |
|---|---|
| README claimed "±2.5% accuracy" and "10% lower RMSE"; the notebook measures neither | Reports MAPE, hit-rate within ±2.5%, directional accuracy, IC, and skill against a baseline |
| Min-Max scaler was fit on the **full** history, so the test period's later, higher prices set the scale | Scaling lives inside the model pipeline and is fit on **training rows only** (tested) |
| Predicted the next *price* from 100 prior scaled prices; copying yesterday's price is already near-perfect | Predicts the next-day **return**, where skill (or its absence) is visible |
| No baseline | Naive and drift baselines; skill = out-of-sample R² vs the zero forecast, with a 95% block-bootstrap CI |
| Moving averages were plotted but never fed to the model | 20/50/200-day MA ratios are model inputs |
| 2 epochs, batch size 1, no validation set | Early stopping on a chronologically later validation block, batch 64 |
| Single random-looking 70/30 split, labels could straddle the boundary | Chronological train/val/test with a 1-row embargo (tested) |

## Does the evaluation work? Checked on series with a known answer

I cannot hand a metric to anyone until it can tell "no skill" from "some skill". So the harness is validated on
synthetic prices where the truth is planted (`results/validation.md`, reproduce with `python scripts/validate_harness.py`;
20,000 days, last 15% held out):

| Series | Model | Skill vs naive [95% CI] | Directional accuracy | Within ±2.5% of actual |
|---|---|---|---|---|
| Random walk (no signal exists) | naive | n/a | n/a | 84.4% |
| | ridge | -0.0005 [-0.0020, +0.0011] | 50.1% | 84.4% |
| AR(1) returns, φ = 0.15 | naive | n/a | n/a | 83.6% |
| | ridge | **+0.0192 [+0.0083, +0.0298]** | 54.3% | 84.3% |

Two things follow:

1. **"Within ±2.5%" is a bad headline metric.** A forecast that says *nothing will change* is within ±2.5% of the
   actual next-day price about 84% of the time. High hit rates on price level are expected for any forecaster.
2. **The harness is honest in both directions.** It finds no skill where none exists (the CI straddles 0) and detects
   the planted effect where one does (CI above 0, magnitude small and bounded, as it should be for φ = 0.15).

An LSTM, tried on the planted-signal series, recovered less of it than the ridge model did (skill +0.007 vs +0.016
on a 3,000-day run, CI including 0), so added capacity is not automatically an improvement.

## Real-data results

Run `python -m stock_forecast.run --ticker NVDA`; it writes `results/NVDA_metrics.csv` and prints the skill-vs-naive
confidence interval. No real-market numbers are quoted in this README because the table above is the only evidence
the repo itself can vouch for until the command has been run on your data. When you run it, expect the pattern a
random walk gives: skill near zero with a CI that straddles it, and a high within-±2.5% rate for every model including
naive. Anything much better than that on daily data deserves suspicion before celebration.

## Layout

```
src/stock_forecast/
  data.py        yfinance download with CSV cache + validation
  features.py    leakage-safe features/targets, chronological split with embargo
  models.py      Naive, Drift, Ridge, optional Keras LSTM
  metrics.py     price + return metrics, skill vs naive, block bootstrap
  synthetic.py   random-walk and AR(1) generators for harness validation
  run.py         CLI
tests/           no-lookahead test (corrupts all future prices), split, scaler, planted-signal tests
archive/         the original notebook, unchanged
```

## Limitations

- Next-day, single-ticker, daily bars. No transaction costs or trading rule: forecast skill is not strategy profit.
- Prices are adjusted `Close` from Yahoo Finance; adjustments can be revised retroactively.
- One chronological test block can be unrepresentative of other regimes; the bootstrap CI captures sampling noise, not regime change.
