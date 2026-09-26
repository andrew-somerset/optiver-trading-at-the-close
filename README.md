# Closing-Auction Return Prediction (Optiver Kaggle Dataset)

An independent research exercise using the archived Kaggle competition [Optiver – Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close) (2023). It is not an official competition entry and is not affiliated with Optiver.

**Task:** Use order-book and auction data from the last ten minutes of Nasdaq trading to predict each stock's 60-second future price move relative to a synthetic index, in basis points. Error is measured as mean absolute error (MAE).

## Plan

1. Data audit: shape, unique keys, chronology, missing values, and when each field becomes available.
2. A chronological split by trading day into train, validation and final holdout sets, with all stocks from a given day in the same set.
3. Baselines (zero and training median) compared with a regularized linear model and one tree-based model.
4. A few interpretable features (spreads, imbalances, price differences, past changes within each stock and day). Errors broken down by day, stock and time within the auction.

## Results

Full write-up: [reports/REPORT.md](reports/REPORT.md).

- Chronological split by trading day: train days 0–360, validation 361–420, final holdout 421–480 (scored once).
- Holdout MAE: training-median baseline 5.815 bps, ridge 5.768 (−0.8%), gradient boosting **5.708 (−1.8%)**.
- The gain is small but consistent: boosting beat the baseline on all 60 holdout days and for 184 of 200 stocks.
- The documented target definition was verified against the data. Error depends strongly on time in the auction and peaks for predictions made at 230–290 s.

![Holdout error through the auction](reports/figures/holdout_by_second.png)

## Layout

```
src/closing_auction/  data (load + audit + target check), split, features, models, plotting
scripts/              audit_data, train_evaluate
tests/                synthetic-data checks: split, no lookahead, zero denominators, target check
reports/              REPORT.md, data_audit.md, results.md/json, figures
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a Kaggle account that has accepted the competition rules.

```bash
uv sync
uv run kaggle competitions download -c optiver-trading-at-the-close -p data/raw
(cd data/raw && unzip -q optiver-trading-at-the-close.zip)
uv run pytest
uv run python scripts/audit_data.py
uv run python scripts/train_evaluate.py
```

## Data

Competition data is subject to Kaggle's competition rules and is not included in this repository.
