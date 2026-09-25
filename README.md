# Closing-Auction Return Prediction (Optiver Kaggle Dataset)

An independent research exercise using the archived Kaggle competition [Optiver – Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close) (2023). It is not an official competition entry and is not affiliated with Optiver.

**Task:** Use order-book and auction data from the last ten minutes of Nasdaq trading to predict each stock's 60-second future price move relative to a synthetic index, in basis points. Error is measured as mean absolute error (MAE).

## Plan

1. Data audit: shape, unique keys, chronology, missing values, and when each field becomes available.
2. A chronological split by trading day into train, validation and final holdout sets, with all stocks from a given day in the same set.
3. Baselines (zero and training median) compared with a regularized linear model and one tree-based model.
4. A few interpretable features (spreads, imbalances, price differences, past changes within each stock and day). Errors broken down by day, stock and time within the auction.

## Status

In progress. No results yet.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a Kaggle account that has accepted the competition rules.

```bash
uv sync
uv run kaggle competitions download -c optiver-trading-at-the-close -p data/raw
```

## Data

Competition data is subject to Kaggle's competition rules and is not included in this repository.
