# Closing-auction return prediction: findings

An independent research exercise on the archived Kaggle dataset *Optiver – Trading at the Close* (2023). It is not a competition entry, and no leaderboard score is claimed.

**Task.** For each Nasdaq stock and each 10-second snapshot of the last 10 minutes before the close, predict the stock's 60-second future WAP return minus a synthetic index's return, in basis points. The metric is mean absolute error (MAE).

## 1. Data contract (`reports/data_audit.md`)

- 5,237,980 rows: 200 stocks × 481 trading days × 55 snapshots (0–540 s). The key (stock, day, second) is unique. 88 rows have no target (dropped).
- Prices are normalized so WAP = 1 at the start of each stock-day; only returns and ratios are meaningful.
- `far_price`/`near_price` are missing for every row before 300 s and appear from 300 s on. That's structural, so the models get a "was missing" signal.
- **The target definition was checked against the data, not just read.** If target = 10⁴ × (stock return − index return), then each stock's own 60-second return minus its target must equal a single index return shared by all stocks at that moment. Across stocks at the same timestamp, that difference varies by a median of 0.005 bps, compared with 8.4 bps for the stock returns themselves. Definition confirmed.

## 2. Method

- **Split by whole days, in time order:** train = days 0–360 (361 days), validation = 361–420 (60), holdout = 421–480 (60). All stocks on a day share a set, so the model always predicts unseen days. The target never crosses a day boundary, so no gap is needed.
- **Features (19, all available at prediction time):** bid–ask spread, order-book imbalance, WAP vs mid, auction imbalance (signed, relative to matched size and to book depth), matched size, far/near price vs reference price, 10/30/60-second WAP and reference-price changes within the same stock and day, and each stock's values relative to the cross-stock average at the same second. Tests on synthetic data confirm that changing any data after time t leaves every feature at t unchanged.
- **Models:** zero, training median (the best constant under MAE), ridge regression (winsorized, median-imputed with missing flags, standardized, all fit on training rows only), and gradient boosting (scikit-learn `HistGradientBoostingRegressor`, absolute-error loss, early stopping off because its random hold-out would mix days).
- **Protocol:** 3 ridge and 3 boosting settings, each fit on train and compared on validation. The chosen settings were refit on train + validation and scored **once** on the holdout.

## 3. Results

| Model | Validation MAE | Holdout MAE | Holdout vs median |
|---|---:|---:|---:|
| Zero | 6.469 | 5.815 | 0.0% |
| Training median | 6.469 | 5.815 | — |
| Ridge (α = 10⁴) | 6.412 | 5.768 | −0.8% |
| **Gradient boosting** (127 leaves, 300 iterations, lr 0.05) | **6.340** | **5.708** | **−1.8%** |

- **Small but consistent.** Boosting cut holdout MAE by 0.107 bps per day on average (standard error 0.004, with days as the unit). It beat the median on **all 60 holdout days** and for 184 of 200 stocks (per-stock change: median −1.2%, range −7.5% to +1.2%).
- The holdout period is calmer than validation (baseline MAE 5.8 vs 6.5). That's why every comparison is against the baseline **on the same days**.
- The ridge penalty barely mattered (6.4124 → 6.4120 across α = 1 to 10⁴): with about 4 million rows and 19 features, there's little to regularize. Gains from here would come from better features or more flexible models, not from shrinking coefficients.

### Where the errors are (`figures/holdout_by_second.png`, `holdout_by_day.png`)
- **Error depends strongly on time in the auction.** Predictions made at 230–290 s have MAE of about 9 bps vs about 5 elsewhere. Their 60-second window crosses the 300 s mark, when near/far indicative prices start being published. That timing is our reading of why; we didn't test the mechanism.
- The biggest relative gain (9%) is at 540 s, whose target window runs into the closing print.
- A few holdout days (e.g. date_id 458, 470, 474) have daily MAE of 8–10 bps for every model. The day's market conditions matter far more than the model choice.

### What the models use (`figures/feature_importance.png`, ridge coefficients in `results.md`)
- Permutation importance (boosting, 300k holdout rows): time in the auction ranks first, then WAP vs mid, the stock's WAP vs reference price relative to other stocks, and auction imbalance.
- Ridge signs are intuitive. A stock trading above its auction reference price, relative to the average stock, tends to fall back relative to the index afterwards. A buy-side auction imbalance goes with a subsequent rise. These are correlations in the features, not causal effects, and correlated features share credit.

## 4. Limitations and what this does not show

- An MAE reduction is a forecasting result. It is not evidence of a profitable strategy: no execution, costs, auction participation or capacity are modelled.
- A small feature set and hyperparameter grid, one model per family, one chronological split. Top competition solutions used far more features, online retraining and ensembles; they haven't been reproduced here.
- The synthetic index weights aren't used. The "relative to other stocks" features use an equal-weighted average as a stand-in.
- Holdout scores come from one 60-day period. Results on a different period could differ.

## Reproduce

```bash
uv sync
uv run kaggle competitions download -c optiver-trading-at-the-close -p data/raw && (cd data/raw && unzip -q optiver-trading-at-the-close.zip)
uv run pytest                              # synthetic-data checks
uv run python scripts/audit_data.py        # writes reports/data_audit.md and a parquet cache
uv run python scripts/train_evaluate.py    # a few minutes on a 12-core laptop
```
