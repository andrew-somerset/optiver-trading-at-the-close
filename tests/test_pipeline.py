"""Checks on SYNTHETIC data shaped like the competition file (no real data needed)."""

import numpy as np
import pandas as pd
import pytest

from closing_auction import features, models, split
from closing_auction.data import EXPECTED_COLUMNS, check_target_definition


def fake_auction(n_stocks=4, n_days=6, seed=0):
    rng = np.random.default_rng(seed)
    seconds = np.arange(0, 550, 10)
    rows = []
    for day in range(n_days):
        index_path = np.cumprod(1 + rng.normal(0, 2e-4, len(seconds)))
        for stock in range(n_stocks):
            wap = index_path * np.cumprod(1 + rng.normal(0, 3e-4, len(seconds)))
            for i, s in enumerate(seconds):
                j = min(i + 6, len(seconds) - 1)
                target = ((wap[j] / wap[i]) - (index_path[j] / index_path[i])) * 1e4 if i + 6 < len(seconds) else np.nan
                rows.append({
                    "stock_id": stock, "date_id": day, "seconds_in_bucket": s,
                    "imbalance_size": rng.uniform(0, 1e6), "imbalance_buy_sell_flag": rng.choice([-1, 0, 1]),
                    "reference_price": wap[i] * (1 + rng.normal(0, 1e-4)), "matched_size": rng.uniform(1e6, 5e6),
                    "far_price": np.nan if s < 300 else wap[i], "near_price": np.nan if s < 300 else wap[i],
                    "bid_price": wap[i] - 1e-4, "bid_size": rng.uniform(1e4, 1e5),
                    "ask_price": wap[i] + 1e-4, "ask_size": rng.uniform(1e4, 1e5), "wap": wap[i],
                    "target": target, "time_id": day * len(seconds) + i, "row_id": f"{day}_{s}_{stock}",
                })
    df = pd.DataFrame(rows)
    assert list(df.columns) == EXPECTED_COLUMNS
    return df


def test_split_is_chronological_and_keeps_days_whole():
    days = np.arange(481)
    train, val, hold = split.split_days(days)
    assert train.max() < val.min() and val.max() < hold.min()
    assert len(train) + len(val) + len(hold) == 481
    df = fake_auction(n_days=8)
    labels = split.assign(df, *split.split_days(df["date_id"], 0.25, 0.25))
    assert (pd.Series(labels).groupby(df["date_id"].to_numpy()).nunique() == 1).all()


def test_split_gap():
    train, val, hold = split.split_days(np.arange(100), 0.1, 0.1, gap_days=2)
    assert val.min() - train.max() == 3 and hold.min() - val.max() == 3


def test_lag_features_never_cross_days_or_stocks():
    df = fake_auction()
    f = features.add_features(df)
    first = f["seconds_in_bucket"] == 0
    assert f.loc[first, "wap_ret_1_bps"].isna().all()
    assert f.loc[f["seconds_in_bucket"] < 60, "wap_ret_6_bps"].isna().all()


def test_features_do_not_use_future_rows():
    df = fake_auction()
    base = features.add_features(df)
    later = df["seconds_in_bucket"] > 300
    shocked = df.copy()
    for col in ("wap", "reference_price", "bid_price", "ask_price", "far_price", "near_price"):
        shocked.loc[later, col] *= 1.05
    shocked.loc[later, "imbalance_size"] *= 3
    after = features.add_features(shocked)
    early = ~later
    pd.testing.assert_frame_equal(base.loc[early, features.FEATURES], after.loc[early, features.FEATURES])


def test_ratios_with_zero_denominator_are_nan_not_inf():
    df = fake_auction()
    df.loc[0, ["bid_size", "ask_size"]] = 0
    df.loc[1, "matched_size"] = 0
    f = features.add_features(df)
    assert np.isnan(f.loc[0, "book_imbalance"]) and np.isnan(f.loc[1, "auction_imbalance"])
    assert np.isfinite(f[features.FEATURES].to_numpy(dtype=float)[~np.isnan(f[features.FEATURES].to_numpy(dtype=float))]).all()


def test_target_definition_check_detects_documented_target():
    df = fake_auction(n_stocks=6)
    result = check_target_definition(df)
    assert result["median_cross_stock_std_of_implied_index_ret_bps"] < 1e-3
    assert result["median_cross_stock_std_of_stock_ret_bps"] > 1


def test_winsorizer_uses_training_quantiles_only():
    w = models.Winsorizer(0.1, 0.9).fit(np.arange(11.0).reshape(-1, 1))
    assert w.transform(np.array([[-100.0], [5.0], [100.0]])).ravel().tolist() == [1.0, 5.0, 9.0]


def test_models_fit_and_predict():
    df = features.add_features(fake_auction()).dropna(subset=["target"])
    X, y = df[features.FEATURES], df["target"]
    assert models.ConstantModel("median").fit(X, y).predict(X[:3]).tolist() == [pytest.approx(np.median(y))] * 3
    for model in (models.ridge(1.0), models.boosting(max_iter=20, min_samples_leaf=20)):
        pred = model.fit(X, y).predict(X)
        assert np.isfinite(pred).all() and len(pred) == len(y)
