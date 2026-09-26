"""Load the competition training file and check it matches what the code expects.

One row = one stock at one 10-second snapshot of the closing auction:
key (stock_id, date_id, seconds_in_bucket). `target` is the stock's 60-second
future WAP return minus the synthetic index's, in basis points.
"""

from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["stock_id", "date_id", "seconds_in_bucket"]
EXPECTED_COLUMNS = [
    "stock_id", "date_id", "seconds_in_bucket", "imbalance_size", "imbalance_buy_sell_flag",
    "reference_price", "matched_size", "far_price", "near_price", "bid_price", "bid_size",
    "ask_price", "ask_size", "wap", "target", "time_id", "row_id",
]
INT_COLUMNS = {"stock_id": "int16", "date_id": "int16", "seconds_in_bucket": "int16",
               "imbalance_buy_sell_flag": "int8", "time_id": "int32"}


def load_train(path):
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(EXPECTED_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}. Check the data dictionary.")
    dtypes = {c: INT_COLUMNS.get(c, "float32") for c in header if c != "row_id"}
    df = pd.read_csv(path, dtype=dtypes)
    return df.sort_values(["date_id", "seconds_in_bucket", "stock_id"], ignore_index=True)


def audit(df):
    """Facts about the file that the modelling depends on. Returns a dict of tables/values."""
    out = {}
    out["rows"] = len(df)
    out["stocks"] = df["stock_id"].nunique()
    out["days"] = df["date_id"].nunique()
    out["date_range"] = (int(df["date_id"].min()), int(df["date_id"].max()))
    out["seconds_values"] = np.sort(df["seconds_in_bucket"].unique()).tolist()
    out["duplicate_keys"] = int(df.duplicated(KEY).sum())
    out["missing_target"] = int(df["target"].isna().sum())

    per_day = df.groupby("date_id")["stock_id"].nunique()
    out["stocks_per_day"] = per_day.describe().round(1).to_dict()

    missing = df.isna().mean().mul(100).round(3)
    out["missing_pct"] = missing[missing > 0].to_dict()
    # far/near price only exist once the auction has matched some interest
    out["far_price_missing_by_second"] = (
        df.groupby("seconds_in_bucket")["far_price"].apply(lambda s: s.isna().mean()).round(3).to_dict())

    prices = ["reference_price", "far_price", "near_price", "bid_price", "ask_price", "wap"]
    first = df[df["seconds_in_bucket"] == 0]
    out["price_level_at_start"] = first[["reference_price", "wap"]].describe().round(4).to_dict()
    out["price_summary"] = df[prices].describe(percentiles=[0.001, 0.5, 0.999]).round(4).to_dict()
    out["crossed_book_rows"] = int((df["bid_price"] > df["ask_price"]).sum())
    out["target_summary"] = df["target"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(3).to_dict()
    out["time_id_is_date_and_second"] = bool(
        (df.groupby("time_id")[["date_id", "seconds_in_bucket"]].nunique() == 1).all().all())
    out["target_check"] = check_target_definition(df)
    return out


def check_target_definition(df, horizon_steps=6):
    """Test the documented target: 1e4 * (stock WAP return - index WAP return) over 60 s.

    If that is right, stock_return - target should be the SAME number for every
    stock at a given (date, second): the index return. We measure how much it
    varies across stocks, relative to how much the stock returns themselves vary.
    """
    d = df.sort_values(["stock_id", "date_id", "seconds_in_bucket"])
    future = d.groupby(["stock_id", "date_id"])["wap"].shift(-horizon_steps)
    d = d.assign(stock_ret=(future / d["wap"] - 1) * 1e4).dropna(subset=["stock_ret", "target"])
    d["implied_index_ret"] = d["stock_ret"] - d["target"]
    g = d.groupby(["date_id", "seconds_in_bucket"])
    return {
        "rows_checked": len(d),
        "median_cross_stock_std_of_stock_ret_bps": float(g["stock_ret"].std().median()),
        "median_cross_stock_std_of_implied_index_ret_bps": float(g["implied_index_ret"].std().median()),
    }
