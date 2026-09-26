"""Interpretable features built only from information available at each snapshot.

Rules:
    - backward-looking changes are computed within one (stock, day), sorted by
      time, using shift(k) with k > 0, so they never see the future or another day
    - cross-sectional features use other stocks at the SAME (day, second), which
      are observed at the same moment
    - ratios with a zero or missing denominator become NaN, never inf
"""

import numpy as np
import pandas as pd

BPS = 1e4


def _ratio(num, den):
    num, den = np.asarray(num, dtype="float64"), np.asarray(den, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.isfinite(out), out, np.nan).astype("float32")


def add_features(df):
    """Return a copy of df with feature columns added (see FEATURES)."""
    d = df.sort_values(["stock_id", "date_id", "seconds_in_bucket"]).copy()
    signed_imbalance = d["imbalance_buy_sell_flag"] * d["imbalance_size"]
    depth = d["bid_size"] + d["ask_size"]
    mid = (d["bid_price"] + d["ask_price"]) / 2

    d["spread_bps"] = _ratio(d["ask_price"] - d["bid_price"], mid) * BPS
    d["book_imbalance"] = _ratio(d["bid_size"] - d["ask_size"], depth)
    d["auction_imbalance"] = _ratio(signed_imbalance, d["matched_size"])
    d["imbalance_vs_depth"] = _ratio(signed_imbalance, depth)
    d["log_matched_size"] = np.log1p(d["matched_size"].clip(lower=0)).astype("float32")
    d["wap_vs_mid_bps"] = (_ratio(d["wap"], mid) - 1) * BPS
    d["wap_vs_ref_bps"] = (_ratio(d["wap"], d["reference_price"]) - 1) * BPS
    d["far_vs_ref_bps"] = (_ratio(d["far_price"], d["reference_price"]) - 1) * BPS
    d["near_vs_ref_bps"] = (_ratio(d["near_price"], d["reference_price"]) - 1) * BPS

    grouped = d.groupby(["stock_id", "date_id"], sort=False)
    for k in (1, 3, 6):
        d[f"wap_ret_{k}_bps"] = (_ratio(d["wap"], grouped["wap"].shift(k)) - 1) * BPS
    d["ref_ret_6_bps"] = (_ratio(d["reference_price"], grouped["reference_price"].shift(6)) - 1) * BPS
    d["auction_imbalance_chg_3"] = d["auction_imbalance"] - grouped["auction_imbalance"].shift(3)

    # Relative to the average stock at the same moment: a rough, equal-weighted
    # stand-in for the synthetic index the target is measured against.
    same_time = d.groupby(["date_id", "seconds_in_bucket"], sort=False)
    for col in ("wap_ret_6_bps", "wap_vs_ref_bps", "auction_imbalance"):
        d[f"{col}_rel"] = d[col] - same_time[col].transform("mean")

    return d.sort_index()


FEATURES = [
    "seconds_in_bucket", "imbalance_buy_sell_flag",
    "spread_bps", "book_imbalance", "auction_imbalance", "imbalance_vs_depth", "log_matched_size",
    "wap_vs_mid_bps", "wap_vs_ref_bps", "far_vs_ref_bps", "near_vs_ref_bps",
    "wap_ret_1_bps", "wap_ret_3_bps", "wap_ret_6_bps", "ref_ret_6_bps", "auction_imbalance_chg_3",
    "wap_ret_6_bps_rel", "wap_vs_ref_bps_rel", "auction_imbalance_rel",
]
