"""Chronological train / validation / holdout split on whole trading days.

Every stock on a given day lands in the same set, and the sets follow each
other in time, so validation and holdout scores are "future" scores.
"""

import numpy as np


def split_days(date_ids, validation_frac=0.125, holdout_frac=0.125, gap_days=0):
    """Return (train_days, validation_days, holdout_days) as sorted arrays.

    gap_days drops that many days between consecutive sets. The target only looks
    60 seconds ahead within the same day, so 0 is enough here; the option exists
    for features or targets that span days.
    """
    days = np.sort(np.unique(date_ids))
    n = len(days)
    n_hold = int(round(n * holdout_frac))
    n_val = int(round(n * validation_frac))
    hold = days[n - n_hold:]
    val = days[n - n_hold - gap_days - n_val: n - n_hold - gap_days]
    train = days[: n - n_hold - 2 * gap_days - n_val]
    return train, val, hold


def assign(df, train_days, val_days, hold_days):
    split = np.full(len(df), "", dtype=object)
    split[df["date_id"].isin(train_days).to_numpy()] = "train"
    split[df["date_id"].isin(val_days).to_numpy()] = "validation"
    split[df["date_id"].isin(hold_days).to_numpy()] = "holdout"
    return split
