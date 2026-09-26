"""Baselines and models. Anything learned from data is fit on training rows only."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ConstantModel:
    """Predicts one number: 0, or the training median (the MAE-optimal constant)."""

    def __init__(self, kind="zero"):
        self.kind = kind

    def fit(self, X, y):
        self.value_ = 0.0 if self.kind == "zero" else float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to training-set quantiles so a few extreme rows can't dominate a linear fit."""

    def __init__(self, lower=0.001, upper=0.999):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype="float64")
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype="float64"), self.low_, self.high_)


def ridge(alpha):
    # Missing values -> training median plus a 0/1 "was missing" column (far/near price
    # are structurally missing early in the auction, and that fact is informative).
    return make_pipeline(Winsorizer(), SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(), Ridge(alpha=alpha))


def boosting(learning_rate=0.05, max_iter=300, max_leaf_nodes=63, min_samples_leaf=500, seed=0):
    # Absolute-error loss matches the MAE metric. Early stopping is off: its internal
    # random hold-out would mix days and leak across time.
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=learning_rate, max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
        early_stopping=False, random_state=seed)


def mae(y, pred):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))
