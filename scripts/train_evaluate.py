"""Chronological model comparison with a one-time final holdout.

    uv run python scripts/train_evaluate.py

1. features + day-based split (train / validation / holdout)
2. every candidate fit on train, scored on validation; best settings chosen there
3. chosen settings refit on train + validation, scored ONCE on holdout
4. error analysis on the holdout: by second in the auction, by day, by stock

Writes reports/results.md, reports/results.json and reports/figures/*.png.
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from closing_auction import features, models, plotting, split

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

RIDGE_ALPHAS = [1.0, 100.0, 10_000.0]
BOOSTING_GRID = [
    dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31),
    dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=127),
    dict(learning_rate=0.1, max_iter=600, max_leaf_nodes=63),
]


def load():
    cached = ROOT / "data" / "processed" / "train.parquet"
    if not cached.exists():
        raise SystemExit("Run scripts/audit_data.py first (it caches data/processed/train.parquet).")
    return pd.read_parquet(cached)


def timed_fit(name, model, X, y):
    start = time.time()
    model.fit(X, y)
    print(f"  fit {name}: {time.time() - start:.0f}s")
    return model


def candidates():
    yield "zero", {}, lambda: models.ConstantModel("zero")
    yield "train median", {}, lambda: models.ConstantModel("median")
    for a in RIDGE_ALPHAS:
        yield "ridge", {"alpha": a}, lambda a=a: models.ridge(a)
    for params in BOOSTING_GRID:
        yield "boosting", params, lambda p=params: models.boosting(**p)


def daily_mae(df, pred_col):
    return (df["target"] - df[pred_col]).abs().groupby(df["date_id"]).mean()


def plot_by_second(hold):
    by_sec = hold.groupby("seconds_in_bucket").apply(
        lambda g: pd.Series({m: models.mae(g["target"], g[f"pred_{m}"]) for m in ("median", "ridge", "boosting")}))
    fig, axes = plotting.figure(ncols=2, figsize=(11.5, 4.4))
    for color, (m, label) in zip(plotting.SERIES, [("median", "Train median"), ("ridge", "Ridge"),
                                                     ("boosting", "Boosting")]):
        axes[0].plot(by_sec.index, by_sec[m], color=color, lw=2, label=label)
    improvement = (1 - by_sec["boosting"] / by_sec["median"]) * 100
    axes[1].plot(by_sec.index, improvement, color=plotting.SERIES[2], lw=2)
    axes[1].axhline(0, color=plotting.AXIS, lw=1)
    axes[0].set_ylabel("MAE (bps)")
    axes[1].set_ylabel("Boosting MAE improvement vs median (%)")
    for ax in axes:
        ax.set_xlabel("Seconds into the closing auction")
    axes[0].legend(frameon=False, fontsize=9)
    plotting.title(fig, "Holdout error through the auction",
                   "Holdout days only, scored once after model choices were fixed on validation days.")
    return fig, by_sec


def plot_by_day(hold):
    days = pd.DataFrame({m: daily_mae(hold, f"pred_{m}") for m in ("median", "boosting")})
    fig, axes = plotting.figure(nrows=2, figsize=(10, 6), sharex=True)
    axes[0].plot(days.index, days["median"], color=plotting.SERIES[0], lw=1.5, label="Train median")
    axes[0].plot(days.index, days["boosting"], color=plotting.SERIES[2], lw=1.5, label="Boosting")
    axes[0].set_ylabel("Daily MAE (bps)")
    axes[0].legend(frameon=False, fontsize=9)
    gain = days["median"] - days["boosting"]
    axes[1].bar(days.index, gain, color=np.where(gain >= 0, plotting.SERIES[2], plotting.SERIES[7]), width=0.8)
    axes[1].axhline(0, color=plotting.AXIS, lw=1)
    axes[1].set_ylabel("Median MAE − boosting MAE (bps)")
    axes[1].set_xlabel("date_id (holdout days)")
    plotting.title(fig, "Holdout error by day: hard days are hard for every model",
                   "Positive bars = boosting beat the constant baseline that day.")
    return fig, days


def plot_importance(names, result):
    order = np.argsort(result.importances_mean)
    fig, ax = plotting.figure(figsize=(8, 6))
    ax.barh(np.array(names)[order], result.importances_mean[order], xerr=result.importances_std[order],
            color=plotting.SERIES[0], height=0.6)
    ax.set_xlabel("Increase in holdout MAE when the feature is shuffled (bps)")
    plotting.title(fig, "Which features the boosting model relies on",
                   "Permutation importance on a holdout sample. Correlated features share credit, so read as a ranking.")
    return fig


def main():
    df = load()
    rows_all = len(df)
    df = features.add_features(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    print(f"rows with target: {len(df):,} of {rows_all:,}")

    train_days, val_days, hold_days = split.split_days(df["date_id"])
    df["split"] = split.assign(df, train_days, val_days, hold_days)
    boundaries = {name: [int(d.min()), int(d.max()), len(d)]
                  for name, d in [("train", train_days), ("validation", val_days), ("holdout", hold_days)]}
    print("day boundaries [first, last, n_days]:", boundaries)

    X = df[features.FEATURES]
    y = df["target"].to_numpy()
    is_train = (df["split"] == "train").to_numpy()
    is_val = (df["split"] == "validation").to_numpy()
    is_hold = (df["split"] == "holdout").to_numpy()

    # 2. choose settings on validation
    validation = []
    for family, params, make in candidates():
        model = timed_fit(f"{family} {params}", make(), X[is_train], y[is_train])
        score = models.mae(y[is_val], model.predict(X[is_val]))
        validation.append({"model": family, "params": params, "validation_mae": score})
        print(f"{family:12s} {params} validation MAE {score:.4f}")
    val_table = pd.DataFrame(validation)
    best = {fam: val_table[val_table["model"] == fam].nsmallest(1, "validation_mae").iloc[0]["params"]
            for fam in ("ridge", "boosting")}

    # 3. refit chosen settings on train + validation, score holdout once
    fit_rows = is_train | is_val
    final = {
        "zero": models.ConstantModel("zero"),
        "median": models.ConstantModel("median"),
        "ridge": models.ridge(**best["ridge"]),
        "boosting": models.boosting(**best["boosting"]),
    }
    hold = df.loc[is_hold, ["stock_id", "date_id", "seconds_in_bucket", "target"]].copy()
    holdout_scores = {}
    for name, model in final.items():
        timed_fit(name, model, X[fit_rows], y[fit_rows])
        hold[f"pred_{name}"] = model.predict(X[is_hold])
        holdout_scores[name] = models.mae(hold["target"], hold[f"pred_{name}"])
        print(f"holdout {name}: {holdout_scores[name]:.4f}")

    # paired comparison with days as the independent unit
    diff = daily_mae(hold, "pred_median") - daily_mae(hold, "pred_boosting")
    paired = {"mean_daily_gain_bps": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
              "days_boosting_better": int((diff > 0).sum()), "days": int(len(diff))}

    # 4. error analysis
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, by_sec = plot_by_second(hold)
    fig.savefig(FIGURES / "holdout_by_second.png", dpi=150, bbox_inches="tight")
    fig, by_day = plot_by_day(hold)
    fig.savefig(FIGURES / "holdout_by_day.png", dpi=150, bbox_inches="tight")

    by_stock = hold.groupby("stock_id").apply(
        lambda g: pd.Series({"median": models.mae(g["target"], g["pred_median"]),
                             "boosting": models.mae(g["target"], g["pred_boosting"])}))
    by_stock["gain_pct"] = (1 - by_stock["boosting"] / by_stock["median"]) * 100

    sample = np.random.default_rng(0).choice(np.flatnonzero(is_hold), size=min(300_000, is_hold.sum()), replace=False)
    importance = permutation_importance(final["boosting"], X.iloc[sample], y[sample], n_repeats=3,
                                        scoring="neg_mean_absolute_error", random_state=0, n_jobs=1)
    fig = plot_importance(features.FEATURES, importance)
    fig.savefig(FIGURES / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    ridge_coefs = pd.Series(final["ridge"][-1].coef_[: len(features.FEATURES)], index=features.FEATURES)

    results = {
        "rows_with_target": len(df), "rows_total": rows_all, "day_boundaries": boundaries,
        "validation": validation, "chosen": best, "holdout_mae": holdout_scores, "paired_daily": paired,
        "by_stock_gain_pct": by_stock["gain_pct"].describe().round(2).to_dict(),
        "stocks_worse_than_median": int((by_stock["gain_pct"] < 0).sum()),
    }
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, default=float))

    importance_table = pd.DataFrame({"feature": features.FEATURES, "mae_increase": importance.importances_mean,
                                     "std": importance.importances_std}).sort_values("mae_increase", ascending=False)
    md = [
        "# Results (generated by scripts/train_evaluate.py)", "",
        f"- Rows with a target: {len(df):,} of {rows_all:,}",
        f"- Day split [first date_id, last date_id, number of days]: {boundaries}",
        f"- Chosen on validation: ridge {best['ridge']}, boosting {best['boosting']}",
        "", "## Validation MAE (bps), models fit on train days", "",
        val_table.assign(params=val_table["params"].astype(str)).round(4).to_markdown(index=False),
        "", "## Holdout MAE (bps), models refit on train + validation, scored once", "",
        pd.Series(holdout_scores, name="holdout_mae").round(4).to_markdown(),
        "", "## Boosting vs train median, by holdout day", "",
        f"- Mean daily MAE gain {paired['mean_daily_gain_bps']:.4f} bps (SE {paired['se']:.4f}); "
        f"boosting better on {paired['days_boosting_better']} of {paired['days']} days",
        f"- Per-stock gain (%): {results['by_stock_gain_pct']}; stocks where boosting was worse: "
        f"{results['stocks_worse_than_median']}",
        "", "## Holdout MAE by seconds_in_bucket", "", by_sec.round(3).to_markdown(),
        "", "## Permutation importance (holdout sample of 300k rows)", "",
        importance_table.round(4).to_markdown(index=False),
        "", "## Ridge coefficients (standardized features)", "",
        ridge_coefs.round(4).sort_values().to_markdown(), "",
    ]
    (REPORTS / "results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
