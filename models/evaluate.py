"""Backtest metrics for the price (quantile) and sell (classifier) models.

Reports point accuracy, prediction-interval coverage, and sell-probability calibration,
each broken out per family, and compares the price model against the naive
"median of recent comps" baseline it needs to beat.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.dataset import FEATURE_COLS, PRICE_TARGET, SELL_TARGET


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def evaluate(price_models: dict, sell_model, test_df: pd.DataFrame) -> dict:
    X_test = test_df[FEATURE_COLS]
    y_true_log = test_df[PRICE_TARGET].to_numpy()
    y_true_price = np.exp(y_true_log)

    preds = {name: model.predict(X_test) for name, model in price_models.items()}
    p50_price = np.exp(preds["p50"])
    p10_price = np.exp(preds["p10"])
    p90_price = np.exp(preds["p90"])

    coverage = float(np.mean((y_true_price >= p10_price) & (y_true_price <= p90_price)))
    mae_log = mean_absolute_error(y_true_log, preds["p50"])
    mape = _mape(y_true_price, p50_price)

    naive_pred = test_df["comp_median"].fillna(test_df["comp_median"].median())
    naive_mape = _mape(y_true_price, naive_pred.to_numpy())

    print("\n=== Price model (overall) ===")
    print(f"n = {len(test_df)}")
    print(f"MAE (log price)   = {mae_log:.3f}")
    print(f"MAPE              = {mape:.1f}%   (naive comp-median baseline: {naive_mape:.1f}%)")
    print(f"P10-P90 coverage  = {coverage:.1%}  (target ~80%)")

    print("\n=== Price model (per family) ===")
    for family, idx in test_df.groupby("family", observed=True).groups.items():
        mask = test_df.index.isin(idx)
        if mask.sum() == 0:
            continue
        fam_mape = _mape(y_true_price[mask], p50_price[mask])
        fam_cov = float(np.mean((y_true_price[mask] >= p10_price[mask]) & (y_true_price[mask] <= p90_price[mask])))
        print(f"  {family:12s} n={mask.sum():4d}  MAPE={fam_mape:6.1f}%  coverage={fam_cov:.1%}")

    result = {"mape": mape, "naive_mape": naive_mape, "coverage": coverage}

    y_sell_true = test_df[SELL_TARGET].astype(int).to_numpy()
    if len(np.unique(y_sell_true)) > 1:
        sell_proba = sell_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_sell_true, sell_proba)
        print(f"\n=== Sell model ===\nAUC = {auc:.3f}")
        result["sell_auc"] = auc
    else:
        print("\n=== Sell model ===\nSkipped: test set has only one outcome class.")

    deal_flag = p50_price > test_df["final_high_bid"].to_numpy()
    if deal_flag.sum() > 0:
        print(f"\n=== Deal detector ===")
        print(f"{deal_flag.sum()} / {len(test_df)} lots flagged as underpriced (P50 > final bid)")

    return result


if __name__ == "__main__":
    import joblib

    from models.dataset import load_feature_dataset, time_split
    from models.train import ARTIFACTS_DIR
    from scrapers.common import load_config

    cfg = load_config()
    df = load_feature_dataset(cfg["scrape"]["processed_dir"], cfg)
    _, test_df = time_split(df)
    price_models = joblib.load(ARTIFACTS_DIR / "price_models.joblib")
    sell_model = joblib.load(ARTIFACTS_DIR / "sell_model.joblib")
    evaluate(price_models, sell_model, test_df)
