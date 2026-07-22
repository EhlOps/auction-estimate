"""Trains the two-part hurdle model:
  - price model: LightGBM quantile regressors (P10/P50/P90) on log(final_high_bid),
    always observed for every lot regardless of sold/reserve-not-met status.
  - sell model: LightGBM classifier for P(reserve met).

Both are single models pooled across all three families (site + family are just features),
so high-volume segments (MINI, Golf GTI) regularize the sparse one (BMW wagons).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.dataset import FEATURE_COLS, NUMERIC_COLS, PRICE_TARGET, SELL_TARGET, load_feature_dataset, time_split
from scrapers.common import load_config

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MIN_ROWS_FOR_SPLIT = 30

QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}


def train_price_models(X: pd.DataFrame, y: pd.Series) -> dict[str, lgb.LGBMRegressor]:
    # Note: LightGBM's quantile objective does not support monotone_constraints (raises a
    # fatal error if set), so small-N regularization here relies on shallow trees / high
    # min_child_samples rather than a mileage/age monotonicity prior.
    models = {}
    for name, alpha in QUANTILES.items():
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=300,
            num_leaves=15,
            min_child_samples=5,
            learning_rate=0.05,
            verbose=-1,
        )
        model.fit(X, y)
        models[name] = model
    return models


def train_sell_model(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=15,
        min_child_samples=5,
        learning_rate=0.05,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def main(max_rows_for_split: int = MIN_ROWS_FOR_SPLIT) -> None:
    cfg = load_config()
    df = load_feature_dataset(cfg["scrape"]["processed_dir"], cfg)
    print(f"loaded {len(df)} rows across families: {df['family'].value_counts().to_dict()}")

    if len(df) < max_rows_for_split:
        warnings.warn(
            f"Only {len(df)} rows available -- this is a small test sample, not enough for a "
            "meaningful held-out backtest. Run the scrapers at full volume before trusting metrics."
        )
        train_df, test_df = df, df.iloc[0:0]
    else:
        train_df, test_df = time_split(df)

    X_train = train_df[FEATURE_COLS]
    y_train_price = train_df[PRICE_TARGET]
    y_train_sell = train_df[SELL_TARGET].astype(int)

    price_models = train_price_models(X_train, y_train_price)
    sell_model = train_sell_model(X_train, y_train_sell)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(price_models, ARTIFACTS_DIR / "price_models.joblib")
    joblib.dump(sell_model, ARTIFACTS_DIR / "sell_model.joblib")
    # Persist categorical dtype metadata so predict.py can reconstruct compatible columns.
    joblib.dump({c: X_train[c].cat.categories.tolist() for c in X_train.select_dtypes("category").columns},
                ARTIFACTS_DIR / "categories.joblib")
    print(f"saved models to {ARTIFACTS_DIR}")

    if len(test_df) > 0:
        from models.evaluate import evaluate  # local import to avoid a cycle at module load

        evaluate(price_models, sell_model, test_df)


if __name__ == "__main__":
    main()
