"""Shared feature-engineering + train/test split logic used by train.py, evaluate.py,
and predict.py, so all three stay consistent about what a "feature row" looks like."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.comps import add_comp_features
from features.macro import join_macro, load_macro_table
from scrapers.common import load_config

CATEGORICAL_COLS = [
    "platform", "family", "generation", "trim", "body_style",
    "transmission", "seller_type", "comp_source",
]
BOOL_COLS = ["modified_flag", "no_reserve", "special_edition", "has_inspection", "featured"]
NUMERIC_COLS = [
    "year", "car_age", "mileage", "comp_median",
    "n_comments", "n_views", "n_watchers",
    "vix", "consumer_sentiment", "sp500", "fed_funds_rate", "cpi_used_vehicles", "loan_delinquency_rate",
]
FEATURE_COLS = CATEGORICAL_COLS + BOOL_COLS + NUMERIC_COLS
PRICE_TARGET = "log_price"
SELL_TARGET = "reserve_met"


def build_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Adds comps + macro features and casts columns to model-ready dtypes."""
    cfg = cfg or load_config()
    df = add_comp_features(df)
    macro_table = load_macro_table(cfg["macro"])
    df = join_macro(df, macro_table)

    df["car_age"] = df["sale_date"].dt.year - df["year"]
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("boolean").astype("Int64")
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    return df


def time_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits by sale_date so test = most recent test_frac of lots (no shuffling)."""
    df = df.sort_values("sale_date").reset_index(drop=True)
    cutoff = int(len(df) * (1 - test_frac))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def load_feature_dataset(processed_dir: str | Path, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = Path(processed_dir) / "listings.parquet"
    df = pd.read_parquet(path)
    return build_features(df, cfg)
