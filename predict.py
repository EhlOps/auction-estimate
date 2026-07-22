#!/usr/bin/env python3
"""CLI decision tool: given a car's features, predicts what it will bid to.

Since `platform` is a model feature, the SAME car can be scored for both Bring a
Trailer and Cars & Bids in one call -- that's the point of treating site as a
parameter rather than a fixed data-source label. Usage:

    predict.py --family mini --year 2020 --generation F55/F56 --trim jcw_gp \\
        --mileage 500 --transmission manual --no-reserve --current-bid 65000

    predict.py --family bmw_wagon --year 2003 --generation E39 --trim standard \\
        --mileage 120000 --transmission manual --platforms bat
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features.comps import compute_comp_for_lot
from features.macro import join_macro, load_macro_table
from models.dataset import BOOL_COLS, CATEGORICAL_COLS, FEATURE_COLS, load_feature_dataset
from models.train import ARTIFACTS_DIR
from scrapers.common import load_config


def build_input_row(args: argparse.Namespace, history: pd.DataFrame, cfg: dict, platform: str) -> pd.DataFrame:
    as_of = pd.Timestamp(args.sale_date or date.today())
    comp_median, comp_source = compute_comp_for_lot(
        history,
        {"family": args.family, "generation": args.generation, "trim": args.trim},
        as_of,
    )

    row = {
        "platform": platform,
        "family": args.family,
        "generation": args.generation,
        "trim": args.trim,
        "body_style": args.body_style,
        "transmission": args.transmission,
        "seller_type": args.seller_type,
        "comp_source": comp_source,
        "modified_flag": int(args.modified),
        "no_reserve": int(args.no_reserve),
        "special_edition": int(args.special_edition),
        "has_inspection": np.nan,
        "featured": np.nan,
        "year": args.year,
        "car_age": as_of.year - args.year,
        "mileage": args.mileage,
        "comp_median": comp_median,
        "n_comments": np.nan,
        "n_views": np.nan,
        "n_watchers": np.nan,
        "sale_date": as_of,
    }
    df = pd.DataFrame([row])

    macro_table = load_macro_table(cfg["macro"])
    df = join_macro(df, macro_table)

    from models.dataset import NUMERIC_COLS

    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")

    categories = joblib.load(ARTIFACTS_DIR / "categories.joblib")
    for col in CATEGORICAL_COLS:
        value = df[col].iloc[0]
        cats = categories.get(col, [])
        df[col] = pd.Categorical([value if value in cats else None], categories=cats)
    for col in BOOL_COLS:
        df[col] = df[col].astype("Int64")

    return df[FEATURE_COLS]


def predict_for_platform(price_models: dict, sell_model, X: pd.DataFrame) -> dict:
    p10 = float(np.exp(price_models["p10"].predict(X)[0]))
    p50 = float(np.exp(price_models["p50"].predict(X)[0]))
    p90 = float(np.exp(price_models["p90"].predict(X)[0]))
    # Independently-trained quantile models can "cross" (esp. on small samples) --
    # enforce p10 <= p50 <= p90 since a crossed interval isn't a meaningful range.
    p10, p50, p90 = sorted([p10, p50, p90])
    p_sell = float(sell_model.predict_proba(X)[0, 1])
    return {"p10": p10, "p50": p50, "p90": p90, "p_sell": p_sell}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", required=True, choices=["mini", "vw_golf", "bmw_wagon"])
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--generation", default=None, help="e.g. F55/F56, Mk7, E39 -- see features/taxonomy.py")
    parser.add_argument("--trim", default="unknown")
    parser.add_argument("--body-style", default=None)
    parser.add_argument("--mileage", type=int, required=True)
    parser.add_argument("--transmission", choices=["manual", "automatic"], required=True)
    parser.add_argument("--seller-type", choices=["Dealer", "Private Party"], default=None)
    parser.add_argument("--no-reserve", action="store_true")
    parser.add_argument("--modified", action="store_true")
    parser.add_argument("--special-edition", action="store_true")
    parser.add_argument("--current-bid", type=float, default=None, help="for over/under comparison")
    parser.add_argument("--sale-date", default=None, help="YYYY-MM-DD, default today")
    parser.add_argument("--platforms", nargs="+", default=["bat", "cnb"], choices=["bat", "cnb"])
    args = parser.parse_args()

    if not args.body_style:
        args.body_style = "wagon" if args.family == "bmw_wagon" else "hatch"

    cfg = load_config()
    artifacts = ARTIFACTS_DIR
    if not (artifacts / "price_models.joblib").exists():
        print("No trained models found -- run `python models/train.py` first.", file=sys.stderr)
        sys.exit(1)

    price_models = joblib.load(artifacts / "price_models.joblib")
    sell_model = joblib.load(artifacts / "sell_model.joblib")
    history = load_feature_dataset(cfg["scrape"]["processed_dir"], cfg)

    print(f"\n{args.year} {args.family} {args.generation or ''} {args.trim} -- {args.mileage:,} mi, {args.transmission}\n")

    for platform in args.platforms:
        X = build_input_row(args, history, cfg, platform)
        pred = predict_for_platform(price_models, sell_model, X)
        label = {"bat": "Bring a Trailer", "cnb": "Cars & Bids"}[platform]
        line = (
            f"  {label:16s}  P50 ${pred['p50']:>9,.0f}   "
            f"(P10 ${pred['p10']:>9,.0f} - P90 ${pred['p90']:>9,.0f})   "
            f"P(sells) {pred['p_sell']:.0%}"
        )
        if args.current_bid is not None:
            delta = pred["p50"] - args.current_bid
            verb = "under" if delta > 0 else "over"
            line += f"   vs current bid: {verb} by ${abs(delta):,.0f}"
        print(line)
    print()


if __name__ == "__main__":
    main()
