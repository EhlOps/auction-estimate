#!/usr/bin/env python3
"""CLI decision tool: predict what a car will sell for at auction.

Two ways to use it:

  # From a live listing URL (BaT or Cars & Bids) -- features are auto-extracted:
  predict.py --url https://carsandbids.com/auctions/XXXXXXXX
  predict.py --url https://bringatrailer.com/listing/some-car/

  # From manually entered details:
  predict.py --family mini --year 2020 --generation F55/F56 --trim jcw_gp \\
      --mileage 500 --transmission manual --no-reserve --current-bid 65000

Because `platform` is a model feature, the same car is scored for BOTH Bring a Trailer
and Cars & Bids, so you can see how much the venue itself moves the number.
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
from models.dataset import BOOL_COLS, CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS, load_feature_dataset
from models.train import ARTIFACTS_DIR
from scrapers.common import load_config

PLATFORM_LABELS = {"bat": "Bring a Trailer", "cnb": "Cars & Bids"}


def build_input_row(feat: dict, history: pd.DataFrame, cfg: dict, platform: str) -> pd.DataFrame:
    """Builds a single model-ready feature row from a feature dict (see feat keys below)."""
    as_of = pd.Timestamp(feat.get("sale_date") or date.today())
    comp_median, comp_source = compute_comp_for_lot(
        history,
        {"family": feat["family"], "generation": feat.get("generation"), "trim": feat.get("trim")},
        as_of,
    )

    row = {
        "platform": platform,
        "family": feat["family"],
        "generation": feat.get("generation"),
        "trim": feat.get("trim"),
        "body_style": feat.get("body_style"),
        "transmission": feat.get("transmission"),
        "seller_type": feat.get("seller_type"),
        "comp_source": comp_source,
        "modified_flag": int(bool(feat.get("modified_flag"))),
        "no_reserve": int(bool(feat.get("no_reserve"))),
        "special_edition": int(bool(feat.get("special_edition"))),
        "has_inspection": np.nan,
        "featured": np.nan,
        "year": feat.get("year"),
        "car_age": (as_of.year - feat["year"]) if feat.get("year") else np.nan,
        "mileage": feat.get("mileage"),
        "comp_median": comp_median,
        "n_comments": np.nan,
        "n_views": np.nan,
        "n_watchers": np.nan,
        "sale_date": as_of,
    }
    df = pd.DataFrame([row])
    df = join_macro(df, load_macro_table(cfg["macro"]))
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
    # Independently-trained quantile models can "cross" (esp. on thin segments) --
    # enforce p10 <= p50 <= p90 since a crossed interval isn't a meaningful range.
    p10, p50, p90 = sorted([p10, p50, p90])
    p_sell = float(sell_model.predict_proba(X)[0, 1])
    return {"p10": p10, "p50": p50, "p90": p90, "p_sell": p_sell}


def run_prediction(feat: dict, platforms: list[str], current_bid: float | None,
                   bid_platform: str | None, cfg: dict) -> None:
    if not (ARTIFACTS_DIR / "price_models.joblib").exists():
        print("No trained models found -- run `uv run python models/train.py` first.", file=sys.stderr)
        sys.exit(1)

    price_models = joblib.load(ARTIFACTS_DIR / "price_models.joblib")
    sell_model = joblib.load(ARTIFACTS_DIR / "sell_model.joblib")
    history = load_feature_dataset(cfg["scrape"]["processed_dir"], cfg)

    header = feat.get("title") or (
        f"{feat.get('year')} {feat['family']} {feat.get('generation') or ''} {feat.get('trim')}"
    )
    miles = f"{feat['mileage']:,} mi" if feat.get("mileage") else "mileage unknown"
    print(f"\n{header} -- {miles}, {feat.get('transmission') or 'transmission unknown'}\n")

    for platform in platforms:
        X = build_input_row(feat, history, cfg, platform)
        pred = predict_for_platform(price_models, sell_model, X)
        line = (
            f"  {PLATFORM_LABELS[platform]:16s}  P50 ${pred['p50']:>9,.0f}   "
            f"(P10 ${pred['p10']:>9,.0f} - P90 ${pred['p90']:>9,.0f})   "
            f"P(sells) {pred['p_sell']:.0%}"
        )
        if current_bid is not None:
            delta = pred["p50"] - current_bid
            verb = "under" if delta > 0 else "over"
            line += f"   vs bid ${current_bid:,.0f}: {verb} by ${abs(delta):,.0f}"
        print(line)

    if current_bid is not None and bid_platform:
        print(f"\n  (current bid ${current_bid:,.0f} observed on {PLATFORM_LABELS[bid_platform]})")
    print()


def feat_from_args(args: argparse.Namespace) -> dict:
    body_style = args.body_style or ("wagon" if args.family == "bmw_wagon" else "hatch")
    return {
        "family": args.family,
        "year": args.year,
        "generation": args.generation,
        "trim": args.trim,
        "body_style": body_style,
        "transmission": args.transmission,
        "mileage": args.mileage,
        "no_reserve": args.no_reserve,
        "modified_flag": args.modified,
        "special_edition": args.special_edition,
        "seller_type": args.seller_type,
        "sale_date": args.sale_date,
        "title": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=None, help="A live BaT or Cars & Bids listing URL (auto-extracts features)")
    parser.add_argument("--family", choices=["mini", "vw_golf", "bmw_wagon"])
    parser.add_argument("--year", type=int)
    parser.add_argument("--generation", default=None, help="e.g. F55/F56, Mk7, E39 -- see features/taxonomy.py")
    parser.add_argument("--trim", default="unknown")
    parser.add_argument("--body-style", default=None)
    parser.add_argument("--mileage", type=int)
    parser.add_argument("--transmission", choices=["manual", "automatic"])
    parser.add_argument("--seller-type", choices=["Dealer", "Private Party"], default=None)
    parser.add_argument("--no-reserve", action="store_true")
    parser.add_argument("--modified", action="store_true")
    parser.add_argument("--special-edition", action="store_true")
    parser.add_argument("--current-bid", type=float, default=None, help="for over/under comparison (manual mode)")
    parser.add_argument("--sale-date", default=None, help="YYYY-MM-DD, default today")
    parser.add_argument("--platforms", nargs="+", default=["bat", "cnb"], choices=["bat", "cnb"])
    args = parser.parse_args()

    cfg = load_config()

    if args.url:
        from scrapers.listing import UnsupportedCar, extract

        try:
            feat, current_bid, bid_platform = extract(args.url, cfg["scrape"])
        except UnsupportedCar as exc:
            print(f"That car ('{exc}') isn't in a family this model covers "
                  f"(MINI Cooper, VW Golf, or BMW wagon).", file=sys.stderr)
            sys.exit(1)
        run_prediction(feat, args.platforms, current_bid, bid_platform, cfg)
        return

    if args.family is None or args.year is None or args.mileage is None or args.transmission is None:
        parser.error("without --url you must provide --family, --year, --mileage, and --transmission")
    run_prediction(feat_from_args(args), args.platforms, args.current_bid, None, cfg)


if __name__ == "__main__":
    main()
