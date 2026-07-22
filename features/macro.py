"""Pulls macro / market-uncertainty series from FRED and joins them onto listings by
sale date. Uses the plain fredgraph.csv endpoint (no API key required) since these are
all public series and this is periodic, low-volume, personal-research pulling."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scrapers.common import load_config

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fetch_fred_series(series_id: str, rename_to: str) -> pd.DataFrame:
    df = pd.read_csv(FRED_CSV_URL.format(series_id=series_id))
    df.columns = ["date", rename_to]
    df["date"] = pd.to_datetime(df["date"])
    df[rename_to] = pd.to_numeric(df[rename_to], errors="coerce")
    return df.dropna(subset=[rename_to])


def build_macro_table(macro_cfg: dict) -> pd.DataFrame:
    tables = []
    for series_id, name in macro_cfg["series"].items():
        try:
            tables.append(fetch_fred_series(series_id, name))
        except Exception as exc:  # noqa: BLE001 -- keep going if one series is unavailable
            print(f"warning: failed to fetch {series_id} ({name}): {exc}")

    merged = tables[0]
    for t in tables[1:]:
        merged = pd.merge(merged, t, on="date", how="outer")
    merged = merged.sort_values("date").ffill()

    out_path = Path(macro_cfg["cache_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path)
    return merged


def load_macro_table(macro_cfg: dict, refresh: bool = False) -> pd.DataFrame:
    out_path = Path(macro_cfg["cache_path"])
    if out_path.exists() and not refresh:
        return pd.read_parquet(out_path)
    return build_macro_table(macro_cfg)


def join_macro(df: pd.DataFrame, macro_table: pd.DataFrame) -> pd.DataFrame:
    """As-of join: each lot gets the most recent macro reading on or before its sale_date."""
    left = df.sort_values("sale_date").copy()
    right = macro_table.sort_values("date").copy()
    left["sale_date"] = left["sale_date"].astype("datetime64[ns]")
    right["date"] = right["date"].astype("datetime64[ns]")
    return pd.merge_asof(left, right, left_on="sale_date", right_on="date", direction="backward")


if __name__ == "__main__":
    cfg = load_config()
    table = load_macro_table(cfg["macro"], refresh=True)
    print(table.tail())
    print("wrote", cfg["macro"]["cache_path"])
