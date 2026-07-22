"""Turns raw per-listing JSON (from data/raw/<platform>/<family>/*.json) into one unified
feature table -- one row per lot, same schema regardless of source platform."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.taxonomy import classify

MILEAGE_RE = re.compile(r"([\d,]+)\s*(k)?[\s-]*Miles?", re.IGNORECASE)
TRANSMISSION_TEXT_RE = re.compile(r"\b(manual|automatic|dct|dual-clutch|dsg)\b", re.IGNORECASE)
MODIFIED_KEYWORDS_RE = re.compile(
    r"\bmodifi|aftermarket|coilovers?|turbo kit|supercharger kit|stage [123]|tuned|built engine|custom\b",
    re.IGNORECASE,
)
INT_PREFIX_RE = re.compile(r"([\d,]+)")


def parse_mileage(text: str) -> int | None:
    m = MILEAGE_RE.search(text or "")
    if not m:
        return None
    num = int(m.group(1).replace(",", ""))
    return num * 1000 if m.group(2) else num


def parse_int_prefix(text) -> int | None:
    if text is None:
        return None
    m = INT_PREFIX_RE.search(str(text))
    return int(m.group(1).replace(",", "")) if m else None


def parse_transmission(text: str) -> str | None:
    m = TRANSMISSION_TEXT_RE.search(text or "")
    if not m:
        return None
    val = m.group(1).lower()
    if val in ("dct", "dual-clutch", "dsg"):
        return "automatic"  # paddle-shift auto for our purposes, not a torque-converter auto
    return val


def normalize_bat_record(rec: dict) -> dict:
    bullets_text = " | ".join(rec.get("detail_bullets", []))
    mileage = parse_mileage(bullets_text) or parse_mileage(rec.get("title", ""))
    transmission = parse_transmission(bullets_text) or parse_transmission(rec.get("title", ""))
    modified = bool(MODIFIED_KEYWORDS_RE.search(f"{rec.get('excerpt','')} {bullets_text}"))
    final_bid = rec["sale_price"] if rec["status"] == "sold" else rec.get("current_bid")

    return {
        "platform": "bat",
        "family": rec["family"],
        "id": rec["id"],
        "title": rec.get("title"),
        "url": rec.get("url"),
        "sale_date": rec.get("sale_date"),
        "status": rec.get("status"),
        "reserve_met": rec.get("status") == "sold",
        "final_high_bid": final_bid,
        "no_reserve": rec.get("no_reserve"),
        "mileage": mileage,
        "transmission": transmission,
        "modified_flag": modified,
        "seller_type": rec.get("seller_type"),
        "location": rec.get("location"),
        "n_comments": parse_int_prefix(rec.get("comments")),
        "n_views": parse_int_prefix(rec.get("views")),
        "n_watchers": parse_int_prefix(rec.get("watchers")),
        "has_inspection": None,
        "featured": None,
    }


def normalize_cnb_record(rec: dict) -> dict:
    text = f"{rec.get('title','')} {rec.get('sub_title','')}"
    mileage = parse_mileage(rec.get("mileage_text", "") or "")
    modified = bool(MODIFIED_KEYWORDS_RE.search(text))
    final_bid = rec["sale_price"] if rec["status"] == "sold" else rec.get("current_bid")

    return {
        "platform": "cnb",
        "family": rec["family"],
        "id": rec["id"],
        "title": rec.get("title"),
        "url": rec.get("url"),
        "sale_date": rec.get("sale_date"),
        "status": rec.get("status"),
        "reserve_met": rec.get("status") == "sold",
        "final_high_bid": final_bid,
        "no_reserve": rec.get("no_reserve"),
        "mileage": mileage,
        "transmission": rec.get("transmission"),
        "modified_flag": modified,
        "seller_type": None,
        "location": rec.get("location"),
        "n_comments": None,
        "n_views": None,
        "n_watchers": None,
        "has_inspection": rec.get("has_inspection"),
        "featured": rec.get("featured"),
    }


NORMALIZERS = {"bat": normalize_bat_record, "cnb": normalize_cnb_record}


def load_raw_records(raw_dir: Path) -> list[dict]:
    records = []
    for platform_dir in raw_dir.iterdir():
        if not platform_dir.is_dir() or platform_dir.name not in NORMALIZERS:
            continue
        for family_dir in platform_dir.iterdir():
            if not family_dir.is_dir():
                continue
            for f in family_dir.glob("*.json"):
                with open(f) as fh:
                    records.append(json.load(fh))
    return records


def build_dataset(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    rows = []
    for rec in load_raw_records(raw_dir):
        normalizer = NORMALIZERS[rec["platform"]]
        row = normalizer(rec)
        tax = classify(row["family"], row["title"] or "", rec.get("sub_title", "") if rec["platform"] == "cnb" else "")
        row.update(tax)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["id"] = df["id"].astype(str)
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df = df.dropna(subset=["final_high_bid", "sale_date"])
    df = df.drop_duplicates(subset=["platform", "id"])
    df["log_price"] = df["final_high_bid"].apply(lambda x: None if x is None or x <= 0 else __import__("math").log(x))
    return df.sort_values("sale_date").reset_index(drop=True)


if __name__ == "__main__":
    from scrapers.common import load_config

    cfg = load_config()
    df = build_dataset(cfg["scrape"]["raw_dir"])
    print(df.shape)
    if not df.empty:
        print(df[["platform", "family", "year", "generation", "trim", "mileage", "transmission", "reserve_met", "final_high_bid"]].head(20))
        out_path = Path(cfg["scrape"]["processed_dir"]) / "listings.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        print("wrote", out_path)
