#!/usr/bin/env python3
"""Runs the full scrape (all families x both platforms), then rebuilds the processed
dataset. This is the real, full-volume run -- expect it to take a while (BMW alone is
8000+ BaT listings before family filtering, and Cars & Bids is throttled by real
browser page loads). Safe to interrupt and re-run: both scrapers cache every listing
to disk and skip ids already fetched.

Usage:
    python scrape_all.py                 # everything
    python scrape_all.py --families mini # just one family
    python scrape_all.py --max-pages 5   # cap pages per family/platform (testing)
"""
from __future__ import annotations

import argparse

from features.parse import build_dataset
from scrapers import bat, carsandbids
from scrapers.common import load_config


def main() -> None:
    cfg = load_config()
    all_families = list(cfg["families"].keys())

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", nargs="+", default=all_families, choices=all_families)
    parser.add_argument("--platforms", nargs="+", default=["bat", "cnb"], choices=["bat", "cnb"])
    parser.add_argument("--max-pages", type=int, default=None, help="cap pages per family/platform, for testing")
    args = parser.parse_args()

    for family in args.families:
        family_cfg = cfg["families"][family]
        if "bat" in args.platforms:
            print(f"\n=== Bring a Trailer: {family} ===")
            recs = bat.scrape_family(family, family_cfg, cfg["scrape"], max_pages=args.max_pages)
            print(f"  {len(recs)} matching lots")
        if "cnb" in args.platforms:
            print(f"\n=== Cars & Bids: {family} ===")
            recs = carsandbids.scrape_family(family, family_cfg, cfg["scrape"], max_pages=args.max_pages)
            print(f"  {len(recs)} matching lots")

    print("\n=== Rebuilding processed dataset ===")
    df = build_dataset(cfg["scrape"]["raw_dir"])
    out_path = f"{cfg['scrape']['processed_dir']}/listings.parquet"
    df.to_parquet(out_path)
    print(f"wrote {len(df)} rows to {out_path}")
    print(df["family"].value_counts())


if __name__ == "__main__":
    main()
