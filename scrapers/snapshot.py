"""Records point-in-time snapshots of currently-live auctions (current bid, time
remaining, watchers) for every modeled family, on both platforms.

Every row in data/processed/listings.parquet is a *completed* auction's terminal state --
there has never been a mid-auction observation anywhere in this pipeline, so nothing has
ever let a model (or even a hand-fit curve) learn how a live bid + time-left relates to
the eventual sale price. `models/adjust.py` currently stands in with a hand-picked
time-decay curve for that reason.

Run this on a schedule (e.g. every 30-60 min) to build up a panel of
(mid-auction state -> eventual outcome) pairs. Once enough accrue, join them to
data/processed/listings.parquet on (platform, id) -- see `models/panel.py` -- to fit a
real decay curve or train a proper snapshot-aware model. Append-only; safe to re-run.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scrapers.bat import BASE as BAT_BASE
from scrapers.common import RateLimiter, get_with_retries, load_config, normalize_bid
from scrapers.listing import infer_family

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"


def _extract_current_auctions(html: str) -> dict:
    """Parses the `auctionsCurrentInitialData` blob on bringatrailer.com/auctions/ -- the
    live-auctions counterpart to bat.py's `_extract_initial_data`, which only looks for the
    completed-auctions marker."""
    idx = html.find("auctionsCurrentInitialData")
    if idx == -1:
        raise ValueError("Could not find auctionsCurrentInitialData on /auctions/")
    start = html.find("{", idx)
    end = html.find("/* ]]", idx)
    return json.loads(html[start:end].rstrip().rstrip(";"))


SNAPSHOT_COLS = [
    "platform", "family", "id", "ts", "current_bid", "ends_at", "hours_left",
    "no_reserve", "n_watchers",
]


def _title_filters(family_cfg: dict) -> tuple[re.Pattern | None, re.Pattern | None]:
    include = re.compile(family_cfg["title_include"], re.IGNORECASE) if family_cfg.get("title_include") else None
    exclude = re.compile(family_cfg["title_exclude"], re.IGNORECASE) if family_cfg.get("title_exclude") else None
    return include, exclude


def _matches_family(title: str, family_cfg: dict) -> bool:
    include, exclude = _title_filters(family_cfg)
    if include and not include.search(title):
        return False
    if exclude and exclude.search(title):
        return False
    return True


def _hours_left(ends_at: datetime | None, now: datetime) -> float | None:
    if ends_at is None:
        return None
    return max(0.0, (ends_at - now).total_seconds() / 3600)


def snapshot_bat(families: dict, scrape_cfg: dict, now: datetime) -> list[dict]:
    """Every currently-live BaT auction, across all categories, filtered to our families.

    Reuses the same `auctionsCurrentInitialData` feed already parsed in
    `scrapers/listing.py`'s `_bat_live_meta`, but here scanned wholesale instead of
    looked up by a single listing id.

    Unlike the completed-auction scraper, this feed spans every BaT category at once, so
    the per-family `title_include`/`title_exclude` config (written assuming an
    already-category-scoped page, which is why e.g. MINI's is `null`) can't be used here --
    it would let every listing through as "mini". `infer_family` (make+keyword based, used
    for single-URL extraction) works regardless of scoping, so it's reused here instead.
    """
    headers = {"User-Agent": scrape_cfg["user_agent"]}
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        resp = get_with_retries(client, f"{BAT_BASE}/auctions/")
    data = _extract_current_auctions(resp.text)

    rows = []
    for item in data.get("items", []):
        title = item.get("title", "")
        family_key = infer_family(title, title)
        if family_key is None or family_key not in families:
            continue
        ends_at = datetime.fromtimestamp(item["timestamp_end"], tz=timezone.utc) if item.get("timestamp_end") else None
        rows.append({
            "platform": "bat",
            "family": family_key,
            "id": str(item.get("id")),
            "ts": now,
            "current_bid": normalize_bid(item.get("current_bid")),
            "ends_at": ends_at,
            "hours_left": _hours_left(ends_at, now),
            "no_reserve": bool(item.get("noreserve")),
            "n_watchers": item.get("watchers"),
        })
    return rows


def snapshot_cnb(families: dict, scrape_cfg: dict, now: datetime) -> list[dict]:
    """Every currently-live C&B auction for each family's make, one page load per family.

    Navigating to /search/<make_slug> makes the site's own front-end fire a
    `sort=1&make_slug=...` request that returns only `status: "live"` auctions for that
    make -- same technique `scrapers/carsandbids.py` already uses (read what the browser
    legitimately requests, don't replay the site's signed API by hand).
    """
    from playwright.sync_api import sync_playwright

    rows = []
    limiter = RateLimiter(scrape_cfg["rate_limit_seconds"])
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=scrape_cfg["user_agent"])
        page = ctx.new_page()

        for family_key, family_cfg in families.items():
            captured: list[dict] = []

            def on_response(resp, _captured=captured):
                if "/v2/autos/auctions" in resp.url and "sort=1" in resp.url and "status=" not in resp.url:
                    try:
                        _captured.append(resp.json())
                    except Exception:
                        pass

            page.on("response", on_response)
            limiter.wait()
            page.goto(f"https://carsandbids.com/search/{family_cfg['cnb_make_slug']}",
                      timeout=30000, wait_until="networkidle")
            page.wait_for_timeout(1200)
            page.remove_listener("response", on_response)
            if not captured:
                continue

            for auction in captured[-1].get("auctions", []):
                haystack = f"{auction.get('title','')} {auction.get('sub_title','')}"
                if not _matches_family(haystack, family_cfg):
                    continue
                ends_at = None
                if auction.get("auction_end"):
                    ends_at = datetime.fromisoformat(auction["auction_end"].replace("Z", "+00:00"))
                rows.append({
                    "platform": "cnb",
                    "family": family_key,
                    "id": str(auction["id"]),
                    "ts": now,
                    "current_bid": normalize_bid(auction.get("current_bid")),
                    "ends_at": ends_at,
                    "hours_left": _hours_left(ends_at, now),
                    "no_reserve": bool(auction.get("no_reserve")),
                    "n_watchers": None,  # not present on the list payload, only per-lot detail
                })
        browser.close()
    return rows


def write_snapshots(platform: str, rows: list[dict]) -> Path:
    """Appends `rows` to data/snapshots/<platform>.parquet, deduped on (platform, id, ts)."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SNAPSHOT_DIR / f"{platform}.parquet"
    new_df = pd.DataFrame(rows, columns=SNAPSHOT_COLS)
    if out_path.exists():
        new_df = pd.concat([pd.read_parquet(out_path), new_df], ignore_index=True)
    new_df = new_df.drop_duplicates(subset=["platform", "id", "ts"]).reset_index(drop=True)
    new_df.to_parquet(out_path)
    return out_path


def run(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    now = datetime.now(timezone.utc)

    bat_rows = snapshot_bat(cfg["families"], cfg["scrape"], now)
    path = write_snapshots("bat", bat_rows)
    print(f"bat: {len(bat_rows)} live lots this run -> {path}")

    cnb_rows = snapshot_cnb(cfg["families"], cfg["scrape"], now)
    path = write_snapshots("cnb", cnb_rows)
    print(f"cnb: {len(cnb_rows)} live lots this run -> {path}")


if __name__ == "__main__":
    run()
