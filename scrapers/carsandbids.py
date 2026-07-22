"""Cars & Bids completed-auction scraper.

C&B fronts its API with Cloudflare and signs every `/v2/autos/auctions` request with a
timestamp + HMAC computed by its own front-end JS. Rather than reverse-engineer and
replay that signing scheme outside a browser (which is exactly the anti-automation
mechanism it exists to defeat), this scraper drives a real headless browser through the
site's own paginated search UI -- i.e. it does what a person does when paging through
results (`carsandbids.com/search/<make>?page=N`, throttled), and simply reads the JSON
response that C&B's own client code legitimately requests and receives for that page.

List pages already carry mileage, transmission code, and sale price, so (unlike BaT) no
separate per-listing detail fetch is required for the core fields.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scrapers.common import DiskCache, RateLimiter, load_config

SEARCH_URL = "https://carsandbids.com/search/{make_slug}?page={page}"

# Observed transmission codes on C&B list payloads.
TRANSMISSION_CODES = {1: "automatic", 2: "manual"}

# A "sold_after" lot did NOT meet its reserve during the auction; it was sold in a
# post-auction private deal at a negotiated `sale_amount` that differs from the auction
# high bid. For our purposes that is a reserve-not-met outcome: the sell model should see
# reserve_met=False, and the price target should be the auction high bid (current_bid),
# not the off-auction negotiated price. So it is mapped alongside the other no-sale states.
STATUS_MAP = {
    "sold": "sold",
    "sold_after": "reserve_not_met",
    "reserve_not_met": "reserve_not_met",
    "no_sale": "reserve_not_met",
}


def _is_target_response(url: str) -> bool:
    return (
        "/v2/autos/auctions" in url
        and "status=closed" in url
        and "is_notable" not in url
    )


def iter_completed_pages(
    make_slug: str,
    *,
    per_page: int,
    rate_limiter: RateLimiter,
    user_agent: str,
    max_pages: int | None = None,
) -> Iterable[dict]:
    """Yields raw auction dicts for every completed (closed) auction for a C&B make."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=user_agent)
        page = ctx.new_page()

        page_num = 1
        total = None
        seen_ids: set[str] = set()
        while True:
            if max_pages is not None and page_num > max_pages:
                break

            captured: list[dict] = []

            def on_response(resp, _captured=captured):
                if _is_target_response(resp.url):
                    try:
                        _captured.append(resp.json())
                    except Exception:
                        pass

            page.on("response", on_response)
            rate_limiter.wait()
            page.goto(
                SEARCH_URL.format(make_slug=make_slug, page=page_num),
                timeout=30000,
                wait_until="networkidle",
            )
            page.wait_for_timeout(1200)
            page.remove_listener("response", on_response)

            if not captured:
                break
            body = captured[-1]
            total = body.get("total", total)
            auctions = body.get("auctions", [])
            new_auctions = [a for a in auctions if a["id"] not in seen_ids]
            if not new_auctions:
                break
            for a in new_auctions:
                seen_ids.add(a["id"])
                yield a

            if total is not None and len(seen_ids) >= total:
                break
            if page_num >= math.ceil((total or 0) / per_page) and total is not None:
                break
            page_num += 1

        browser.close()


TRANSMISSION_HINT_RE = re.compile(r"(\d)-Speed (Manual|Automatic)", re.IGNORECASE)


def normalize_item(raw: dict, family: str) -> dict:
    status = STATUS_MAP.get(raw.get("status"), "unknown")
    sale_price = raw.get("sale_amount") if status == "sold" else None
    transmission = TRANSMISSION_CODES.get(raw.get("transmission"))
    if transmission is None:
        hint = TRANSMISSION_HINT_RE.search(raw.get("sub_title") or "")
        if hint:
            transmission = hint.group(2).lower()

    return {
        "platform": "cnb",
        "family": family,
        "id": raw["id"],
        "title": raw.get("title"),
        "sub_title": raw.get("sub_title"),
        "url": f"https://carsandbids.com/auctions/{raw['id']}",
        "current_bid": raw.get("current_bid"),
        "status": status,
        "sale_price": sale_price,
        "sale_date": (raw.get("auction_end") or "")[:10] or None,
        "no_reserve": bool(raw.get("no_reserve")),
        "mileage_text": raw.get("mileage"),
        "transmission": transmission,
        "location": raw.get("location"),
        "has_inspection": bool(raw.get("has_inspection")),
        "featured": bool(raw.get("featured")),
    }


def scrape_family(family_key: str, family_cfg: dict, scrape_cfg: dict, *, max_pages: int | None = None) -> list[dict]:
    title_include = re.compile(family_cfg["title_include"], re.IGNORECASE) if family_cfg.get("title_include") else None
    title_exclude = re.compile(family_cfg["title_exclude"], re.IGNORECASE) if family_cfg.get("title_exclude") else None

    raw_dir = Path(scrape_cfg["raw_dir"]) / "cnb" / family_key
    cache = DiskCache(raw_dir)
    limiter = RateLimiter(scrape_cfg["rate_limit_seconds"])

    results = []
    for raw in iter_completed_pages(
        family_cfg["cnb_make_slug"],
        per_page=scrape_cfg["cnb_per_page"],
        rate_limiter=limiter,
        user_agent=scrape_cfg["user_agent"],
        max_pages=max_pages,
    ):
        haystack = f"{raw.get('title','')} {raw.get('sub_title','')}"
        if title_include and not title_include.search(haystack):
            continue
        if title_exclude and title_exclude.search(haystack):
            continue

        item = normalize_item(raw, family_key)
        key = str(item["id"])
        if not cache.has(key):
            cache.set(key, item)
        results.append(item)

    return results


if __name__ == "__main__":
    cfg = load_config()
    fam_key = sys.argv[1] if len(sys.argv) > 1 else "mini"
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    recs = scrape_family(fam_key, cfg["families"][fam_key], cfg["scrape"], max_pages=max_pages)
    print(f"scraped {len(recs)} {fam_key} records from Cars & Bids")
    if recs:
        print(json.dumps(recs[0], indent=2)[:1200])
