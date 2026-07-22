"""Bring a Trailer completed-auction scraper.

BaT's category archive pages (bringatrailer.com/<make>/) embed a JSON blob
(`auctionsCompletedInitialData`) with the first page of completed listings plus a
`base_filter` describing that category. The same data, paginated, is available with no
authentication from the site's own `listings-filter` REST endpoint by replaying that
base_filter with `get_items=true&page=N` -- this is the same endpoint the "Show More"
button on the site calls, just exercised directly over plain HTTP instead of via
in-browser AJAX. No login, cookies, or bypassing of any protection is involved: this is
a public GET on public completed-auction data.

Per-listing detail pages are then fetched (rate limited, cached) to pull the free-text
"Listing Details" spec bullets (VIN/chassis, mileage, transmission, trim, options).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scrapers.common import DiskCache, RateLimiter, get_with_retries, load_config

BASE = "https://bringatrailer.com"
LISTINGS_FILTER_URL = f"{BASE}/wp-json/bringatrailer/1.0/data/listings-filter"

SOLD_TEXT_RE = re.compile(
    r"(Sold for|Bid to)\s+(?:USD\s+)?\$([\d,]+).*?on\s+(\d{1,2}/\d{1,2}/\d{4})",
    re.IGNORECASE,
)


def _extract_initial_data(html: str) -> dict:
    idx = html.find("auctionsCompletedInitialData")
    if idx == -1:
        raise ValueError("Could not find auctionsCompletedInitialData on category page")
    start = html.find("{", idx)
    end = html.find("/* ]]", idx)
    blob = html[start:end].rstrip().rstrip(";").rstrip()
    return json.loads(blob)


def _base_filter_params(base_filter: dict) -> list[tuple[str, str]]:
    """Serializes BaT's base_filter dict into PHP-bracket-style query params."""
    params: list[tuple[str, str]] = []
    for key, value in base_filter.items():
        if isinstance(value, list):
            for v in value:
                params.append((f"base_filter[{key}][]", str(v)))
        elif value is not None:
            params.append((f"base_filter[{key}]", str(value)))
    return params


def fetch_category_meta(client: httpx.Client, category_slug: str) -> dict:
    """Fetches the category page and returns its base_filter + pagination totals."""
    resp = get_with_retries(client, f"{BASE}/{category_slug}/")
    return _extract_initial_data(resp.text)


def iter_completed_items(
    client: httpx.Client,
    category_slug: str,
    *,
    per_page: int,
    rate_limiter: RateLimiter,
    max_pages: int | None = None,
) -> Iterable[dict]:
    """Yields raw list-level item dicts for every completed auction in a BaT category.

    Every page -- including the first -- is fetched from the REST endpoint at the SAME
    `per_page`, rather than seeding page 1 from the category page's embedded `items` blob.
    The embedded blob always holds only `items_per_page` (24) rows, so mixing it with a
    larger REST `per_page` offset grid would silently skip the rows between the embedded
    count and the first REST offset. `pages_total` is recomputed for our `per_page` since
    the value on the category page is tied to the embedded page size.
    """
    meta = fetch_category_meta(client, category_slug)
    base_filter = meta["base_filter"]
    params_base = _base_filter_params(base_filter)

    items_total = meta.get("items_total", 0)
    pages_total = math.ceil(items_total / per_page) if items_total else 1
    if max_pages is not None:
        pages_total = min(pages_total, max_pages)

    for page in range(1, pages_total + 1):
        rate_limiter.wait()
        params = params_base + [("page", str(page)), ("per_page", str(per_page)), ("get_items", "true")]
        resp = get_with_retries(client, LISTINGS_FILTER_URL, params=params)
        data = resp.json()
        items = data.get("items", [])
        if not items:
            break
        yield from items


def parse_sold_text(sold_text: str) -> tuple[str, int | None, str | None]:
    """Returns (status, price, date_iso) from BaT's 'Sold for ...' / 'Bid to ...' string."""
    match = SOLD_TEXT_RE.search(sold_text or "")
    if not match:
        return "unknown", None, None
    verb, amount, date_str = match.groups()
    status = "sold" if verb.lower() == "sold for" else "reserve_not_met"
    price = int(amount.replace(",", ""))
    month, day, year = date_str.split("/")
    date_iso = f"{year}-{int(month):02d}-{int(day):02d}"
    return status, price, date_iso


DETAIL_LIST_RE = re.compile(r"<strong>Listing Details</strong><ul>(.*?)</ul>", re.DOTALL)
LI_RE = re.compile(r"<li>(.*?)</li>", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
LOT_RE = re.compile(r"Lot</strong>\s*#(\d+)")
SELLER_TYPE_RE = re.compile(r"Private Party or Dealer</strong>:\s*([A-Za-z ]+)")
LOCATION_RE = re.compile(r'Location</strong>:\s*<a[^>]*>([^<]+)</a>')


def parse_listing_detail_html(html: str) -> dict:
    out: dict[str, Any] = {"detail_bullets": [], "lot_number": None, "seller_type": None, "location": None}
    m = DETAIL_LIST_RE.search(html)
    if m:
        bullets = [TAG_RE.sub("", li).strip() for li in LI_RE.findall(m.group(1))]
        out["detail_bullets"] = [b for b in bullets if b]
    lot_m = LOT_RE.search(html)
    if lot_m:
        out["lot_number"] = lot_m.group(1)
    seller_m = SELLER_TYPE_RE.search(html)
    if seller_m:
        out["seller_type"] = seller_m.group(1).strip()
    loc_m = LOCATION_RE.search(html)
    if loc_m:
        out["location"] = loc_m.group(1).strip()
    return out


def fetch_listing_detail(client: httpx.Client, url: str) -> dict:
    resp = get_with_retries(client, url)
    return parse_listing_detail_html(resp.text)


def normalize_list_item(raw: dict, family: str) -> dict:
    status, price, date_iso = parse_sold_text(raw.get("sold_text", ""))
    return {
        "platform": "bat",
        "family": family,
        "id": raw["id"],
        "title": raw.get("title"),
        "url": raw.get("url"),
        "current_bid": raw.get("current_bid"),
        "status": status,
        "sale_price": price,
        "sale_date": date_iso,
        "no_reserve": bool(raw.get("noreserve")),
        "comments": int(raw.get("comments") or 0),
        "views": raw.get("views"),
        "watchers": raw.get("watchers"),
        "excerpt": raw.get("excerpt"),
    }


def scrape_family(family_key: str, family_cfg: dict, scrape_cfg: dict, *, max_pages: int | None = None) -> list[dict]:
    """Scrapes one family end to end: list pages -> title filter -> per-listing detail.

    Returns the list of normalized+detail-enriched records and also writes each to
    `data/raw/bat/<family_key>/<id>.json` (skipping ids already cached).
    """
    title_include = re.compile(family_cfg["title_include"], re.IGNORECASE) if family_cfg.get("title_include") else None
    title_exclude = re.compile(family_cfg["title_exclude"], re.IGNORECASE) if family_cfg.get("title_exclude") else None

    raw_dir = Path(scrape_cfg["raw_dir"]) / "bat" / family_key
    cache = DiskCache(raw_dir)
    limiter = RateLimiter(scrape_cfg["rate_limit_seconds"])
    headers = {"User-Agent": scrape_cfg["user_agent"]}

    results = []
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for raw in iter_completed_items(
            client,
            family_cfg["bat_category"],
            per_page=scrape_cfg["bat_per_page"],
            rate_limiter=limiter,
            max_pages=max_pages,
        ):
            title = raw.get("title", "")
            if title_include and not title_include.search(title):
                continue
            if title_exclude and title_exclude.search(title):
                continue

            item = normalize_list_item(raw, family_key)
            key = str(item["id"])
            cached = cache.get(key)
            if cached is not None:
                results.append(cached)
                continue

            limiter.wait()
            try:
                detail = fetch_listing_detail(client, item["url"])
            except RuntimeError:
                detail = {}
            item.update(detail)
            cache.set(key, item)
            results.append(item)

    return results


if __name__ == "__main__":
    cfg = load_config()
    fam_key = sys.argv[1] if len(sys.argv) > 1 else "mini"
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    recs = scrape_family(fam_key, cfg["families"][fam_key], cfg["scrape"], max_pages=max_pages)
    print(f"scraped {len(recs)} {fam_key} records from BaT")
    if recs:
        print(json.dumps(recs[0], indent=2)[:1500])
