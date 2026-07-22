"""Extract model features from a single live listing URL (BaT or Cars & Bids).

This is the glue that lets `predict.py --url <listing>` work: given one auction URL it
pulls the same fields the training pipeline uses (year/trim/generation, mileage,
transmission, reserve status, seller type) and the current bid, reusing the existing
listing parsers and the make/model/generation taxonomy so a live listing is featurized
exactly the way the trained data was.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.parse import MODIFIED_KEYWORDS_RE, parse_mileage, parse_transmission
from features.taxonomy import classify
from scrapers.bat import parse_listing_detail_html
from scrapers.carsandbids import TRANSMISSION_CODES
from scrapers.common import get_with_retries, load_config

BAT_AUCTIONS_FEED = "https://bringatrailer.com/auctions/"


class UnsupportedCar(ValueError):
    """Raised when a listing is not one of the families the model covers."""


def infer_family(make: str | None, title: str | None) -> str | None:
    """Maps a make + title to one of the three modeled families, or None if unsupported."""
    text = f"{make or ''} {title or ''}".lower()
    if "mini" in text:
        return "mini"
    if ("volkswagen" in text or "vw " in text) and re.search(r"\b(golf|gti|gli|r32)\b", text):
        return "vw_golf"
    if "bmw" in text and re.search(r"touring|sport wagon|wagon|estate", text):
        return "bmw_wagon"
    return None


def _meta_content(html: str, prop: str) -> str | None:
    m = re.search(rf'<meta property="{re.escape(prop)}" content="([^"]*)"', html)
    return m.group(1) if m else None


def _listing_id(html: str) -> str | None:
    m = re.search(r"post;stats;(\d+)", html)
    return m.group(1) if m else None


def _feat_from_taxonomy(family: str, title: str, sub_title: str) -> dict:
    tax = classify(family, title, sub_title)
    return {
        "family": family,
        "year": tax["year"],
        "generation": tax["generation"],
        "trim": tax["trim"],
        "body_style": tax["body_style"],
        "special_edition": tax["special_edition"],
        "title": title,
    }


# ---------------------------------------------------------------------------
# Bring a Trailer
# ---------------------------------------------------------------------------

def _bat_live_meta(client: httpx.Client, listing_id: str | None) -> tuple[float | None, bool | None]:
    """Looks up a live listing's current bid + reserve status from BaT's current-auctions
    feed (the `auctionsCurrentInitialData` blob on /auctions/). Returns (None, None) if the
    listing isn't currently live (e.g. it already ended)."""
    if not listing_id:
        return None, None
    import json

    resp = get_with_retries(client, BAT_AUCTIONS_FEED)
    html = resp.text
    idx = html.find("auctionsCurrentInitialData")
    if idx == -1:
        return None, None
    start = html.find("{", idx)
    end = html.find("/* ]]", idx)
    data = json.loads(html[start:end].rstrip().rstrip(";"))
    for item in data.get("items", []):
        if str(item.get("id")) == str(listing_id):
            return item.get("current_bid"), bool(item.get("noreserve"))
    return None, None


def extract_bat(url: str, scrape_cfg: dict) -> tuple[dict, float | None, str]:
    headers = {"User-Agent": scrape_cfg["user_agent"]}
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        resp = get_with_retries(client, url)
        html = resp.text
        detail = parse_listing_detail_html(html)
        title = _meta_content(html, "og:title") or ""
        excerpt = _meta_content(html, "og:description") or ""
        current_bid, no_reserve = _bat_live_meta(client, _listing_id(html))

    family = infer_family(title, title)
    if family is None:
        raise UnsupportedCar(title)

    bullets_text = " | ".join(detail.get("detail_bullets", []))
    if no_reserve is None:  # not live in the feed -- fall back to a page badge
        no_reserve = bool(re.search(r">\s*No Reserve\s*<", html))

    feat = _feat_from_taxonomy(family, title, "")
    feat.update(
        transmission=parse_transmission(bullets_text) or parse_transmission(title),
        mileage=parse_mileage(bullets_text) or parse_mileage(title),
        modified_flag=bool(MODIFIED_KEYWORDS_RE.search(f"{excerpt} {bullets_text}")),
        no_reserve=bool(no_reserve),
        seller_type=detail.get("seller_type"),
    )
    return feat, current_bid, "bat"


# ---------------------------------------------------------------------------
# Cars & Bids
# ---------------------------------------------------------------------------

def _cnb_fetch_detail(url: str, auction_id: str, user_agent: str) -> dict:
    """Drives a headless browser to a C&B auction page and captures the detail JSON the
    site's own front-end fetches (same approach as scrapers/carsandbids.py)."""
    from playwright.sync_api import sync_playwright

    box: dict = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=user_agent)
        page = ctx.new_page()

        def on_response(resp):
            if f"/v2/autos/auctions/{auction_id}" in resp.url.split("?")[0]:
                try:
                    box["data"] = resp.json()
                except Exception:
                    pass

        page.on("response", on_response)
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2000)
        browser.close()

    if "data" not in box:
        raise RuntimeError(f"Could not capture Cars & Bids auction data for {url}")
    return box["data"]


def extract_cnb(url: str, scrape_cfg: dict) -> tuple[dict, float | None, str]:
    m = re.search(r"/auctions/([A-Za-z0-9]+)", url)
    if not m:
        raise ValueError(f"Not a recognizable Cars & Bids auction URL: {url}")
    data = _cnb_fetch_detail(url, m.group(1), scrape_cfg["user_agent"])

    listing = data.get("listing", {})
    title = listing.get("title", "")
    sub_title = listing.get("sub_title", "") or ""

    family = infer_family(listing.get("make"), title)
    if family is None:
        raise UnsupportedCar(title)

    current_bid = (data.get("stats", {}).get("current_bid") or {}).get("amount")

    feat = _feat_from_taxonomy(family, title, sub_title)
    feat.update(
        transmission=TRANSMISSION_CODES.get(listing.get("transmission")),
        mileage=listing.get("mileage") or parse_mileage(listing.get("mileage_text", "") or ""),
        modified_flag=bool(MODIFIED_KEYWORDS_RE.search(sub_title)),
        no_reserve=bool(data.get("no_reserve")),
        # Training left C&B seller_type unset, so keep it None to match the model's categories.
        seller_type=None,
    )
    return feat, current_bid, "cnb"


def extract(url: str, scrape_cfg: dict | None = None) -> tuple[dict, float | None, str]:
    """Dispatches to the right extractor by domain. Returns (feature_dict, current_bid, platform)."""
    scrape_cfg = scrape_cfg or load_config()["scrape"]
    if "bringatrailer.com" in url:
        return extract_bat(url, scrape_cfg)
    if "carsandbids.com" in url:
        return extract_cnb(url, scrape_cfg)
    raise ValueError(f"Unrecognized listing URL (expected bringatrailer.com or carsandbids.com): {url}")


if __name__ == "__main__":
    feat, bid, platform = extract(sys.argv[1])
    print(f"platform={platform} current_bid={bid}")
    for k, v in feat.items():
        print(f"  {k}: {v}")
