# Auction Estimate

Predicts what an enthusiast car will sell for at auction on **Bring a Trailer (BaT)** and
**Cars & Bids (C&B)**, covering three families:

- **MINI Cooper** -- all generations (R50/R53 → F55/F56/F57), One/Cooper/Cooper S/JCW, incl. JCW GP.
- **VW Golf** -- GTI / R / R32 / GLI, Mk4-Mk8.
- **BMW wagons** -- 3- and 5-series Touring, E36 through G21, incl. Alpina/M variants.

It's a two-part model: a **price model** (quantile regression → P10/P50/P90 range) trained on
every completed lot's final high bid (sold or not), and a **sell model** (probability the
reserve is met). **Platform (BaT vs. C&B) is a model feature**, so the same car can be scored
for both sites -- the two auction houses genuinely price the same cars differently.

## Setup

This project uses [uv](https://docs.astral.sh/uv/). `uv run` auto-syncs the environment
from `pyproject.toml` / `uv.lock` on first use, so you don't need to create a venv manually.

```bash
uv sync                                   # install dependencies into .venv
uv run playwright install chromium        # browser for the C&B scraper only (one-time)
```

## Pipeline

```bash
# 1. Scrape (safe to interrupt/resume -- every listing is cached to data/raw/ by id)
uv run python scrape_all.py                    # full run, all families/platforms
uv run python scrape_all.py --families mini    # just one family
uv run python scrape_all.py --max-pages 3       # small test run

# 2. Train (rebuilds comps + macro features, trains price + sell models)
uv run python models/train.py

# 3. Predict

# (a) From a live listing URL -- features are auto-extracted from BaT or Cars & Bids:
uv run python predict.py --url https://carsandbids.com/auctions/XXXXXXXX
uv run python predict.py --url https://bringatrailer.com/listing/some-car/

# (b) From manually entered details:
uv run python predict.py --family mini --year 2020 --generation F55/F56 --trim jcw_gp \
    --mileage 500 --transmission manual --no-reserve --special-edition --current-bid 65000
```

Either way it prints a predicted sale price (P50) with a P10-P90 range and sell probability
for **both** BaT and Cars & Bids, plus how that compares to the current bid. URL mode only
works for the three modeled families (MINI Cooper, VW Golf, BMW wagon) and reuses the same
listing parsers (`scrapers/listing.py`) and taxonomy the training data was built with, so a
live car is featurized exactly like the training set. C&B URLs drive a headless browser
(needs the one-time `playwright install chromium`); BaT URLs are plain HTTP.

`predict.py --platforms bat cnb` (the default) scores the same car for both sites so you can
see how much the venue itself moves the number; pass `--platforms bat` to only price one.

## How the data is collected

- **BaT**: category archive pages (`bringatrailer.com/<make>/`) embed the first page of
  completed listings as JSON. Further pages come from BaT's own `listings-filter` REST
  endpoint (`get_items=true`) -- the same endpoint the site's "Show More" button calls,
  hit directly over plain HTTP with no login or cookies. Per-listing detail pages (also
  plain HTML, no JS needed) are then fetched for mileage/transmission/trim/VIN, rate
  limited and cached so nothing is re-fetched across runs.
- **Cars & Bids**: C&B fronts its API behind Cloudflare and signs every request
  (timestamp + HMAC computed client-side). Rather than reverse-engineer and replay that
  signature outside a browser, the scraper drives a real headless browser through the
  site's own paginated search (`carsandbids.com/search/<make>?page=N`, throttled) and
  reads the JSON response the site's own front-end code legitimately requests for that
  page -- i.e. it pages through results the way a person does, just automated and rate
  limited.
- Both scrapers respect a configurable delay between requests (`config.yaml:
  scrape.rate_limit_seconds`) and this is intended for personal, low-volume research use.

## Vehicle scope / family filtering (`config.yaml`)

MINI's BaT/C&B category *is* the family, so nothing extra is filtered. VW Golf and BMW
wagon are matched by a title/sub-title regex on top of the broader Volkswagen/BMW
category (`title_include` / `title_exclude` in `config.yaml`) -- adjust these if you want
to loosen or tighten scope (e.g. include Jetta GLI, exclude a specific edition).

## Feature engineering

- `features/taxonomy.py` -- canonicalizes messy titles into make/model/generation/trim
  (heuristic: explicit chassis-code mention if present, else body-style + model-year
  lookup table). Not a VIN decoder -- good enough for modeling, not authoritative.
- `features/comps.py` -- rolling median of comparable recent sales, computed strictly
  as-of each lot's sale date (no future leakage), falling back from
  family+generation+trim → family+generation → family when there isn't enough recent
  history in the tightest group. This matters most for the sparse BMW-wagon segment.
- `features/macro.py` -- joins FRED series onto each lot by sale date: VIX and
  U.Michigan Consumer Sentiment (market uncertainty), S&P 500 and Fed Funds rate
  (market/economic conditions), used-vehicle CPI (long-run trend proxy), consumer-loan
  delinquency rate (credit-stress signal). All public, no API key required.

## Known limitations / next steps

- **This repo currently only has a small smoke-test sample** (~130 rows, 1-2 pages per
  family/platform) so you can exercise the whole pipeline end to end. Run
  `scrape_all.py` with no `--max-pages` cap for a real training set before trusting any
  prediction -- `models/evaluate.py` will print MAPE/coverage per family so you can see
  when it's ready.
- Quantile models are trained independently per quantile, which can occasionally
  "cross" (P10 > P90) on thin data; `predict.py` sorts the three outputs defensively,
  but a crossed/degenerate prediction is a sign that segment needs more training data,
  not a number to trust.
- `predict.py` takes manual feature entry, not a live listing URL. A given car's
  current bid-momentum signals (views/watchers/comments) aren't knowable ahead of time
  for a hypothetical entry, so those are left missing (LightGBM handles this natively,
  but it does mean the model can't use momentum for a not-yet-run auction).
- BMW wagons will likely remain the thinnest segment even at full scrape volume --
  pooling with MINI/Golf regularizes it, but treat its predictions as wider/rougher
  ranges rather than precise points.
