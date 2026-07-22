"""Shared scraping helpers: rate limiting, disk caching, retrying HTTP calls."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class RateLimiter:
    """Sleeps as needed so calls to `wait()` are spaced >= min_interval apart."""

    def __init__(self, min_interval_seconds: float):
        self.min_interval = min_interval_seconds
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


class DiskCache:
    """One JSON file per key under `root`. Never re-fetches a cached key."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> Any | None:
        p = self.path_for(key)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    def set(self, key: str, value: Any) -> None:
        with open(self.path_for(key), "w") as f:
            json.dump(value, f)

    def has(self, key: str) -> bool:
        return self.path_for(key).exists()


def get_with_retries(
    client: httpx.Client,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    max_retries: int = 4,
    timeout: float = 20.0,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_exc = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to GET {url} after {max_retries} attempts") from last_exc
