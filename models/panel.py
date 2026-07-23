"""Joins live-auction snapshots (`scrapers/snapshot.py`) to their eventual outcomes, once
those lots complete -- the panel this project needs before a real bid/time-aware model (or
even a fitted decay curve, replacing the hand-picked one in `models/adjust.py`) is possible.

Not enough snapshot history has accrued yet for that fit/retrain; this module only defines
and documents the join.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scrapers.snapshot import SNAPSHOT_DIR


def load_snapshots(snapshot_dir: Path | str = SNAPSHOT_DIR) -> pd.DataFrame:
    snapshot_dir = Path(snapshot_dir)
    frames = [pd.read_parquet(f) for f in snapshot_dir.glob("*.parquet")]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_panel(snapshots: pd.DataFrame, completed: pd.DataFrame) -> pd.DataFrame:
    """Inner-joins snapshot rows to completed-auction outcomes on (platform, id).

    A snapshot row only appears in the result once its lot has actually completed --
    still-live lots are dropped here (there is no outcome for them yet). Returns one row
    per (lot, snapshot timestamp) with the snapshot's mid-auction state alongside the
    lot's final `final_high_bid` / `reserve_met`.
    """
    if snapshots.empty or completed.empty:
        return pd.DataFrame()
    outcomes = completed[["platform", "id", "final_high_bid", "reserve_met", "sale_date"]]
    return snapshots.merge(outcomes, on=["platform", "id"], how="inner")


if __name__ == "__main__":
    from features.parse import build_dataset
    from scrapers.common import load_config

    cfg = load_config()
    snapshots = load_snapshots()
    completed = build_dataset(cfg["scrape"]["raw_dir"])
    panel = build_panel(snapshots, completed)
    print(f"{len(snapshots)} snapshot rows, {len(completed)} completed lots -> {len(panel)} joined panel rows")
