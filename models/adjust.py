"""Adjusts a static fair-value prediction using live auction state (current bid, time
remaining). The base price/sell models never see either signal -- every training row is a
*completed* auction with no mid-auction snapshot, so there is nothing to learn a live decay
curve from yet. This module is a closed-form stand-in until enough live snapshots
(`scrapers/snapshot.py`) accrue to fit one.

Two independent adjustments:
  - Floor: the final price can never be below the current high bid, so P10/P50/P90 are
    always clamped to >= current_bid. This is a correctness fix, not a heuristic.
  - Time-decay blend: only in the final `blend_window_hours` of an auction, the point
    estimate is pulled from the model's fair value toward the current bid as the close
    approaches -- outside that window the bid is still far from final and the fair-value
    estimate is left alone (beyond the floor).
"""
from __future__ import annotations


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def adjust_prediction(
    pred: dict,
    current_bid: float | None,
    hours_left: float | None,
    no_reserve: bool,
    cfg: dict,
) -> dict:
    """Returns a new pred dict (does not mutate `pred`) with p10/p50/p90/p_sell adjusted
    for live auction state. `cfg` is the `live_adjust` block of config.yaml."""
    p10, p50, p90, p_sell = pred["p10"], pred["p50"], pred["p90"], pred["p_sell"]

    if current_bid is not None:
        # Floor + fix crossing first, using the *original* p50 -- the blend below only ever
        # pulls p50 down, so applying it before the floor's sort would let a collapsed p50
        # get relabeled as p10 by sorted(), silently discarding the blend.
        p10, p50, p90 = sorted([max(current_bid, p10), max(current_bid, p50), max(current_bid, p90)])

        if hours_left is not None:
            window = cfg["blend_window_hours"]
            w = _clip01(hours_left / window) if window > 0 else 0.0
            p50 = max(current_bid, w * p50 + (1 - w) * current_bid)
            # p50 may now sit below the floored p10 (or above p90 if window <= 0 edge cases
            # ever put w oddly) -- pull the other two in to preserve p10 <= p50 <= p90
            # instead of re-sorting, which would relabel rather than converge.
            p10 = min(p10, p50)
            p90 = max(p90, p50)

        if current_bid >= p50 and not no_reserve:
            # Bid has already reached fair value -- strong evidence the reserve clears.
            p_sell = _clip01(p_sell + cfg["p_sell_bid_above_fair_boost"])

    if no_reserve:
        p_sell = 1.0

    return {"p10": p10, "p50": p50, "p90": p90, "p_sell": p_sell}
