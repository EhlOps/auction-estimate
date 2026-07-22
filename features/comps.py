"""Rolling median of comparable recent sales -- usually the single strongest predictor.

Computed strictly as-of each lot's sale_date (shifted, trailing window only) so no future
sale ever leaks into a lot's own comp figure. Falls back from the tightest comparable group
(family+generation+trim) to broader ones when there isn't enough recent history, which
matters most for the sparser BMW-wagon segment.

Both the training-time batch computation (`add_comp_features`) and the single-lot
prediction-time computation (`compute_comp_for_lot`) share the SAME window, min-periods,
and fallback tiers defined here, so the feature a model is trained on matches the feature
it is served -- avoiding train/serve skew on this dominant predictor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 10
DEFAULT_MIN_PERIODS = 3

# Fallback tiers, tightest comparable group first. Shared by both entry points below.
COMP_TIERS: list[tuple[str, list[str]]] = [
    ("tight", ["family", "generation", "trim"]),
    ("broad", ["family", "generation"]),
    ("family", ["family"]),
]


def _trailing_median(df: pd.DataFrame, group_cols: list[str], window: int, min_periods: int) -> pd.Series:
    def _comp(g: pd.DataFrame) -> pd.Series:
        return g["final_high_bid"].shift(1).rolling(window=window, min_periods=min_periods).median()

    return df.groupby(group_cols, group_keys=False).apply(_comp)


def add_comp_features(df: pd.DataFrame, window: int = DEFAULT_WINDOW, min_periods: int = DEFAULT_MIN_PERIODS) -> pd.DataFrame:
    df = df.sort_values("sale_date").reset_index(drop=True).copy()

    tier_cols = []
    for source, group_cols in COMP_TIERS:
        col = f"comp_median_{source}"
        df[col] = _trailing_median(df, group_cols, window, min_periods)
        tier_cols.append((source, col))

    df["comp_median"] = df[tier_cols[0][1]]
    for _, col in tier_cols[1:]:
        df["comp_median"] = df["comp_median"].fillna(df[col])

    df["comp_source"] = np.select(
        [df[col].notna() for _, col in tier_cols],
        [source for source, _ in tier_cols],
        default="none",
    )
    return df


def compute_comp_for_lot(
    history: pd.DataFrame,
    values: dict,
    as_of: pd.Timestamp,
    window: int = DEFAULT_WINDOW,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> tuple[float | None, str]:
    """Comp for a single hypothetical lot, matching `add_comp_features` exactly.

    Mirrors `.shift(1).rolling(window).median()`: takes the trailing `window` prior sales
    in the tightest group with at least `min_periods` of them, else falls back to broader
    tiers. `values` supplies the grouping keys (family/generation/trim).
    """
    prior = history[history["sale_date"] < as_of]
    for source, group_cols in COMP_TIERS:
        mask = pd.Series(True, index=prior.index)
        for col in group_cols:
            mask &= prior[col].astype(str) == str(values[col])
        recent = prior[mask].sort_values("sale_date")["final_high_bid"].tail(window)
        if recent.notna().sum() >= min_periods:
            return float(recent.median()), source
    return None, "none"
