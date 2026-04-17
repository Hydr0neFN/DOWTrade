"""
PDF p.10 fixed-risk position sizing for MYM paper-trading bot.

Formula:
    contracts = floor( fixed_risk_usd / (|entry - stop| * point_value_usd) )

Cap at max_contracts. Skip (return 0 contracts) when stop distance is zero
or when the raw floor is < 1 (stop too wide for the risk budget).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.config import (
    FIXED_RISK_PER_TRADE_USD,
    MAX_OPEN_CONTRACTS,
    POINT_VALUE_USD,
)


@dataclass(frozen=True)
class SizingResult:
    contracts: int                    # 0 means skip the trade
    risk_usd: float                   # expected $ risk at this size
    stop_distance_points: float
    skip_reason: Optional[str]        # None when contracts > 0


def compute_size(
    entry: float,
    stop: float,
    *,
    fixed_risk_usd: float = FIXED_RISK_PER_TRADE_USD,
    point_value_usd: float = POINT_VALUE_USD,
    max_contracts: int = MAX_OPEN_CONTRACTS,
) -> SizingResult:
    """
    PDF p.10 fixed-risk unit sizing.

    contracts = floor( fixed_risk_usd / (|entry - stop| * point_value_usd) )

    - If |entry - stop| == 0          → skip ("zero stop distance").
    - If computed contracts < 1       → skip ("stop too wide for risk unit").
    - Cap at max_contracts.
    - entry, stop, fixed_risk_usd, point_value_usd must all be > 0.
    - max_contracts must be >= 1.
    """
    # ------------------------------------------------------------------ #
    # Validate inputs                                                      #
    # ------------------------------------------------------------------ #
    if entry <= 0:
        raise ValueError(f"entry must be positive, got {entry!r}")
    if stop <= 0:
        raise ValueError(f"stop must be positive, got {stop!r}")
    if fixed_risk_usd <= 0:
        raise ValueError(f"fixed_risk_usd must be positive, got {fixed_risk_usd!r}")
    if point_value_usd <= 0:
        raise ValueError(f"point_value_usd must be positive, got {point_value_usd!r}")
    if max_contracts < 1:
        raise ValueError(f"max_contracts must be >= 1, got {max_contracts!r}")

    # ------------------------------------------------------------------ #
    # Core sizing math                                                     #
    # ------------------------------------------------------------------ #
    stop_distance_points = abs(entry - stop)

    if stop_distance_points == 0.0:
        return SizingResult(
            contracts=0,
            risk_usd=0.0,
            stop_distance_points=0.0,
            skip_reason="zero stop distance",
        )

    risk_per_contract = stop_distance_points * point_value_usd
    raw = fixed_risk_usd / risk_per_contract
    floored = math.floor(raw)

    if floored < 1:
        return SizingResult(
            contracts=0,
            risk_usd=0.0,
            stop_distance_points=stop_distance_points,
            skip_reason="stop too wide for risk unit",
        )

    capped = min(floored, max_contracts)
    risk_usd = capped * risk_per_contract

    return SizingResult(
        contracts=capped,
        risk_usd=risk_usd,
        stop_distance_points=stop_distance_points,
        skip_reason=None,
    )
