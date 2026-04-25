"""
Feature extraction: swing-point detection (PDF p.8 HH/HL rules),
ATR-14 Wilder, SMA-200, Donchian-20, and MarketSnapshot assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.data.bars import Bar


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SwingPoint:
    ts: int
    price: float
    kind: Literal["HH", "HL", "LH", "LL"]


@dataclass(frozen=True)
class MarketSnapshot:
    bars: List[Bar]
    swings: List[SwingPoint]
    atr14: Optional[float]
    sma200: Optional[float]
    donchian_upper20: Optional[float]
    donchian_lower20: Optional[float]
    current_price: float
    current_ts: int


# ---------------------------------------------------------------------------
# Swing detection (PDF p.8)
# ---------------------------------------------------------------------------

def _find_pivot_highs(bars: List[Bar], k: int) -> List[int]:
    """Pivot-high indices via vectorized sliding-window comparison.

    A bar at index i is a pivot-high iff bars[i].h is strictly greater than
    bars[i-k..i-1].h AND bars[i+1..i+k].h.  Equivalent to the prior loop;
    ~10x faster on 500-bar windows via numpy stride tricks.
    """
    n = len(bars)
    if n < 2 * k + 1:
        return []
    h = np.fromiter((b.h for b in bars), dtype=np.float64, count=n)
    win = sliding_window_view(h, 2 * k + 1)
    centers = win[:, k]
    left_max = win[:, :k].max(axis=1)
    right_max = win[:, k + 1:].max(axis=1)
    mask = (centers > left_max) & (centers > right_max)
    return (np.flatnonzero(mask) + k).tolist()


def _find_pivot_lows(bars: List[Bar], k: int) -> List[int]:
    """Pivot-low indices via vectorized sliding-window comparison."""
    n = len(bars)
    if n < 2 * k + 1:
        return []
    lo = np.fromiter((b.l for b in bars), dtype=np.float64, count=n)
    win = sliding_window_view(lo, 2 * k + 1)
    centers = win[:, k]
    left_min = win[:, :k].min(axis=1)
    right_min = win[:, k + 1:].min(axis=1)
    mask = (centers < left_min) & (centers < right_min)
    return (np.flatnonzero(mask) + k).tolist()


def detect_swings(bars: List[Bar], pivot_k: int = 2) -> List[SwingPoint]:
    """Return chronological list of CONFIRMED swing points only.

    Algorithm (PDF p.8):
    1. Detect pivot highs/lows with k-bar fractal.
    2. Walk candidates chronologically; track pending_high / pending_low
       and last confirmed high / low levels.
    3. A pivot-high becomes pending; confirmed only when a subsequent bar
       closes ABOVE the pending high price.
    4. Same logic for pivot-lows (close BELOW pending low price).
    5. Between consecutive confirmed highs, emit the bar with the minimum
       low as the intervening HL/LL swing point.
    """
    if len(bars) < 2 * pivot_k + 1:
        return []

    pivot_high_idx = set(_find_pivot_highs(bars, pivot_k))
    pivot_low_idx = set(_find_pivot_lows(bars, pivot_k))

    candidate_map: dict[int, List[str]] = {}
    for idx in pivot_high_idx:
        candidate_map.setdefault(idx, []).append("pivot_high")
    for idx in pivot_low_idx:
        candidate_map.setdefault(idx, []).append("pivot_low")

    confirmed: List[SwingPoint] = []

    last_conf_high: Optional[float] = None
    last_conf_high_bar_idx: Optional[int] = None
    last_conf_low: Optional[float] = None

    # (bar_idx, price, ts)
    pending_high: Optional[Tuple[int, float, int]] = None
    pending_low: Optional[Tuple[int, float, int]] = None

    for i, bar in enumerate(bars):
        # --- Confirmation checks ---
        if pending_high is not None:
            ph_idx, ph_price, ph_ts = pending_high
            if i > ph_idx and bar.c > ph_price:
                if last_conf_high is None or ph_price > last_conf_high:
                    kind_h: Literal["HH", "LH"] = "HH"
                else:
                    kind_h = "LH"
                _maybe_add_intervening_low(
                    bars, confirmed,
                    last_conf_high_bar_idx, ph_idx,
                    last_conf_low,
                )
                # Keep last_conf_low updated to the deepest point seen
                if last_conf_high_bar_idx is not None:
                    seg_lo = min(bars[j].l for j in range(last_conf_high_bar_idx + 1, ph_idx + 1))
                    if last_conf_low is None or seg_lo < last_conf_low:
                        last_conf_low = seg_lo
                confirmed.append(SwingPoint(ts=ph_ts, price=ph_price, kind=kind_h))
                last_conf_high = ph_price
                last_conf_high_bar_idx = ph_idx
                pending_high = None

        if pending_low is not None:
            pl_idx, pl_price, pl_ts = pending_low
            if i > pl_idx and bar.c < pl_price:
                if last_conf_low is None or pl_price > last_conf_low:
                    kind_l: Literal["HL", "LL"] = "HL"
                else:
                    kind_l = "LL"
                confirmed.append(SwingPoint(ts=pl_ts, price=pl_price, kind=kind_l))
                last_conf_low = pl_price
                pending_low = None

        # --- New candidate pivots at this bar ---
        for kind in candidate_map.get(i, []):
            if kind == "pivot_high":
                ph_price_new = bars[i].h
                if pending_high is None or ph_price_new >= pending_high[1]:
                    pending_high = (i, ph_price_new, bars[i].ts)
            elif kind == "pivot_low":
                pl_price_new = bars[i].l
                if pending_low is None or pl_price_new <= pending_low[1]:
                    pending_low = (i, pl_price_new, bars[i].ts)

    confirmed.sort(key=lambda sp: sp.ts)
    return confirmed


def _maybe_add_intervening_low(
    bars: List[Bar],
    confirmed: List[SwingPoint],
    prev_high_idx: Optional[int],
    curr_high_idx: int,
    last_conf_low: Optional[float],
) -> None:
    """Between two confirmed highs, emit the bar with the minimum low as HL/LL
    if no confirmed low already covers that interval."""
    if prev_high_idx is None:
        return
    start = prev_high_idx + 1
    end = curr_high_idx
    if start >= end:
        return

    t_start = bars[start].ts
    t_end = bars[end - 1].ts
    existing_lows = [
        sp for sp in confirmed
        if t_start <= sp.ts <= t_end and sp.kind in ("HL", "LL")
    ]
    if existing_lows:
        return

    min_bar_idx = min(range(start, end), key=lambda j: bars[j].l)
    min_bar = bars[min_bar_idx]
    if last_conf_low is None or min_bar.l > last_conf_low:
        kind: Literal["HL", "LL"] = "HL"
    else:
        kind = "LL"
    confirmed.append(SwingPoint(ts=min_bar.ts, price=min_bar.l, kind=kind))


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def atr(bars: List[Bar], period: int = 14) -> Optional[float]:
    """Wilder's ATR.  Returns None if len(bars) <= period.

    Vectorized: TRs computed via numpy on h/l/c arrays; Wilder smoothing
    is intrinsically sequential so it loops in pure-Python over the tail.
    """
    n = len(bars)
    if n <= period:
        return None
    h = np.fromiter((b.h for b in bars), dtype=np.float64, count=n)
    lo = np.fromiter((b.l for b in bars), dtype=np.float64, count=n)
    c = np.fromiter((b.c for b in bars), dtype=np.float64, count=n)
    prev_c = c[:-1]
    h1 = h[1:] - lo[1:]
    h2 = np.abs(h[1:] - prev_c)
    h3 = np.abs(lo[1:] - prev_c)
    trs = np.maximum(np.maximum(h1, h2), h3)
    atr_val = float(trs[:period].mean())
    inv = 1.0 / period
    pm1 = period - 1
    for tr in trs[period:]:
        atr_val = (atr_val * pm1 + float(tr)) * inv
    return atr_val


def sma(bars: List[Bar], period: int = 200) -> Optional[float]:
    """Simple moving average of the last *period* closes.  None if insufficient bars."""
    if len(bars) < period:
        return None
    return sum(b.c for b in bars[-period:]) / period


def donchian(
    bars: List[Bar], period: int = 20
) -> Tuple[Optional[float], Optional[float]]:
    """Donchian channel over last *period* bars: (upper=max high, lower=min low)."""
    if len(bars) < period:
        return (None, None)
    window = bars[-period:]
    return (max(b.h for b in window), min(b.l for b in window))


# ---------------------------------------------------------------------------
# Snapshot assembly
# ---------------------------------------------------------------------------

def build_snapshot(bars: List[Bar]) -> MarketSnapshot:
    """Build a MarketSnapshot from a list of bars."""
    if not bars:
        raise ValueError("bars must not be empty")
    upper, lower = donchian(bars, period=20)
    return MarketSnapshot(
        bars=bars,
        swings=detect_swings(bars),
        atr14=atr(bars, period=14),
        sma200=sma(bars, period=200),
        donchian_upper20=upper,
        donchian_lower20=lower,
        current_price=bars[-1].c,
        current_ts=bars[-1].ts,
    )
