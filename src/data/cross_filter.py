"""
Golden Cross / Death Cross filter for DOWTrade.

Dad's rule: ???????????- Golden cross (SMA20 > SMA50) = bullish
- Death cross (SMA20 < SMA50) = bearish

Filter mode: LLM proposes trade, cross must agree.
- open_long requires BOTH 15m golden AND 1hr golden
- open_short requires BOTH 15m death AND 1hr death
- Otherwise: blocked

Usage in runner.py:
    from src.data.cross_filter import CrossFilter
    self._cross = CrossFilter()
    # on each bar:
    self._cross.update(bar)
    cross_state = self._cross.state()
    # cross_state = {"15m": "golden"|"death"|"neutral", "1hr": "golden"|"death"|"neutral"}
    # check:
    allowed = self._cross.allows(action)  # action = "open_long" | "open_short"
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import List, Literal

from src.data.bars import Bar


def _sma(closes: list[float], period: int) -> float | None:
    """Simple moving average of last `period` values. None if not enough data."""
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def _resample_1hr(bars_15m: list[Bar]) -> list[Bar]:
    """Resample 15m bars into 1hr bars (groups of 4).

    Groups by floor(ts / 3600). Each group produces one 1hr bar:
    O=first.o, H=max(h), L=min(l), C=last.c, V=sum(v).
    """
    if not bars_15m:
        return []

    buckets: dict[int, list[Bar]] = {}
    for b in bars_15m:
        hour_key = (b.ts // 3600) * 3600
        buckets.setdefault(hour_key, []).append(b)

    result = []
    for ts_key in sorted(buckets):
        group = buckets[ts_key]
        result.append(Bar(
            ts=ts_key + 3600,  # close of hour bucket
            o=group[0].o,
            h=max(b.h for b in group),
            l=min(b.l for b in group),
            c=group[-1].c,
            v=sum(b.v for b in group),
        ))
    return result


CrossState = Literal["golden", "death", "neutral"]


@dataclass
class CrossInfo:
    """Cross state for one timeframe."""
    state: CrossState
    sma_fast: float | None
    sma_slow: float | None


class CrossFilter:
    """Track SMA 20/50 cross state on 15m and 1hr timeframes.

    Feed it 15m bars. It internally resamples to 1hr.
    """

    FAST = 20
    SLOW = 50
    # Need at least 50 1hr bars = 200 15m bars for 1hr SMA50.
    # 15m SMA50 needs 50 15m bars. So 200 15m bars covers both.

    def __init__(self, maxlen: int = 500) -> None:
        self._bars_15m: deque[Bar] = deque(maxlen=maxlen)

    def update(self, bar: Bar) -> None:
        """Append a new 15m bar."""
        self._bars_15m.append(bar)

    def bulk_load(self, bars: list[Bar]) -> None:
        """Load historical bars (e.g. from hydration)."""
        for b in bars:
            self._bars_15m.append(b)

    def _cross_state(self, closes: list[float]) -> CrossInfo:
        """Determine cross state from a series of closes."""
        fast = _sma(closes, self.FAST)
        slow = _sma(closes, self.SLOW)
        if fast is None or slow is None:
            return CrossInfo(state="neutral", sma_fast=fast, sma_slow=slow)
        if fast > slow:
            return CrossInfo(state="golden", sma_fast=fast, sma_slow=slow)
        elif fast < slow:
            return CrossInfo(state="death", sma_fast=fast, sma_slow=slow)
        else:
            return CrossInfo(state="neutral", sma_fast=fast, sma_slow=slow)

    def state(self) -> dict[str, CrossInfo]:
        """Return cross state for both timeframes."""
        bars_15m = list(self._bars_15m)
        closes_15m = [b.c for b in bars_15m]

        bars_1hr = _resample_1hr(bars_15m)
        closes_1hr = [b.c for b in bars_1hr]

        return {
            "15m": self._cross_state(closes_15m),
            "1hr": self._cross_state(closes_1hr),
        }

    def allows(self, action: str) -> tuple[bool, str]:
        """Check if cross state allows the proposed action.

        Returns (allowed: bool, reason: str).
        """
        if action not in ("open_long", "open_short"):
            return True, "no_filter_needed"

        cs = self.state()
        s15 = cs["15m"].state
        s1h = cs["1hr"].state

        if action == "open_long":
            if s15 == "golden" and s1h == "golden":
                return True, f"cross_ok: 15m={s15} 1hr={s1h}"
            return False, f"CROSS_BLOCKED: need golden/golden, got 15m={s15} 1hr={s1h}"

        if action == "open_short":
            if s15 == "death" and s1h == "death":
                return True, f"cross_ok: 15m={s15} 1hr={s1h}"
            return False, f"CROSS_BLOCKED: need death/death, got 15m={s15} 1hr={s1h}"

        return True, "no_filter_needed"

