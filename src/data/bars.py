"""
Bar dataclass, tick aggregation, and rolling BarWindow.

Timestamps are UNIX epoch seconds aligned to the CLOSE of a 15-min bucket
(i.e. bucket_start + timeframe_seconds - 1 rounded up, or more precisely:
bucket_close = floor(ts / bucket_sec) * bucket_sec + bucket_sec).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

from src.config import TIMEFRAME_MINUTES


@dataclass(frozen=True)
class Bar:
    ts: int    # unix epoch seconds, aligned to the CLOSE of a 15-min bucket
    o: float
    h: float
    l: float
    c: float
    v: int


def align_to_timeframe(ts: int, timeframe_min: int = TIMEFRAME_MINUTES) -> int:
    """Return the bucket-CLOSE timestamp for *ts* given a timeframe in minutes.

    The bucket close is defined as: floor(ts / bucket_sec) * bucket_sec + bucket_sec
    so the close of the bucket that *contains* ts.
    """
    bucket_sec = timeframe_min * 60
    return (ts // bucket_sec) * bucket_sec + bucket_sec


def aggregate_ticks(
    ticks: List[Tuple[int, float, int]],
    timeframe_min: int = TIMEFRAME_MINUTES,
) -> List[Bar]:
    """Aggregate (ts, price, volume) ticks into OHLCV bars.

    Each bar's timestamp is the bucket-CLOSE.  Buckets with no ticks are
    omitted.  Within each bucket: first price = O, max = H, min = L,
    last = C, volumes summed.
    """
    if not ticks:
        return []

    bucket_sec = timeframe_min * 60
    buckets: dict[int, dict] = {}

    for ts, price, volume in ticks:
        bucket_close = (ts // bucket_sec) * bucket_sec + bucket_sec
        if bucket_close not in buckets:
            buckets[bucket_close] = {
                "o": price,
                "h": price,
                "l": price,
                "c": price,
                "v": volume,
            }
        else:
            b = buckets[bucket_close]
            if price > b["h"]:
                b["h"] = price
            if price < b["l"]:
                b["l"] = price
            b["c"] = price
            b["v"] += volume

    return [
        Bar(ts=bc, o=b["o"], h=b["h"], l=b["l"], c=b["c"], v=b["v"])
        for bc, b in sorted(buckets.items())
    ]


class BarWindow:
    """Rolling fixed-size deque of Bar objects."""

    def __init__(self, maxlen: int = 500) -> None:
        self._dq: deque[Bar] = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def append(self, bar: Bar) -> None:
        self._dq.append(bar)

    def as_list(self) -> List[Bar]:
        return list(self._dq)

    def __len__(self) -> int:
        return len(self._dq)
