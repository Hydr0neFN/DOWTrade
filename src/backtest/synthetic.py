"""
Synthetic 15-min MYM bar generator for backtesting.

Generates deterministic OHLCV bars respecting Tradovate/CME Globex
trading hours: Sun 18:00 ET -> Fri 17:00 ET, with a daily 17:00-18:00 ET
maintenance break.  Timestamps are UNIX epoch seconds at bar CLOSE.

Session layout (one "trading day" = one Globex session):
  - Mon session : Sun 18:00 ET open -> Mon 17:00 ET close
  - Tue session : Mon 18:00 ET open -> Tue 17:00 ET close
  - ...
  - Fri session : Thu 18:00 ET open -> Fri 17:00 ET close
  - Saturday and Sunday are NOT counted as sessions.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from src.data.bars import Bar

ET = ZoneInfo("America/New_York")

# ~252 trading days × 23 trading hours × 4 bars/hour = 23,184 bars/year
BARS_PER_YEAR: int = 23_184

# Per-bar drift for each regime.
# 0.0001 per bar gives ~10-15% directional drift over a 30-day run
# while staying well within the ±30% price-sanity bound.
REGIME_DRIFTS = {
    "up":   +0.0001,
    "down": -0.0001,
    "chop":  0.0,
}
REGIME_VOL_MULT = {
    "up":   1.0,
    "down": 1.0,
    "chop": 1.2,
}


@dataclass(frozen=True)
class SyntheticConfig:
    start_date: date
    num_days: int
    start_price: float = 42_000.0
    annual_vol: float = 0.18
    regimes: Optional[List[Tuple[str, int]]] = None
    seed: int = 42


def _default_regimes(num_days: int) -> List[Tuple[str, int]]:
    """Build a default alternating-regime list covering at least *num_days*."""
    base = [
        ("up",   5),
        ("chop", 2),
        ("down", 4),
        ("chop", 2),
        ("up",   6),
        ("chop", 3),
    ]
    used = sum(d for _, d in base)
    remainder = max(1, num_days - used)
    return base + [("up", remainder)]


def _next_weekday(d: date) -> date:
    """Advance *d* until it lands on a Mon-Fri weekday."""
    while d.weekday() > 4:   # 5=Sat, 6=Sun
        d += timedelta(days=1)
    return d


def _iter_trading_slots(start: date, num_days: int):
    """
    Yield (bar_close_dt_et, is_first_bar_of_session) for every 15-min bar.

    Only Mon-Fri calendar days are counted as sessions.  Each session:
      open  = previous calendar day 18:00 ET  (Mon -> Sun 18:00)
      close = this calendar day 17:00 ET

    The 17:00 ET bar IS included (final bar of the session).
    Bars with close times 17:01–17:59 ET are skipped (maintenance break).

    Saturday and Sunday are never counted as sessions.
    """
    STEP = timedelta(minutes=15)

    # Ensure we start on a weekday
    cal_day = _next_weekday(start)
    days_yielded = 0

    while days_yielded < num_days:
        # Session open = previous calendar day 18:00 ET
        # (for Monday that is Sunday 18:00 ET — correct Globex open)
        session_open_et = datetime(
            cal_day.year, cal_day.month, cal_day.day,
            18, 0, 0, tzinfo=ET
        ) - timedelta(days=1)

        session_close_et = datetime(
            cal_day.year, cal_day.month, cal_day.day,
            17, 0, 0, tzinfo=ET
        )

        bar_close = session_open_et + STEP
        first_bar = True

        while bar_close <= session_close_et:
            h, m = bar_close.hour, bar_close.minute
            # Skip maintenance slots 17:01–17:59 ET
            # (17:00 close is included; 17:15 and later up to 17:59 are skipped)
            if h == 17 and m > 0:
                bar_close += STEP
                continue

            yield bar_close, first_bar
            first_bar = False
            bar_close += STEP

        days_yielded += 1
        # Advance to next weekday
        cal_day = _next_weekday(cal_day + timedelta(days=1))


def generate_bars(cfg: SyntheticConfig) -> List[Bar]:
    """
    Generate 15-min OHLCV bars for *cfg.num_days* trading days.

    All timestamps are UNIX epoch seconds (UTC) at bar close.
    """
    rng = np.random.default_rng(cfg.seed)

    per_bar_sigma = cfg.annual_vol / math.sqrt(BARS_PER_YEAR)

    # Expand regimes into a per-day (drift, vol_mult) list
    regimes = cfg.regimes if cfg.regimes is not None else _default_regimes(cfg.num_days)
    day_params: List[Tuple[float, float]] = []
    for regime_name, regime_days in regimes:
        drift = REGIME_DRIFTS.get(regime_name, 0.0)
        vmult = REGIME_VOL_MULT.get(regime_name, 1.0)
        for _ in range(regime_days):
            day_params.append((drift, vmult))
        if len(day_params) >= cfg.num_days:
            break
    while len(day_params) < cfg.num_days:
        day_params.append((0.0, 1.0))
    day_params = day_params[: cfg.num_days]

    bars: List[Bar] = []
    prev_close = cfg.start_price
    day_index = 0

    for bar_close_et, is_new_day in _iter_trading_slots(cfg.start_date, cfg.num_days):
        # Advance day_index at the start of each new session (except the very first)
        if is_new_day and bars:
            day_index += 1

        drift, vmult = day_params[min(day_index, len(day_params) - 1)]
        sigma = per_bar_sigma * vmult

        # Log-return for this bar
        r = drift + sigma * rng.standard_normal()
        close = prev_close * math.exp(r)
        open_ = prev_close   # gap-free

        # Intra-bar high/low noise
        n1 = abs(rng.normal(0, sigma / 2))
        n2 = abs(rng.normal(0, sigma / 2))
        high = max(open_, close) * (1.0 + n1)
        low  = min(open_, close) * (1.0 - n2)

        # Volume: Poisson(3000)
        vol = int(rng.poisson(3000))

        # Convert bar close ET datetime -> UTC unix epoch seconds
        ts = int(bar_close_et.timestamp())

        bars.append(Bar(
            ts=ts,
            o=round(open_, 2),
            h=round(high, 2),
            l=round(low, 2),
            c=round(close, 2),
            v=vol,
        ))
        prev_close = close

    return bars


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def write_csv(bars: List[Bar], path: str) -> None:
    """Write bars to CSV.  Header: ts,o,h,l,c,v"""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ts", "o", "h", "l", "c", "v"])
        for b in bars:
            writer.writerow([b.ts, b.o, b.h, b.l, b.c, b.v])


def read_csv(path: str) -> List[Bar]:
    """Read bars from CSV produced by write_csv."""
    bars: List[Bar] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bars.append(Bar(
                ts=int(row["ts"]),
                o=float(row["o"]),
                h=float(row["h"]),
                l=float(row["l"]),
                c=float(row["c"]),
                v=int(row["v"]),
            ))
    return bars
