"""
Tests for src/backtest/synthetic.py
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from src.backtest.synthetic import (
    SyntheticConfig,
    generate_bars,
    read_csv,
    write_csv,
)

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bars_30d():
    cfg = SyntheticConfig(start_date=date(2025, 1, 6), num_days=30, seed=42)
    return generate_bars(cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bar_count(bars_30d):
    """30-day run must produce at least 30 × 92 = 2760 bars."""
    assert len(bars_30d) >= 2760, f"Only {len(bars_30d)} bars generated"


def test_15min_alignment(bars_30d):
    """Every bar timestamp must be divisible by 900 (15 min)."""
    bad = [b for b in bars_30d if b.ts % 900 != 0]
    assert not bad, f"{len(bad)} bars not 15-min aligned: first={bad[0]}"


def test_no_saturday_bars(bars_30d):
    """No bar may fall on a Saturday (UTC datetime weekday == 5)."""
    bad = []
    for b in bars_30d:
        dt_et = datetime.fromtimestamp(b.ts, tz=ET)
        if dt_et.weekday() == 5:
            bad.append(b)
    assert not bad, f"{len(bad)} bars fall on Saturday"


def test_no_maintenance_break_bars(bars_30d):
    """No bars should have a close time strictly between 17:00 and 18:00 ET."""
    bad = []
    for b in bars_30d:
        dt_et = datetime.fromtimestamp(b.ts, tz=ET)
        h, m = dt_et.hour, dt_et.minute
        # 17:15 .. 17:59 are the forbidden slots (17:00 is the last valid bar)
        if h == 17 and m > 0:
            bad.append(dt_et)
        elif h > 17 and h < 18:
            bad.append(dt_et)
    assert not bad, f"{len(bad)} bars in maintenance window: first={bad[0]}"


def test_no_sunday_before_1800(bars_30d):
    """No bar close should be on Sunday before 18:00 ET."""
    bad = []
    for b in bars_30d:
        dt_et = datetime.fromtimestamp(b.ts, tz=ET)
        # Sunday = weekday 6
        if dt_et.weekday() == 6 and (dt_et.hour < 18):
            bad.append(dt_et)
    assert not bad, f"{len(bad)} bars on Sunday before 18:00 ET"


def test_chronological(bars_30d):
    """Bars must be strictly increasing in time."""
    for i in range(1, len(bars_30d)):
        assert bars_30d[i].ts > bars_30d[i - 1].ts, (
            f"Non-monotonic at index {i}: {bars_30d[i-1].ts} >= {bars_30d[i].ts}"
        )


def test_ohlc_invariants(bars_30d):
    """low <= min(open, close) and high >= max(open, close) for every bar."""
    for i, b in enumerate(bars_30d):
        assert b.l <= min(b.o, b.c), f"Bar {i}: low {b.l} > min(o,c) {min(b.o, b.c)}"
        assert b.h >= max(b.o, b.c), f"Bar {i}: high {b.h} < max(o,c) {max(b.o, b.c)}"


def test_determinism():
    """Two calls with the same config must produce identical bars."""
    cfg = SyntheticConfig(start_date=date(2025, 1, 6), num_days=10, seed=99)
    bars_a = generate_bars(cfg)
    bars_b = generate_bars(cfg)
    assert bars_a == bars_b, "Non-deterministic output for identical configs"


def test_csv_round_trip(bars_30d):
    """write_csv then read_csv must return bars equal to the original."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        path = tmp.name
    try:
        write_csv(bars_30d, path)
        restored = read_csv(path)
        assert restored == list(bars_30d), (
            f"CSV round-trip mismatch: {len(restored)} vs {len(bars_30d)} bars"
        )
    finally:
        os.unlink(path)


def test_price_sanity(bars_30d):
    """No NaN values and prices stay within ±30% of start_price (42000)."""
    import math
    start = 42_000.0
    low_bound  = start * 0.70
    high_bound = start * 1.30
    for b in bars_30d:
        for val in (b.o, b.h, b.l, b.c):
            assert not math.isnan(val), f"NaN in bar {b}"
            assert low_bound <= val <= high_bound, (
                f"Price {val:.2f} outside ±30% band [{low_bound:.0f}, {high_bound:.0f}]"
            )
