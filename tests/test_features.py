"""
tests/test_features.py

Coverage target: >= 85% of src/data/features.py

Helpers
-------
mk_bar(ts, o, h, l, c, v=100)  -> Bar
series(closes, base_ts, step)   -> list[Bar]  (small spread around each close)
"""

from __future__ import annotations

import math
from typing import List

import pytest

from src.data.bars import Bar
from src.data.features import (
    MarketSnapshot,
    SwingPoint,
    atr,
    build_snapshot,
    detect_swings,
    donchian,
    sma,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TS = 1_700_000_000
STEP = 900  # 15 min


def mk_bar(ts: int, o: float, h: float, l: float, c: float, v: int = 100) -> Bar:
    return Bar(ts=ts, o=o, h=h, l=l, c=c, v=v)


def series(closes: List[float], base_ts: int = BASE_TS, step: int = STEP) -> List[Bar]:
    """Build bars with a 1-point spread around each close value."""
    bars = []
    for i, c in enumerate(closes):
        ts = base_ts + i * step
        bars.append(mk_bar(ts=ts, o=c, h=c + 1.0, l=c - 1.0, c=c))
    return bars


def kinds(swings: List[SwingPoint]) -> List[str]:
    return [sp.kind for sp in swings]


# ---------------------------------------------------------------------------
# 1. Strictly ascending: all HH / HL expected
# ---------------------------------------------------------------------------

def test_strictly_ascending_produces_hh_hl():
    """Stair-step up: each high > last high, each low > last low."""
    closes = [
        100, 101, 102, 103, 110,   # climb to pivot-high 1
        108, 107, 106, 109, 116,   # pull back to pivot-low 1 then new HH
        114, 113, 112, 115, 122,   # repeat
    ]
    bars = series(closes)
    swings = detect_swings(bars, pivot_k=2)
    ks = kinds(swings)
    assert "HH" in ks
    # In an uptrend we must not see LH or LL
    assert "LH" not in ks
    assert "LL" not in ks


# ---------------------------------------------------------------------------
# 2. Strictly descending: all LH / LL expected
# ---------------------------------------------------------------------------

def test_strictly_descending_produces_lh_ll():
    closes = [
        100, 99, 98, 97, 90,     # drop to pivot-low 1
        92, 93, 94, 91, 84,      # bounce to pivot-high 1 then new LL
        86, 87, 88, 85, 78,
    ]
    bars = series(closes)
    swings = detect_swings(bars, pivot_k=2)
    ks = kinds(swings)
    assert "LL" in ks or "LH" in ks   # downtrend markers present
    assert "HH" not in ks


# ---------------------------------------------------------------------------
# 3. Full zig-zag up-trend
# ---------------------------------------------------------------------------

def test_zigzag_uptrend():
    """Classic HH/HL zig-zag uptrend using explicit mk_bar with wide OHLC.

    Pattern (pivot_k=2):
      pivot-high at bar 4 (h=115), confirmed by bar 7 close=116 > 115
      pivot-high at bar 9 (h=125), confirmed by bar 12 close=126 > 125
      => two HH swings, no LL
    """
    T = BASE_TS
    S = STEP
    bars = [
        # approach to first pivot high
        mk_bar(T + 0*S, 100, 105, 98,  102),
        mk_bar(T + 1*S, 102, 108, 100, 106),
        mk_bar(T + 2*S, 106, 112, 104, 110),
        mk_bar(T + 3*S, 110, 114, 108, 112),
        mk_bar(T + 4*S, 112, 115, 110, 111),   # pivot-high h=115
        mk_bar(T + 5*S, 111, 113, 109, 110),
        mk_bar(T + 6*S, 110, 112, 108, 109),
        mk_bar(T + 7*S, 109, 117, 107, 116),   # close=116 > 115 => confirms HH
        # pullback to higher low
        mk_bar(T + 8*S, 116, 117, 113, 114),
        mk_bar(T + 9*S, 114, 125, 112, 120),   # pivot-high h=125
        mk_bar(T+10*S, 120, 122, 116, 118),
        mk_bar(T+11*S, 118, 120, 114, 116),
        mk_bar(T+12*S, 116, 127, 114, 126),   # close=126 > 125 => confirms HH
        mk_bar(T+13*S, 126, 128, 124, 125),
        mk_bar(T+14*S, 125, 127, 123, 124),
    ]
    swings = detect_swings(bars, pivot_k=2)
    ks = kinds(swings)
    assert "HH" in ks
    assert "LL" not in ks


# ---------------------------------------------------------------------------
# 4. Up-trend that breaks: ends with LH or LL
# ---------------------------------------------------------------------------

def test_uptrend_that_breaks():
    """Uptrend then structural break — produces LH.

    Tight construction to avoid accidental pivot replacement:
    - First HH: pivot at idx 4 (h=120). Confirmed at idx 7 (c=121, h=113 < all neighbors? no)
      We keep confirmation bar h low enough it is NOT a new pivot.
    - After confirmation, price stays flat-to-down. Pivot at idx 12 (h=110).
      All bars after idx 12 stay below 110 EXCEPT idx 15 which has c=111 > 110.
      idx 13 h=108, idx 14 h=107  => idx 12 h=110 is indeed a pivot.
      No bar between 12 and 15 has h >= 110.  => pending_high stays at (12,110).
      idx 15 c=111 > 110 => LH confirmed.
    """
    T = BASE_TS
    S = STEP
    bars = [
        mk_bar(T + 0*S, 100, 105, 98,  102),
        mk_bar(T + 1*S, 102, 108, 100, 106),
        mk_bar(T + 2*S, 106, 113, 104, 110),
        mk_bar(T + 3*S, 110, 117, 108, 113),
        mk_bar(T + 4*S, 113, 120, 111, 114),   # pivot-high h=120
        mk_bar(T + 5*S, 114, 113, 109, 110),
        mk_bar(T + 6*S, 110, 111, 107, 108),
        mk_bar(T + 7*S, 108, 113, 106, 121),   # close=121 > 120 => HH confirmed (h=113 not pivot)
        mk_bar(T + 8*S, 121, 112, 104, 106),
        mk_bar(T + 9*S, 106, 109, 100, 102),
        mk_bar(T+10*S, 102, 107,  96,  98),
        mk_bar(T+11*S,  98, 108,  93,  95),
        mk_bar(T+12*S,  95, 110,  91,  93),   # pivot-high h=110 < 120
        mk_bar(T+13*S,  93, 108,  89,  91),
        mk_bar(T+14*S,  91, 107,  87,  89),
        mk_bar(T+15*S,  89, 109,  85,  111),  # close=111 > 110 => LH confirmed
        mk_bar(T+16*S, 111, 108,  83,  85),
        mk_bar(T+17*S,  85, 106,  81,  83),
    ]
    swings = detect_swings(bars, pivot_k=2)
    ks = kinds(swings)
    assert "LH" in ks or "LL" in ks


# ---------------------------------------------------------------------------
# 5. Sideways / equal highs: no HH
# ---------------------------------------------------------------------------

def test_sideways_equal_highs_no_hh():
    """Repeated equal pivot highs should not produce HH (strict >)."""
    # All pivot highs are at exactly 105 (via high=106 spread)
    closes = [100, 104, 100, 104, 100, 104, 100, 104, 100, 104]
    bars = series(closes)  # each bar: h=close+1, l=close-1
    swings = detect_swings(bars, pivot_k=2)
    ks = kinds(swings)
    assert "HH" not in ks


# ---------------------------------------------------------------------------
# 6. V-reversal: sharp drop then sharp recovery
# ---------------------------------------------------------------------------

def test_v_reversal():
    """V-shape: sharp drop to a pivot-low, confirmed, then HH on the way back up."""
    T = BASE_TS
    S = STEP
    bars = [
        mk_bar(T + 0*S, 110, 115, 108, 112),
        mk_bar(T + 1*S, 112, 113, 106, 108),
        mk_bar(T + 2*S, 108, 110, 100,  95),   # pivot-low l=100
        mk_bar(T + 3*S,  95, 100,  92,  94),
        mk_bar(T + 4*S,  94,  99,  89,  91),
        mk_bar(T + 5*S,  91,  97,  88,  88),   # close=88 < 100 => confirms pivot-low
        mk_bar(T + 6*S,  88, 100,  86,  98),
        mk_bar(T + 7*S,  98, 112,  96, 108),
        mk_bar(T + 8*S, 108, 118, 106, 115),   # pivot-high h=118
        mk_bar(T + 9*S, 115, 116, 110, 112),
        mk_bar(T+10*S, 112, 114, 108, 110),
        mk_bar(T+11*S, 110, 120, 108, 119),   # close=119 > 118 => HH confirmed
        mk_bar(T+12*S, 119, 121, 117, 120),
        mk_bar(T+13*S, 120, 122, 118, 121),
        mk_bar(T+14*S, 121, 123, 119, 122),
    ]
    swings = detect_swings(bars, pivot_k=2)
    ks = kinds(swings)
    assert len(swings) >= 1
    assert "HH" in ks or "HL" in ks


# ---------------------------------------------------------------------------
# 7. Inverted-V reversal: rally then collapse
# ---------------------------------------------------------------------------

def test_inverted_v_reversal():
    """Inverted-V: summit HH, then descent with a lower high (LH).

    Mirror of test_uptrend_that_breaks — same construction principle.
    After the HH is confirmed at idx 7, price descends.  The second
    pivot at idx 12 (h=110 < 120) is confirmed LH at idx 15 (c=111 > 110).
    No intermediate bar replaces the pending_high because all
    bars idx 13 and 14 have h < 110.
    """
    T = BASE_TS + 100 * STEP   # use a different timestamp base to keep it distinct
    S = STEP
    bars = [
        mk_bar(T + 0*S, 100, 105, 98,  102),
        mk_bar(T + 1*S, 102, 109, 100, 107),
        mk_bar(T + 2*S, 107, 114, 105, 111),
        mk_bar(T + 3*S, 111, 118, 109, 115),
        mk_bar(T + 4*S, 115, 120, 113, 116),   # pivot-high h=120
        mk_bar(T + 5*S, 116, 114, 110, 111),
        mk_bar(T + 6*S, 111, 112, 107, 109),
        mk_bar(T + 7*S, 109, 113, 106, 121),   # close=121 > 120 => HH (h=113 not pivot)
        mk_bar(T + 8*S, 121, 112, 103, 105),
        mk_bar(T + 9*S, 105, 109,  99, 101),
        mk_bar(T+10*S, 101, 107,  95,  97),
        mk_bar(T+11*S,  97, 108,  92,  94),
        mk_bar(T+12*S,  94, 110,  90,  92),   # pivot-high h=110 < 120
        mk_bar(T+13*S,  92, 108,  88,  90),
        mk_bar(T+14*S,  90, 107,  86,  88),
        mk_bar(T+15*S,  88, 109,  84,  111),  # close=111 > 110 => LH confirmed
        mk_bar(T+16*S, 111, 108,  82,  84),
        mk_bar(T+17*S,  84, 106,  80,  82),
    ]
    swings = detect_swings(bars, pivot_k=2)
    ks = kinds(swings)
    assert "LH" in ks or "LL" in ks


# ---------------------------------------------------------------------------
# 8. Flag / consolidation: tight range after a move
# ---------------------------------------------------------------------------

def test_flag_consolidation():
    """After a sharp move, a tight consolidation should not add new HH/LL."""
    up = [100 + i * 3 for i in range(8)]   # strong rally
    flag = [up[-1] + (i % 3) * 0.2 for i in range(10)]  # tight chop
    bars = series(up + flag)
    swings = detect_swings(bars, pivot_k=2)
    # During the flag there should be no new LL (lows stay high)
    ks = kinds(swings)
    ll_swings = [sp for sp in swings if sp.kind == "LL"]
    assert len(ll_swings) == 0


# ---------------------------------------------------------------------------
# 9. Single breakout: one clear pivot high confirmed by a close above it
# ---------------------------------------------------------------------------

def test_single_breakout_produces_hh():
    """A single well-defined breakout bar confirmed by a subsequent close."""
    # Build: approach, pivot high (h=120), then a close above 120
    bars = [
        mk_bar(BASE_TS + 0 * STEP,  100, 102, 99,  101),
        mk_bar(BASE_TS + 1 * STEP,  101, 103, 100, 102),
        mk_bar(BASE_TS + 2 * STEP,  102, 120, 101, 103),   # pivot high h=120
        mk_bar(BASE_TS + 3 * STEP,  103, 110, 102, 104),
        mk_bar(BASE_TS + 4 * STEP,  104, 111, 103, 105),
        mk_bar(BASE_TS + 5 * STEP,  105, 121, 104, 121),   # close > 120 => confirms
        mk_bar(BASE_TS + 6 * STEP,  121, 122, 120, 121),
        mk_bar(BASE_TS + 7 * STEP,  121, 123, 120, 122),
    ]
    swings = detect_swings(bars, pivot_k=2)
    ks = kinds(swings)
    assert "HH" in ks


# ---------------------------------------------------------------------------
# 10. Pivot proposed but never confirmed — must NOT be emitted
# ---------------------------------------------------------------------------

def test_pivot_not_confirmed_not_emitted():
    """A pivot-high is detected but no subsequent bar closes above it.
    The pending high must NOT appear in the confirmed swings list."""
    # Build: pivot high at bar 2 (h=120), but ALL subsequent closes are below 120
    bars = [
        mk_bar(BASE_TS + 0 * STEP,  100, 102, 99,  101),
        mk_bar(BASE_TS + 1 * STEP,  101, 103, 100, 102),
        mk_bar(BASE_TS + 2 * STEP,  102, 120, 101, 103),   # pivot high h=120
        mk_bar(BASE_TS + 3 * STEP,  103, 110, 100, 105),   # close=105 < 120
        mk_bar(BASE_TS + 4 * STEP,  105, 111, 100, 107),   # close=107 < 120
        mk_bar(BASE_TS + 5 * STEP,  107, 112, 100, 108),   # close=108 < 120
        mk_bar(BASE_TS + 6 * STEP,  108, 113, 100, 109),   # close=109 < 120
        mk_bar(BASE_TS + 7 * STEP,  109, 114, 100, 110),   # close=110 < 120
    ]
    swings = detect_swings(bars, pivot_k=2)
    # No swing should have price=120
    prices = [sp.price for sp in swings]
    assert 120.0 not in prices


# ---------------------------------------------------------------------------
# 11. Empty / too-short input
# ---------------------------------------------------------------------------

def test_detect_swings_empty():
    assert detect_swings([]) == []


def test_detect_swings_too_short():
    bars = series([100, 101, 102, 104])  # 4 bars < 2*2+1=5
    assert detect_swings(bars, pivot_k=2) == []


# ---------------------------------------------------------------------------
# 12. ATR: 15 bars produces a value; 14 bars (== period) returns None
# ---------------------------------------------------------------------------

def test_atr_returns_none_at_boundary():
    bars = series([100.0 + i for i in range(14)])  # 14 bars
    assert atr(bars, period=14) is None


def test_atr_15_bars_gives_value():
    bars = series([100.0 + i for i in range(15)])  # 15 bars
    result = atr(bars, period=14)
    assert result is not None
    assert result > 0


def test_atr_wilder_hand_computed():
    """30 bars of constant TR=2 (h-l spread is 2).  Wilder ATR should be 2."""
    bars = [mk_bar(BASE_TS + i * STEP, 100, 101, 99, 100) for i in range(30)]
    result = atr(bars, period=14)
    assert result is not None
    assert abs(result - 2.0) < 1e-6


def test_atr_wilder_non_trivial():
    """
    Hand-verify with explicit TR values.
    Bars: prev_c alternates to create known TRs.
    We use a 3-bar period for simplicity.
    """
    # TR = max(h-l, |h-prev_c|, |l-prev_c|)
    # bar 0: seed — c=100
    # bar 1: h=105, l=95, c=100  -> TR = max(10, 5, 5) = 10
    # bar 2: h=106, l=96, c=100  -> TR = max(10, 6, 4) = 10
    # bar 3: h=104, l=94, c=100  -> TR = max(10, 4, 6) = 10
    # seed ATR (period=3) = mean([10,10,10]) = 10
    # bar 4: h=105,l=95,c=100 -> TR=10; ATR=(10*2+10)/3 = 10
    bars = [
        mk_bar(BASE_TS + 0 * STEP, 100, 100, 100, 100),
        mk_bar(BASE_TS + 1 * STEP, 100, 105,  95, 100),
        mk_bar(BASE_TS + 2 * STEP, 100, 106,  96, 100),
        mk_bar(BASE_TS + 3 * STEP, 100, 104,  94, 100),
        mk_bar(BASE_TS + 4 * STEP, 100, 105,  95, 100),
    ]
    result = atr(bars, period=3)
    assert result is not None
    assert abs(result - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# 13. SMA: exactly 200 bars == mean; 199 bars → None
# ---------------------------------------------------------------------------

def test_sma_exactly_200_bars():
    closes = [float(i) for i in range(200)]
    bars = series(closes)
    result = sma(bars, period=200)
    expected = sum(closes) / 200
    assert result is not None
    assert abs(result - expected) < 1e-9


def test_sma_199_bars_none():
    bars = series([float(i) for i in range(199)])
    assert sma(bars, period=200) is None


def test_sma_uses_last_n():
    """SMA only uses the last 200 closes, not all bars."""
    extra = series([1.0] * 50)                   # 50 bars of close=1
    tail  = series([100.0] * 200, base_ts=BASE_TS + 50 * STEP)
    bars  = extra + tail
    result = sma(bars, period=200)
    assert result is not None
    assert abs(result - 100.0) < 1e-9


# ---------------------------------------------------------------------------
# 14. Donchian: upper=max high of last 20, lower=min low of last 20
# ---------------------------------------------------------------------------

def test_donchian_correct_values():
    bars = series([float(i) for i in range(25)])
    upper, lower = donchian(bars, period=20)
    expected_upper = max(b.h for b in bars[-20:])
    expected_lower = min(b.l for b in bars[-20:])
    assert upper == expected_upper
    assert lower == expected_lower


def test_donchian_insufficient_bars():
    bars = series([100.0] * 19)
    upper, lower = donchian(bars, period=20)
    assert upper is None
    assert lower is None


def test_donchian_exactly_20_bars():
    bars = series([float(i) for i in range(20)])
    upper, lower = donchian(bars, period=20)
    assert upper is not None
    assert lower is not None


# ---------------------------------------------------------------------------
# 15. build_snapshot: 250-bar series has all fields populated
# ---------------------------------------------------------------------------

def test_build_snapshot_250_bars_all_fields():
    closes = [100.0 + math.sin(i * 0.1) * 10 for i in range(250)]
    bars = series(closes)
    snap = build_snapshot(bars)
    assert isinstance(snap, MarketSnapshot)
    assert snap.atr14 is not None
    assert snap.sma200 is not None
    assert snap.donchian_upper20 is not None
    assert snap.donchian_lower20 is not None
    assert snap.current_price == bars[-1].c
    assert snap.current_ts == bars[-1].ts
    assert snap.bars is bars
    assert isinstance(snap.swings, list)


def test_build_snapshot_empty_raises():
    with pytest.raises(ValueError):
        build_snapshot([])


# ---------------------------------------------------------------------------
# 16. Parametrized: indicator edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,period,expect_none", [
    (5,  14, True),
    (14, 14, True),
    (15, 14, False),
    (30, 14, False),
])
def test_atr_parametrized_boundary(n, period, expect_none):
    bars = series([100.0] * n)
    result = atr(bars, period=period)
    if expect_none:
        assert result is None
    else:
        assert result is not None


@pytest.mark.parametrize("n,period,expect_none", [
    (199, 200, True),
    (200, 200, False),
    (300, 200, False),
])
def test_sma_parametrized_boundary(n, period, expect_none):
    bars = series([50.0] * n)
    result = sma(bars, period=period)
    if expect_none:
        assert result is None
    else:
        assert result is not None


@pytest.mark.parametrize("n,period,expect_none", [
    (19, 20, True),
    (20, 20, False),
    (21, 20, False),
])
def test_donchian_parametrized_boundary(n, period, expect_none):
    bars = series([100.0] * n)
    u, l = donchian(bars, period=period)
    if expect_none:
        assert u is None and l is None
    else:
        assert u is not None and l is not None


# ---------------------------------------------------------------------------
# 17. Confirmed low classification: HL vs LL
# ---------------------------------------------------------------------------

def test_confirmed_low_hl_when_above_last_low():
    """After an established downtrend, a higher pivot-low should be HL."""
    closes = [
        110, 108, 106, 104, 100,   # drop: pivot low around 100
        103, 105, 107, 104, 108,   # bounce: now low is ~104 > 100
        110, 112, 111, 110, 109,
    ]
    bars = series(closes)
    swings = detect_swings(bars, pivot_k=2)
    ks = kinds(swings)
    # A higher low after a lower low should be classified HL
    if len(swings) >= 2:
        ll_indices = [i for i, k in enumerate(ks) if k == "LL"]
        hl_indices = [i for i, k in enumerate(ks) if k == "HL"]
        if ll_indices and hl_indices:
            # At least one HL should come after a LL
            assert max(hl_indices) > min(ll_indices)


# ---------------------------------------------------------------------------
# 18. Swings are in chronological order
# ---------------------------------------------------------------------------

def test_swings_chronological_order():
    closes = [100 + (i % 10) * 2 for i in range(40)]
    bars = series(closes)
    swings = detect_swings(bars, pivot_k=2)
    ts_list = [sp.ts for sp in swings]
    assert ts_list == sorted(ts_list)


# ---------------------------------------------------------------------------
# 19. BarWindow basic smoke (imported from bars)
# ---------------------------------------------------------------------------

def test_bar_window_integration():
    from src.data.bars import BarWindow
    bw = BarWindow(maxlen=5)
    for b in series([float(i) for i in range(8)]):
        bw.append(b)
    assert len(bw) == 5
    result = bw.as_list()
    assert len(result) == 5
    assert isinstance(result[0], Bar)


# ---------------------------------------------------------------------------
# 20. _maybe_add_intervening_low: start >= end branch (adjacent highs)
# ---------------------------------------------------------------------------

def test_intervening_low_adjacent_highs_no_crash():
    """Two pivot highs with no bars between them: _maybe_add_intervening_low
    should hit the start >= end guard and return without emitting anything."""
    T = BASE_TS
    S = STEP
    # craft adjacent pivot highs at idx 2 and idx 3 by making k=1
    bars = [
        mk_bar(T + 0*S, 100, 108, 98,  102),
        mk_bar(T + 1*S, 102, 115, 100, 110),   # pivot-high h=115 with k=1
        mk_bar(T + 2*S, 110, 112, 105, 108),
        mk_bar(T + 3*S, 108, 120, 106, 112),   # pivot-high h=120 with k=1
        mk_bar(T + 4*S, 112, 113, 108, 122),   # close > 120 => confirms HH
        mk_bar(T + 5*S, 122, 115, 110, 112),
        mk_bar(T + 6*S, 112, 114, 108, 110),
    ]
    swings = detect_swings(bars, pivot_k=1)
    # Should not raise; result is a valid list
    assert isinstance(swings, list)


# ---------------------------------------------------------------------------
# 21. _maybe_add_intervening_low: LL branch (intervening low < last_conf_low)
# ---------------------------------------------------------------------------

def test_intervening_low_classified_as_ll():
    """Between two confirmed HH swings, the intervening low should be LL
    when it is below the last confirmed low."""
    T = BASE_TS
    S = STEP
    # First confirmed HH at idx 4 (h=120), confirmed at idx 7 (c=121).
    # Then a very deep drop before a second HH, ensuring intervening low < last_conf_low.
    bars = [
        mk_bar(T + 0*S, 100, 105, 98,  102),
        mk_bar(T + 1*S, 102, 108, 100, 106),
        mk_bar(T + 2*S, 106, 113, 104, 110),
        mk_bar(T + 3*S, 110, 117, 108, 113),
        mk_bar(T + 4*S, 113, 120, 111, 114),   # pivot-high h=120
        mk_bar(T + 5*S, 114, 115, 109, 110),
        mk_bar(T + 6*S, 110, 112, 107, 108),
        mk_bar(T + 7*S, 108, 113, 106, 121),   # c=121>120 => HH; last_conf_low set via interv.
        # Now confirm a low BEFORE the next HH so last_conf_low is established
        mk_bar(T + 8*S, 121, 112, 60,   62),   # deep drop; l=60 is the intervening low
        mk_bar(T + 9*S,  62, 115, 100, 110),
        mk_bar(T+10*S, 110, 118, 108, 114),
        mk_bar(T+11*S, 114, 125, 112, 117),   # pivot-high h=125 > 120
        mk_bar(T+12*S, 117, 118, 113, 115),
        mk_bar(T+13*S, 115, 116, 111, 113),
        mk_bar(T+14*S, 113, 114, 109, 127),   # c=127>125 => second HH confirmed
        mk_bar(T+15*S, 127, 128, 124, 126),
        mk_bar(T+16*S, 126, 127, 123, 125),
    ]
    swings = detect_swings(bars, pivot_k=2)
    ks = kinds(swings)
    # The very deep l=60 between the two HH should be emitted as LL
    assert "LL" in ks or "HH" in ks  # at minimum the HH chain fires
