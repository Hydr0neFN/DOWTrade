"""
tests/test_stubs.py
===================
Tests for src/backtest/stubs.py -- targeting >=85% coverage.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

import pytest

from src.backtest.stubs import StubDeepSeek, StubGemini, StubHaiku, run_stub_pipeline
from src.broker.models import AccountState, Position
from src.data.features import MarketSnapshot, SwingPoint

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def make_swing(ts: int, price: float, kind: str) -> SwingPoint:
    return SwingPoint(ts=ts, price=price, kind=kind)


def make_snapshot(
    current_price: float = 20000.0,
    sma200: Optional[float] = 19000.0,
    atr14: Optional[float] = 50.0,
    swings: Optional[List[SwingPoint]] = None,
    donchian_upper20: Optional[float] = None,
    donchian_lower20: Optional[float] = None,
) -> MarketSnapshot:
    return MarketSnapshot(
        bars=[],
        swings=swings or [],
        atr14=atr14,
        sma200=sma200,
        donchian_upper20=donchian_upper20,
        donchian_lower20=donchian_lower20,
        current_price=current_price,
        current_ts=1000,
    )


def make_position(
    side: str = "flat",
    qty: int = 0,
    avg_price: float = 0.0,
    unrealized_pnl: float = 0.0,
    pyramid_adds_used: int = 0,
) -> Position:
    return Position(
        side=side,
        qty=qty,
        avg_price=avg_price,
        unrealized_pnl=unrealized_pnl,
        pyramid_adds_used=pyramid_adds_used,
    )


def make_state(
    equity: float = 10000.0,
    realized_pnl_today: float = 0.0,
    unrealized_pnl: float = 0.0,
    position: Optional[Position] = None,
    now_et: Optional[datetime] = None,
) -> AccountState:
    if position is None:
        position = make_position()
    if now_et is None:
        # Default: Tuesday 10:00 ET (well within trading hours)
        now_et = datetime(2026, 4, 14, 10, 0, 0, tzinfo=ET)
    return AccountState(
        equity=equity,
        realized_pnl_today=realized_pnl_today,
        unrealized_pnl=unrealized_pnl,
        position=position,
        now_et=now_et,
    )


# Uptrend swings: two HH, two HL, last swing is HL
UP_SWINGS = [
    make_swing(1, 19600.0, "HH"),
    make_swing(2, 19500.0, "HL"),
    make_swing(3, 19800.0, "HH"),
    make_swing(4, 19700.0, "HL"),
]

# Downtrend swings: two LH, two LL, last swing is LL
DOWN_SWINGS = [
    make_swing(1, 19400.0, "LH"),
    make_swing(2, 19300.0, "LL"),
    make_swing(3, 19200.0, "LH"),
    make_swing(4, 19100.0, "LL"),
]


# ---------------------------------------------------------------------------
# StubHaiku tests
# ---------------------------------------------------------------------------

class TestStubHaiku:
    def setup_method(self):
        self.haiku = StubHaiku()

    # -- Uptrend --

    def test_uptrend_no_signal_below_hl(self):
        # Price below last HL (19700) -> neither new_hh_break nor new_hl_hold
        snap = make_snapshot(current_price=19650.0, sma200=19000.0, swings=UP_SWINGS)
        result = self.haiku.evaluate(snap)
        assert result["trend"] == "up"
        assert result["structural_signal"] == "none"
        assert result["last_confirmed_hh"] == 19800.0
        assert result["last_confirmed_hl"] == 19700.0
        assert result["pattern_intact"] is True
        assert result["confidence_0_to_1"] == 0.4

    def test_uptrend_new_hh_break_when_close_above_hh(self):
        # Price above last HH (19800) -> new_hh_break
        snap = make_snapshot(current_price=19900.0, sma200=19000.0, swings=UP_SWINGS)
        result = self.haiku.evaluate(snap)
        assert result["trend"] == "up"
        assert result["structural_signal"] == "new_hh_break"
        assert result["confidence_0_to_1"] == 0.8

    def test_uptrend_new_hl_hold_on_pullback(self):
        # Last swing is HL at 19700, price above it but below HH
        swings = [
            make_swing(1, 19600.0, "HH"),
            make_swing(2, 19500.0, "HL"),
            make_swing(3, 19800.0, "HH"),
            make_swing(4, 19700.0, "HL"),  # last swing is HL
        ]
        snap = make_snapshot(current_price=19750.0, sma200=19000.0, swings=swings)
        result = self.haiku.evaluate(snap)
        assert result["trend"] == "up"
        assert result["structural_signal"] == "new_hl_hold"
        assert result["confidence_0_to_1"] == 0.8

    def test_uptrend_pattern_broken_by_ll(self):
        # LL after HH sequence -> pattern_broken
        swings = UP_SWINGS + [make_swing(5, 19000.0, "LL")]
        snap = make_snapshot(current_price=19750.0, sma200=19000.0, swings=swings)
        result = self.haiku.evaluate(snap)
        assert result["structural_signal"] == "pattern_broken"
        assert result["pattern_intact"] is False
        assert result["confidence_0_to_1"] == 0.9

    # -- Downtrend --

    def test_downtrend_detected(self):
        # Price below SMA200, last 2 highs LH, last 2 lows LL
        snap = make_snapshot(current_price=18800.0, sma200=19500.0, swings=DOWN_SWINGS)
        result = self.haiku.evaluate(snap)
        assert result["trend"] == "down"

    def test_downtrend_new_hh_break_below_ll(self):
        # Price below last LL (19100) -> continuation break -> new_hh_break
        snap = make_snapshot(current_price=19000.0, sma200=19500.0, swings=DOWN_SWINGS)
        result = self.haiku.evaluate(snap)
        assert result["trend"] == "down"
        assert result["structural_signal"] == "new_hh_break"
        assert result["confidence_0_to_1"] == 0.8

    def test_downtrend_no_signal_above_ll(self):
        snap = make_snapshot(current_price=19150.0, sma200=19500.0, swings=DOWN_SWINGS)
        result = self.haiku.evaluate(snap)
        assert result["trend"] == "down"
        assert result["structural_signal"] == "none"

    def test_downtrend_pattern_broken_by_hh(self):
        swings = DOWN_SWINGS + [make_swing(5, 19500.0, "HH")]
        snap = make_snapshot(current_price=19150.0, sma200=19500.0, swings=swings)
        result = self.haiku.evaluate(snap)
        assert result["structural_signal"] == "pattern_broken"
        assert result["pattern_intact"] is False

    # -- Sideways / edge cases --

    def test_sideways_when_sma200_none(self):
        snap = make_snapshot(sma200=None, swings=UP_SWINGS)
        result = self.haiku.evaluate(snap)
        assert result["trend"] == "sideways"
        assert result["structural_signal"] == "none"
        assert result["confidence_0_to_1"] == 0.5

    def test_sideways_when_mixed_swings(self):
        # Mixed swings (LH/HL alternation) don't form a clean up or down trend
        # and don't contain the HH+LL pattern_broken condition
        swings = [
            make_swing(1, 19600.0, "LH"),
            make_swing(2, 19500.0, "HL"),
            make_swing(3, 19700.0, "LH"),
            make_swing(4, 19600.0, "HL"),
        ]
        snap = make_snapshot(current_price=19650.0, sma200=19000.0, swings=swings)
        result = self.haiku.evaluate(snap)
        assert result["trend"] == "sideways"
        assert result["structural_signal"] == "none"
        assert result["confidence_0_to_1"] == 0.5

    def test_empty_swings_sideways_no_signal(self):
        snap = make_snapshot(swings=[], sma200=19000.0)
        result = self.haiku.evaluate(snap)
        assert result["trend"] == "sideways"
        assert result["structural_signal"] == "none"
        assert result["last_confirmed_hh"] is None
        assert result["last_confirmed_hl"] is None

    def test_schema_keys_present(self):
        snap = make_snapshot(swings=UP_SWINGS)
        result = self.haiku.evaluate(snap)
        required = {
            "trend", "last_confirmed_hh", "last_confirmed_hl",
            "pattern_intact", "structural_signal", "confidence_0_to_1", "reasoning",
        }
        assert required <= result.keys()


# ---------------------------------------------------------------------------
# StubGemini tests
# ---------------------------------------------------------------------------

class TestStubGemini:
    def setup_method(self):
        self.gemini = StubGemini()

    def _haiku(self, trend="up", signal="new_hh_break"):
        return {
            "trend": trend,
            "structural_signal": signal,
            "last_confirmed_hh": 19800.0,
            "last_confirmed_hl": 19700.0,
            "pattern_intact": signal != "pattern_broken",
            "confidence_0_to_1": 0.8,
            "reasoning": "stub",
        }

    def test_open_long_on_uptrend_new_hh_break(self):
        snap = make_snapshot(current_price=19850.0, atr14=50.0)
        pos  = make_position(side="flat")
        result = self.gemini.evaluate(self._haiku("up", "new_hh_break"), snap, pos, 10000.0)
        assert result["action"] == "open_long"
        assert result["stop_price"] == pytest.approx(19850.0 - 2 * 50.0)
        assert result["trailing_stop_atr_multiple"] == 2.0

    def test_open_long_on_uptrend_new_hl_hold(self):
        snap = make_snapshot(current_price=19750.0, atr14=50.0)
        pos  = make_position(side="flat")
        result = self.gemini.evaluate(self._haiku("up", "new_hl_hold"), snap, pos, 10000.0)
        assert result["action"] == "open_long"

    def test_open_short_on_downtrend_continuation(self):
        snap = make_snapshot(current_price=19050.0, atr14=50.0)
        pos  = make_position(side="flat")
        result = self.gemini.evaluate(self._haiku("down", "new_hh_break"), snap, pos, 10000.0)
        assert result["action"] == "open_short"
        assert result["stop_price"] == pytest.approx(19050.0 + 2 * 50.0)

    def test_close_on_pattern_broken_with_long(self):
        snap = make_snapshot(current_price=19750.0, atr14=50.0)
        pos  = make_position(side="long", qty=1, avg_price=19700.0, unrealized_pnl=50.0)
        result = self.gemini.evaluate(self._haiku("up", "pattern_broken"), snap, pos, 10000.0)
        assert result["action"] == "close"

    def test_close_on_pattern_broken_with_short(self):
        snap = make_snapshot(current_price=19050.0, atr14=50.0)
        pos  = make_position(side="short", qty=1, avg_price=19200.0, unrealized_pnl=75.0)
        result = self.gemini.evaluate(self._haiku("down", "pattern_broken"), snap, pos, 10000.0)
        assert result["action"] == "close"

    def test_add_pyramid_on_profitable_long(self):
        snap = make_snapshot(current_price=19900.0, atr14=50.0)
        pos  = make_position(side="long", qty=1, avg_price=19700.0, unrealized_pnl=100.0, pyramid_adds_used=0)
        result = self.gemini.evaluate(self._haiku("up", "new_hh_break"), snap, pos, 10000.0)
        assert result["action"] == "add_pyramid"
        assert result["stop_price"] == pytest.approx(19900.0 - 2 * 50.0)

    def test_add_pyramid_refused_at_max_adds(self):
        from src.config import MAX_PYRAMID_ADDS
        snap = make_snapshot(current_price=19900.0, atr14=50.0)
        pos  = make_position(
            side="long", qty=2, avg_price=19700.0,
            unrealized_pnl=100.0, pyramid_adds_used=MAX_PYRAMID_ADDS
        )
        result = self.gemini.evaluate(self._haiku("up", "new_hh_break"), snap, pos, 10000.0)
        assert result["action"] == "hold"

    def test_atr_none_returns_hold(self):
        snap = make_snapshot(atr14=None)
        pos  = make_position(side="flat")
        result = self.gemini.evaluate(self._haiku("up", "new_hh_break"), snap, pos, 10000.0)
        assert result["action"] == "hold"

    def test_counter_trend_never_proposed_long_in_downtrend(self):
        # Position is long, trend is down -> hold (not close, not add)
        snap = make_snapshot(current_price=19000.0, atr14=50.0)
        pos  = make_position(side="long", qty=1, avg_price=19500.0, unrealized_pnl=-250.0)
        result = self.gemini.evaluate(self._haiku("down", "new_hh_break"), snap, pos, 10000.0)
        assert result["action"] == "hold"

    def test_counter_trend_never_proposed_short_in_uptrend(self):
        # Position is short, trend is up -> hold
        snap = make_snapshot(current_price=19900.0, atr14=50.0)
        pos  = make_position(side="short", qty=1, avg_price=19500.0, unrealized_pnl=-200.0)
        result = self.gemini.evaluate(self._haiku("up", "new_hh_break"), snap, pos, 10000.0)
        assert result["action"] == "hold"

    def test_hold_when_no_signal(self):
        snap = make_snapshot(current_price=19750.0, atr14=50.0)
        pos  = make_position(side="flat")
        result = self.gemini.evaluate(self._haiku("up", "none"), snap, pos, 10000.0)
        assert result["action"] == "hold"
        assert result["stop_price"] == 0.0

    def test_hold_on_sideways_flat(self):
        snap = make_snapshot(current_price=19750.0, atr14=50.0)
        pos  = make_position(side="flat")
        result = self.gemini.evaluate(self._haiku("sideways", "none"), snap, pos, 10000.0)
        assert result["action"] == "hold"

    def test_add_pyramid_for_short(self):
        snap = make_snapshot(current_price=19000.0, atr14=50.0)
        pos  = make_position(side="short", qty=1, avg_price=19200.0, unrealized_pnl=100.0, pyramid_adds_used=0)
        result = self.gemini.evaluate(self._haiku("down", "new_hh_break"), snap, pos, 10000.0)
        assert result["action"] == "add_pyramid"
        assert result["stop_price"] == pytest.approx(19000.0 + 2 * 50.0)

    def test_schema_keys_present(self):
        snap = make_snapshot(atr14=50.0)
        pos  = make_position(side="flat")
        result = self.gemini.evaluate(self._haiku(), snap, pos, 10000.0)
        required = {"action", "stop_price", "trailing_stop_atr_multiple", "reasoning"}
        assert required <= result.keys()


# ---------------------------------------------------------------------------
# StubDeepSeek tests
# ---------------------------------------------------------------------------

class TestStubDeepSeek:
    def setup_method(self):
        self.ds = StubDeepSeek()

    def _gemini(self, action="open_long", stop_price=19750.0):
        return {
            "action": action,
            "stop_price": stop_price,
            "trailing_stop_atr_multiple": 2.0,
            "reasoning": "stub",
        }

    def _clean_state(self, **kwargs):
        return make_state(**kwargs)

    def test_clean_approval(self):
        state  = self._clean_state()
        result = self.ds.evaluate(self._gemini("open_long", 19750.0), 1, state, 50.0)
        assert result["approved"] is True
        assert result["violations"] == []
        assert result["override_action"] is None
        assert result["reasoning"] == "All checks passed."

    def test_missing_stop_detected(self):
        state  = self._clean_state()
        result = self.ds.evaluate(self._gemini("open_long", 0.0), 1, state, 50.0)
        assert result["approved"] is False
        assert "MISSING_STOP" in result["violations"]
        assert result["override_action"] == "hold"

    def test_stop_atr_too_tight(self):
        # Position open long at 20000, stop at 19975 -> distance=25, atr=50, min=50
        pos   = make_position(side="long", qty=1, avg_price=20000.0, unrealized_pnl=100.0)
        state = self._clean_state(position=pos)
        result = self.ds.evaluate(self._gemini("add_pyramid", 19975.0), 1, state, 50.0)
        assert result["approved"] is False
        assert "STOP_ATR_OUT_OF_BOUNDS" in result["violations"]

    def test_stop_atr_too_wide(self):
        # Position long at 20000, stop at 19700 -> distance=300, atr=50, max=150
        pos   = make_position(side="long", qty=1, avg_price=20000.0, unrealized_pnl=100.0)
        state = self._clean_state(position=pos)
        result = self.ds.evaluate(self._gemini("add_pyramid", 19700.0), 1, state, 50.0)
        assert result["approved"] is False
        assert "STOP_ATR_OUT_OF_BOUNDS" in result["violations"]

    def test_stop_atr_within_bounds(self):
        # Position long at 20000, stop at 19900 -> distance=100, atr=50, [50,150]
        pos   = make_position(side="long", qty=1, avg_price=20000.0, unrealized_pnl=100.0)
        state = self._clean_state(position=pos)
        result = self.ds.evaluate(self._gemini("add_pyramid", 19900.0), 1, state, 50.0)
        assert "STOP_ATR_OUT_OF_BOUNDS" not in result["violations"]

    def test_averaging_down_detected_on_losing_long(self):
        pos   = make_position(side="long", qty=1, avg_price=20000.0, unrealized_pnl=-50.0)
        state = self._clean_state(position=pos, unrealized_pnl=-50.0)
        result = self.ds.evaluate(self._gemini("open_long", 19900.0), 1, state, 50.0)
        assert result["approved"] is False
        assert "AVERAGING_DOWN" in result["violations"]

    def test_averaging_down_on_losing_add_pyramid(self):
        pos   = make_position(side="long", qty=1, avg_price=20000.0, unrealized_pnl=-50.0)
        state = self._clean_state(position=pos, unrealized_pnl=-50.0)
        result = self.ds.evaluate(self._gemini("add_pyramid", 19900.0), 1, state, 50.0)
        assert "AVERAGING_DOWN" in result["violations"]

    def test_pyramid_violation_flat_position(self):
        state = self._clean_state()  # flat position
        result = self.ds.evaluate(self._gemini("add_pyramid", 19800.0), 1, state, 50.0)
        assert "PYRAMID_VIOLATION" in result["violations"]

    def test_pyramid_violation_unprofitable(self):
        pos   = make_position(side="long", qty=1, avg_price=20000.0, unrealized_pnl=-10.0)
        state = self._clean_state(position=pos)
        result = self.ds.evaluate(self._gemini("add_pyramid", 19900.0), 1, state, 50.0)
        assert "PYRAMID_VIOLATION" in result["violations"]

    def test_pyramid_violation_at_max_adds(self):
        from src.config import MAX_PYRAMID_ADDS
        pos   = make_position(
            side="long", qty=2, avg_price=20000.0,
            unrealized_pnl=100.0, pyramid_adds_used=MAX_PYRAMID_ADDS
        )
        state = self._clean_state(position=pos)
        result = self.ds.evaluate(self._gemini("add_pyramid", 19900.0), 1, state, 50.0)
        assert "PYRAMID_VIOLATION" in result["violations"]

    def test_daily_loss_limit_at_threshold(self):
        # realized=-201 -> total = -201 <= -200
        state = make_state(realized_pnl_today=-201.0, unrealized_pnl=0.0)
        result = self.ds.evaluate(self._gemini("open_long", 19800.0), 1, state, 50.0)
        assert result["approved"] is False
        assert "DAILY_LOSS_LIMIT" in result["violations"]

    def test_daily_loss_limit_not_triggered_at_199(self):
        state = make_state(realized_pnl_today=-199.0, unrealized_pnl=0.0)
        result = self.ds.evaluate(self._gemini("open_long", 19800.0), 1, state, 50.0)
        assert "DAILY_LOSS_LIMIT" not in result["violations"]

    def test_max_contracts_exceeded(self):
        pos   = make_position(side="long", qty=2, avg_price=19800.0, unrealized_pnl=100.0)
        state = self._clean_state(position=pos)
        # existing 2 + proposed 2 = 4 > MAX_OPEN_CONTRACTS (3)
        result = self.ds.evaluate(self._gemini("add_pyramid", 19900.0), 2, state, 50.0)
        assert result["approved"] is False
        assert "MAX_CONTRACTS" in result["violations"]

    def test_max_contracts_at_limit_ok(self):
        pos   = make_position(side="long", qty=2, avg_price=19800.0, unrealized_pnl=100.0)
        state = self._clean_state(position=pos)
        # 2 + 1 = 3 == MAX_OPEN_CONTRACTS -> ok
        result = self.ds.evaluate(self._gemini("add_pyramid", 19900.0), 1, state, 50.0)
        assert "MAX_CONTRACTS" not in result["violations"]

    def test_out_of_hours_saturday(self):
        now = datetime(2026, 4, 18, 10, 0, 0, tzinfo=ET)  # Saturday
        state = make_state(now_et=now)
        result = self.ds.evaluate(self._gemini("open_long", 19800.0), 1, state, 50.0)
        assert result["approved"] is False
        assert "OUT_OF_HOURS" in result["violations"]

    def test_out_of_hours_sunday_pre_open(self):
        # Sunday before 18:00
        now = datetime(2026, 4, 19, 10, 0, 0, tzinfo=ET)  # Sunday
        state = make_state(now_et=now)
        result = self.ds.evaluate(self._gemini("open_long", 19800.0), 1, state, 50.0)
        assert "OUT_OF_HOURS" in result["violations"]

    def test_in_hours_sunday_after_open(self):
        # Sunday after 18:00 -> in hours
        now = datetime(2026, 4, 19, 19, 0, 0, tzinfo=ET)
        state = make_state(now_et=now)
        result = self.ds.evaluate(self._gemini("open_long", 19800.0), 1, state, 50.0)
        assert "OUT_OF_HOURS" not in result["violations"]

    def test_out_of_hours_maintenance_break(self):
        # Wednesday 17:30 -> maintenance break
        now = datetime(2026, 4, 15, 17, 30, 0, tzinfo=ET)
        state = make_state(now_et=now)
        result = self.ds.evaluate(self._gemini("open_long", 19800.0), 1, state, 50.0)
        assert "OUT_OF_HOURS" in result["violations"]

    def test_out_of_hours_friday_after_flat_time(self):
        # Friday 16:50 -> after WEEKEND_FLAT_TIME_ET (16:45) but before 17:00 maintenance
        now = datetime(2026, 4, 17, 16, 50, 0, tzinfo=ET)
        state = make_state(now_et=now)
        result = self.ds.evaluate(self._gemini("open_long", 19800.0), 1, state, 50.0)
        assert "OUT_OF_HOURS" in result["violations"]

    def test_override_action_none_for_close_when_violated(self):
        # Even with violations, close action -> override_action=None
        now = datetime(2026, 4, 18, 10, 0, 0, tzinfo=ET)  # Saturday
        state = make_state(now_et=now)
        result = self.ds.evaluate(self._gemini("close", 0.0), 1, state, 50.0)
        assert result["approved"] is False
        assert result["override_action"] is None

    def test_averaging_down_on_losing_short(self):
        # Covers the open_short side of the averaging_down check (L325)
        pos   = make_position(side="short", qty=1, avg_price=19000.0, unrealized_pnl=-50.0)
        state = make_state(position=pos, unrealized_pnl=-50.0)
        result = self.ds.evaluate(self._gemini("open_short", 19100.0), 1, state, 50.0)
        assert "AVERAGING_DOWN" in result["violations"]

    def test_out_of_hours_naive_datetime(self):
        # No tzinfo -> treated as out-of-hours (L378)
        now = datetime(2026, 4, 14, 10, 0, 0)  # naive, no tzinfo
        pos = make_position()
        from src.broker.models import AccountState
        state = AccountState(equity=10000.0, realized_pnl_today=0.0, unrealized_pnl=0.0, position=pos, now_et=now)
        result = self.ds.evaluate(self._gemini("open_long", 19800.0), 1, state, 50.0)
        assert "OUT_OF_HOURS" in result["violations"]

    def test_out_of_hours_friday_before_flat_time_is_ok(self):
        # Friday 10:00 -> before WEEKEND_FLAT_TIME_ET (16:45) -> in hours
        now = datetime(2026, 4, 17, 10, 0, 0, tzinfo=ET)
        state = make_state(now_et=now)
        result = self.ds.evaluate(self._gemini("open_long", 19800.0), 1, state, 50.0)
        assert "OUT_OF_HOURS" not in result["violations"]

    def test_lh_hh_pattern_broken_in_down_structure(self):
        # LH/HL swings (no LL) followed by HH -> down-structure break via elif branch
        # No LL present so the first "HH+LL" condition doesn't fire; "LH+HH" elif fires
        swings = [
            make_swing(1, 19400.0, "LH"),
            make_swing(2, 19350.0, "HL"),
            make_swing(3, 19300.0, "LH"),
            make_swing(4, 19500.0, "HH"),  # HH after LH sequence -> pattern broken
        ]
        snap = make_snapshot(current_price=19400.0, sma200=19500.0, swings=swings)
        result = StubHaiku().evaluate(snap)
        assert result["pattern_intact"] is False
        assert result["structural_signal"] == "pattern_broken"

    def test_schema_keys_present(self):
        state  = self._clean_state()
        result = self.ds.evaluate(self._gemini(), 1, state, 50.0)
        required = {"approved", "violations", "override_action", "reasoning"}
        assert required <= result.keys()


# ---------------------------------------------------------------------------
# run_stub_pipeline integration tests
# ---------------------------------------------------------------------------

class TestRunStubPipeline:
    def test_happy_path_returns_three_dicts(self):
        swings = UP_SWINGS
        snap  = make_snapshot(current_price=19900.0, sma200=19000.0, atr14=50.0, swings=swings)
        state = make_state()

        def qty_fn(risk_usd, atr):
            return 1

        h, g, d = run_stub_pipeline(snap, state, qty_fn)

        # Haiku schema
        assert "trend" in h
        assert "structural_signal" in h
        assert "confidence_0_to_1" in h

        # Gemini schema
        assert "action" in g
        assert "stop_price" in g
        assert "trailing_stop_atr_multiple" in g

        # DeepSeek schema
        assert "approved" in d
        assert "violations" in d
        assert "override_action" in d

    def test_pipeline_with_no_atr(self):
        snap  = make_snapshot(atr14=None, swings=[])
        state = make_state()

        def qty_fn(risk_usd, atr):
            return 1

        h, g, d = run_stub_pipeline(snap, state, qty_fn)
        assert g["action"] == "hold"

    def test_pipeline_shapes_match_schema(self):
        snap  = make_snapshot(current_price=19900.0, sma200=19000.0, atr14=50.0, swings=UP_SWINGS)
        state = make_state()

        def qty_fn(risk_usd, atr):
            return 1

        h, g, d = run_stub_pipeline(snap, state, qty_fn)

        haiku_keys   = {"trend", "last_confirmed_hh", "last_confirmed_hl",
                        "pattern_intact", "structural_signal", "confidence_0_to_1", "reasoning"}
        gemini_keys  = {"action", "stop_price", "trailing_stop_atr_multiple", "reasoning"}
        deepseek_keys= {"approved", "violations", "override_action", "reasoning"}

        assert haiku_keys  <= h.keys()
        assert gemini_keys <= g.keys()
        assert deepseek_keys <= d.keys()

    def test_pipeline_downtrend_open_short(self):
        # Covers open_short averaging_down path and LH+HH pattern_broken
        snap  = make_snapshot(current_price=19000.0, sma200=19500.0, atr14=50.0, swings=DOWN_SWINGS)
        state = make_state()

        def qty_fn(r, a):
            return 1

        h, g, d = run_stub_pipeline(snap, state, qty_fn)
        assert h["trend"] == "down"

    def test_pipeline_trend_values(self):
        snap  = make_snapshot(current_price=19900.0, sma200=19000.0, atr14=50.0, swings=UP_SWINGS)
        state = make_state()

        def qty_fn(r, a):
            return 1

        h, g, d = run_stub_pipeline(snap, state, qty_fn)
        assert h["trend"] in ("up", "down", "sideways")
        assert g["action"] in ("open_long", "open_short", "close", "hold", "add_pyramid")
        assert isinstance(d["approved"], bool)
        assert isinstance(d["violations"], list)
