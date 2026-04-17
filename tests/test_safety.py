"""
tests/test_safety.py
====================
Comprehensive unit tests for src/safety/guards.py.

Factory helpers keep individual test bodies short.
Covers every rule: happy-path approval + every rejection branch.
Target: >= 90% line coverage on src/safety/guards.py.
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on sys.path so src.* imports resolve.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from src.broker.models import AccountState, Position, ProposedOrder
from src.safety.guards import GuardDecision, final_check

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _flat_pos() -> Position:
    return Position(side="flat", qty=0, avg_price=0.0, unrealized_pnl=0.0, pyramid_adds_used=0)


def _long_pos(qty: int = 1, unrealized_pnl: float = 50.0, pyramid_adds_used: int = 0) -> Position:
    return Position(
        side="long",
        qty=qty,
        avg_price=40000.0,
        unrealized_pnl=unrealized_pnl,
        pyramid_adds_used=pyramid_adds_used,
    )


def _short_pos(qty: int = 1, unrealized_pnl: float = 50.0, pyramid_adds_used: int = 0) -> Position:
    return Position(
        side="short",
        qty=qty,
        avg_price=40000.0,
        unrealized_pnl=unrealized_pnl,
        pyramid_adds_used=pyramid_adds_used,
    )


def _weekday_et(hour: int = 10, minute: int = 0) -> datetime:
    """Return a timezone-aware Tuesday 10:00 ET datetime (safe trading hours)."""
    return datetime(2026, 4, 14, hour, minute, 0, tzinfo=ET)   # Tue Apr 14 2026


def make_state(
    *,
    realized_pnl_today: float = 0.0,
    unrealized_pnl: float = 0.0,
    position: Position | None = None,
    now_et: datetime | None = None,
    equity: float = 10_000.0,
) -> AccountState:
    return AccountState(
        equity=equity,
        realized_pnl_today=realized_pnl_today,
        unrealized_pnl=unrealized_pnl,
        position=position if position is not None else _flat_pos(),
        now_et=now_et if now_et is not None else _weekday_et(),
    )


def make_proposed(
    *,
    side: str = "long",
    action: str = "open",
    entry_price: float = 40000.0,
    stop_price: float = 39980.0,   # 20 pts below entry; ATR default = 10 -> 2x ATR
    qty: int = 1,
    atr: float = 10.0,
    symbol: str = "MYM",
) -> ProposedOrder:
    return ProposedOrder(
        side=side,
        action=action,
        entry_price=entry_price,
        stop_price=stop_price,
        qty=qty,
        atr=atr,
        symbol=symbol,
    )


# ---------------------------------------------------------------------------
# Helper asserts
# ---------------------------------------------------------------------------

def assert_approved(decision: GuardDecision) -> None:
    assert decision.approved, f"Expected approval; violations={decision.violations}"
    assert decision.order is not None
    assert decision.violations == []


def assert_rejected(decision: GuardDecision, *expected_violations: str) -> None:
    assert not decision.approved, "Expected rejection"
    assert decision.order is None
    for v in expected_violations:
        assert any(v in viol for viol in decision.violations), (
            f"Expected violation containing {v!r}; got {decision.violations}"
        )


# ===========================================================================
# 1. Happy-path approvals
# ===========================================================================

class TestHappyPath:
    def test_clean_long_open(self):
        decision = final_check(make_proposed(side="long", action="open"), make_state())
        assert_approved(decision)
        assert decision.order.side == "long"
        assert decision.order.action == "open"

    def test_clean_short_open(self):
        proposed = make_proposed(
            side="short", action="open",
            entry_price=40000.0, stop_price=40020.0  # stop above for short
        )
        decision = final_check(proposed, make_state())
        assert_approved(decision)

    def test_close_order_approved(self):
        """Close orders bypass ATR stop-distance check."""
        proposed = make_proposed(
            side="long", action="close",
            entry_price=40000.0, stop_price=0.0,  # no stop needed for close
            atr=10.0,
        )
        state = make_state(position=_long_pos())
        decision = final_check(proposed, state)
        assert_approved(decision)

    def test_approved_order_has_uuid(self):
        decision = final_check(make_proposed(), make_state())
        assert_approved(decision)
        import uuid
        uuid.UUID(decision.order.order_id)  # raises if invalid


# ===========================================================================
# 2. Symbol check (Rule 2)
# ===========================================================================

class TestSymbolCheck:
    def test_wrong_symbol_rejected(self):
        decision = final_check(make_proposed(symbol="ES"), make_state())
        assert_rejected(decision, "WRONG_SYMBOL")

    def test_correct_symbol_passes(self):
        decision = final_check(make_proposed(symbol="MYM"), make_state())
        assert_approved(decision)


# ===========================================================================
# 3. Mandatory stop (Rule 3)
# ===========================================================================

class TestMandatoryStop:
    def test_missing_stop_rejected(self):
        proposed = make_proposed(stop_price=0.0)
        assert_rejected(final_check(proposed, make_state()), "MISSING_STOP_LOSS")

    def test_stop_wrong_side_long(self):
        # For long: stop must be < entry; here stop > entry
        proposed = make_proposed(side="long", entry_price=40000.0, stop_price=40010.0)
        assert_rejected(final_check(proposed, make_state()), "STOP_WRONG_SIDE_LONG")

    def test_stop_equal_entry_long(self):
        proposed = make_proposed(side="long", entry_price=40000.0, stop_price=40000.0)
        assert_rejected(final_check(proposed, make_state()), "STOP_WRONG_SIDE_LONG")

    def test_stop_wrong_side_short(self):
        # For short: stop must be > entry; here stop < entry
        proposed = make_proposed(side="short", entry_price=40000.0, stop_price=39990.0)
        assert_rejected(final_check(proposed, make_state()), "STOP_WRONG_SIDE_SHORT")

    def test_valid_long_stop(self):
        proposed = make_proposed(side="long", entry_price=40000.0, stop_price=39980.0)
        assert_approved(final_check(proposed, make_state()))

    def test_valid_short_stop(self):
        proposed = make_proposed(side="short", entry_price=40000.0, stop_price=40020.0, atr=10.0)
        assert_approved(final_check(proposed, make_state()))


# ===========================================================================
# 4. ATR stop-distance bounds (Rule 4)
# ===========================================================================

class TestStopAtrBounds:
    """ATR=10; valid range: 10-30 pts. STOP_ATR_MIN_MULT=1.0, MAX=3.0."""

    def test_stop_too_tight(self):
        # distance = 5 < 1 * 10
        proposed = make_proposed(entry_price=40000.0, stop_price=39995.0, atr=10.0)
        assert_rejected(final_check(proposed, make_state()), "STOP_TOO_TIGHT")

    def test_stop_too_wide(self):
        # distance = 50 > 3 * 10
        proposed = make_proposed(entry_price=40000.0, stop_price=39950.0, atr=10.0)
        assert_rejected(final_check(proposed, make_state()), "STOP_TOO_WIDE")

    def test_stop_exactly_1x_atr_passes(self):
        proposed = make_proposed(entry_price=40000.0, stop_price=39990.0, atr=10.0)
        assert_approved(final_check(proposed, make_state()))

    def test_stop_exactly_3x_atr_passes(self):
        proposed = make_proposed(entry_price=40000.0, stop_price=39970.0, atr=10.0)
        assert_approved(final_check(proposed, make_state()))

    def test_close_bypasses_atr_check(self):
        """Close orders are exempt from ATR stop-distance check."""
        proposed = make_proposed(
            action="close", entry_price=40000.0, stop_price=39995.0, atr=10.0
        )
        state = make_state(position=_long_pos())
        assert_approved(final_check(proposed, state))

    def test_zero_atr_rejected(self):
        proposed = make_proposed(atr=0.0, stop_price=39990.0)
        assert_rejected(final_check(proposed, make_state()), "ATR_ZERO_OR_NEGATIVE")

    @pytest.mark.parametrize("distance,should_pass", [
        (10.0, True),   # exactly 1x
        (20.0, True),   # 2x
        (30.0, True),   # exactly 3x
        (9.9,  False),  # just under 1x
        (30.1, False),  # just over 3x
    ])
    def test_atr_boundary(self, distance: float, should_pass: bool):
        entry = 40000.0
        atr = 10.0
        stop = entry - distance  # long direction
        proposed = make_proposed(side="long", entry_price=entry, stop_price=stop, atr=atr)
        decision = final_check(proposed, make_state())
        if should_pass:
            assert_approved(decision)
        else:
            assert not decision.approved


# ===========================================================================
# 5. No averaging down (Rule 5)
# ===========================================================================

class TestNoAveragingDown:
    def test_averaging_down_long_rejected(self):
        state = make_state(position=_long_pos(unrealized_pnl=-10.0))
        proposed = make_proposed(side="long", action="open")
        assert_rejected(final_check(proposed, state), "AVERAGING_DOWN")

    def test_averaging_down_short_rejected(self):
        state = make_state(position=_short_pos(unrealized_pnl=-10.0))
        proposed = make_proposed(side="short", action="open", stop_price=40020.0)
        assert_rejected(final_check(proposed, state), "AVERAGING_DOWN")

    def test_averaging_into_profitable_long_ok(self):
        state = make_state(position=_long_pos(unrealized_pnl=50.0, qty=1))
        proposed = make_proposed(side="long", action="open", qty=1)
        assert_approved(final_check(proposed, state))

    def test_open_opposite_side_losing_ok(self):
        """Opening short while holding losing long is allowed (hedging)."""
        state = make_state(position=_long_pos(unrealized_pnl=-10.0))
        proposed = make_proposed(side="short", action="open", stop_price=40020.0)
        # May be rejected for other reasons (max contracts) but NOT averaging_down
        decision = final_check(proposed, state)
        assert "AVERAGING_DOWN" not in decision.violations

    def test_close_exempt_from_averaging_down(self):
        state = make_state(position=_long_pos(unrealized_pnl=-10.0))
        proposed = make_proposed(side="long", action="close", stop_price=0.0)
        decision = final_check(proposed, state)
        assert "AVERAGING_DOWN" not in decision.violations


# ===========================================================================
# 6. Max open contracts (Rule 6)
# ===========================================================================

class TestMaxContracts:
    def test_exceeds_max_rejected(self):
        # MAX_OPEN_CONTRACTS = 3; already hold 3
        state = make_state(position=_long_pos(qty=3))
        proposed = make_proposed(side="long", action="open", qty=1)
        assert_rejected(final_check(proposed, state), "MAX_CONTRACTS_EXCEEDED")

    def test_at_limit_rejected(self):
        # 2 existing + 2 proposed = 4 > 3
        state = make_state(position=_long_pos(qty=2))
        proposed = make_proposed(side="long", action="open", qty=2)
        assert_rejected(final_check(proposed, state), "MAX_CONTRACTS_EXCEEDED")

    def test_within_limit_passes(self):
        state = make_state(position=_long_pos(qty=1))
        proposed = make_proposed(side="long", action="open", qty=1)
        assert_approved(final_check(proposed, state))

    def test_close_exempt_from_contract_limit(self):
        state = make_state(position=_long_pos(qty=3))
        proposed = make_proposed(side="long", action="close", qty=3, stop_price=0.0)
        decision = final_check(proposed, state)
        assert "MAX_CONTRACTS_EXCEEDED" not in decision.violations

    def test_flat_position_counts_zero(self):
        state = make_state(position=_flat_pos())
        proposed = make_proposed(side="long", action="open", qty=3)
        # 0 + 3 = 3 == MAX_OPEN_CONTRACTS -> allowed
        assert_approved(final_check(proposed, state))


# ===========================================================================
# 7. Pyramid rules (Rule 7)
# ===========================================================================

class TestPyramidCheck:
    def _profitable_long_state(self, pyramid_adds_used: int = 0, qty: int = 1) -> AccountState:
        return make_state(position=_long_pos(
            unrealized_pnl=100.0, pyramid_adds_used=pyramid_adds_used, qty=qty
        ))

    def test_valid_pyramid_long(self):
        state = self._profitable_long_state(pyramid_adds_used=0, qty=1)
        proposed = make_proposed(side="long", action="add_pyramid", qty=1)
        assert_approved(final_check(proposed, state))

    def test_pyramid_into_flat_rejected(self):
        state = make_state(position=_flat_pos())
        proposed = make_proposed(side="long", action="add_pyramid")
        assert_rejected(final_check(proposed, state), "PYRAMID_INTO_FLAT")

    def test_pyramid_wrong_side_rejected(self):
        state = make_state(position=_short_pos(unrealized_pnl=100.0))
        proposed = make_proposed(side="long", action="add_pyramid", stop_price=39980.0)
        assert_rejected(final_check(proposed, state), "PYRAMID_WRONG_SIDE")

    def test_pyramid_into_losing_rejected(self):
        state = make_state(position=_long_pos(unrealized_pnl=-5.0))
        proposed = make_proposed(side="long", action="add_pyramid")
        assert_rejected(final_check(proposed, state), "PYRAMID_INTO_LOSING")

    def test_pyramid_into_breakeven_rejected(self):
        """unrealized_pnl == 0 counts as not profitable."""
        state = make_state(position=_long_pos(unrealized_pnl=0.0))
        proposed = make_proposed(side="long", action="add_pyramid")
        assert_rejected(final_check(proposed, state), "PYRAMID_INTO_LOSING")

    def test_pyramid_limit_reached_rejected(self):
        # MAX_PYRAMID_ADDS = 2; already used 2
        state = self._profitable_long_state(pyramid_adds_used=2, qty=2)
        proposed = make_proposed(side="long", action="add_pyramid", qty=1)
        assert_rejected(final_check(proposed, state), "PYRAMID_LIMIT_REACHED")

    def test_pyramid_second_add_allowed(self):
        # used 1 of 2 allowed
        state = self._profitable_long_state(pyramid_adds_used=1, qty=2)
        proposed = make_proposed(side="long", action="add_pyramid", qty=1)
        assert_approved(final_check(proposed, state))


# ===========================================================================
# 8. Daily loss limit (Rule 8)
# ===========================================================================

class TestDailyLossLimit:
    def test_at_limit_rejected(self):
        # -200 == -MAX_DAILY_LOSS_USD (200)
        state = make_state(realized_pnl_today=-200.0, unrealized_pnl=0.0)
        assert_rejected(final_check(make_proposed(), state), "DAILY_LOSS_LIMIT")

    def test_over_limit_rejected(self):
        state = make_state(realized_pnl_today=-150.0, unrealized_pnl=-60.0)
        assert_rejected(final_check(make_proposed(), state), "DAILY_LOSS_LIMIT")

    def test_just_under_limit_passes(self):
        state = make_state(realized_pnl_today=-199.0, unrealized_pnl=0.0)
        assert_approved(final_check(make_proposed(), state))

    def test_positive_pnl_passes(self):
        state = make_state(realized_pnl_today=100.0, unrealized_pnl=50.0)
        assert_approved(final_check(make_proposed(), state))


# ===========================================================================
# 9. Trading hours (Rule 9)
# ===========================================================================

def _dt_et(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=ET)


class TestTradingHours:
    # --- Saturday ---
    def test_saturday_rejected(self):
        # Sat Apr 18 2026
        state = make_state(now_et=_dt_et(2026, 4, 18, 10, 0))
        assert_rejected(final_check(make_proposed(), state), "MARKET_CLOSED_SATURDAY")

    def test_saturday_midnight_rejected(self):
        state = make_state(now_et=_dt_et(2026, 4, 18, 0, 0))
        assert_rejected(final_check(make_proposed(), state), "MARKET_CLOSED_SATURDAY")

    # --- Sunday before 18:00 ---
    def test_sunday_10am_rejected(self):
        # Sun Apr 19 2026 10:00 ET
        state = make_state(now_et=_dt_et(2026, 4, 19, 10, 0))
        assert_rejected(final_check(make_proposed(), state), "MARKET_CLOSED_SUNDAY_PRE_OPEN")

    def test_sunday_1759_rejected(self):
        state = make_state(now_et=_dt_et(2026, 4, 19, 17, 59))
        assert_rejected(final_check(make_proposed(), state), "MARKET_CLOSED_SUNDAY_PRE_OPEN")

    def test_sunday_1800_approved(self):
        state = make_state(now_et=_dt_et(2026, 4, 19, 18, 0))
        assert_approved(final_check(make_proposed(), state))

    # --- Daily maintenance break 17:00-18:00 Mon-Fri ---
    def test_daily_break_1730_rejected(self):
        # Wed Apr 15 2026 17:30 ET
        state = make_state(now_et=_dt_et(2026, 4, 15, 17, 30))
        assert_rejected(final_check(make_proposed(), state), "DAILY_MAINTENANCE_BREAK")

    def test_daily_break_1700_rejected(self):
        state = make_state(now_et=_dt_et(2026, 4, 15, 17, 0))
        assert_rejected(final_check(make_proposed(), state), "DAILY_MAINTENANCE_BREAK")

    def test_daily_break_1759_rejected(self):
        state = make_state(now_et=_dt_et(2026, 4, 15, 17, 59))
        assert_rejected(final_check(make_proposed(), state), "DAILY_MAINTENANCE_BREAK")

    def test_after_break_1800_approved(self):
        state = make_state(now_et=_dt_et(2026, 4, 15, 18, 0))
        assert_approved(final_check(make_proposed(), state))

    # --- Friday WEEKEND_FLAT_CUTOFF (16:45 ET) ---
    def test_friday_1645_open_rejected(self):
        # Fri Apr 17 2026 16:45 ET
        state = make_state(now_et=_dt_et(2026, 4, 17, 16, 45))
        proposed = make_proposed(action="open")
        assert_rejected(final_check(proposed, state), "WEEKEND_FLAT_CUTOFF")

    def test_friday_1700_open_rejected(self):
        state = make_state(now_et=_dt_et(2026, 4, 17, 16, 50))
        proposed = make_proposed(action="open")
        assert_rejected(final_check(proposed, state), "WEEKEND_FLAT_CUTOFF")

    def test_friday_1645_close_allowed(self):
        """Closes should pass the WEEKEND_FLAT_CUTOFF rule."""
        state = make_state(
            position=_long_pos(), now_et=_dt_et(2026, 4, 17, 16, 45)
        )
        proposed = make_proposed(action="close", stop_price=0.0)
        decision = final_check(proposed, state)
        assert "WEEKEND_FLAT_CUTOFF" not in decision.violations

    def test_friday_before_cutoff_approved(self):
        # Fri 10:00 ET - well before 16:45
        state = make_state(now_et=_dt_et(2026, 4, 17, 10, 0))
        assert_approved(final_check(make_proposed(), state))

    # --- Normal weekday midday ---
    def test_tuesday_midday_approved(self):
        state = make_state(now_et=_dt_et(2026, 4, 14, 14, 0))
        assert_approved(final_check(make_proposed(), state))

    def test_monday_morning_approved(self):
        state = make_state(now_et=_dt_et(2026, 4, 13, 9, 30))
        assert_approved(final_check(make_proposed(), state))


# ===========================================================================
# 10. Multiple violations accumulate
# ===========================================================================

class TestMultipleViolations:
    def test_symbol_and_daily_loss(self):
        state = make_state(realized_pnl_today=-200.0)
        proposed = make_proposed(symbol="ES")
        decision = final_check(proposed, state)
        assert not decision.approved
        assert any("WRONG_SYMBOL" in v for v in decision.violations)
        assert any("DAILY_LOSS_LIMIT" in v for v in decision.violations)


# ===========================================================================
# 11. Order fields on approval
# ===========================================================================

class TestApprovedOrderFields:
    def test_order_fields_match_proposed(self):
        proposed = make_proposed(
            side="long", action="open",
            entry_price=40000.0, stop_price=39980.0, qty=1, atr=10.0
        )
        decision = final_check(proposed, make_state())
        assert_approved(decision)
        o = decision.order
        assert o.symbol == "MYM"
        assert o.side == "long"
        assert o.action == "open"
        assert o.qty == 1
        assert o.entry_price == 40000.0
        assert o.stop_price == 39980.0
        assert o.atr == 10.0
        assert o.status == "pending"

    def test_reason_contains_action(self):
        decision = final_check(make_proposed(action="open"), make_state())
        assert_approved(decision)
        assert "open" in decision.reason.lower()

    def test_rejection_reason_mentions_violation(self):
        proposed = make_proposed(stop_price=0.0)
        decision = final_check(proposed, make_state())
        assert "MISSING_STOP_LOSS" in decision.reason
