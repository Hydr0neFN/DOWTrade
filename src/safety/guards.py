"""
src/safety/guards.py
====================
FINAL WORD safety layer.

If final_check() rejects an order it is dropped unconditionally --
regardless of what the LLM ensemble decided.

Design constraints
------------------
* Pure function: no network, no DB, no LLM calls.
* Only side-effect: logging (logger "safety.guards").
* Deterministic given the same inputs.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, time
from typing import Callable, List, Optional
from zoneinfo import ZoneInfo

from src.config import (
    BROKER_ENV,
    MAX_DAILY_LOSS_USD,
    MAX_OPEN_CONTRACTS,
    MAX_PYRAMID_ADDS,
    MANDATORY_STOP_LOSS,
    NO_AVERAGING_DOWN,
    PAPER_ONLY,
    STOP_ATR_MAX_MULT,
    STOP_ATR_MIN_MULT,
    SYMBOL,
    TRADING_HOURS_ET,
    WEEKEND_FLAT_DAY,
    WEEKEND_FLAT_TIME_ET,
)
from src.broker.models import AccountState, Order, ProposedOrder

logger = logging.getLogger("safety.guards")

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class GuardDecision:
    approved: bool
    order: Optional[Order]      # None if rejected
    violations: List[str]       # rule names that tripped
    reason: str                 # short human-readable summary


# ---------------------------------------------------------------------------
# Individual rule checks
# Each returns None on PASS, or a violation-name string on FAIL.
# ---------------------------------------------------------------------------

def _check_paper_only() -> Optional[str]:
    """Re-assert paper-only / demo posture (defence-in-depth)."""
    if not PAPER_ONLY:
        return "PAPER_ONLY_DISABLED"
    if BROKER_ENV != "demo":
        return "BROKER_ENV_NOT_DEMO"
    return None


def _check_symbol(proposed: ProposedOrder) -> Optional[str]:
    """Proposed symbol must match the configured contract."""
    if proposed.symbol != SYMBOL:
        return "WRONG_SYMBOL"
    return None


def _check_mandatory_stop(proposed: ProposedOrder) -> Optional[str]:
    """
    Stop-loss must be present (non-zero) and on the correct side:
      long  -> stop < entry
      short -> stop > entry
    Close orders do not carry a meaningful stop; skip this check.
    """
    if not MANDATORY_STOP_LOSS:
        return None
    if proposed.action == "close":
        return None
    if not proposed.stop_price:
        return "MISSING_STOP_LOSS"
    if proposed.side == "long" and proposed.stop_price >= proposed.entry_price:
        return "STOP_WRONG_SIDE_LONG"
    if proposed.side == "short" and proposed.stop_price <= proposed.entry_price:
        return "STOP_WRONG_SIDE_SHORT"
    return None


def _check_stop_atr_bounds(proposed: ProposedOrder) -> Optional[str]:
    """
    Stop distance must be within [STOP_ATR_MIN_MULT, STOP_ATR_MAX_MULT] x ATR.
    Skipped for close orders (they do not carry a meaningful stop).
    """
    if proposed.action == "close":
        return None
    if proposed.atr <= 0:
        return "ATR_ZERO_OR_NEGATIVE"
    distance = abs(proposed.entry_price - proposed.stop_price)
    min_dist = STOP_ATR_MIN_MULT * proposed.atr
    max_dist = STOP_ATR_MAX_MULT * proposed.atr
    if distance < min_dist:
        return "STOP_TOO_TIGHT"
    if distance > max_dist:
        return "STOP_TOO_WIDE"
    return None


def _check_no_averaging_down(
    proposed: ProposedOrder, state: AccountState
) -> Optional[str]:
    """
    Reject any size-add into a losing position in the same direction.
    Covers action='open' (re-entry) and action='add_pyramid'.
    """
    if not NO_AVERAGING_DOWN:
        return None
    if proposed.action not in {"open", "add_pyramid"}:
        return None
    pos = state.position
    if pos.side == "flat":
        return None
    if pos.side == proposed.side and pos.unrealized_pnl < 0:
        return "AVERAGING_DOWN"
    return None


def _check_max_contracts(
    proposed: ProposedOrder, state: AccountState
) -> Optional[str]:
    """Existing qty + proposed qty must not exceed MAX_OPEN_CONTRACTS."""
    if proposed.action == "close":
        return None
    current_qty = state.position.qty if state.position.side != "flat" else 0
    if current_qty + proposed.qty > MAX_OPEN_CONTRACTS:
        return "MAX_CONTRACTS_EXCEEDED"
    return None


def _check_pyramid(
    proposed: ProposedOrder, state: AccountState
) -> Optional[str]:
    """
    Pyramid adds are only valid when:
      1. There is an open position on the same side.
      2. That position is profitable (unrealized_pnl > 0).
      3. pyramid_adds_used < MAX_PYRAMID_ADDS.
    """
    if proposed.action != "add_pyramid":
        return None
    pos = state.position
    if pos.side == "flat":
        return "PYRAMID_INTO_FLAT"
    if pos.side != proposed.side:
        return "PYRAMID_WRONG_SIDE"
    if pos.unrealized_pnl <= 0:
        return "PYRAMID_INTO_LOSING"
    if pos.pyramid_adds_used >= MAX_PYRAMID_ADDS:
        return "PYRAMID_LIMIT_REACHED"
    return None


def _check_daily_loss(state: AccountState) -> Optional[str]:
    """Total P&L today must not breach the daily loss limit."""
    total_pnl = state.realized_pnl_today + state.unrealized_pnl
    if total_pnl <= -MAX_DAILY_LOSS_USD:
        return "DAILY_LOSS_LIMIT"
    return None


def _parse_time(t: str) -> time:
    """Parse HH:MM string to datetime.time."""
    h, m = t.split(":")
    return time(int(h), int(m))


def _check_trading_hours(
    proposed: ProposedOrder, state: AccountState
) -> Optional[str]:
    """
    Valid trading window: Sun 18:00 ET -> Fri 17:00 ET.
    Daily maintenance break: 17:00-18:00 ET every day.
    Saturday: all orders rejected.
    Sunday before 18:00 ET: rejected.
    Friday after WEEKEND_FLAT_TIME_ET: new opens/adds rejected; closes allowed.

    state.now_et must be timezone-aware.
    """
    now = state.now_et
    if now.tzinfo is None:
        raise ValueError("state.now_et must be timezone-aware")
    now_et = now.astimezone(ET)

    dow = now_et.weekday()   # 0=Mon ... 6=Sun
    t   = now_et.time().replace(second=0, microsecond=0)

    session_open  = _parse_time(TRADING_HOURS_ET[0])   # 18:00
    session_close = _parse_time(TRADING_HOURS_ET[1])   # 17:00
    flat_time     = _parse_time(WEEKEND_FLAT_TIME_ET)  # 16:45

    # Saturday (5): no trading at all
    if dow == 5:
        return "MARKET_CLOSED_SATURDAY"

    # Sunday (6): only open after 18:00
    if dow == 6:
        if t < session_open:
            return "MARKET_CLOSED_SUNDAY_PRE_OPEN"
        return None  # Sun >= 18:00 is fine

    # Daily maintenance break 17:00-18:00 (applies Mon-Fri)
    if session_close <= t < session_open:
        return "DAILY_MAINTENANCE_BREAK"

    # Friday (4) after flat-time: closes still allowed, opens/adds rejected
    if dow == WEEKEND_FLAT_DAY and t >= flat_time:
        if proposed.action != "close":
            return "WEEKEND_FLAT_CUTOFF"

    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def final_check(proposed: ProposedOrder, state: AccountState) -> GuardDecision:
    """
    Run every safety rule.  Returns a GuardDecision -- the caller MUST
    honour the `approved` field and drop the order if False.

    This function is PURE except for logging.
    """
    violations: list[str] = []

    def _collect(v: Optional[str]) -> None:
        if v is not None:
            violations.append(v)

    _collect(_check_paper_only())
    _collect(_check_symbol(proposed))
    _collect(_check_mandatory_stop(proposed))
    _collect(_check_stop_atr_bounds(proposed))
    _collect(_check_no_averaging_down(proposed, state))
    _collect(_check_max_contracts(proposed, state))
    _collect(_check_pyramid(proposed, state))
    _collect(_check_daily_loss(state))
    _collect(_check_trading_hours(proposed, state))

    if violations:
        reason = "Rejected: " + "; ".join(violations)
        logger.warning(
            "Order REJECTED symbol=%s side=%s action=%s qty=%d | %s",
            proposed.symbol, proposed.side, proposed.action, proposed.qty, reason,
        )
        return GuardDecision(
            approved=False,
            order=None,
            violations=violations,
            reason=reason,
        )

    order = Order(
        order_id=str(uuid.uuid4()),
        symbol=proposed.symbol,
        side=proposed.side,
        action=proposed.action,
        qty=proposed.qty,
        entry_price=proposed.entry_price,
        stop_price=proposed.stop_price,
        atr=proposed.atr,
        status="pending",
    )
    reason = (
        "Approved: " + proposed.action + " " + str(proposed.qty) + "x "
        + proposed.symbol + " " + proposed.side
        + " entry=" + str(proposed.entry_price)
        + " stop=" + str(proposed.stop_price)
    )
    logger.info(
        "Order APPROVED order_id=%s symbol=%s side=%s action=%s qty=%d",
        order.order_id, order.symbol, order.side, order.action, order.qty,
    )
    return GuardDecision(
        approved=True,
        order=order,
        violations=[],
        reason=reason,
    )
