"""
src/broker/models.py
Lightweight dataclasses shared across the trading-bot codebase.
frozen=True where mutation would be a bug (ProposedOrder, Fill).
Order is mutable so status can be updated by the broker layer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Inbound (pre-validation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProposedOrder:
    """
    An order candidate produced by the LLM ensemble / signal layer.
    Not yet validated. The safety guard operates on this object.
    """
    side: Literal["long", "short"]
    action: Literal["open", "close", "add_pyramid"]
    entry_price: float
    stop_price: float
    qty: int
    atr: float          # current ATR used for stop-distance validation
    symbol: str = "MYM"


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """
    Current open position.  side='flat' means no open position.
    """
    side: Literal["long", "short", "flat"]
    qty: int
    avg_price: float
    unrealized_pnl: float
    pyramid_adds_used: int  # count of pyramid fills on this position


@dataclass
class AccountState:
    """
    Snapshot of account at the moment an order is evaluated.
    now_et MUST be timezone-aware (America/New_York).
    """
    equity: float
    realized_pnl_today: float
    unrealized_pnl: float
    position: Position
    now_et: datetime       # timezone-aware ET


# ---------------------------------------------------------------------------
# Post-validation (accepted / working orders)
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """
    An order approved by the safety guard and submitted (or to be submitted)
    to the broker.  status flows: pending -> submitted -> filled | rejected.
    """
    order_id: str
    symbol: str
    side: Literal["long", "short"]
    action: Literal["open", "close", "add_pyramid"]
    qty: int
    entry_price: float
    stop_price: float
    atr: float
    status: Literal["pending", "submitted", "filled", "rejected"] = "pending"
    created_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            object.__setattr__(self, "created_at", datetime.utcnow())


@dataclass(frozen=True)
class Fill:
    """
    Immutable record of an executed fill returned by the broker adapter.
    Matches the §9 schema fields used downstream (journal, P&L accounting).
    """
    fill_id: str
    order_id: str
    symbol: str
    side: Literal["long", "short"]
    action: Literal["open", "close", "add_pyramid"]
    qty: int
    fill_price: float
    commission_usd: float
    filled_at: datetime     # UTC
    realized_pnl: float     # 0.0 for opens; populated for closes
