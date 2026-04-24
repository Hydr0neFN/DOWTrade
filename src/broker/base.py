"""
src/broker/base.py
Abstract Broker interface that ALL broker implementations must satisfy.

Implementations MUST:
  - Refuse to instantiate against non-paper / non-sandbox endpoints.
  - Never log credentials, tokens, or account numbers in plaintext.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.broker.models import AccountState, Order, Position


# ---------------------------------------------------------------------------
# Thin data class for OHLCV bars (used by fetch_historical_bars)
# ---------------------------------------------------------------------------

class Bar:
    """
    Single OHLCV bar.
    ts_utc: unix epoch seconds (bar open time, UTC).
    """
    __slots__ = ("ts_utc", "open", "high", "low", "close", "volume")

    def __init__(
        self,
        ts_utc: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        self.ts_utc = ts_utc
        self.open   = open
        self.high   = high
        self.low    = low
        self.close  = close
        self.volume = volume

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Bar(ts={self.ts_utc}, O={self.open}, H={self.high}, "
            f"L={self.low}, C={self.close}, V={self.volume})"
        )


# ---------------------------------------------------------------------------
# Abstract Broker
# ---------------------------------------------------------------------------

class Broker(ABC):
    """
    Abstract broker interface.

    All implementations MUST refuse non-paper/non-sandbox endpoints at
    construction time — before any network call can be made.
    """

    @abstractmethod
    def connect(self) -> None:
        """
        Authenticate against the broker API and store the session token.
        Must raise RuntimeError if authentication fails (without leaking credentials).
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """
        Gracefully terminate the session (best-effort) and release resources.
        """
        ...

    @abstractmethod
    def get_account_state(self) -> AccountState:
        """
        Return a current snapshot of equity, P&L, and open position.
        now_et MUST be timezone-aware (America/New_York).
        """
        ...

    @abstractmethod
    def submit_bracket_order(self, order: Order) -> Order:
        """
        Submit a bracket order (entry + stop-loss) to the broker.
        Returns the same Order with broker_order_id set and status updated.
        """
        ...

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> None:
        """Cancel a working order by its broker-assigned ID."""
        ...

    @abstractmethod
    def get_open_orders(self) -> list[Order]:
        """Return all open / working orders on this account."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Position:
        """
        Return the current position for *symbol*.
        Must return a flat Position (side='flat', qty=0) if no position exists.
        """
        ...

    @abstractmethod
    def fetch_historical_bars(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        timeframe_min: int = 15,
    ) -> list[Bar]:
        """
        Return OHLCV bars for *symbol* between *start_ts* and *end_ts*
        (unix epoch seconds, UTC) with a bar width of *timeframe_min* minutes.
        """
        ...
