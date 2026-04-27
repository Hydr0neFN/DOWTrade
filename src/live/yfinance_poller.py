"""
src/live/yfinance_poller.py
============================
Drop-in replacement for DxLinkStreamer that sources 15m MYM bars from
yfinance. Used when the cert dxLink has no live market data subscription.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Callable, Optional

import yfinance as yf

log = logging.getLogger(__name__)


class YFinancePoller:
    """Polls yfinance for 15m bars and delivers completed ones via on_candle."""

    def __init__(self, ticker: str, on_candle: Callable[[str, dict], None]):
        self._ticker = ticker
        self.on_candle = on_candle
        self._last_ts_ms: int = 0
        self._running: bool = False
        self._symbol: str = ticker
        self._period_ms: int = 15 * 60 * 1000  # 15 minutes in ms

    async def connect(self) -> None:
        """No-op — yfinance needs no persistent connection."""
        log.info("YFinancePoller: ready (ticker=%s)", self._ticker)

    async def subscribe_candles(
        self, symbol: str, period: str = "15m", from_time_ms: int = 0
    ) -> None:
        """Index history to set _last_ts_ms so the polling loop knows the cutoff.

        Historical bars are already loaded into the BarWindow and DB by
        LiveRunner._hydrate_window. Re-delivering them here causes _process_loop
        to drop every one as a duplicate and never fire decisions.
        """
        self._symbol = symbol
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None, lambda: yf.download(self._ticker, period="5d", interval="15m", progress=False)
        )
        if df.empty:
            log.warning("YFinancePoller: initial download returned no data")
            return
        if hasattr(df.columns, "get_level_values"):
            df.columns = df.columns.get_level_values(0)
        now_ms = int(time.time() * 1000)
        indexed = 0
        for row in df.itertuples():
            ts_ms = int(row.Index.timestamp() * 1000)
            if ts_ms + self._period_ms <= now_ms:
                self._last_ts_ms = ts_ms
                indexed += 1
        log.info("YFinancePoller: indexed %d bars, _last_ts_ms=%d", indexed, self._last_ts_ms)

    async def run(self) -> None:
        """Poll yfinance every 60s and deliver any new completed bars."""
        self._running = True
        loop = asyncio.get_event_loop()
        log.info("YFinancePoller: polling loop started (60s interval)")

        while self._running:
            await asyncio.sleep(60)
            if not self._running:
                break
            try:
                df = await loop.run_in_executor(
                    None, lambda: yf.download(self._ticker, period="1d", interval="15m", progress=False)
                )
                if df.empty:
                    continue
                if hasattr(df.columns, "get_level_values"):
                    df.columns = df.columns.get_level_values(0)

                now_ms = int(time.time() * 1000)
                new_bars = 0
                for row in df.itertuples():
                    ts_ms = int(row.Index.timestamp() * 1000)
                    if ts_ms <= self._last_ts_ms:
                        continue
                    if ts_ms + self._period_ms > now_ms:
                        continue  # still in progress
                    candle = self._row_to_candle(row, ts_ms)
                    if candle is None:
                        continue
                    self.on_candle(self._symbol, candle)
                    self._last_ts_ms = ts_ms
                    new_bars += 1

                if new_bars:
                    log.info("YFinancePoller: poll found %d new bar(s)", new_bars)
            except Exception as exc:
                log.error("YFinancePoller: poll error: %s", exc)

    async def close(self) -> None:
        self._running = False
        log.info("YFinancePoller: stopped")

    # ------------------------------------------------------------------

    def _row_to_candle(self, row, ts_ms: int) -> Optional[dict]:
        """Convert a DataFrame row to the candle dict expected by on_candle."""
        def _f(x):
            try:
                v = float(x)
                return None if math.isnan(v) else v
            except (TypeError, ValueError):
                return None

        o = _f(getattr(row, "Open", None))
        h = _f(getattr(row, "High", None))
        lo = _f(getattr(row, "Low", None))
        c = _f(getattr(row, "Close", None))
        if None in (o, h, lo, c):
            return None

        v_raw = _f(getattr(row, "Volume", 0)) or 0.0
        return {
            "eventSymbol": self._symbol,
            "time": ts_ms,
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": v_raw,
        }
