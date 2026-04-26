"""
src/broker/tastytrade.py
Tastytrade certification/sandbox broker implementation.

PAPER-ONLY: This module HARD-REFUSES any non-sandbox URL at construction time.
Only api.cert.tastyworks.com is on the allowlist.

Tastytrade API quirks documented here:
  - Auth endpoint: POST /sessions -> {data: {session-token: ...}}
  - Header name: Authorization with NO Bearer prefix (unusual).
  - Token TTL: ~24h; we re-login automatically on 401.
  - Futures symbols: /MYMcurrent_front_month e.g. /MYMM5 for Jun-2025.
  - Order price-effect: Debit for buys (long), Credit for sells (short).
    TODO(Phase 5): Verify Tastytrade exact futures order payload shape
    against https://developer.tastytrade.com -- the bracket/OCO structure
    for futures may differ from equity options.
  - Historical bars: available via dxLink CANDLE subscription (Phase 6).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx

from src.broker.base import Bar, Broker
from src.broker.models import AccountState, Order, Position
from src.config import Settings

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

BROKER_NAME: str = "tastytrade-cert"  # injected into every log line
_ET = ZoneInfo("America/New_York")

# Front-month letter codes for futures (IMM calendar)
# Jan=F Feb=G Mar=H Apr=J May=K Jun=M Jul=N Aug=Q Sep=U Oct=V Nov=X Dec=Z
_MONTH_CODES: dict[int, str] = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}

log = logging.getLogger(BROKER_NAME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _redact(text: str, *secrets: str) -> str:
    """Replace every occurrence of a secret in text with ***."""
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return text



def _current_front_month_code(now: Optional[datetime] = None) -> str:
    """
    Return the approximate front-month letter+year-digit for MYM.
    MYM front-months roll quarterly: Mar(H), Jun(M), Sep(U), Dec(Z).
    We advance to the next quarterly month after the 3rd Friday of expiry month.
    """
    if now is None:
        now = datetime.now(_ET)
    year = now.year
    month = now.month
    day = now.day

    import calendar
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    monthcal = c.monthdatescalendar(year, month)
    fridays = [d for week in monthcal for d in week if d.weekday() == calendar.FRIDAY and d.month == month]
    third_friday = fridays[2] if len(fridays) >= 3 else now.date()
    
    roll = False
    if day > third_friday.day:
        roll = True

    quarterly = [3, 6, 9, 12]
    target_q = None
    target_year = year
    for q in quarterly:
        if month < q or (month == q and not roll):
            target_q = q
            break
    if target_q is None:
        target_q = 3
        target_year += 1

    return _MONTH_CODES[target_q] + str(target_year % 10)

def dxfeed_symbol(root: str, now: Optional[datetime] = None) -> str:
    return to_tastytrade_symbol(root, now) + ":XCME"


def to_tastytrade_symbol(root: str, now: Optional[datetime] = None) -> str:
    """
    Convert a root symbol like MYM to a Tastytrade futures symbol like /MYMM5.
    Always prefixed with / for futures.

    Examples (assuming today is 2026-04-25):
      to_tastytrade_symbol("MYM") -> "/MYMM6"   (Jun 2026 front month)
    """
    code = _current_front_month_code(now)
    return f"/{root}{code}"


# ---------------------------------------------------------------------------
# TastytradeBroker
# ---------------------------------------------------------------------------

class TastytradeBroker(Broker):
    """
    Tastytrade certification-environment broker.

    Construction raises RuntimeError immediately for any non-sandbox URL,
    before any attribute is set (defence-in-depth).
    """

    SANDBOX_HOSTS: frozenset = frozenset({"api.cert.tastyworks.com"})

    def __init__(self, settings: Settings) -> None:
        # --- URL guard FIRST -- no attributes set before this check ---
        parsed = urlparse(settings.tastytrade_base_url)
        host: str = (parsed.hostname or "").lower()
        if host not in self.SANDBOX_HOSTS:
            raise RuntimeError(
                f"[{BROKER_NAME}] Refusing to instantiate TastytradeBroker with "
                f"non-sandbox host {host!r}. "
                f"PAPER_ONLY safety: only {set(self.SANDBOX_HOSTS)} are allowed."
            )

        # --- Config-level paper posture assertion ---
        from src.config import assert_safety_posture
        assert_safety_posture(settings)

        self._settings = settings
        self._session_token: Optional[str] = None
        self._account_number: Optional[str] = None
        self._client = httpx.Client(
            base_url=settings.tastytrade_base_url,
            timeout=15.0,
        )
        log.info("[%s] Broker instantiated (host=%s)", BROKER_NAME, host)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        POST /sessions with username + password.
        Stores the session-token. Raises RuntimeError on auth failure
        WITHOUT logging the password.
        """
        payload = {
            "login": self._settings.tastytrade_cert_username,
            "password": self._settings.tastytrade_cert_password,
        }
        try:
            resp = self._client.post("/sessions", json=payload)
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"[{BROKER_NAME}] Network error during connect: {exc}"
            ) from exc

        if resp.status_code != 201:
            raise RuntimeError(
                f"[{BROKER_NAME}] Authentication failed (HTTP {resp.status_code}). "
                "Check TASTYTRADE_CERT_USERNAME / TASTYTRADE_CERT_PASSWORD."
            )

        try:
            token = resp.json()["data"]["session-token"]
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                f"[{BROKER_NAME}] Unexpected /sessions response shape: {exc}"
            ) from exc

        self._session_token = token
        log.info("[%s] Session established (token=***)", BROKER_NAME)

    def disconnect(self) -> None:
        """
        DELETE /sessions (best-effort) then close the httpx client.
        Errors are swallowed -- disconnect must never raise.
        """
        if self._session_token:
            try:
                self._client.delete(
                    "/sessions",
                    headers={"Authorization": self._session_token},
                )
            except Exception:
                pass
        try:
            self._client.close()
        except Exception:
            pass
        self._session_token = None
        log.info("[%s] Session disconnected", BROKER_NAME)

    # ------------------------------------------------------------------
    # Internal request helper
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """
        Execute method HTTP request against path, injecting Authorization header.
        On 401, reconnects once and retries.
        Raises RuntimeError for non-2xx responses (body excerpt, token redacted).
        """
        if self._session_token is None:
            self.connect()

        headers = kwargs.pop("headers", {})
        headers["Authorization"] = self._session_token

        resp = self._client.request(method, path, headers=headers, **kwargs)

        if resp.status_code == 401:
            log.warning("[%s] 401 received -- reconnecting", BROKER_NAME)
            self.connect()
            headers["Authorization"] = self._session_token
            resp = self._client.request(method, path, headers=headers, **kwargs)

        if resp.status_code >= 400:
            body_excerpt = resp.text[:200]
            safe_excerpt = _redact(
                body_excerpt,
                self._session_token or "",
                self._settings.tastytrade_cert_password,
                self._account_number or "",
            )
            raise RuntimeError(
                f"[{BROKER_NAME}] HTTP {resp.status_code} {method} {path}: "
                f"{safe_excerpt}"
            )

        return resp

    # ------------------------------------------------------------------
    # Account helpers
    # ------------------------------------------------------------------

    def _get_account_number(self) -> str:
        """
        Fetch and cache the cert account number from /customers/me/accounts.
        Raises if no accounts are found.
        """
        if self._account_number:
            return self._account_number

        resp = self._request("GET", "/customers/me/accounts")
        accounts = resp.json().get("data", {}).get("items", [])
        if not accounts:
            raise RuntimeError(f"[{BROKER_NAME}] No accounts found for this user.")

        acct = accounts[0].get("account", {}).get("account-number", "")
        if not acct:
            raise RuntimeError(f"[{BROKER_NAME}] Could not parse account number.")
        self._account_number = acct
        log.info("[%s] Account number cached (***)", BROKER_NAME)
        return acct

    # ------------------------------------------------------------------
    # Broker interface implementation
    # ------------------------------------------------------------------

    def get_account_state(self) -> AccountState:
        """
        Fetch balances and positions; return an AccountState snapshot.

        Tastytrade balance fields used:
          cash-balance          -> base cash
          long-equity-value     -> long equity mark value
          realized-day-gain     -> today realized P&L (may be absent -> 0)
        Position unrealized-day-gain-value -> sum across all positions.
        """
        acct = self._get_account_number()

        bal_resp = self._request("GET", f"/accounts/{acct}/balances")
        pos_resp = self._request("GET", f"/accounts/{acct}/positions")

        bal = bal_resp.json().get("data", {})
        positions_raw = pos_resp.json().get("data", {}).get("items", [])

        cash_balance: float = float(bal.get("cash-balance", 0) or 0)
        long_equity: float = float(bal.get("long-equity-value", 0) or 0)
        equity: float = cash_balance + long_equity

        realized_today: float = float(bal.get("realized-day-gain", 0) or 0)

        unrealized_total: float = sum(
            float(p.get("unrealized-day-gain-value", 0) or 0)
            for p in positions_raw
        )

        mym_positions = [
            p for p in positions_raw
            if p.get("instrument-type") == "Future"
        ]

        position: Position
        if mym_positions:
            raw = mym_positions[0]
            qty_signed: int = int(raw.get("quantity", 0) or 0)
            direction = raw.get("quantity-direction", "Long")
            side_str = "long" if direction == "Long" else "short"
            avg_price: float = float(raw.get("average-open-price", 0) or 0)
            unreal: float = float(raw.get("unrealized-day-gain-value", 0) or 0)
            position = Position(
                side=side_str,
                qty=abs(qty_signed),
                avg_price=avg_price,
                unrealized_pnl=unreal,
                pyramid_adds_used=0,
            )
        else:
            position = Position(
                side="flat",
                qty=0,
                avg_price=0.0,
                unrealized_pnl=0.0,
                pyramid_adds_used=0,
            )

        now_et = datetime.now(_ET)

        return AccountState(
            equity=equity,
            realized_pnl_today=realized_today,
            unrealized_pnl=unrealized_total,
            position=position,
            now_et=now_et,
        )

    def submit_bracket_order(self, order: Order) -> Order:
        """
        POST /accounts/{acct}/orders with entry order + stop-loss order.

        Tastytrade uses price-effect = Debit for buys, Credit for sells.

        TODO(Phase 5): Tastytrade bracket/OCO for futures is not fully
        documented in the sandbox. Currently submitting two separate orders
        (entry + stop). If native OCO exists for futures, use it.
        Verify against https://developer.tastytrade.com/order-management/

        Tastytrade returns HTTP 201 on order creation.
        """
        acct = self._get_account_number()
        tt_symbol = to_tastytrade_symbol(order.symbol)

        is_buy = order.side == "long"
        price_effect = "Debit" if is_buy else "Credit"
        action = "Buy to Open" if is_buy else "Sell to Open"
        stop_action = "Sell to Close" if is_buy else "Buy to Close"
        stop_price_effect = "Credit" if is_buy else "Debit"

        order_type = "Limit" if order.entry_price > 0 else "Market"

        payload: dict = {
            "order-type": order_type,
            "price-effect": price_effect,
            "time-in-force": "GTC",
            "legs": [
                {
                    "instrument-type": "Future",
                    "symbol": tt_symbol,
                    "quantity": str(order.qty),
                    "action": action,
                },
            ],
        }
        if order_type == "Limit":
            payload["price"] = str(order.entry_price)

        log.info(
            "[%s] Submitting entry order symbol=%s side=%s qty=%d",
            BROKER_NAME, tt_symbol, order.side, order.qty,
        )

        entry_resp = self._request("POST", f"/accounts/{acct}/orders", json=payload)
        entry_data = entry_resp.json().get("data", {}).get("order", {})
        broker_order_id: str = str(entry_data.get("id", ""))
        raw_status: str = str(entry_data.get("status", "Received")).lower()

        status_map = {
            "received": "submitted",
            "live": "submitted",
            "filled": "filled",
            "cancelled": "rejected",
            "rejected": "rejected",
        }
        our_status = status_map.get(raw_status, "submitted")

        stop_payload = {
            "order-type": "Stop",
            "stop-trigger": str(order.stop_price),
            "price-effect": stop_price_effect,
            "time-in-force": "GTC",
            "legs": [
                {
                    "instrument-type": "Future",
                    "symbol": tt_symbol,
                    "quantity": str(order.qty),
                    "action": stop_action,
                },
            ],
        }
        try:
            self._request("POST", f"/accounts/{acct}/orders", json=stop_payload)
            log.info(
                "[%s] Stop-loss order submitted symbol=%s stop_price=%s",
                BROKER_NAME, tt_symbol, order.stop_price,
            )
        except RuntimeError as exc:
            log.warning(
                "[%s] Stop-loss submission failed (entry order still placed): %s",
                BROKER_NAME, exc,
            )

        order.status = our_status
        # TODO(Phase 5): Add broker_order_id field to Order dataclass in models.py
        # For now, overwrite order_id with broker id if available
        if broker_order_id:
            order.order_id = broker_order_id
        log.info("[%s] Order submitted broker_id=*** status=%s", BROKER_NAME, our_status)
        return order

    def cancel_order(self, broker_order_id: str) -> None:
        """DELETE /accounts/{acct}/orders/{id}."""
        acct = self._get_account_number()
        self._request("DELETE", f"/accounts/{acct}/orders/{broker_order_id}")
        log.info("[%s] Order *** cancelled", BROKER_NAME)

    def get_open_orders(self) -> list[Order]:
        """
        GET /accounts/{acct}/orders/live -- returns all working orders.

        TODO(Phase 5): Map Tastytrade order JSON fully to our Order dataclass.
        Currently returns best-effort list; some fields may default.
        """
        acct = self._get_account_number()
        resp = self._request("GET", f"/accounts/{acct}/orders/live")
        items = resp.json().get("data", {}).get("items", [])

        orders: list[Order] = []
        for item in items:
            try:
                legs = item.get("legs", [{}])
                leg = legs[0] if legs else {}
                action_raw = leg.get("action", "Buy to Open")
                side: str = "long" if "Buy" in action_raw else "short"
                qty = int(leg.get("remaining-quantity", leg.get("quantity", 1)) or 1)
                symbol_raw = leg.get("symbol", "MYM")
                root = re.sub(r"^/([A-Z]+)[A-Z]\d$", r"\1", symbol_raw)
                o = Order(
                    order_id=str(item.get("id", "")),
                    symbol=root,
                    side=side,
                    action="open",
                    qty=qty,
                    entry_price=float(item.get("price", 0) or 0),
                    stop_price=float(item.get("stop-trigger", 0) or 0),
                    atr=0.0,
                    status="submitted",
                )
                orders.append(o)
            except Exception as exc:
                log.warning("[%s] Could not parse open order item: %s", BROKER_NAME, exc)

        return orders

    def get_position(self, symbol: str) -> Position:
        """
        Return current position for symbol root (e.g. MYM).
        Returns flat Position if no matching position exists.

        Tastytrade futures symbols look like /MYMM5; we match on root prefix.
        """
        acct = self._get_account_number()
        resp = self._request("GET", f"/accounts/{acct}/positions")
        items = resp.json().get("data", {}).get("items", [])

        symbol_upper = symbol.upper()
        for p in items:
            raw_symbol: str = p.get("symbol", "")
            root = re.sub(r"^/([A-Z]+)[A-Z]\d$", r"\1", raw_symbol)
            if root == symbol_upper:
                qty_signed = int(p.get("quantity", 0) or 0)
                direction = p.get("quantity-direction", "Long")
                side_str = "long" if direction == "Long" else "short"
                return Position(
                    side=side_str,
                    qty=abs(qty_signed),
                    avg_price=float(p.get("average-open-price", 0) or 0),
                    unrealized_pnl=float(p.get("unrealized-day-gain-value", 0) or 0),
                    pyramid_adds_used=0,
                )

        return Position(side="flat", qty=0, avg_price=0.0, unrealized_pnl=0.0, pyramid_adds_used=0)


    def get_dxlink_token(self) -> dict:
        """GET /api-quote-tokens with the session token. Returns {token, dxlink-url, level}."""
        resp = self._request("GET", "/api-quote-tokens")
        data = resp.json().get("data", {})
        return {
            "token": data.get("token", ""),
            "dxlink-url": data.get("dxlink-url", ""),
            "level": data.get("level", "")
        }

    def fetch_historical_bars(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        timeframe_min: int = 15,
    ) -> list[Bar]:
        import asyncio
        from src.live.dxlink import DxLinkStreamer
        
        token_data = self.get_dxlink_token()
        dxlink_url = token_data.get("dxlink-url", "")
        dxlink_token = token_data.get("token", "")
        
        bars = []
        
        def on_candle(sym: str, candle: dict):
            ts = candle.get("time", 0) // 1000
            if not (start_ts <= ts <= end_ts):
                return
            # Drop dxFeed snapshot tombstones: a single event at exactly fromTime
            # with all OHLC = NaN/0 means "no history available" (cert sandbox).
            import math as _m
            def _f(x):
                try:
                    v = float(x)
                    return None if _m.isnan(v) else v
                except (TypeError, ValueError):
                    return None
            o, h, lo, c = (_f(candle.get(k, 0.0)) for k in ("open", "high", "low", "close"))
            if None in (o, h, lo, c) or (o == 0.0 and h == 0.0 and lo == 0.0 and c == 0.0):
                return
            v = _f(candle.get("volume", 0))
            vol_int = 0 if v is None else int(v)
            bars.append(Bar(ts_utc=ts, open=o, high=h, low=lo, close=c, volume=vol_int))

        async def run_fetch():
            streamer = DxLinkStreamer(dxlink_url, dxlink_token, on_candle)
            await streamer.connect()
            period_str = f"{timeframe_min}m"
            target_sym = dxfeed_symbol(symbol) + f"{{={period_str}}}"
            await streamer.subscribe_candles(target_sym, period_str, from_time_ms=start_ts * 1000)
            
            run_task = asyncio.create_task(streamer.run())
            
            # Wait up to 30s, but exit early once the stream stalls (no new bars
            # for 3s) AND we have at least some data, OR we have hit end_ts.
            stall_window = 3.0
            last_count = 0
            last_change = asyncio.get_event_loop().time()
            deadline = asyncio.get_event_loop().time() + 30.0
            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.25)
                cur_count = len(bars)
                if cur_count != last_count:
                    last_count = cur_count
                    last_change = asyncio.get_event_loop().time()
                if bars and bars[-1].ts_utc >= end_ts:
                    break
                if cur_count > 0 and (asyncio.get_event_loop().time() - last_change) > stall_window:
                    break
            
            await streamer.close()
            try:
                await run_task
            except Exception:
                pass
            
        import threading
        def _run_in_thread():
            asyncio.run(run_fetch())
        t = threading.Thread(target=_run_in_thread)
        t.start()
        t.join()
        
        unique_bars = {}
        for b in bars:
            unique_bars[b.ts_utc] = b
        res = list(unique_bars.values())
        res.sort(key=lambda b: b.ts_utc)
        return res
