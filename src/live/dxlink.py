import asyncio
import json
import logging
from typing import Callable, Optional

import websockets

log = logging.getLogger(__name__)

class DxLinkStreamer:
    def __init__(self, dxlink_url: str, dxlink_token: str, on_candle: Callable[[str, dict], None]):
        self.dxlink_url = dxlink_url
        self.dxlink_token = dxlink_token
        self.on_candle = on_candle
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._keepalive_task = None
        self._subscribed = False
        self._target_symbol = ""
        self._from_time_ms = 0
        self._period = "15m"
        self._channel = 3

    async def connect(self):
        self.ws = await websockets.connect(self.dxlink_url)
        # 1. Send SETUP
        await self._send({"type": "SETUP", "channel": 0, "version": "0.1", "keepaliveTimeout": 60, "acceptKeepaliveTimeout": 60})
        # Wait for SETUP and AUTH_STATE UNAUTHORIZED
        while True:
            msg = await self._recv()
            if msg.get("type") == "AUTH_STATE" and msg.get("state") == "UNAUTHORIZED":
                break

        # 3. Send AUTH
        await self._send({"type": "AUTH", "channel": 0, "token": self.dxlink_token})
        # Wait for AUTH_STATE AUTHORIZED
        while True:
            msg = await self._recv()
            if msg.get("type") == "AUTH_STATE" and msg.get("state") == "AUTHORIZED":
                break

        # 5. Send CHANNEL_REQUEST
        await self._send({"type": "CHANNEL_REQUEST", "channel": self._channel, "service": "FEED", "parameters": {"contract": "AUTO"}})
        # Wait for CHANNEL_OPENED
        while True:
            msg = await self._recv()
            if msg.get("type") == "CHANNEL_OPENED" and msg.get("channel") == self._channel:
                break

        # 7. Send FEED_SETUP
        await self._send({
            "type": "FEED_SETUP",
            "channel": self._channel,
            "acceptAggregationPeriod": 0.1,
            "acceptDataFormat": "COMPACT",
            "acceptEventFields": {
                "Candle": ["eventSymbol", "time", "open", "high", "low", "close", "volume"]
            }
        })
        # Wait for FEED_CONFIG
        while True:
            msg = await self._recv()
            if msg.get("type") == "FEED_CONFIG":
                break

        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        log.info("DxLink connected and configured.")

    async def _send(self, data: dict):
        if self.ws:
            payload = json.dumps(data)
            # Do not log tokens
            if "token" in payload:
                log.debug("Sending: %s", payload.replace(self.dxlink_token, "***"))
            else:
                log.debug("Sending: %s", payload)
            await self.ws.send(payload)

    async def _recv(self) -> dict:
        if self.ws:
            raw = await self.ws.recv()
            msg = json.loads(raw)
            log.debug("Received: %s", raw)
            return msg
        raise ConnectionError("WebSocket is not connected")

    async def subscribe_candles(self, symbol: str, period: str = "15m", from_time_ms: int = 0):
        self._target_symbol = symbol
        self._period = period
        self._from_time_ms = from_time_ms
        await self._send({
            "type": "FEED_SUBSCRIPTION",
            "channel": self._channel,
            "add": [{"symbol": symbol, "type": "Candle", "fromTime": from_time_ms}]
        })
        self._subscribed = True
        log.info(f"Subscribed to {symbol} from {from_time_ms}")

    async def _keepalive_loop(self):
        try:
            while self._running:
                await asyncio.sleep(30)
                await self._send({"type": "KEEPALIVE", "channel": 0})
        except Exception as e:
            log.error(f"Keepalive error: {e}")

    def _process_feed_data(self, data: list):
        # COMPACT format: ["Candle", [fields...]]
        if len(data) >= 2 and data[0] == "Candle":
            fields = data[1]
            # acceptEventFields: ["eventSymbol", "time", "open", "high", "low", "close", "volume"]
            num_fields = 7
            for i in range(0, len(fields), num_fields):
                chunk = fields[i:i+num_fields]
                if len(chunk) == num_fields:
                    candle = {
                        "eventSymbol": chunk[0],
                        "time": chunk[1],
                        "open": chunk[2],
                        "high": chunk[3],
                        "low": chunk[4],
                        "close": chunk[5],
                        "volume": chunk[6],
                    }
                    # Filter completed bars (time + period < now)
                    # For simplicity, we pass to on_candle and let runner handle,
                    # or filter here if period is given.
                    # period_ms: 15m = 15 * 60 * 1000 = 900000
                    period_ms = int(self._period.replace("m", "")) * 60 * 1000
                    now_ms = int(asyncio.get_event_loop().time() * 1000) # This is loop time, not real wall clock.
                    import time
                    real_now_ms = int(time.time() * 1000)
                    if candle["time"] + period_ms <= real_now_ms:
                        self.on_candle(candle["eventSymbol"], candle)

    async def run(self):
        self._running = True
        backoff = 1
        while self._running:
            try:
                if not self.ws:
                    log.info("Reconnecting DxLinkStreamer...")
                    await self.connect()
                    if self._subscribed:
                        await self.subscribe_candles(self._target_symbol, self._period, self._from_time_ms)
                    backoff = 1
                
                msg = await self._recv()
                msg_type = msg.get("type")
                
                if msg_type == "FEED_DATA":
                    data = msg.get("data", [])
                    self._process_feed_data(data)
                elif msg_type == "AUTH_STATE" and msg.get("state") == "UNAUTHORIZED":
                    log.warning("AUTH_STATE UNAUTHORIZED mid-stream. Re-authenticating.")
                    await self._send({"type": "AUTH", "channel": 0, "token": self.dxlink_token})

            except Exception as e:
                import websockets
                if isinstance(e, websockets.exceptions.ConnectionClosedOK):
                    log.info("DxLink closed normally.")
                else:
                    log.error(f"DxLink connection error: {e}")
                if self.ws:
                    try:
                        await self.ws.close()
                    except:
                        pass
                if not self._running:
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def close(self):
        self._running = False
        if self._keepalive_task:
            self._keepalive_task.cancel()
        if self.ws:
            await self.ws.close()
