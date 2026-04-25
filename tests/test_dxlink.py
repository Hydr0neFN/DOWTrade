import asyncio
import json
import logging
import pytest
from datetime import date
from unittest.mock import patch, MagicMock, AsyncMock

import websockets

from src.live.dxlink import DxLinkStreamer
from src.broker.tastytrade import to_tastytrade_symbol, dxfeed_symbol

class DummyWS:
    def __init__(self):
        self.sent = []
        self.recv_queue = asyncio.Queue()
        self.closed = False

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def recv(self):
        return await self.recv_queue.get()

    async def close(self):
        self.closed = True

@pytest.fixture
def dummy_ws():
    return DummyWS()

@pytest.mark.asyncio
async def test_dxlink_handshake(dummy_ws, caplog):
    caplog.set_level(logging.DEBUG)

    with patch("websockets.connect", new_callable=AsyncMock, return_value=dummy_ws):
        streamer = DxLinkStreamer("wss://test", "secret_token", lambda s, c: None)

        # Enqueue fake responses for connect()
        dummy_ws.recv_queue.put_nowait(json.dumps({"type": "AUTH_STATE", "state": "UNAUTHORIZED"}))
        dummy_ws.recv_queue.put_nowait(json.dumps({"type": "AUTH_STATE", "state": "AUTHORIZED"}))
        dummy_ws.recv_queue.put_nowait(json.dumps({"type": "CHANNEL_OPENED", "channel": 3}))
        dummy_ws.recv_queue.put_nowait(json.dumps({"type": "FEED_CONFIG"}))     

        await streamer.connect()

        assert len(dummy_ws.sent) == 4
        assert dummy_ws.sent[0]["type"] == "SETUP"
        assert dummy_ws.sent[1]["type"] == "AUTH"
        assert dummy_ws.sent[1]["token"] == "secret_token"
        assert dummy_ws.sent[2]["type"] == "CHANNEL_REQUEST"
        assert dummy_ws.sent[3]["type"] == "FEED_SETUP"

        # Token redaction check
        for record in caplog.records:
            assert "secret_token" not in record.message

@pytest.mark.asyncio
async def test_compact_decode():
    candles = []
    def on_candle(sym, c):
        candles.append(c)

    streamer = DxLinkStreamer("wss://test", "tok", on_candle)
    streamer._period = "15m"

    now_ms = int(asyncio.get_event_loop().time() * 1000)
    import time
    real_now_ms = int(time.time() * 1000)

    # Time should be in the past to be considered "completed"
    past_ms = real_now_ms - 20 * 60 * 1000

    data = ["Candle", [
        "/MYMM6:XCME", past_ms, 100.0, 110.0, 90.0, 105.0, 500
    ]]

    streamer._process_feed_data(data)

    assert len(candles) == 1
    assert candles[0]["eventSymbol"] == "/MYMM6:XCME"
    assert candles[0]["close"] == 105.0
    assert candles[0]["volume"] == 500

@pytest.mark.parametrize("dt,expected", [
    (date(2026, 4, 25), "/MYMM6"),
    (date(2026, 6, 20), "/MYMU6"),
])
def test_to_tastytrade_symbol(dt, expected):
    from datetime import datetime
    dt_time = datetime(dt.year, dt.month, dt.day)
    assert to_tastytrade_symbol("MYM", dt_time) == expected
