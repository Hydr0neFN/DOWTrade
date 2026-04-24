"""
tests/test_tastytrade.py
Tests for the TastytradeBroker implementation.

All HTTP calls are mocked via unittest.mock.patch on httpx.Client.
No real network calls are made. No real credentials are used.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch, call
from zoneinfo import ZoneInfo

import httpx
import pytest

from src.broker.models import Order, Position
from src.broker.tastytrade import TastytradeBroker, to_tastytrade_symbol, _redact
from src.config import Settings

_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cert_settings() -> Settings:
    """Settings pointing at the cert sandbox with dummy credentials."""
    return Settings(
        tastytrade_base_url="https://api.cert.tastyworks.com",
        tastytrade_cert_username="test-user",
        tastytrade_cert_password="test-pass",
        tradovate_base_url="https://demo.tradovateapi.com/v1",
    )


def _make_mock_client():
    """Return a MagicMock that stands in for httpx.Client."""
    client = MagicMock()
    return client


def _ok_response(body: dict, status: int = 200) -> MagicMock:
    """Build a mock httpx.Response with given body and status."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


# ---------------------------------------------------------------------------
# URL guard tests
# ---------------------------------------------------------------------------

class TestURLGuard:
    """Instantiation must refuse non-sandbox URLs."""

    def test_refuses_live_url(self):
        """Production tastytrade URL must be rejected."""
        settings = Settings(
            tastytrade_base_url="https://api.tastyworks.com",
            tastytrade_cert_username="u",
            tastytrade_cert_password="p",
            tradovate_base_url="https://demo.tradovateapi.com/v1",
        )
        with pytest.raises(RuntimeError, match="non-sandbox"):
            TastytradeBroker(settings)

    def test_refuses_arbitrary_hostname(self):
        """Arbitrary hostnames must also be rejected."""
        settings = Settings(
            tastytrade_base_url="https://evil.example.com",
            tastytrade_cert_username="u",
            tastytrade_cert_password="p",
            tradovate_base_url="https://demo.tradovateapi.com/v1",
        )
        with pytest.raises(RuntimeError, match="non-sandbox"):
            TastytradeBroker(settings)

    def test_accepts_cert_url(self, cert_settings):
        """Cert URL must pass the guard (no RuntimeError raised)."""
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value = _make_mock_client()
            broker = TastytradeBroker(cert_settings)
            assert broker is not None


# ---------------------------------------------------------------------------
# connect() tests
# ---------------------------------------------------------------------------

class TestConnect:

    def test_connect_posts_credentials_and_stores_token(self, cert_settings):
        """connect() must POST /sessions and store the returned token."""
        with patch("httpx.Client") as mock_cls:
            mock_client = _make_mock_client()
            mock_cls.return_value = mock_client

            login_resp = _ok_response(
                {"data": {"session-token": "tok-abc123"}}, status=201
            )
            mock_client.post.return_value = login_resp

            broker = TastytradeBroker(cert_settings)
            broker.connect()

            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "/sessions"
            body = call_args[1]["json"]
            assert body["login"] == "test-user"
            assert body["password"] == "test-pass"
            assert broker._session_token == "tok-abc123"

    def test_connect_raises_on_auth_failure(self, cert_settings):
        """connect() must raise RuntimeError on non-201, without leaking password."""
        with patch("httpx.Client") as mock_cls:
            mock_client = _make_mock_client()
            mock_cls.return_value = mock_client

            bad_resp = _ok_response({}, status=401)
            mock_client.post.return_value = bad_resp

            broker = TastytradeBroker(cert_settings)
            with pytest.raises(RuntimeError) as exc_info:
                broker.connect()

            err_msg = str(exc_info.value)
            assert "test-pass" not in err_msg, "Password leaked in error message"
            assert "Authentication failed" in err_msg


# ---------------------------------------------------------------------------
# 401 retry test
# ---------------------------------------------------------------------------

class TestRetryOn401:

    def test_request_retries_on_401(self, cert_settings):
        """A 401 should trigger one reconnect and then succeed on retry."""
        with patch("httpx.Client") as mock_cls:
            mock_client = _make_mock_client()
            mock_cls.return_value = mock_client

            # First connect: returns token1
            login_resp1 = _ok_response(
                {"data": {"session-token": "token1"}}, status=201
            )
            # Second connect (after 401): returns token2
            login_resp2 = _ok_response(
                {"data": {"session-token": "token2"}}, status=201
            )
            mock_client.post.side_effect = [login_resp1, login_resp2]

            # request() call: first returns 401, then 200
            resp_401 = _ok_response({}, status=401)
            resp_ok = _ok_response({"data": {"items": []}}, status=200)
            mock_client.request.side_effect = [resp_401, resp_ok]

            broker = TastytradeBroker(cert_settings)
            result = broker._request("GET", "/some-endpoint")

            # Should have connected twice
            assert mock_client.post.call_count == 2
            # Second token should be used after reconnect
            assert broker._session_token == "token2"


# ---------------------------------------------------------------------------
# get_account_state() tests
# ---------------------------------------------------------------------------

class TestGetAccountState:

    def test_happy_path(self, cert_settings):
        """get_account_state() must build a valid AccountState from mocked responses."""
        with patch("httpx.Client") as mock_cls:
            mock_client = _make_mock_client()
            mock_cls.return_value = mock_client

            # Pre-set token so we skip the connect() call
            broker = TastytradeBroker(cert_settings)
            broker._session_token = "tok-test"
            broker._account_number = "ACCT001"

            accounts_resp = _ok_response({
                "data": {"items": [{"account": {"account-number": "ACCT001"}}]}
            })
            balances_resp = _ok_response({
                "data": {
                    "cash-balance": "50000.00",
                    "long-equity-value": "2500.00",
                    "realized-day-gain": "125.50",
                }
            })
            positions_resp = _ok_response({
                "data": {
                    "items": [
                        {
                            "instrument-type": "Future",
                            "symbol": "/MYMM6",
                            "quantity": "2",
                            "quantity-direction": "Long",
                            "average-open-price": "43200.00",
                            "unrealized-day-gain-value": "75.00",
                        }
                    ]
                }
            })

            # Calls: GET /accounts/ACCT001/balances, GET /accounts/ACCT001/positions
            mock_client.request.side_effect = [balances_resp, positions_resp]

            state = broker.get_account_state()

            assert state.equity == 52500.0
            assert state.realized_pnl_today == 125.50
            assert state.unrealized_pnl == 75.0
            assert state.position.side == "long"
            assert state.position.qty == 2
            assert state.position.avg_price == 43200.0
            assert state.now_et.tzinfo is not None


# ---------------------------------------------------------------------------
# submit_bracket_order() tests
# ---------------------------------------------------------------------------

class TestSubmitBracketOrder:

    def test_happy_path(self, cert_settings):
        """submit_bracket_order() must POST and return Order with broker id."""
        with patch("httpx.Client") as mock_cls:
            mock_client = _make_mock_client()
            mock_cls.return_value = mock_client

            broker = TastytradeBroker(cert_settings)
            broker._session_token = "tok-test"
            broker._account_number = "ACCT001"

            entry_resp = _ok_response({
                "data": {
                    "order": {"id": "ORD-999", "status": "Received"}
                }
            }, status=201)
            stop_resp = _ok_response({
                "data": {"order": {"id": "STOP-888", "status": "Received"}}
            }, status=201)
            mock_client.request.side_effect = [entry_resp, stop_resp]

            order = Order(
                order_id="local-1",
                symbol="MYM",
                side="long",
                action="open",
                qty=1,
                entry_price=43200.0,
                stop_price=43100.0,
                atr=50.0,
            )
            result = broker.submit_bracket_order(order)

            assert mock_client.request.call_count == 2
            assert result.status == "submitted"
            assert result.order_id == "ORD-999"


# ---------------------------------------------------------------------------
# fetch_historical_bars() raises NotImplementedError
# ---------------------------------------------------------------------------

class TestFetchHistoricalBars:

    def test_raises_not_implemented(self, cert_settings):
        """fetch_historical_bars must raise NotImplementedError mentioning Phase 6."""
        with patch("httpx.Client") as mock_cls:
            mock_client = _make_mock_client()
            mock_cls.return_value = mock_client

            broker = TastytradeBroker(cert_settings)
            with pytest.raises(NotImplementedError) as exc_info:
                broker.fetch_historical_bars("MYM", 0, 1, 15)

            assert "Phase 6" in str(exc_info.value)


# ---------------------------------------------------------------------------
# get_position() with no matching symbol
# ---------------------------------------------------------------------------

class TestGetPosition:

    def test_returns_flat_when_symbol_not_in_account(self, cert_settings):
        """get_position() must return flat Position when symbol is absent."""
        with patch("httpx.Client") as mock_cls:
            mock_client = _make_mock_client()
            mock_cls.return_value = mock_client

            broker = TastytradeBroker(cert_settings)
            broker._session_token = "tok"
            broker._account_number = "ACCT001"

            positions_resp = _ok_response({
                "data": {"items": []}
            })
            mock_client.request.return_value = positions_resp

            pos = broker.get_position("MYM")
            assert pos.side == "flat"
            assert pos.qty == 0

    def test_returns_position_when_symbol_matches(self, cert_settings):
        """get_position() must map Tastytrade position data correctly."""
        with patch("httpx.Client") as mock_cls:
            mock_client = _make_mock_client()
            mock_cls.return_value = mock_client

            broker = TastytradeBroker(cert_settings)
            broker._session_token = "tok"
            broker._account_number = "ACCT001"

            positions_resp = _ok_response({
                "data": {
                    "items": [{
                        "instrument-type": "Future",
                        "symbol": "/MYMM6",
                        "quantity": "1",
                        "quantity-direction": "Long",
                        "average-open-price": "43500",
                        "unrealized-day-gain-value": "25",
                    }]
                }
            })
            mock_client.request.return_value = positions_resp

            pos = broker.get_position("MYM")
            assert pos.side == "long"
            assert pos.qty == 1
            assert pos.avg_price == 43500.0


# ---------------------------------------------------------------------------
# Token redaction in logs
# ---------------------------------------------------------------------------

class TestTokenRedaction:

    def test_token_not_leaked_in_logs_on_error(self, cert_settings, caplog):
        """A 401 error must not include the session token in any log record."""
        with patch("httpx.Client") as mock_cls:
            mock_client = _make_mock_client()
            mock_cls.return_value = mock_client

            broker = TastytradeBroker(cert_settings)
            broker._session_token = "secret123"
            broker._account_number = "ACCT001"

            # Arrange: request returns 500 with token in body
            bad_resp = MagicMock()
            bad_resp.status_code = 500
            bad_resp.text = "Internal error. Token: secret123"
            bad_resp.json.return_value = {}
            mock_client.request.return_value = bad_resp

            with caplog.at_level(logging.WARNING, logger="tastytrade-cert"):
                with pytest.raises(RuntimeError) as exc_info:
                    broker._request("GET", "/some-path")

            err_msg = str(exc_info.value)
            assert "secret123" not in err_msg, "Token leaked in RuntimeError message"
            for record in caplog.records:
                assert "secret123" not in record.getMessage(), (
                    f"Token leaked in log record: {record.getMessage()!r}"
                )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_to_tastytrade_symbol_jun_2026(self):
        """Verify symbol generation for a known date."""
        from src.broker.tastytrade import to_tastytrade_symbol
        now = datetime(2026, 4, 25, tzinfo=_ET)
        sym = to_tastytrade_symbol("MYM", now=now)
        # April 2026 -> next quarterly is June (M), year digit 6
        assert sym == "/MYMM6"

    def test_redact_replaces_secret(self):
        assert _redact("my token=abc123 here", "abc123") == "my token=*** here"

    def test_redact_ignores_empty_secret(self):
        assert _redact("unchanged", "") == "unchanged"
