"""Regression tests for the 2026-09-26 correctness audit (vault doc
dowtrade-correctness-audit): #1 stop-leg failure, #5 position root match,
#6 NaN prices / invalid Gemini stop, #7 monthly LLM budget, #8 journal
agreement, #9 safety posture survives python -O.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.broker.models import Order
from src.broker.tastytrade import TastytradeBroker, _STOP_RETRY_DELAYS
from src.data.bars import Bar
from src.journal.daily import _votes_agree
from src.live.dxlink import DxLinkStreamer
from src.live.runner import _valid_stop
from src.llm.base import CostBudgetExceeded, CostTracker, LLMCallResult
from tests.test_runner import mock_broker, mock_db, runner  # noqa: F401  (fixtures)
from tests.test_tastytrade import _make_mock_client, _ok_response, cert_settings  # noqa: F401

_ET = ZoneInfo("America/New_York")


def _err_response(status: int = 500) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = '{"error": "boom"}'
    return resp


def _broker(cert_settings):
    mock_client = _make_mock_client()
    with patch("httpx.Client", return_value=mock_client):
        broker = TastytradeBroker(cert_settings)
    broker._session_token = "tok"
    broker._account_number = "ACCT001"
    broker._sleep = MagicMock()
    return broker, mock_client


def _order(side="long", qty=1):
    return Order(order_id="local-1", symbol="MYM", side=side, action="open",
                 qty=qty, entry_price=0.0, stop_price=43100.0, atr=50.0)


def _positions(*items):
    return _ok_response({"data": {"items": list(items)}})


def _fut(symbol, qty, direction="Long"):
    return {"instrument-type": "Future", "symbol": symbol, "quantity": str(qty),
            "quantity-direction": direction, "average-open-price": "43200",
            "unrealized-day-gain-value": "10"}


# --------------------------------------------------------------------- #1

class TestStopLegFailure:

    def test_stop_retried_then_succeeds(self, cert_settings):
        broker, client = _broker(cert_settings)
        entry = _ok_response({"data": {"order": {"id": "E1", "status": "Filled"}}}, 201)
        ok = _ok_response({"data": {"order": {"id": "S1"}}}, 201)
        client.request.side_effect = [_positions(), entry, _err_response(), ok]

        result = broker.submit_bracket_order(_order())

        assert client.request.call_count == 4
        assert result.order_id == "E1"
        broker._sleep.assert_called_once_with(_STOP_RETRY_DELAYS[0])

    def test_filled_entry_is_flattened_when_stop_never_lands(self, cert_settings):
        broker, client = _broker(cert_settings)
        entry = _ok_response({"data": {"order": {"id": "E1", "status": "Filled"}}}, 201)
        stop_fails = [_err_response() for _ in range(len(_STOP_RETRY_DELAYS) + 1)]
        cancel_fails = _err_response(422)  # already filled
        pos = _positions(_fut("/MYMZ6", 1))
        flatten_ok = _ok_response({"data": {"order": {"id": "F1"}}}, 201)
        client.request.side_effect = [_positions(), entry, *stop_fails, cancel_fails, pos, flatten_ok]

        with pytest.raises(RuntimeError, match="stop-loss leg failed"):
            broker.submit_bracket_order(_order())

        last = client.request.call_args_list[-1]
        assert last.args[0] == "POST"
        leg = last.kwargs["json"]["legs"][0]
        assert last.kwargs["json"]["order-type"] == "Market"
        assert leg["action"] == "Sell to Close"
        assert leg["quantity"] == "1"
        cancel = client.request.call_args_list[2 + len(stop_fails)]
        assert cancel.args == ("DELETE", "/accounts/ACCT001/orders/E1")

    def test_unfilled_entry_is_cancelled_not_flattened(self, cert_settings):
        broker, client = _broker(cert_settings)
        entry = _ok_response({"data": {"order": {"id": "E1", "status": "Live"}}}, 201)
        stop_fails = [_err_response() for _ in range(len(_STOP_RETRY_DELAYS) + 1)]
        cancel_ok = _ok_response({}, 200)
        flat = _positions()
        client.request.side_effect = [_positions(), entry, *stop_fails, cancel_ok, flat]

        with pytest.raises(RuntimeError):
            broker.submit_bracket_order(_order(side="short"))

        methods = [c.args[0] for c in client.request.call_args_list]
        # baseline GET, entry POST, stop POSTs, DELETE, positions GET -- no flatten
        assert methods == ["GET"] + ["POST"] * (1 + len(stop_fails)) + ["DELETE", "GET"]

    def test_flatten_never_exceeds_the_entry_qty(self, cert_settings):
        broker, client = _broker(cert_settings)
        entry = _ok_response({"data": {"order": {"id": "E1", "status": "Filled"}}}, 201)
        stop_fails = [_err_response() for _ in range(len(_STOP_RETRY_DELAYS) + 1)]
        pre = _positions(_fut("/MYMZ6", 2))
        pos = _positions(_fut("/MYMZ6", 4))  # 2 pre-existing + 2 new, entry was qty 2
        client.request.side_effect = [pre, entry, *stop_fails, _err_response(422), pos,
                                      _ok_response({}, 201)]

        with pytest.raises(RuntimeError):
            broker.submit_bracket_order(_order(qty=2))

        assert client.request.call_args_list[-1].kwargs["json"]["legs"][0]["quantity"] == "2"

    def test_unfilled_pyramid_entry_never_touches_the_existing_lot(self, cert_settings):
        broker, client = _broker(cert_settings)
        entry = _ok_response({"data": {"order": {"id": "E1", "status": "Live"}}}, 201)
        stop_fails = [_err_response() for _ in range(len(_STOP_RETRY_DELAYS) + 1)]
        existing = _fut("/MYMZ6", 1)
        client.request.side_effect = [_positions(existing), entry, *stop_fails,
                                      _ok_response({}, 200), _positions(existing)]

        with pytest.raises(RuntimeError):
            broker.submit_bracket_order(_order(qty=1))

        methods = [c.args[0] for c in client.request.call_args_list]
        assert methods[-2:] == ["DELETE", "GET"]  # no flatten POST after the lookup

    def test_network_error_on_stop_is_retried(self, cert_settings):
        import httpx
        broker, client = _broker(cert_settings)
        entry = _ok_response({"data": {"order": {"id": "E1", "status": "Filled"}}}, 201)
        client.request.side_effect = [_positions(), entry, httpx.ReadTimeout("slow"),
                                      _ok_response({}, 201)]

        broker.submit_bracket_order(_order())

        assert client.request.call_count == 4

    def test_no_entry_when_baseline_lookup_fails(self, cert_settings):
        broker, client = _broker(cert_settings)
        client.request.side_effect = [_err_response()]

        with pytest.raises(RuntimeError):
            broker.submit_bracket_order(_order())

        assert client.request.call_count == 1  # never POSTed an entry


# --------------------------------------------------------------------- #5

class TestAccountStateRootMatch:

    def test_stray_future_of_other_root_is_ignored(self, cert_settings):
        broker, client = _broker(cert_settings)
        bal = _ok_response({"data": {"cash-balance": "1000", "long-equity-value": "0"}})
        pos = _positions(_fut("/ESZ6", 4, "Short"), _fut("/MYMZ6", 1, "Long"))
        client.request.side_effect = [bal, pos]

        state = broker.get_account_state()

        assert state.position.side == "long"
        assert state.position.qty == 1

    def test_stray_unrealized_pnl_is_excluded(self, cert_settings):
        broker, client = _broker(cert_settings)
        bal = _ok_response({"data": {"cash-balance": "1000", "long-equity-value": "0"}})
        stray = dict(_fut("/ESZ6", 4, "Short"), **{"unrealized-day-gain-value": "-900"})
        client.request.side_effect = [bal, _positions(stray, _fut("/MYMZ6", 1))]

        assert broker.get_account_state().unrealized_pnl == 10.0

    @pytest.mark.parametrize("sym", ["/MYMZ6", "/MYMZ26", "/MYMZ6:XCME"])
    def test_symbol_root_forms(self, sym):
        from src.broker.tastytrade import _symbol_root
        assert _symbol_root(sym) == "MYM"

    def test_only_other_roots_means_flat(self, cert_settings):
        broker, client = _broker(cert_settings)
        bal = _ok_response({"data": {"cash-balance": "1000", "long-equity-value": "0"}})
        client.request.side_effect = [bal, _positions(_fut("/ESZ6", 4, "Short"))]

        assert broker.get_account_state().position.side == "flat"


# --------------------------------------------------------------------- #6

def test_dxlink_drops_candle_with_nan_close():
    candles = []
    s = DxLinkStreamer("wss://test", "tok", lambda sym, c: candles.append(c))
    s._period = "15m"
    past = int(time.time() * 1000) - 20 * 60 * 1000
    s._process_feed_data(["Candle", [
        "/MYMZ6:XCME", past, 100.0, 110.0, 90.0, "NaN", 5,
        "/MYMZ6:XCME", past - 900000, 100.0, 110.0, 90.0, 105.0, "NaN",
    ]])
    assert len(candles) == 1
    assert candles[0]["close"] == 105.0
    assert candles[0]["volume"] == 0.0


@pytest.mark.parametrize("raw,expected", [
    (43100.0, 43100.0), ("43100", 43100.0), (None, 0.0), (0, 0.0),
    (-5, 0.0), (float("nan"), 0.0), (float("inf"), 0.0), ("abc", 0.0),
])
def test_valid_stop(raw, expected):
    assert _valid_stop(raw) == expected


def _llm(parsed):
    return LLMCallResult(parsed=parsed, raw_response="", latency_ms=0, input_tokens=0,
                         output_tokens=0, cost_usd=0, error=None, used_fallback=False,
                         model_used="")


async def _run_one(runner, candle):
    runner._on_candle("MYM", candle)
    with patch("src.live.runner.SIM_FILLS", False), \
         patch("src.live.runner.final_check", return_value=MagicMock(approved=True, reason="ok")):
        task = asyncio.create_task(runner._process_loop())
        await asyncio.sleep(0.2)
        task.cancel()


def _seed(runner):
    for i in range(30):
        runner.window.append(Bar(1000 + i * 900, 38000, 38010, 37990, 38005, 10))
    runner._cross = MagicMock()
    runner._cross.allows.return_value = (True, "forced by test")


@pytest.mark.asyncio
async def test_runner_drops_zero_close_bar(runner):
    _seed(runner)
    await _run_one(runner, {"time": 30000000, "open": 38005, "high": 38010,
                            "low": 38000, "close": 0.0, "volume": 10})
    assert not runner.db.insert_bar.called
    assert not runner.db.insert_decision.called
    assert not runner.broker.submit_bracket_order.called


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_stop", [-5.0, float("nan"), "garbage"])
async def test_invalid_gemini_stop_uses_atr_fallback(runner, bad_stop):
    _seed(runner)
    runner.gemini.evaluate.return_value = _llm({"action": "open_long", "stop_price": bad_stop})
    await _run_one(runner, {"time": 30000000, "open": 38005, "high": 38010,
                            "low": 38000, "close": 38005, "volume": 10})
    assert runner.broker.submit_bracket_order.called
    order = runner.broker.submit_bracket_order.call_args[0][0]
    assert 0 < order.stop_price < 38005  # side-aware long fallback, below entry


# --------------------------------------------------------------------- #7

class TestMonthlyBudget:

    def test_spend_resets_when_the_et_month_changes(self):
        now = [datetime(2026, 9, 30, 23, 0, tzinfo=_ET)]
        t = CostTracker(cap_usd=1.0, clock=lambda: now[0])
        t.record(0.99)
        with pytest.raises(CostBudgetExceeded):
            t.authorize(0.05)
        now[0] = datetime(2026, 10, 1, 0, 5, tzinfo=_ET)
        t.authorize(0.05)  # new month: must not raise
        assert t.total_usd == 0.0

    def test_same_month_keeps_accumulating(self):
        now = [datetime(2026, 9, 1, tzinfo=_ET)]
        t = CostTracker(cap_usd=10.0, clock=lambda: now[0])
        t.record(3.0)
        now[0] = datetime(2026, 9, 30, 23, 59, tzinfo=_ET)
        t.record(2.0)
        assert t.total_usd == pytest.approx(5.0)

    def test_month_boundary_is_eastern_not_utc(self):
        # 2026-10-01 02:00 UTC is still Sept 30 in New York.
        from datetime import timezone
        now = [datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)]
        t = CostTracker(cap_usd=10.0, clock=lambda: now[0])
        t.record(4.0)
        now[0] = datetime(2026, 10, 1, 2, 0, tzinfo=timezone.utc)
        assert t.total_usd == pytest.approx(4.0)


# --------------------------------------------------------------------- #8

@pytest.mark.parametrize("votes,agree", [
    ({"haiku": {"trend": "up"}, "gemini": {"action": "open_long"}, "ds": {"approved": True}}, True),
    ({"haiku": {"trend": "down"}, "gemini": {"action": "open_short"}, "ds": {"approved": True}}, True),
    ({"haiku": {"trend": "up"}, "gemini": {"action": "open_short"}, "ds": {"approved": True}}, False),
    ({"haiku": {"trend": "up"}, "gemini": {"action": "open_long"}, "ds": {"approved": False}}, False),
    ({"haiku": {"trend": "range"}, "gemini": {"action": "hold"}, "ds": {"approved": False}}, True),
    ({"haiku": {"trend": "up"}, "gemini": {"action": "hold"}, "ds": {"approved": True}}, True),
    ({"haiku": {"trend": "up"}, "gemini": {"action": "hold"}, "ds": {"approved": False}}, True),  # veto of a hold is no dissent
    ([{"direction": "LONG"}] * 3, False),
])
def test_votes_agree(votes, agree):
    assert _votes_agree(votes) is agree


def test_journal_agreement_pct_counts_dict_votes(tmp_path):
    from src.db.repo import Database
    from src.journal.daily import generate_daily_journal
    db = Database(str(tmp_path / "j.db"))
    ts = str(int(datetime(2026, 9, 25, 10, 0, tzinfo=_ET).timestamp()))
    votes = json.dumps({"haiku": {"trend": "up"}, "gemini": {"action": "open_long"},
                        "ds": {"approved": True}})
    db._conn.execute("INSERT INTO bars (ts, open, high, low, close) VALUES (?, 1, 1, 1, 1)", (ts,))
    db._conn.execute(
        "INSERT INTO decisions (bar_ts, direction, confidence, raw_votes, safety_ok) "
        "VALUES (?, 'LONG', 0.9, ?, 1)", (ts, votes))
    db._conn.commit()
    client = MagicMock()
    client.messages.create.return_value = MagicMock(content=[MagicMock(text="ok")])
    with patch("src.journal.daily.Path") as p:
        p.return_value = tmp_path
        generate_daily_journal("2026-09-25", db, anthropic_client=client)
    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Agreement %: 100.0%" in prompt
    db.close()


# --------------------------------------------------------------------- #9

def test_safety_posture_raises_systemexit_not_assert(cert_settings, monkeypatch):
    import src.config as cfg
    monkeypatch.setattr(cfg, "PAPER_ONLY", False)
    with pytest.raises(SystemExit, match="PAPER_ONLY"):
        cfg.assert_safety_posture(cert_settings)


def test_safety_posture_has_no_bare_asserts():
    import inspect
    import src.config as cfg
    src = inspect.getsource(cfg.assert_safety_posture)
    assert "assert " not in src.replace("assert_safety_posture", "")
