"""Regression tests for the post-fix ultrareview of 2026-09-26 (vault doc
dowtrade-correctness-audit, section "Ultrareview 2026-09-26 (post-fix)"):
#2 dashboard avg price, #3 pyramid gate, #4 rejected entry, #6 decision_id,
#7 ledger lock, #8 single cross-filter call, #9 weighted avg price.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from src.broker.tastytrade import TastytradeBroker
from src.db.repo import Database, init_db
from src.live.runner import _PositionState
from src.llm.base import LLMCallResult
from tests.test_audit3_fixes import _broker, _order, _positions  # noqa: F401
from tests.test_runner import (TWO_CONTRACT_STOP_DISTANCE, _arm_pyramid,  # noqa: F401
                               _run_one_bar, mock_broker, mock_db, runner)
from tests.test_tastytrade import _ok_response, cert_settings  # noqa: F401


def _llm(parsed):
    return LLMCallResult(parsed=parsed, raw_response="", latency_ms=0, input_tokens=0,
                         output_tokens=0, cost_usd=0, error=None, used_fallback=False,
                         model_used="")


# --------------------------------------------------------------------- #2

def test_dashboard_avg_price_uses_open_lots_only(tmp_path):
    from src.dashboard.app import _snapshot
    path = str(tmp_path / "d.db")
    init_db(path)
    db = Database(path)
    c = db._conn
    c.execute("CREATE TABLE sim_positions (id INTEGER PRIMARY KEY, side TEXT, qty INTEGER,"
              " avg_price REAL, current_stop REAL, entry_ts INTEGER)")
    # closed round trip at 100 -> 200, then an open lot of 2 at 300
    for oid, side, qty, px in ((1, "BUY", 1, 100.0), (2, "SELL", 1, 200.0), (3, "BUY", 2, 300.0)):
        c.execute("INSERT INTO orders (id, ts, symbol, side, qty, order_type, status)"
                  " VALUES (?, '1', 'MYM', ?, ?, 'market', 'filled')", (oid, side, qty))
        c.execute("INSERT INTO fills (order_id, ts, qty, price) VALUES (?, '1', ?, ?)",
                  (oid, qty, px))
    c.execute("INSERT INTO sim_positions (side, qty, avg_price, current_stop, entry_ts)"
              " VALUES ('long', 2, 300.0, 250.0, 1)")
    c.commit()

    _, position, _ = _snapshot(db)

    assert position["side"] == "LONG" and position["qty"] == 2
    assert position["avg_price"] == 300.0  # all-fills average would be 225
    db.close()


# --------------------------------------------------------------------- #3

@pytest.mark.asyncio
async def test_pyramid_refused_when_aggregate_is_losing(runner):
    _arm_pyramid(runner, open_qty=1)  # lot 0: long 1 @ 37000, well in profit
    runner._positions.append(_PositionState(
        side="long", qty=1, avg_price=48000.0, current_stop=36000.0,
        pyramid_adds_used=0, entry_ts=1000))  # lot 1: deep loser, net < 0
    with patch("src.live.runner.MAX_OPEN_CONTRACTS", 5), \
         patch("src.live.runner.MAX_PYRAMID_ADDS", 3):
        await _run_one_bar(runner)
    assert len(runner._positions) == 2, "added to a net-losing position"


@pytest.mark.asyncio
async def test_pyramid_allowed_when_aggregate_is_winning(runner):
    # Lot 0 is UNDER water, the aggregate is well in profit: the old lot-0
    # gate refused this add, the aggregate gate allows it.
    _arm_pyramid(runner, open_qty=1)
    runner._positions[0].avg_price = 39000.0              # -995 pts at 38005
    runner._positions.append(_PositionState(
        side="long", qty=1, avg_price=35000.0, current_stop=34000.0,
        pyramid_adds_used=0, entry_ts=1000))               # +3005 pts
    with patch("src.live.runner.MAX_OPEN_CONTRACTS", 5), \
         patch("src.live.runner.MAX_PYRAMID_ADDS", 3):
        await _run_one_bar(runner)
    assert len(runner._positions) == 3


# --------------------------------------------------------------------- #4

def test_rejected_entry_gets_no_stop_leg(cert_settings):
    broker, client = _broker(cert_settings)
    rejected = _ok_response({"data": {"order": {"id": "E1", "status": "Rejected"}}}, 201)
    client.request.side_effect = [_positions(), rejected]

    result = broker.submit_bracket_order(_order())

    assert result.status == "rejected"
    assert client.request.call_count == 2  # baseline GET + entry POST, no stop


# --------------------------------------------------------------------- #6 #8

@pytest.mark.asyncio
async def test_orders_carry_the_decision_id(runner):
    for i in range(30):
        from src.data.bars import Bar
        runner.window.append(Bar(1000 + i * 900, 38000, 38010, 37990, 38005, 10))
    runner._cross = MagicMock()
    runner._cross.allows.return_value = (True, "forced by test")
    runner.db.insert_decision.return_value = 4242
    runner.gemini.evaluate.return_value = _llm({"action": "open_long", "stop_price": 37955.0})
    runner._on_candle("MYM", {"time": 30000000, "open": 38005, "high": 38010,
                              "low": 38000, "close": 38005, "volume": 10})
    with patch("src.live.runner.final_check",
               return_value=MagicMock(approved=True, reason="ok")):
        await _run_one_bar(runner)

    assert runner.db.insert_order.called
    assert runner.db.insert_order.call_args[0][0]["decision_id"] == 4242
    assert runner._cross.allows.call_count == 1  # #8: one verdict per bar


@pytest.mark.asyncio
async def test_blocked_decision_row_keeps_votes(runner):
    for i in range(30):
        from src.data.bars import Bar
        runner.window.append(Bar(1000 + i * 900, 38000, 38010, 37990, 38005, 10))
    runner._cross = MagicMock()
    runner._cross.allows.return_value = (False, "death cross")
    runner.gemini.evaluate.return_value = _llm({"action": "open_long", "stop_price": 37955.0})
    runner._on_candle("MYM", {"time": 30000000, "open": 38005, "high": 38010,
                              "low": 38000, "close": 38005, "volume": 10})
    await _run_one_bar(runner)

    row = runner.db.insert_decision.call_args[0][0]
    assert row["safety_ok"] == 0 and row["safety_notes"] == "death cross"
    assert row["direction"] == "LONG" and row["raw_votes"]
    assert not runner.db.insert_order.called


def test_decision_upsert_keeps_id_and_referencing_orders(tmp_path):
    path = str(tmp_path / "u.db")
    init_db(path)
    db = Database(path)
    db._conn.execute("INSERT INTO bars (ts, open, high, low, close) VALUES ('100', 1, 1, 1, 1)")
    row = {"bar_ts": "100", "direction": "LONG", "confidence": 0.5, "stop_price": 1.0,
           "entry_price": 2.0, "raw_votes": "{}", "safety_ok": 1, "safety_notes": ""}
    first = db.insert_decision(row)
    db.insert_order({"ts": "100", "decision_id": first, "broker_id": "sim", "symbol": "MYM",
                     "side": "BUY", "qty": 1, "order_type": "market", "limit_price": 0.0,
                     "stop_price": 1.0, "status": "filled", "raw_response": ""})
    second = db.insert_decision({**row, "safety_ok": 0, "safety_notes": "blocked"})
    assert second == first
    got = db._conn.execute("SELECT safety_notes FROM decisions WHERE id=?", (first,)).fetchone()
    assert got["safety_notes"] == "blocked"
    db.close()


@pytest.mark.asyncio
async def test_rejected_broker_entry_is_recorded_as_rejected(runner):
    from src.data.bars import Bar
    for i in range(30):
        runner.window.append(Bar(1000 + i * 900, 38000, 38010, 37990, 38005, 10))
    runner._cross = MagicMock()
    runner._cross.allows.return_value = (True, "forced by test")
    runner.gemini.evaluate.return_value = _llm({"action": "open_long", "stop_price": 37955.0})
    runner.broker.submit_bracket_order.return_value = MagicMock(order_id="E9", status="rejected")
    runner._on_candle("MYM", {"time": 30000000, "open": 38005, "high": 38010,
                              "low": 38000, "close": 38005, "volume": 10})
    with patch("src.live.runner.SIM_FILLS", False), \
         patch("src.live.runner.final_check", return_value=MagicMock(approved=True, reason="ok")):
        await _run_one_bar(runner)
    rec = runner.db.insert_order.call_args[0][0]
    assert rec["status"] == "rejected" and rec["broker_id"] == "E9"


# --------------------------------------------------------------------- #9

@pytest.mark.asyncio
async def test_aggregate_avg_price_is_qty_weighted(runner):
    _arm_pyramid(runner, open_qty=1)  # 1 @ 37000
    runner._positions.append(_PositionState(
        side="long", qty=3, avg_price=38000.0, current_stop=36000.0,
        pyramid_adds_used=0, entry_ts=1000))
    runner.gemini.evaluate.return_value = _llm({"action": "hold", "stop_price": 0.0})
    await _run_one_bar(runner)
    pos_arg = runner.gemini.evaluate.call_args[0][2]
    assert pos_arg.avg_price == pytest.approx((37000 + 3 * 38000) / 4)


# --------------------------------------------------------------------- #7

def _ledger_worker(path: str, n: int) -> None:
    import src.llm.claude_sdk as sdk
    from pathlib import Path
    sdk.CREDIT_FILE = Path(path)
    for _ in range(n):
        sdk._add(0.01)


def test_credit_ledger_add_is_serialised_across_processes(tmp_path, monkeypatch):
    pytest.importorskip("fcntl")
    import multiprocessing as mp
    import src.llm.claude_sdk as sdk
    path = tmp_path / "credit.json"
    n_procs, n_each = 6, 40
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=_ledger_worker, args=(str(path), n_each))
             for _ in range(n_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
        assert p.exitcode == 0
    monkeypatch.setattr(sdk, "CREDIT_FILE", path)
    assert sdk._spent() == pytest.approx(n_procs * n_each * 0.01)
