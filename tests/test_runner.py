import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
from src.live.runner import LiveRunner, _PositionState
from src.config import FIXED_RISK_PER_TRADE_USD, POINT_VALUE_USD
from src.data.bars import Bar
from src.broker.models import AccountState, Position
from src.llm.base import CostBudgetExceeded, LLMCallResult


# Stop distance that makes compute_size return exactly 2 contracts at whatever
# risk budget is configured: risk/contract = FIXED_RISK/2, so floor(2.0) = 2.
# Derived rather than hard-coded so a future budget change does not turn these
# into tests of a different code path.
TWO_CONTRACT_STOP_DISTANCE = FIXED_RISK_PER_TRADE_USD / (2 * POINT_VALUE_USD)


@pytest.fixture
def mock_db():
    db = MagicMock()
    return db

@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.get_account_state.return_value = AccountState(
        equity=100000.0,
        realized_pnl_today=0.0,
        unrealized_pnl=0.0,
        position=Position("flat", 0, 0.0, 0.0, 0),
        now_et=None
    )
    return broker

@pytest.fixture
def runner(mock_db, mock_broker):
    r = LiveRunner()
    r.MIN_WARMUP_BARS = 20
    r.db = mock_db
    r.broker = mock_broker
    r.haiku = MagicMock()
    r.gemini = MagicMock()
    r.deepseek = MagicMock()
    
    r.haiku.evaluate.return_value = LLMCallResult(parsed={"regime": "trend_following"}, raw_response="", latency_ms=0, input_tokens=0, output_tokens=0, cost_usd=0, error=None, used_fallback=False, model_used="")
    r.gemini.evaluate.return_value = LLMCallResult(parsed={"action": "open_long", "stop_price": 37000.0}, raw_response="", latency_ms=0, input_tokens=0, output_tokens=0, cost_usd=0, error=None, used_fallback=False, model_used="")
    r.deepseek.evaluate.return_value = LLMCallResult(parsed={"approved": True, "suggested_stop_price": 38000.0}, raw_response="", latency_ms=0, input_tokens=0, output_tokens=0, cost_usd=0, error=None, used_fallback=False, model_used="")
    return r

@pytest.mark.asyncio
async def test_on_candle_submits_order(runner):
    # Setup window to allow attr calculation
    for i in range(30):
        runner.window.append(Bar(1000 + i*900, 38000, 38010, 37990, 38005, 10))

    # Flat synthetic bars never produce a golden/death cross, and execution is
    # now gated on the cross-filter (DeepSeek demoted to advisory) — force it
    # open. The stop must sit within the $50 fixed risk budget (MYM $0.50/pt:
    # 38005-37955 = 50 pts = $25/contract) or compute_size returns 0 and the
    # open is skipped before final_check. SIM_FILLS is patched off to exercise
    # the real broker submit path.
    runner._cross = MagicMock()
    runner._cross.allows.return_value = (True, "forced by test")
    runner.gemini.evaluate.return_value = LLMCallResult(parsed={"action": "open_long", "stop_price": 37955.0}, raw_response="", latency_ms=0, input_tokens=0, output_tokens=0, cost_usd=0, error=None, used_fallback=False, model_used="")

    # Process one candle
    runner._on_candle("MYM", {"time": 30000000, "open": 38005, "high": 38010, "low": 38000, "close": 38005, "volume": 10})

    with patch("src.live.runner.SIM_FILLS", False), \
         patch("src.live.runner.final_check", return_value=MagicMock(approved=True, reason="ok")):
        loop_task = asyncio.create_task(runner._process_loop())
        await asyncio.sleep(0.2)
        loop_task.cancel()

    assert runner.broker.submit_bracket_order.called
    assert runner.db.insert_decision.called
    assert runner.db.insert_order.called

@pytest.mark.asyncio
async def test_on_candle_rejected_by_final_check(runner):
    for i in range(30):
        runner.window.append(Bar(1000 + i*900, 38000, 38010, 37990, 38005, 10))

    runner._cross = MagicMock()
    runner._cross.allows.return_value = (True, "forced by test")
    runner.gemini.evaluate.return_value = LLMCallResult(parsed={"action": "open_long", "stop_price": 37955.0}, raw_response="", latency_ms=0, input_tokens=0, output_tokens=0, cost_usd=0, error=None, used_fallback=False, model_used="")

    # Make final_check fail by patching it
    with patch("src.live.runner.SIM_FILLS", False), \
         patch("src.live.runner.final_check", return_value=MagicMock(approved=False, reason="rejected")):
        runner._on_candle("MYM", {"time": 30000000, "open": 38005, "high": 38010, "low": 38000, "close": 38005, "volume": 10})

        loop_task = asyncio.create_task(runner._process_loop())
        await asyncio.sleep(0.2)
        loop_task.cancel()

    assert not runner.broker.submit_bracket_order.called
    # The reject path re-inserts the decision row (bar_ts is UNIQUE + OR
    # REPLACE) with the reason recorded in safety_notes.
    last_decision = runner.db.insert_decision.call_args[0][0]
    assert "final_check rejected" in last_decision["safety_notes"]

def _arm_pyramid(runner, open_qty: int):
    """Warm the window and stage an open, profitable `open_qty`-contract long.

    The stop sits far below the test bar's low so the pre-decision stop check
    does not close the position before the add is considered.
    """
    for i in range(30):
        runner.window.append(Bar(1000 + i * 900, 38000, 38010, 37990, 38005, 10))

    runner._cross = MagicMock()
    runner._cross.allows.return_value = (True, "forced by test")

    runner._positions = [_PositionState(
        side="long",
        qty=open_qty,
        avg_price=37000.0,
        current_stop=36000.0,
        pyramid_adds_used=0,
        entry_ts=1000,
    )]

    runner.gemini.evaluate.return_value = LLMCallResult(
        parsed={"action": "add_pyramid",
                "stop_price": 38005.0 - TWO_CONTRACT_STOP_DISTANCE},
        raw_response="", latency_ms=0, input_tokens=0, output_tokens=0,
        cost_usd=0, error=None, used_fallback=False, model_used="",
    )

    runner._on_candle("MYM", {"time": 30000000, "open": 38005, "high": 38010,
                              "low": 38000, "close": 38005, "volume": 10})


async def _run_one_bar(runner):
    loop_task = asyncio.create_task(runner._process_loop())
    await asyncio.sleep(0.2)
    loop_task.cancel()


@pytest.mark.asyncio
async def test_pyramid_add_clamped_to_remaining_capacity(runner):
    """Risk unit > remaining capacity must clamp the add, not drop it.

    The backtest harness has always sized an add as min(risk_unit, remaining)
    (harness.py:317-320); this path used to reject the whole add whenever the
    full risk unit did not fit, which at a $250 budget meant 2 + 2 > 3 and no
    add could ever fill. With 2 of 3 contracts already open and a 2-contract
    risk unit, exactly 1 contract must fill.
    """
    _arm_pyramid(runner, open_qty=2)

    with patch("src.live.runner.MAX_OPEN_CONTRACTS", 3), \
         patch("src.live.runner.MAX_PYRAMID_ADDS", 2):
        await _run_one_bar(runner)

    assert len(runner._positions) == 2, "the add was dropped instead of clamped"
    added = runner._positions[-1]
    assert added.qty == 1, f"expected the add clamped to 1 contract, got {added.qty}"
    assert added.side == "long"
    assert sum(p.qty for p in runner._positions) == 3, "must land exactly on the cap"
    # The fill row must carry the clamped size, not the unclamped risk unit.
    assert runner.db.insert_fill.call_args[0][0]["qty"] == 1


@pytest.mark.asyncio
async def test_pyramid_add_refused_when_already_at_cap(runner):
    """Clamping must not become "always allow one more": at the cap, refuse."""
    _arm_pyramid(runner, open_qty=3)

    with patch("src.live.runner.MAX_OPEN_CONTRACTS", 3), \
         patch("src.live.runner.MAX_PYRAMID_ADDS", 2):
        await _run_one_bar(runner)

    assert len(runner._positions) == 1, "no add may fill once gross qty is at the cap"
    assert sum(p.qty for p in runner._positions) == 3


@pytest.mark.asyncio
async def test_cost_budget_exceeded(runner):
    runner._budget_exceeded = True
    
    runner._on_candle("MYM", {"time": 30000000, "open": 38005, "high": 38010, "low": 38000, "close": 38005, "volume": 10})
    
    loop_task = asyncio.create_task(runner._process_loop())
    await asyncio.sleep(0.2)
    loop_task.cancel()
    
    # Should not call LLMs
    assert not runner.haiku.evaluate.called
