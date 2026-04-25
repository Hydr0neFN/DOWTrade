import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
from src.live.runner import LiveRunner
from src.data.bars import Bar
from src.broker.models import AccountState, Position
from src.llm.base import CostBudgetExceeded, LLMCallResult


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
    
    # Process one candle
    runner._on_candle("MYM", {"time": 30000000, "open": 38005, "high": 38010, "low": 38000, "close": 38005, "volume": 10})
    
    with patch("src.live.runner.final_check", return_value=MagicMock(approved=True, reason="ok")):
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
    
    # Make final_check fail by patching it
    with patch("src.live.runner.final_check", return_value=MagicMock(approved=False, reason="rejected")):
        runner._on_candle("MYM", {"time": 30000000, "open": 38005, "high": 38010, "low": 38000, "close": 38005, "volume": 10})
        
        loop_task = asyncio.create_task(runner._process_loop())
        await asyncio.sleep(0.2)
        loop_task.cancel()
    
    assert not runner.broker.submit_bracket_order.called
    call_args = runner.db.insert_decision.call_args[1]
    assert "final_check" in call_args["disagreement_flags"]

@pytest.mark.asyncio
async def test_cost_budget_exceeded(runner):
    runner._budget_exceeded = True
    
    runner._on_candle("MYM", {"time": 30000000, "open": 38005, "high": 38010, "low": 38000, "close": 38005, "volume": 10})
    
    loop_task = asyncio.create_task(runner._process_loop())
    await asyncio.sleep(0.2)
    loop_task.cancel()
    
    # Should not call LLMs
    assert not runner.haiku.evaluate.called
