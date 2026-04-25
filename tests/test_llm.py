"""
tests/test_llm.py
==================
Mocked unit tests for Phase 4 LLM integration layer.
No real API calls are made.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.llm.base import (
    CostBudgetExceeded,
    CostTracker,
    LLMCallResult,
    LLMClient,
    parse_json_strict,
    render_prompt,
    strip_json_fences,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).parent.parent / "src" / "llm" / "prompts"

STRUCTURAL_KEYS = {"trend", "structural_signal", "pattern_intact", "confidence_0_to_1", "reasoning"}
EXECUTION_KEYS = {"action", "stop_price", "trailing_stop_atr_multiple", "reasoning"}
RISK_KEYS = {"approved", "violations", "reasoning"}


def _make_haiku_response(data: dict) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(data))]
    msg.usage.input_tokens = 100
    msg.usage.output_tokens = 50
    msg.usage.cache_read_input_tokens = 0
    return msg


def _make_gemini_response(data: dict, model: str = "gemini-2.5-flash") -> MagicMock:
    resp = MagicMock()
    resp.text = json.dumps(data)
    resp.usage_metadata.prompt_token_count = 80
    resp.usage_metadata.candidates_token_count = 40
    return resp


def _make_hf_response(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=json.dumps(data)))]
    resp.usage = MagicMock(prompt_tokens=60, completion_tokens=30)
    return resp


# ---------------------------------------------------------------------------
# parse_json_strict
# ---------------------------------------------------------------------------

class TestParseJsonStrict:
    def test_bare_json(self):
        result = parse_json_strict('{"key": "value"}')
        assert result == {"key": "value"}

    def test_fenced_json(self):
        raw = '```json\n{"key": "value"}\n```'
        result = parse_json_strict(raw)
        assert result == {"key": "value"}

    def test_prose_then_json(self):
        raw = 'Here is my analysis:\n{"key": "value"}\nEnd.'
        result = parse_json_strict(raw)
        assert result == {"key": "value"}

    def test_malformed_returns_none(self):
        result = parse_json_strict("{not valid json}")
        assert result is None

    def test_empty_returns_none(self):
        result = parse_json_strict("")
        assert result is None

    def test_no_braces_returns_none(self):
        result = parse_json_strict("just some text")
        assert result is None

    def test_nested_json(self):
        raw = '{"outer": {"inner": 42}}'
        result = parse_json_strict(raw)
        assert result["outer"]["inner"] == 42


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class TestCostTracker:
    def test_authorize_passes_under_cap(self):
        tracker = CostTracker(cap_usd=1.0)
        tracker.authorize(0.5)  # should not raise

    def test_authorize_raises_over_cap(self):
        tracker = CostTracker(cap_usd=1.0)
        tracker.record(0.8)
        with pytest.raises(CostBudgetExceeded):
            tracker.authorize(0.3)

    def test_authorize_raises_exactly_at_cap(self):
        tracker = CostTracker(cap_usd=1.0)
        tracker.record(0.999)
        with pytest.raises(CostBudgetExceeded):
            tracker.authorize(0.002)

    def test_record_accumulates(self):
        tracker = CostTracker(cap_usd=10.0)
        tracker.record(1.5)
        tracker.record(2.5)
        assert tracker.total_usd == pytest.approx(4.0)

    def test_budget_zero_raises_immediately(self):
        tracker = CostTracker(cap_usd=0.0)
        with pytest.raises(CostBudgetExceeded):
            tracker.authorize(0.0001)


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------

class TestRenderPrompt:
    def test_splits_markers_correctly(self, tmp_path):
        template = tmp_path / "test.txt"
        template.write_text(
            "---SYSTEM---\nYou are {role}.\n---USER---\nHello {name}.\n"
        )
        system, user = render_prompt(template, role="a bot", name="Alice")
        assert system == "You are a bot."
        assert user == "Hello Alice."

    def test_missing_system_marker_raises(self, tmp_path):
        template = tmp_path / "bad.txt"
        template.write_text("---USER---\nHello.\n")
        with pytest.raises(ValueError, match="SYSTEM"):
            render_prompt(template)

    def test_missing_user_marker_raises(self, tmp_path):
        template = tmp_path / "bad.txt"
        template.write_text("---SYSTEM---\nSystem.\n")
        with pytest.raises(ValueError, match="USER"):
            render_prompt(template)

    def test_real_structural_template(self):
        """Smoke-test that the real structural.txt loads and formats."""
        system, user = render_prompt(
            PROMPTS_DIR / "structural.txt",
            bars_csv="ts,o,h,l,c,v",
            swings_json="[]",
            sma200=40000.0,
            atr14=150.0,
            current_price=40100.0,
        )
        assert "trend" in system
        assert "40100.0" in user

    def test_real_execution_template(self):
        system, user = render_prompt(
            PROMPTS_DIR / "execution.txt",
            trend="up",
            structural_signal="new_hh_break",
            pattern_intact=True,
            confidence=0.8,
            last_confirmed_hh=40000.0,
            last_confirmed_hl=39500.0,
            current_price=40100.0,
            atr14=150.0,
            sma200=39000.0,
            position_side="flat",
            position_qty=0,
            avg_price=0.0,
            unrealized_pnl=0.0,
            pyramid_adds_used=0,
            equity=10000.0,
        )
        assert "action" in system
        assert "40100.0" in user

    def test_real_risk_template(self):
        system, user = render_prompt(
            PROMPTS_DIR / "risk_audit.txt",
            action="open_long",
            stop_price=39850.0,
            trailing_stop_atr_multiple=2.0,
            gemini_reasoning="Uptrend confirmed.",
            proposed_qty=1,
            position_side="flat",
            position_qty=0,
            avg_price=0.0,
            unrealized_pnl=0.0,
            pyramid_adds_used=0,
            equity=10000.0,
            realized_pnl_today=0.0,
            atr14=150.0,
        )
        assert "approved" in system
        assert "open_long" in user


# ---------------------------------------------------------------------------
# HaikuStructural mock tests
# ---------------------------------------------------------------------------

HAIKU_GOOD = {
    "trend": "up",
    "structural_signal": "new_hh_break",
    "pattern_intact": True,
    "confidence_0_to_1": 0.8,
    "last_confirmed_hh": 40000.0,
    "last_confirmed_hl": 39500.0,
    "reasoning": "Uptrend confirmed.",
}


class TestHaikuStructural:
    def _make_client(self, tracker=None):
        from src.llm.haiku_structural import HaikuStructural
        return HaikuStructural(api_key="test-key-redacted", tracker=tracker)

    def test_happy_path(self):
        client = self._make_client()
        with patch.object(client._client.messages, "create", return_value=_make_haiku_response(HAIKU_GOOD)):
            result = client.evaluate_raw("system", "user", bar_ts=12345)
        assert result.error is None
        assert result.parsed["trend"] == "up"
        assert result.used_fallback is False
        assert result.cost_usd > 0

    def test_timeout_returns_safe_default(self):
        client = self._make_client()
        with patch.object(client._client.messages, "create", side_effect=Exception("timeout")):
            result = client.evaluate_raw("system", "user", bar_ts=12345)
        assert result.used_fallback is True
        assert result.parsed["trend"] == "sideways"
        assert "timeout" in result.error

    def test_malformed_json_returns_safe_default(self):
        client = self._make_client()
        bad_msg = MagicMock()
        bad_msg.content = [MagicMock(text="not json at all")]
        bad_msg.usage.input_tokens = 10
        bad_msg.usage.output_tokens = 5
        bad_msg.usage.cache_read_input_tokens = 0
        with patch.object(client._client.messages, "create", return_value=bad_msg):
            result = client.evaluate_raw("system", "user", bar_ts=12345)
        assert result.used_fallback is True
        assert result.error == "json_parse_failed"

    def test_budget_exceeded_returns_safe_default_no_sdk_call(self):
        tracker = CostTracker(cap_usd=0.0)
        client = self._make_client(tracker=tracker)
        with patch.object(client._client.messages, "create") as mock_create:
            result = client.evaluate_raw("system", "user", bar_ts=12345)
            mock_create.assert_not_called()
        assert result.error == "budget_exceeded"
        assert result.used_fallback is True

    def test_db_persistence_row_inserted(self):
        client = self._make_client()
        mock_db = MagicMock()
        client._db = mock_db
        with patch.object(client._client.messages, "create", return_value=_make_haiku_response(HAIKU_GOOD)):
            client.evaluate_raw("system", "user", bar_ts=12345)
        mock_db.insert_llm_call.assert_called_once()
        call_kwargs = mock_db.insert_llm_call.call_args[0][0]
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs["error"] is None

    def test_api_key_not_in_logs(self, caplog):
        client = self._make_client()
        bad_msg = MagicMock()
        bad_msg.content = [MagicMock(text="bad json")]
        bad_msg.usage.input_tokens = 10
        bad_msg.usage.output_tokens = 5
        bad_msg.usage.cache_read_input_tokens = 0
        with caplog.at_level(logging.WARNING):
            with patch.object(client._client.messages, "create", return_value=bad_msg):
                client.evaluate_raw("system", "user", bar_ts=12345)
        for record in caplog.records:
            assert "test-key-redacted" not in record.message


# ---------------------------------------------------------------------------
# GeminiExecution mock tests
# ---------------------------------------------------------------------------

GEMINI_GOOD = {
    "action": "open_long",
    "stop_price": 39700.0,
    "trailing_stop_atr_multiple": 2.0,
    "reasoning": "Strong uptrend break.",
}


class TestGeminiExecution:
    def _make_client(self, tracker=None):
        from src.llm.gemini_execution import GeminiExecution
        return GeminiExecution(api_key="test-key-redacted", tracker=tracker)

    def test_happy_path(self):
        client = self._make_client()
        with patch.object(client._client.models, "generate_content",
                          return_value=_make_gemini_response(GEMINI_GOOD)):
            result = client.evaluate_raw("system", "user", bar_ts=12345)
        assert result.error is None
        assert result.parsed["action"] == "open_long"
        assert result.used_fallback is False

    def test_timeout_returns_safe_default(self):
        client = self._make_client()
        with patch.object(client._client.models, "generate_content",
                          side_effect=Exception("timeout")):
            result = client.evaluate_raw("system", "user", bar_ts=12345)
        assert result.used_fallback is True
        assert result.parsed["action"] == "hold"

    def test_malformed_json_returns_safe_default(self):
        client = self._make_client()
        bad_resp = MagicMock()
        bad_resp.text = "not json"
        bad_resp.usage_metadata.prompt_token_count = 10
        bad_resp.usage_metadata.candidates_token_count = 5
        with patch.object(client._client.models, "generate_content", return_value=bad_resp):
            result = client.evaluate_raw("system", "user", bar_ts=12345)
        assert result.used_fallback is True
        assert result.error == "json_parse_failed"

    def test_budget_exceeded_no_sdk_call(self):
        tracker = CostTracker(cap_usd=0.0)
        client = self._make_client(tracker=tracker)
        with patch.object(client._client.models, "generate_content") as mock_gen:
            result = client.evaluate_raw("system", "user", bar_ts=12345)
            mock_gen.assert_not_called()
        assert result.error == "budget_exceeded"

    def test_fallback_chain_first_two_fail_third_succeeds(self):
        """First 2 models fail; 3rd succeeds. used_fallback=False (got a parsed result)."""
        from src.llm.gemini_execution import GEMINI_MODELS
        client = self._make_client()

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise Exception("429 quota exceeded")
            return _make_gemini_response(GEMINI_GOOD)

        with patch.object(client._client.models, "generate_content", side_effect=side_effect):
            result = client.evaluate_raw("system", "user", bar_ts=12345)

        assert call_count[0] == 3
        assert result.parsed["action"] == "open_long"
        assert result.error is None

    def test_db_persistence(self):
        client = self._make_client()
        mock_db = MagicMock()
        client._db = mock_db
        with patch.object(client._client.models, "generate_content",
                          return_value=_make_gemini_response(GEMINI_GOOD)):
            client.evaluate_raw("system", "user", bar_ts=12345)
        mock_db.insert_llm_call.assert_called_once()

    def test_api_key_not_in_logs(self, caplog):
        client = self._make_client()
        bad_resp = MagicMock()
        bad_resp.text = "bad json"
        bad_resp.usage_metadata.prompt_token_count = 10
        bad_resp.usage_metadata.candidates_token_count = 5
        with caplog.at_level(logging.WARNING):
            with patch.object(client._client.models, "generate_content", return_value=bad_resp):
                client.evaluate_raw("system", "user", bar_ts=12345)
        for record in caplog.records:
            assert "test-key-redacted" not in record.message


# ---------------------------------------------------------------------------
# DeepSeekRisk mock tests
# ---------------------------------------------------------------------------

RISK_GOOD = {
    "approved": True,
    "violations": [],
    "override_action": None,
    "reasoning": "All checks passed.",
}

RISK_VIOLATION = {
    "approved": False,
    "violations": ["AVERAGING_DOWN"],
    "override_action": "hold",
    "reasoning": "Adding to losing position.",
}


class TestDeepSeekRisk:
    def _make_client(self, tracker=None):
        from src.llm.deepseek_risk import DeepSeekRisk
        return DeepSeekRisk(api_key="test-key-redacted", tracker=tracker)

    def test_happy_path_approved(self):
        client = self._make_client()
        with patch.object(client._client, "chat_completion",
                          return_value=_make_hf_response(RISK_GOOD)):
            result = client.evaluate_raw("system", "user", bar_ts=12345)
        assert result.error is None
        assert result.parsed["approved"] is True
        assert result.used_fallback is False

    def test_violation_parsed_correctly(self):
        client = self._make_client()
        with patch.object(client._client, "chat_completion",
                          return_value=_make_hf_response(RISK_VIOLATION)):
            result = client.evaluate_raw("system", "user", bar_ts=12345)
        assert result.parsed["approved"] is False
        assert "AVERAGING_DOWN" in result.parsed["violations"]

    def test_timeout_returns_safe_default(self):
        client = self._make_client()
        with patch.object(client._client, "chat_completion",
                          side_effect=Exception("connection timeout")):
            result = client.evaluate_raw("system", "user", bar_ts=12345)
        assert result.used_fallback is True
        assert result.parsed["approved"] is False
        assert "llm_unavailable" in result.parsed["violations"]

    def test_malformed_json_returns_safe_default(self):
        client = self._make_client()
        bad_resp = MagicMock()
        bad_resp.choices = [MagicMock(message=MagicMock(content="not json"))]
        bad_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        with patch.object(client._client, "chat_completion", return_value=bad_resp):
            result = client.evaluate_raw("system", "user", bar_ts=12345)
        assert result.used_fallback is True

    def test_budget_exceeded_no_sdk_call(self):
        tracker = CostTracker(cap_usd=0.0)
        client = self._make_client(tracker=tracker)
        with patch.object(client._client, "chat_completion") as mock_cc:
            result = client.evaluate_raw("system", "user", bar_ts=12345)
            mock_cc.assert_not_called()
        assert result.error == "budget_exceeded"

    def test_fallback_chain_first_two_fail_third_succeeds(self):
        client = self._make_client()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise Exception("503 service unavailable")
            return _make_hf_response(RISK_GOOD)

        with patch.object(client._client, "chat_completion", side_effect=side_effect):
            result = client.evaluate_raw("system", "user", bar_ts=12345)

        assert call_count[0] == 3
        assert result.parsed["approved"] is True

    def test_db_persistence(self):
        client = self._make_client()
        mock_db = MagicMock()
        client._db = mock_db
        with patch.object(client._client, "chat_completion",
                          return_value=_make_hf_response(RISK_GOOD)):
            client.evaluate_raw("system", "user", bar_ts=12345)
        mock_db.insert_llm_call.assert_called_once()

    def test_api_key_not_in_logs(self, caplog):
        client = self._make_client()
        bad_resp = MagicMock()
        bad_resp.choices = [MagicMock(message=MagicMock(content="bad json"))]
        bad_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        with caplog.at_level(logging.WARNING):
            with patch.object(client._client, "chat_completion", return_value=bad_resp):
                client.evaluate_raw("system", "user", bar_ts=12345)
        for record in caplog.records:
            assert "test-key-redacted" not in record.message
