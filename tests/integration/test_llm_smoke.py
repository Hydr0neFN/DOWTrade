"""
tests/integration/test_llm_smoke.py
=====================================
Real connectivity smoke tests. Skipped unless RUN_LLM_SMOKE=1.

Makes ONE real call per LLM with a tiny prompt.
Asserts non-empty parsed dict with expected keys.
"""
from __future__ import annotations

import os
import pytest

from src.config import Settings

RUN_SMOKE = os.environ.get("RUN_LLM_SMOKE", "0") == "1"
skip_unless_smoke = pytest.mark.skipif(not RUN_SMOKE, reason="Set RUN_LLM_SMOKE=1 to run")

settings = Settings()


@skip_unless_smoke
def test_haiku_structural_smoke():
    """Real call to Claude Haiku with a minimal structural prompt."""
    from src.llm.haiku_structural import HaikuStructural
    from src.llm.base import CostTracker

    tracker = CostTracker(cap_usd=1.0)
    client = HaikuStructural(api_key=settings.anthropic_api_key, tracker=tracker)

    system = (
        "You are a structural trend judge. Output JSON only. "
        'Schema: {"trend": "up|down|sideways", "structural_signal": "none", '
        '"pattern_intact": true, "confidence_0_to_1": 0.5, '
        '"last_confirmed_hh": null, "last_confirmed_hl": null, "reasoning": "smoke test"}'
    )
    user = "Price: 40000. No bars available. Return the safe fallback JSON."

    result = client.evaluate_raw(system, user, bar_ts=0)

    print(f"\n[haiku] model={result.model_used} latency={result.latency_ms}ms "
          f"cost=${result.cost_usd:.6f} error={result.error}")
    assert result.parsed is not None, f"parsed is None, error={result.error}"
    assert "trend" in result.parsed
    assert result.cost_usd >= 0


@skip_unless_smoke
def test_gemini_execution_smoke():
    """Real call to Gemini with a minimal execution prompt."""
    from src.llm.gemini_execution import GeminiExecution
    from src.llm.base import CostTracker

    tracker = CostTracker(cap_usd=1.0)
    client = GeminiExecution(api_key=settings.google_api_key, tracker=tracker)

    system = (
        "You are an execution judge for a futures bot. Output JSON only. "
        'Schema: {"action": "hold", "stop_price": 0.0, '
        '"trailing_stop_atr_multiple": 2.0, "reasoning": "smoke test"}'
    )
    user = "No position. No signal. Return the hold JSON."

    result = client.evaluate_raw(system, user, bar_ts=0)

    print(f"\n[gemini] model={result.model_used} latency={result.latency_ms}ms "
          f"cost=${result.cost_usd:.6f} error={result.error}")
    assert result.parsed is not None, f"parsed is None, error={result.error}"
    assert "action" in result.parsed
    assert result.cost_usd >= 0


@skip_unless_smoke
def test_deepseek_risk_smoke():
    """Real call to HuggingFace DeepSeek with a minimal risk audit prompt."""
    from src.llm.deepseek_risk import DeepSeekRisk
    from src.llm.base import CostTracker

    tracker = CostTracker(cap_usd=1.0)
    client = DeepSeekRisk(api_key=settings.huggingface_api_key, tracker=tracker)

    system = (
        "You are a risk auditor for a futures bot. Output JSON only. "
        'Schema: {"approved": true, "violations": [], '
        '"override_action": null, "reasoning": "smoke test"}'
    )
    user = "Action: hold. No position. No violations. Return approved=true JSON."

    result = client.evaluate_raw(system, user, bar_ts=0)

    print(f"\n[deepseek] model={result.model_used} latency={result.latency_ms}ms "
          f"cost=${result.cost_usd:.6f} error={result.error}")
    assert result.parsed is not None, f"parsed is None, error={result.error}"
    assert "approved" in result.parsed
    assert result.cost_usd >= 0
