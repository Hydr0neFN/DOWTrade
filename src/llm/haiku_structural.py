"""
src/llm/haiku_structural.py
============================
Claude Haiku structural trend judge.
Replaces StubHaiku from Phase 2.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

from anthropic import Anthropic

from src.config import LLM_TIMEOUT_SEC
from src.data.features import MarketSnapshot
from src.llm.base import LLMCallResult, LLMClient, CostTracker, render_prompt
from src.llm.claude_sdk import sdk_available, sdk_complete, scope_allows

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "structural.txt"


class HaikuStructural(LLMClient):
    """
    Claude Haiku 4.5 structural trend judge.

    Output schema (brief §6.1):
      trend, last_confirmed_hh, last_confirmed_hl, pattern_intact,
      structural_signal, confidence_0_to_1, reasoning
    """

    name = "haiku"
    schema_keys = {
        "trend",
        "structural_signal",
        "pattern_intact",
        "confidence_0_to_1",
        "reasoning",
    }
    safe_default = {
        "trend": "sideways",
        "structural_signal": "none",
        "pattern_intact": True,
        "confidence_0_to_1": 0.0,
        "last_confirmed_hh": None,
        "last_confirmed_hl": None,
        "reasoning": "fallback",
    }

    def __init__(
        self,
        api_key: str,
        tracker: Optional[CostTracker] = None,
        db=None,
        prompt_path: Optional[Path] = None,
    ) -> None:
        super().__init__(tracker=tracker, db=db, prompt_path=prompt_path or _PROMPT_PATH)
        self._client = Anthropic(api_key=api_key)
        self._last_model = "claude-haiku-4-5-20251001"

    def _call(self, system: str, user: str) -> Tuple[str, int, int]:
        # Primary: Claude Sonnet via the subscription SDK (the $20 credit), but
        # only on crucial bars (see DOWTRADE_SDK_FOR / the `crucial` flag).
        if sdk_available() and scope_allows(getattr(self, "_crucial", False)):
            try:
                raw, in_tok, out_tok, _cost = sdk_complete(system, user, max_tokens=400)
                self._last_model = "sonnet-sdk"
                return raw, in_tok, out_tok
            except Exception as exc:
                log.warning("[haiku] SDK path failed (%s) — falling back to Haiku API",
                            str(exc)[:160])
        # Fallback: metered Anthropic API on Haiku.
        msg = self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            timeout=LLM_TIMEOUT_SEC,
        )
        self._last_model = "claude-haiku-4-5-20251001"
        raw = msg.content[0].text.strip()
        in_tok = msg.usage.input_tokens + getattr(msg.usage, "cache_read_input_tokens", 0)
        out_tok = msg.usage.output_tokens
        return raw, in_tok, out_tok

    def _actual_cost_usd(self, in_tok: int, out_tok: int) -> float:
        # Sonnet (subscription SDK path) vs Haiku (API fallback) pricing.
        if getattr(self, "_last_model", "") == "sonnet-sdk":
            return in_tok * 3e-6 + out_tok * 15e-6
        # Haiku 4.5: $1/M input, $5/M output
        return in_tok * 1e-6 + out_tok * 5e-6

    def _estimated_cost_usd(self, prompt_chars: int) -> float:
        # Rough estimate: ~4 chars per token, 400 output tokens max
        est_in = prompt_chars / 4
        est_out = 400
        return self._actual_cost_usd(int(est_in), est_out)

    def evaluate(
        self,
        snapshot: MarketSnapshot,
        *,
        bar_ts: int,
        crucial: bool = False,
    ) -> LLMCallResult:
        """
        Build prompt from snapshot and call the model.
        `crucial=True` (e.g. a live position is open) routes this call to Sonnet
        under DOWTRADE_SDK_FOR='positions'; otherwise it stays on Haiku.
        Returns LLMCallResult with parsed structural analysis.
        """
        self._crucial = crucial
        bars_csv = "\n".join(
            f"{b.ts},{b.o},{b.h},{b.l},{b.c},{b.v}"
            for b in snapshot.bars[-40:]
        )
        swings_json = json.dumps(
            [{"ts": s.ts, "price": s.price, "kind": s.kind} for s in snapshot.swings]
        )
        system, user = render_prompt(
            self._prompt_path,
            bars_csv=bars_csv,
            swings_json=swings_json,
            sma200=snapshot.sma200,
            atr14=snapshot.atr14,
            current_price=snapshot.current_price,
        )
        return self.evaluate_raw(system, user, bar_ts=bar_ts)
