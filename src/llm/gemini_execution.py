"""
src/llm/gemini_execution.py
============================
Gemini execution judge with model fallback chain.
Replaces StubGemini from Phase 2.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

from google import genai
from google.genai import types as genai_types

from src.broker.models import Position
from src.config import LLM_TIMEOUT_SEC
from src.data.features import MarketSnapshot
from src.llm.base import LLMCallResult, LLMClient, CostTracker, render_prompt

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "execution.txt"

GEMINI_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]


class GeminiExecution(LLMClient):
    """
    Gemini execution judge — translates structural signal into trade action.

    Output schema (brief §6.2):
      action, stop_price, trailing_stop_atr_multiple, reasoning
    """

    name = "gemini"
    schema_keys = {"action", "stop_price", "trailing_stop_atr_multiple", "reasoning"}
    safe_default = {
        "action": "hold",
        "stop_price": 0.0,
        "trailing_stop_atr_multiple": 2.0,
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
        self._client = genai.Client(api_key=api_key)
        self._last_model = GEMINI_MODELS[-1]  # default to final fallback

    def _call(self, system: str, user: str) -> Tuple[str, int, int]:
        """Try each model in GEMINI_MODELS; fall through on quota/availability errors."""
        last_exc = None
        for model in GEMINI_MODELS:
            try:
                resp = self._client.models.generate_content(
                    model=model,
                    contents=user,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type="application/json",
                    ),
                )
                self._last_model = model
                in_tok = resp.usage_metadata.prompt_token_count if resp.usage_metadata else 0
                out_tok = resp.usage_metadata.candidates_token_count if resp.usage_metadata else 0
                return resp.text, in_tok, out_tok
            except Exception as exc:
                err_str = str(exc).lower()
                log.warning("[gemini] Model %s failed (%s) -- trying next", model, str(exc)[:80])
                last_exc = exc
                continue
        raise RuntimeError(f"All Gemini models failed: {last_exc}")

    def _actual_cost_usd(self, in_tok: int, out_tok: int) -> float:
        # Gemini 2.5 Flash pricing placeholder: $0.075/M input, $0.30/M output
        return in_tok * 0.075e-6 + out_tok * 0.30e-6

    def _estimated_cost_usd(self, prompt_chars: int) -> float:
        est_in = prompt_chars / 4
        est_out = 300
        return self._actual_cost_usd(int(est_in), est_out)

    def evaluate(
        self,
        haiku_dict: dict,
        snapshot: MarketSnapshot,
        position: Position,
        equity: float,
        *,
        bar_ts: int,
    ) -> LLMCallResult:
        """
        Build execution prompt from haiku analysis + position context and call Gemini.
        """
        system, user = render_prompt(
            self._prompt_path,
            trend=haiku_dict.get("trend", "sideways"),
            structural_signal=haiku_dict.get("structural_signal", "none"),
            pattern_intact=haiku_dict.get("pattern_intact", True),
            confidence=haiku_dict.get("confidence_0_to_1", 0.0),
            last_confirmed_hh=haiku_dict.get("last_confirmed_hh"),
            last_confirmed_hl=haiku_dict.get("last_confirmed_hl"),
            current_price=snapshot.current_price,
            atr14=snapshot.atr14,
            sma200=snapshot.sma200,
            position_side=position.side,
            position_qty=position.qty,
            avg_price=position.avg_price,
            unrealized_pnl=position.unrealized_pnl,
            pyramid_adds_used=position.pyramid_adds_used,
            equity=equity,
        )
        return self.evaluate_raw(system, user, bar_ts=bar_ts)
