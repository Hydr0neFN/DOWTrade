"""
src/llm/deepseek_risk.py
=========================
DeepSeek (via HuggingFace Inference API) risk auditor with model fallback.
Replaces StubDeepSeek from Phase 2.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from huggingface_hub import InferenceClient as HFClient

from src.broker.models import AccountState
from src.llm.base import LLMCallResult, LLMClient, CostTracker, render_prompt

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "risk_audit.txt"

HF_MODELS = [
    "deepseek-ai/DeepSeek-V3.2-Exp",
    "Qwen/Qwen3-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]


class DeepSeekRisk(LLMClient):
    """
    HuggingFace-hosted risk auditor (primary: DeepSeek-V3.2-Exp).

    Output schema (brief §6.3):
      approved, violations, override_action, reasoning
    """

    name = "deepseek"
    schema_keys = {"approved", "violations", "reasoning"}
    safe_default = {
        "approved": False,
        "violations": ["llm_unavailable"],
        "override_action": "hold",
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
        self._client = HFClient(token=api_key)
        self._last_model = HF_MODELS[0]

    def _call(self, system: str, user: str) -> Tuple[str, int, int]:
        """Try each HF model in order; fall through on error."""
        last_exc = None
        for model in HF_MODELS:
            try:
                resp = self._client.chat_completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=300,
                    temperature=0.1,
                )
                self._last_model = model
                raw = resp.choices[0].message.content or ""
                in_tok = getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0
                out_tok = getattr(getattr(resp, "usage", None), "completion_tokens", 0) or 0
                return raw, in_tok, out_tok
            except Exception as exc:
                log.warning("[deepseek] Model %s failed (%s) -- trying next", model, str(exc)[:80])
                last_exc = exc
                continue
        raise RuntimeError(f"All HF models failed: {last_exc}")

    def _actual_cost_usd(self, in_tok: int, out_tok: int) -> float:
        # HF Inference API is on user's plan; $0.001 placeholder per call
        return 0.001

    def _estimated_cost_usd(self, prompt_chars: int) -> float:
        return 0.001

    def evaluate(
        self,
        gemini_dict: dict,
        proposed_qty: int,
        state: AccountState,
        atr14: float,
        *,
        bar_ts: int,
        mark_price: float = 0.0,
    ) -> LLMCallResult:
        """
        Build risk audit prompt from execution decision + account state and call DeepSeek.
        mark_price is the current bar close — used by the LLM to correctly measure
        stop distance for pyramid adds (stop is relative to new entry, not avg_price).
        """
        system, user = render_prompt(
            self._prompt_path,
            action=gemini_dict.get("action", "hold"),
            stop_price=gemini_dict.get("stop_price", 0.0),
            trailing_stop_atr_multiple=gemini_dict.get("trailing_stop_atr_multiple", 2.0),
            gemini_reasoning=gemini_dict.get("reasoning", ""),
            proposed_qty=proposed_qty,
            position_side=state.position.side,
            position_qty=state.position.qty,
            avg_price=state.position.avg_price,
            unrealized_pnl=state.position.unrealized_pnl,
            pyramid_adds_used=state.position.pyramid_adds_used,
            equity=state.equity,
            realized_pnl_today=state.realized_pnl_today,
            atr14=atr14,
            mark_price=mark_price,
        )
        return self.evaluate_raw(system, user, bar_ts=bar_ts)
