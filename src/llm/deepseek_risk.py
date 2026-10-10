"""
src/llm/deepseek_risk.py
=========================
Advisory risk auditor. Provider chain (first valid answer wins):

  1. NVIDIA NIM  openai/gpt-oss-20b      (20 s)
  2. NVIDIA NIM  z-ai/glm-5.3-flash      (25 s)
  3. Cloudflare  llama-4-scout-17b       (free plan)
  4. Cloudflare  mistral-small-3.1-24b   (free plan)

The class keeps its historical name (DeepSeekRisk); DeepSeek on Hugging Face was
the original provider and was removed on 2026-10-11: that account has no credit
and returned HTTP 402 on every call.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from src.broker.models import AccountState
from src.llm import nim
from src.llm.base import LLMCallResult, LLMClient, CostTracker, render_prompt
from src.llm.cf import cf_chat, make_cf_client

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "risk_audit.txt"

# NIM models, picked from /root/nim-bench: on the risk-audit prompt gpt-oss-20b
# and glm-5.3-flash both caught 18/18 true violations with no false positives,
# against 9/18 for Cloudflare mistral-small and 10/18 for llama-4-scout.
NIM_CHAIN = (
    nim.NimSpec(nim.GPT_OSS_20B, 20.0),
    nim.NimSpec(nim.GLM_FLASH, 25.0),
)

# Workers AI free plan. llama-4-scout goes first: it was the more accurate and
# more deterministic of the two on this exact prompt. Models that put their
# answer in `reasoning` and leave `content` empty (gpt-oss, glm-4.7-flash,
# nemotron-3, gemma-4, qwen3-30b) are unusable on Cloudflare -- json.loads()
# gets "".
CF_MODELS = [
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
]


class DeepSeekRisk(LLMClient):
    """
    Risk auditor (NIM first, Cloudflare Workers AI as fallback).

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
        tracker: Optional[CostTracker] = None,
        db=None,
        prompt_path: Optional[Path] = None,
    ) -> None:
        super().__init__(tracker=tracker, db=db, prompt_path=prompt_path or _PROMPT_PATH)
        self._last_model = NIM_CHAIN[0].model
        self._cf_client = make_cf_client("deepseek")

    def _call(self, system: str, user: str) -> Tuple[str, int, int]:
        """NIM chain, then the Cloudflare models; fall through on any failure."""
        res = nim.run_chain(NIM_CHAIN, system, user, self.schema_keys)
        if res is not None:
            self._last_model = f"nim:{res.model}"
            return res.raw, res.in_tok, res.out_tok
        last_exc = None
        if self._cf_client is not None:
            for model in CF_MODELS:
                try:
                    raw, in_tok, out_tok = cf_chat(self._cf_client, model, system, user)
                    self._last_model = model
                    return raw, in_tok, out_tok
                except Exception as exc:
                    log.warning("[deepseek] Model %s failed (%s) -- trying next", model, str(exc)[:80])
                    last_exc = exc
        raise RuntimeError(f"All risk-audit models failed: {last_exc or 'no provider available'}")

    def _actual_cost_usd(self, in_tok: int, out_tok: int) -> float:
        # Every provider left in the chain is free (NIM free tier, Cloudflare's
        # daily Neuron allowance). Charging a placeholder would walk the shared
        # CostTracker toward MAX_LLM_SPEND_USD over spend that never happened.
        return 0.0

    def _estimated_cost_usd(self, prompt_chars: int) -> float:
        return 0.0

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
        Build risk audit prompt from execution decision + account state and run the provider chain.
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
