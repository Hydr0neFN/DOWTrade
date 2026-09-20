"""
src/llm/deepseek_risk.py
=========================
DeepSeek (via HuggingFace Inference API) risk auditor with model fallback.
Replaces StubDeepSeek from Phase 2.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from huggingface_hub import InferenceClient as HFClient

from src.broker.models import AccountState
from src.llm.base import LLMCallResult, LLMClient, CostTracker, render_prompt

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "risk_audit.txt"

# Provider chain: HuggingFace first (DeepSeek V4.1-Flash), then Cloudflare
# Workers AI as a free fallback. The HF free tier returns 402 for *every*
# model once the monthly credit allowance is spent, so the old all-HF chain
# died as a unit -- it had been failing since 2026-09-15 unnoticed.
HF_MODEL = os.environ.get("HF_RISK_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")

# Workers AI free plan, OpenAI-compatible endpoint. Both picked by measured
# accuracy on this exact risk_audit prompt: mistral-small has the best
# violation recall, llama-4-scout is the most deterministic. Models that put
# their answer in `reasoning` and leave `content` empty (gpt-oss, glm-4.7-flash,
# nemotron-3, gemma-4, qwen3-30b) are unusable here -- json.loads() gets "".
CF_MODELS = [
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
]
CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()


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
        self._last_model = HF_MODEL
        # Cloudflare Workers AI speaks the OpenAI chat-completions schema, so
        # the same InferenceClient call shape works with a swapped base_url.
        self._cf_client = None
        if CF_ACCOUNT_ID and CF_API_TOKEN:
            self._cf_client = HFClient(
                base_url=f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1",
                token=CF_API_TOKEN,
            )
        else:
            log.warning("[deepseek] Cloudflare fallback disabled "
                        "(CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN unset)")

    def _providers(self):
        """(client, model) in priority order: HF primary, Cloudflare free fallback."""
        yield self._client, HF_MODEL
        if self._cf_client is not None:
            for model in CF_MODELS:
                yield self._cf_client, model

    def _call(self, system: str, user: str) -> Tuple[str, int, int]:
        """Try each provider/model in order; fall through on error."""
        last_exc = None
        for client, model in self._providers():
            try:
                resp = client.chat_completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=500,
                    temperature=0.1,
                )
                raw = resp.choices[0].message.content or ""
                if not raw.strip():
                    # Reasoning-only models answer in `reasoning` and leave
                    # `content` empty; that is a failure for our JSON schema.
                    raise ValueError("empty content (reasoning-only response)")
                self._last_model = model
                in_tok = getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0
                out_tok = getattr(getattr(resp, "usage", None), "completion_tokens", 0) or 0
                return raw, in_tok, out_tok
            except Exception as exc:
                log.warning("[deepseek] Model %s failed (%s) -- trying next", model, str(exc)[:80])
                last_exc = exc
                continue
        raise RuntimeError(f"All risk-audit models failed: {last_exc}")

    def _actual_cost_usd(self, in_tok: int, out_tok: int) -> float:
        # Cloudflare Workers AI runs inside the free daily Neuron allowance, so
        # charging the HF placeholder for it would walk the shared CostTracker
        # into MAX_LLM_SPEND_USD and halt risk auditing over spend that never
        # happened. HF Inference API is on the user's plan; $0.001 placeholder.
        if self._last_model.startswith("@cf/"):
            return 0.0
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
