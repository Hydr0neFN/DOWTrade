"""
src/llm/base.py
===============
Base infrastructure for all LLM clients:
  - LLMCallResult dataclass
  - CostTracker / CostBudgetExceeded
  - Utility helpers: strip_json_fences, parse_json_strict, render_prompt, retry
  - LLMClient ABC
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from src.config import MAX_LLM_SPEND_USD

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LLMCallResult:
    parsed: Optional[dict]
    raw_response: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: Optional[str]
    used_fallback: bool
    model_used: str


class CostBudgetExceeded(RuntimeError):
    """Raised when an LLM call would exceed the cumulative cost cap."""


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------

class CostTracker:
    def __init__(self, cap_usd: float = MAX_LLM_SPEND_USD) -> None:
        self._total: float = 0.0
        self._cap: float = cap_usd

    def authorize(self, estimated_cost_usd: float) -> None:
        if self._total + estimated_cost_usd > self._cap:
            raise CostBudgetExceeded(
                f"Budget cap would be exceeded "
                f"(current={self._total:.4f}, estimated={estimated_cost_usd:.6f})"
            )

    def record(self, actual_cost_usd: float) -> None:
        self._total += actual_cost_usd

    @property
    def total_usd(self) -> float:
        return self._total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_json_fences(text: str) -> str:
    """Remove markdown code fences around JSON."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        inner = parts[1] if len(parts) > 1 else text
        if inner.startswith("json"):
            inner = inner[4:]
        return inner.strip()
    return text


def parse_json_strict(raw: str) -> Optional[dict]:
    """Strip fences, find first {...} block, parse JSON. Returns None on any failure."""
    try:
        cleaned = strip_json_fences(raw)
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group(0))
    except Exception:
        return None


def render_prompt(template_path: Path, **kwargs: Any) -> Tuple[str, str]:
    """
    Load a prompt template file and split on ---SYSTEM--- / ---USER--- markers.
    Returns (system_text, user_text) with kwargs substituted.
    """
    content = template_path.read_text(encoding="utf-8")
    parts = re.split(r'---SYSTEM---\s*', content, maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Template {template_path} missing ---SYSTEM--- marker")
    after_system = parts[1]
    halves = re.split(r'---USER---\s*', after_system, maxsplit=1)
    if len(halves) != 2:
        raise ValueError(f"Template {template_path} missing ---USER--- marker")
    system_tmpl, user_tmpl = halves
    system = system_tmpl.strip().format(**kwargs)
    user = user_tmpl.strip().format(**kwargs)
    return system, user


def retry(fn, retries: int = 1, delay: float = 3.0):
    """Call fn; retry up to `retries` times on any exception."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries:
                raise
            log.warning(
                "Attempt %d/%d failed (%s). Retrying in %.0fs",
                attempt + 1, retries + 1, exc, delay,
            )
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Abstract base client
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """
    Abstract base for Haiku / Gemini / DeepSeek clients.

    Subclasses implement:
      _call(system, user) -> (raw_text, in_tokens, out_tokens)
      _actual_cost_usd(in_tok, out_tok) -> float
      _estimated_cost_usd(prompt_chars) -> float
    """

    name: str = "base"
    safe_default: dict = {}
    schema_keys: set = set()

    def __init__(
        self,
        tracker: Optional[CostTracker] = None,
        db=None,
        prompt_path: Optional[Path] = None,
    ) -> None:
        self._tracker = tracker or CostTracker()
        self._db = db
        self._prompt_path = prompt_path
        self._last_model: str = self.name

    def _make_safe_result(
        self,
        error: str,
        latency_ms: int = 0,
        used_fallback: bool = True,
    ) -> LLMCallResult:
        return LLMCallResult(
            parsed=dict(self.safe_default),
            raw_response="",
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            error=error,
            used_fallback=used_fallback,
            model_used=self.name,
        )

    def evaluate_raw(
        self,
        system: str,
        user: str,
        *,
        bar_ts: int,
    ) -> LLMCallResult:
        """
        Full call pipeline:
        1. Estimate cost and authorize against budget.
        2. Call _call() with timeout tracking.
        3. Parse JSON response and validate schema keys.
        4. Record cost and optionally persist to DB.
        """
        # Step 1: budget check
        est_cost = self._estimated_cost_usd(len(system) + len(user))
        try:
            self._tracker.authorize(est_cost)
        except CostBudgetExceeded as exc:
            log.warning("[%s] Budget exceeded: %s", self.name, exc)
            return self._make_safe_result(error="budget_exceeded", used_fallback=True)

        # Step 2: call with timing
        t0 = time.monotonic()
        try:
            raw, in_tok, out_tok = self._call(system, user)
        except Exception as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            log.warning("[%s] _call() failed: %s", self.name, str(exc)[:200])
            return self._make_safe_result(
                error=str(exc)[:300],
                latency_ms=latency_ms,
                used_fallback=True,
            )
        latency_ms = int((time.monotonic() - t0) * 1000)

        # The API was already hit, so tokens were spent regardless of whether the
        # response parses. Record that spend on the failure paths too, or the
        # budget cap (MAX_LLM_SPEND_USD) can be bypassed by a model that keeps
        # returning malformed/partial JSON.
        actual_cost = self._actual_cost_usd(in_tok, out_tok)

        # Step 3: parse JSON
        parsed = parse_json_strict(raw)
        if parsed is None:
            log.warning("[%s] JSON parse failed. raw[:200]=%s", self.name, raw[:200])
            self._tracker.record(actual_cost)
            return LLMCallResult(
                parsed=dict(self.safe_default),
                raw_response=raw[:500],
                latency_ms=latency_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=actual_cost,
                error="json_parse_failed",
                used_fallback=True,
                model_used=getattr(self, '_last_model', self.name),
            )

        # Validate schema keys
        missing = self.schema_keys - parsed.keys()
        if missing:
            log.warning("[%s] Missing schema keys: %s", self.name, missing)
            self._tracker.record(actual_cost)
            return LLMCallResult(
                parsed=dict(self.safe_default),
                raw_response=raw[:500],
                latency_ms=latency_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=actual_cost,
                error=f"missing_keys:{missing}",
                used_fallback=True,
                model_used=getattr(self, '_last_model', self.name),
            )

        # Step 4: record cost
        actual_cost = self._actual_cost_usd(in_tok, out_tok)
        self._tracker.record(actual_cost)

        if self._db is not None:
            prompt_hash = hashlib.sha256((system + user).encode()).hexdigest()[:16]
            try:
                self._db.insert_llm_call({
                    "bar_ts": str(bar_ts),
                    "model": getattr(self, '_last_model', self.name),
                    "prompt_hash": prompt_hash,
                    "raw_response": raw[:2000],
                    "parsed_json": json.dumps(parsed),
                    "latency_ms": latency_ms,
                    "cost_usd": actual_cost,
                    "error": None,
                })
            except Exception as db_exc:
                log.warning("[%s] DB insert failed: %s", self.name, db_exc)

        return LLMCallResult(
            parsed=parsed,
            raw_response=raw[:500],
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=actual_cost,
            error=None,
            used_fallback=False,
            model_used=getattr(self, '_last_model', self.name),
        )

    @abstractmethod
    def _call(self, system: str, user: str) -> Tuple[str, int, int]:
        """Return (raw_text, input_tokens, output_tokens)."""

    @abstractmethod
    def _actual_cost_usd(self, in_tok: int, out_tok: int) -> float:
        """Compute actual cost from token counts."""

    @abstractmethod
    def _estimated_cost_usd(self, prompt_chars: int) -> float:
        """Estimate cost from prompt character count (before the call)."""
