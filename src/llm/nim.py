"""
src/llm/nim.py
==============
NVIDIA NIM (build.nvidia.com hosted, OpenAI-compatible) client and the guard
that every leg shares.

NIM is free but rate limited (~40 requests/min per key), and the three legs run
back to back from one trading loop, so the guard is a process-wide singleton:

  * sliding-window limiter -- at most NIM_MAX_REQUESTS per rolling
    NIM_WINDOW_SEC across all legs (the structural shadow call included). When
    the window is full NIM is "unavailable" for that call: the caller moves on
    to its next non-NIM provider immediately, never sleeps.
  * 429 cooldown -- NIM is globally unavailable for 30 s (Retry-After honoured
    if larger, capped at 120 s).
  * per-model circuit breaker -- 2 consecutive timeouts / 5xx / connection
    errors open the breaker for 15 min, after which exactly one probe call is
    let through. Schema and parse failures are per-prompt, not a sign that the
    model is down, so they never trip it.

There are no retries anywhere: every failure falls through at once. Without a
key (env NVIDIA_NIM_API_KEY, else ~/.nvidia-nim-key) NIM is disabled and the
callers behave exactly as they did before NIM existed. The key is only ever put
in the Authorization header; nothing here logs it.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import httpx

from src.llm.base import parse_json_strict, strip_json_fences

log = logging.getLogger(__name__)

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
KEY_ENV = "NVIDIA_NIM_API_KEY"
KEY_FILE = Path.home() / ".nvidia-nim-key"

GPT_OSS_20B = "openai/gpt-oss-20b"
GLM_FLASH = "z-ai/glm-5.3-flash"
NEMOTRON_SUPER = "nvidia/nemotron-3-super-120b-a12b"

# Per-model request settings, taken from the benchmark in /root/nim-bench.
#   gpt-oss-20b : reasoning kept at "low"; the default burns the whole budget on
#                 reasoning and leaves `content` empty.
#   glm-5.3-flash: reasoning cannot be switched off, so it needs headroom.
#   nemotron-3-super: thinking off makes it a ~1 s answer instead of an empty one.
_MODEL_SETTINGS = {
    GPT_OSS_20B: {"max_tokens": 1024, "reasoning_effort": "low"},
    GLM_FLASH: {"max_tokens": 2048},
    NEMOTRON_SUPER: {
        "max_tokens": 1024,
        "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
    },
}

# Guard parameters.
NIM_MAX_REQUESTS = 30
NIM_WINDOW_SEC = 60.0
COOLDOWN_429_SEC = 30.0
COOLDOWN_429_MAX_SEC = 120.0
BREAKER_THRESHOLD = 2
BREAKER_OPEN_SEC = 15 * 60.0

# Failure kinds that count towards the breaker: the model/endpoint itself is
# unreachable or unhealthy, as opposed to this one prompt getting a bad answer.
BREAKER_KINDS = frozenset({"timeout", "5xx", "connection"})

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class NimSpec:
    """One model in a leg's chain, with that leg's hard timeout for it."""
    model: str
    timeout: float


@dataclass
class NimResult:
    model: str
    raw: str = ""                      # cleaned content (fences / <think> stripped)
    in_tok: int = 0
    out_tok: int = 0
    latency_ms: int = 0
    error: Optional[str] = None        # failure kind; None on success
    skipped: Optional[str] = None      # "cooldown" | "ratelimit" | "breaker"; no request was made

    @property
    def ok(self) -> bool:
        return self.error is None and self.skipped is None


# ---------------------------------------------------------------------------
# Guard: limiter + 429 cooldown + circuit breaker + stats
# ---------------------------------------------------------------------------

class _Breaker:
    __slots__ = ("fails", "open_until", "probing")

    def __init__(self) -> None:
        self.fails = 0
        self.open_until: Optional[float] = None   # None = closed
        self.probing = False


class NimGuard:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        max_requests: int = NIM_MAX_REQUESTS,
        window_sec: float = NIM_WINDOW_SEC,
    ) -> None:
        self._clock = clock
        self._max = max_requests
        self._window = window_sec
        self._lock = threading.Lock()
        self._stamps: deque = deque()
        self._unavailable_until = 0.0
        self._breakers: dict = {}
        self._counts: Counter = Counter()
        self._failures: Counter = Counter()

    # -- admission ---------------------------------------------------------

    def acquire(self, model: str) -> Optional[str]:
        """None if a request may go out now (a limiter slot is consumed), else
        the reason it may not: "cooldown", "breaker" or "ratelimit"."""
        with self._lock:
            now = self._clock()
            if now < self._unavailable_until:
                self._counts["skipped_cooldown"] += 1
                return "cooldown"
            b = self._breakers.get(model)
            half_open = b is not None and b.open_until is not None and now >= b.open_until
            if b is not None and b.open_until is not None and (not half_open or b.probing):
                self._counts["skipped_breaker"] += 1
                return "breaker"
            while self._stamps and now - self._stamps[0] >= self._window:
                self._stamps.popleft()
            if len(self._stamps) >= self._max:
                self._counts["skipped_ratelimit"] += 1
                return "ratelimit"
            if half_open:
                b.probing = True            # the one probe call
            self._stamps.append(now)
            self._counts["calls"] += 1
            return None

    # -- outcomes ----------------------------------------------------------

    def record_success(self, model: str) -> None:
        """The model answered (even if the answer later failed validation): it is
        reachable, so the consecutive-failure run is broken and a probe closes
        the breaker."""
        with self._lock:
            b = self._breakers.get(model)
            if b is None:
                return
            if b.open_until is not None:
                log.info("[nim] breaker CLOSED for %s (probe answered)", model)
            b.fails, b.open_until, b.probing = 0, None, False

    def record_failure(self, model: str, kind: str) -> None:
        """A timeout / 5xx / connection failure."""
        with self._lock:
            self._failures[kind] += 1
            b = self._breakers.setdefault(model, _Breaker())
            b.fails += 1
            was_probe = b.probing
            b.probing = False
            if b.fails >= BREAKER_THRESHOLD or was_probe:
                b.open_until = self._clock() + BREAKER_OPEN_SEC
                log.info("[nim] breaker OPEN for %s: %d consecutive failures (last=%s), "
                         "skipping for %.0f min", model, b.fails, kind, BREAKER_OPEN_SEC / 60)

    def record_other_failure(self, model: str, kind: str) -> None:
        """A parse / schema / 4xx failure: counted, but it does not trip the breaker."""
        with self._lock:
            self._failures[kind] += 1
            b = self._breakers.get(model)
            if b is not None:
                b.probing = False           # let the next call be the probe instead

    def record_rate_limited(self, model: str, retry_after: Optional[float]) -> float:
        """HTTP 429: NIM is unavailable for every caller for a while."""
        with self._lock:
            self._failures["429"] += 1
            wait = min(max(COOLDOWN_429_SEC, retry_after or 0.0), COOLDOWN_429_MAX_SEC)
            self._unavailable_until = max(self._unavailable_until, self._clock() + wait)
            b = self._breakers.get(model)
            if b is not None:
                b.probing = False           # a 429 says nothing about model health
            return wait

    # -- stats -------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            return {
                "calls": self._counts["calls"],
                "skipped_ratelimit": self._counts["skipped_ratelimit"],
                "skipped_breaker": self._counts["skipped_breaker"],
                "skipped_cooldown": self._counts["skipped_cooldown"],
                "failures": dict(self._failures),
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._counts.clear()
            self._failures.clear()

    def clear(self) -> None:
        """Forget all state (window, cooldown, breakers, stats). For tests."""
        with self._lock:
            self._stamps.clear()
            self._unavailable_until = 0.0
            self._breakers.clear()
            self._counts.clear()
            self._failures.clear()

    def summary_line(self) -> str:
        s = self.stats()
        fails = ",".join(f"{k}={v}" for k, v in sorted(s["failures"].items())) or "none"
        return (f"NIM stats: calls={s['calls']} skipped_ratelimit={s['skipped_ratelimit']} "
                f"skipped_breaker={s['skipped_breaker']} skipped_cooldown={s['skipped_cooldown']} "
                f"failures={fails}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _clean_content(content: str) -> str:
    text = _THINK_RE.sub("", content).strip()
    return strip_json_fences(text)


def _retry_after(resp) -> Optional[float]:
    try:
        return float(resp.headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


class NimClient:
    def __init__(
        self,
        api_key: str,
        guard: Optional[NimGuard] = None,
        post: Optional[Callable] = None,
    ) -> None:
        self._headers = {"Authorization": "Bearer " + api_key}
        self.guard = guard or GUARD
        self._post = post or self._http_post
        self._http: Optional[httpx.Client] = None

    def _http_post(self, url: str, headers: dict, body: dict, timeout: float):
        if self._http is None:
            self._http = httpx.Client()
        return self._http.post(
            url, headers=headers, json=body,
            timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)),
        )

    def complete(
        self,
        spec: NimSpec,
        system: str,
        user: str,
        schema_keys: Optional[set] = None,
    ) -> NimResult:
        """One non-streaming chat completion. Never raises, never retries."""
        model = spec.model
        skipped = self.guard.acquire(model)
        if skipped:
            log.warning("[nim] %s skipped (%s) -- falling through", model, skipped)
            return NimResult(model, skipped=skipped)

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "stream": False,
            **_MODEL_SETTINGS.get(model, {"max_tokens": 1024}),
        }
        t0 = time.monotonic()

        def fail(kind: str, detail: str = "") -> NimResult:
            ms = int((time.monotonic() - t0) * 1000)
            log.warning("[nim] %s failed (%s%s) after %d ms -- falling through",
                        model, kind, f": {detail}" if detail else "", ms)
            return NimResult(model, latency_ms=ms, error=kind)

        try:
            resp = self._post(NIM_URL, self._headers, body, spec.timeout)
        except httpx.TimeoutException:
            self.guard.record_failure(model, "timeout")
            return fail("timeout")
        except httpx.TransportError as exc:
            self.guard.record_failure(model, "connection")
            return fail("connection", type(exc).__name__)
        except Exception as exc:  # anything unexpected must not reach the trading path
            self.guard.record_other_failure(model, "error")
            return fail("error", type(exc).__name__)

        status = resp.status_code
        if status == 429:
            wait = self.guard.record_rate_limited(model, _retry_after(resp))
            log.warning("[nim] HTTP 429 -- NIM unavailable for %.0f s", wait)
            return fail("429")
        if status >= 500:
            self.guard.record_failure(model, "5xx")
            return fail("5xx", f"HTTP {status}")
        if status != 200:
            self.guard.record_other_failure(model, "http_4xx")
            return fail("http_4xx", f"HTTP {status}")

        # HTTP 200: the model is reachable whatever the content turns out to be.
        self.guard.record_success(model)
        latency_ms = int((time.monotonic() - t0) * 1000)
        try:
            data = resp.json()
            content = data["choices"][0]["message"].get("content")
            usage = data.get("usage") or {}
        except Exception:
            self.guard.record_other_failure(model, "parse")
            return fail("parse", "bad response envelope")
        in_tok = int(usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or 0)

        # Only `content` counts. Reasoning models put their working in
        # `reasoning` / `reasoning_content`; that is never an answer.
        if not isinstance(content, str) or not content.strip():
            self.guard.record_other_failure(model, "empty")
            return fail("empty")
        raw = _clean_content(content)
        parsed = parse_json_strict(raw) if "<think>" not in raw.lower() else None
        if parsed is None:
            self.guard.record_other_failure(model, "parse")
            res = fail("parse")
            res.raw, res.in_tok, res.out_tok = raw[:2000], in_tok, out_tok
            return res
        missing = (schema_keys or set()) - parsed.keys()
        if missing:
            self.guard.record_other_failure(model, "schema")
            res = fail("schema", f"missing keys {sorted(missing)}")
            res.raw, res.in_tok, res.out_tok = raw[:2000], in_tok, out_tok
            return res
        return NimResult(model, raw=raw, in_tok=in_tok, out_tok=out_tok, latency_ms=latency_ms)


# ---------------------------------------------------------------------------
# Process-wide singletons
# ---------------------------------------------------------------------------

GUARD = NimGuard()

_client_lock = threading.Lock()
_client: Optional[NimClient] = None
_resolved = False


def _load_key() -> str:
    key = os.environ.get(KEY_ENV, "").strip()
    if key:
        return key
    try:
        return KEY_FILE.read_text().strip()
    except OSError:
        return ""


def get_client() -> Optional[NimClient]:
    """The shared client, or None when NIM is disabled (no key)."""
    global _client, _resolved
    with _client_lock:
        if not _resolved:
            key = _load_key()
            _client = NimClient(key) if key else None
            _resolved = True
            if _client is None:
                log.info("[nim] disabled (no %s and no %s)", KEY_ENV, KEY_FILE)
            else:
                log.info("[nim] enabled")
        return _client


def set_client(client: Optional[NimClient]) -> None:
    """Install (or, with None, disable) the shared client. For tests."""
    global _client, _resolved
    with _client_lock:
        _client, _resolved = client, True


def reset() -> None:
    """Forget the resolved client and clear guard state. For tests."""
    global _client, _resolved
    with _client_lock:
        _client, _resolved = None, False
    GUARD.clear()


def stats() -> dict:
    return GUARD.stats()


def summary_and_reset() -> str:
    """One-line usage summary since the last call (for the daily log), then zero it."""
    line = GUARD.summary_line()
    GUARD.reset_stats()
    return line


def run_chain(
    specs: Sequence[NimSpec],
    system: str,
    user: str,
    schema_keys: Optional[set] = None,
) -> Optional[NimResult]:
    """Try each NIM model in order; the first valid answer wins, else None.

    Falls through to the next model on a failure or an open breaker. When the
    limiter or the 429 cooldown says NIM as a whole is unavailable (or NIM just
    answered 429) the remaining NIM models are skipped too, since they would be
    refused for the same reason.
    """
    client = get_client()
    if client is None:
        return None
    for spec in specs:
        res = client.complete(spec, system, user, schema_keys)
        if res.ok:
            return res
        if res.skipped in ("ratelimit", "cooldown") or res.error == "429":
            break
    return None
