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
  * per-model circuit breaker -- 2 consecutive timeouts / connection errors /
    HTTP 502, 504, 529 open the breaker for 15 min, after which exactly one probe
    call is let through. Anything else (a 200 of any content, 429, other 4xx,
    other 5xx) is a response, so it resets the consecutive count to 0 and never
    trips it: schema/parse failures are per-prompt, not a sign the model is down.
  * hard wall-clock deadline -- every request is cut off at its spec.timeout in
    total (DNS + connect + headers + body), not per phase.
  * shadow admission -- the structural shadow call is only let in while the
    window has spare room, NIM is not cooling down and its model's breaker is
    closed, and it never feeds the breaker, so it cannot change which provider a
    real decision ends up using.

There are no retries anywhere: every failure falls through at once. Without a
key (env NVIDIA_NIM_API_KEY, else ~/.nvidia-nim-key) NIM is disabled and the
callers behave exactly as they did before NIM existed. The key is only ever put
in the Authorization header; nothing here logs it.
"""
from __future__ import annotations

import concurrent.futures
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

# The shadow call is admitted only while the window holds at most this many
# requests, i.e. it always leaves >= NIM_MAX_REQUESTS - SHADOW_MAX_USED slots free
# for real decisions.
SHADOW_MAX_USED = 20

# Failure kinds that count towards the breaker: the model/endpoint itself is
# unreachable or unhealthy, as opposed to this one prompt getting a bad answer.
# For HTTP only these gateway/overload statuses qualify; any other 5xx is a
# response like any other and does not (see record_other_failure).
BREAKER_KINDS = frozenset({"timeout", "5xx", "connection"})
BREAKER_STATUSES = frozenset({502, 504, 529})

# Hard cap on concurrent HTTP workers. A request that blows its deadline is
# abandoned (its worker ends on its own via the httpx per-phase timeouts), so the
# pool is deliberately small: a hung endpoint can pin at most this many threads.
HTTP_WORKERS = 4

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
    skipped: Optional[str] = None      # "cooldown" | "ratelimit" | "breaker" | "shadow"; no request was made

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

    def acquire(self, model: str, shadow: bool = False) -> Optional[str]:
        """None if a request may go out now (a limiter slot is consumed), else
        the reason it may not: "cooldown", "breaker" or "ratelimit".

        shadow=True is the observation-only caller. It is admitted only if it
        cannot crowd out or alter a real decision: NIM not cooling down, the
        model's breaker fully closed (never a half-open probe), and no more than
        SHADOW_MAX_USED requests already in the window. Otherwise it is refused
        with "shadow" and counted as shadow_skipped."""
        with self._lock:
            now = self._clock()
            if shadow:
                while self._stamps and now - self._stamps[0] >= self._window:
                    self._stamps.popleft()
                b = self._breakers.get(model)
                if (now < self._unavailable_until
                        or (b is not None and b.open_until is not None)
                        or len(self._stamps) > SHADOW_MAX_USED):
                    self._counts["shadow_skipped"] += 1
                    return "shadow"
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
        """The model answered 200 (even if the answer later failed validation):
        it is reachable, so the consecutive-failure run is broken and a probe
        closes the breaker."""
        with self._lock:
            b = self._breakers.get(model)
            if b is None:
                return
            if b.open_until is not None:
                log.info("[nim] breaker CLOSED for %s (probe answered)", model)
            b.fails, b.open_until, b.probing = 0, None, False

    def record_failure(self, model: str, kind: str, shadow: bool = False) -> None:
        """A timeout / connection error / HTTP 502, 504, 529: the only outcomes
        that count towards the breaker. A shadow failure is only tallied."""
        with self._lock:
            self._failures[kind] += 1
            if shadow:
                return
            b = self._breakers.setdefault(model, _Breaker())
            b.fails += 1
            was_probe = b.probing
            b.probing = False
            if b.fails >= BREAKER_THRESHOLD or was_probe:
                b.open_until = self._clock() + BREAKER_OPEN_SEC
                log.info("[nim] breaker OPEN for %s: %d consecutive failures (last=%s), "
                         "skipping for %.0f min", model, b.fails, kind, BREAKER_OPEN_SEC / 60)

    def record_other_failure(self, model: str, kind: str, shadow: bool = False,
                             got_response: bool = True) -> None:
        """A failure that does not trip the breaker (parse / schema / 4xx / other
        5xx / unexpected error): counted. If the server did respond, that breaks
        the consecutive run, so the count goes back to 0; a breaker that is
        already open stays open (a probe that gets such an answer is retried on
        the next call). A shadow failure is only tallied."""
        with self._lock:
            self._failures[kind] += 1
            if shadow:
                return
            b = self._breakers.get(model)
            if b is not None:
                b.probing = False           # let the next call be the probe instead
                if got_response:
                    b.fails = 0

    def record_rate_limited(self, model: str, retry_after: Optional[float],
                            shadow: bool = False) -> float:
        """HTTP 429: NIM is unavailable for every caller for a while. That is a
        true signal about the key, so a shadow 429 sets the cooldown too; it says
        nothing about model health, so only a real call touches the breaker (it
        breaks the consecutive run)."""
        with self._lock:
            self._failures["429"] += 1
            wait = min(max(COOLDOWN_429_SEC, retry_after or 0.0), COOLDOWN_429_MAX_SEC)
            self._unavailable_until = max(self._unavailable_until, self._clock() + wait)
            b = self._breakers.get(model)
            if b is not None and not shadow:
                b.probing = False
                b.fails = 0
            return wait

    # -- stats -------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            return {
                "calls": self._counts["calls"],
                "skipped_ratelimit": self._counts["skipped_ratelimit"],
                "skipped_breaker": self._counts["skipped_breaker"],
                "skipped_cooldown": self._counts["skipped_cooldown"],
                "shadow_skipped": self._counts["shadow_skipped"],
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
                f"shadow_skipped={s['shadow_skipped']} failures={fails}")


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


def _tokens(value) -> int:
    """A token count from untrusted JSON: anything that is not a plain
    non-negative number is 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    try:
        return max(int(value), 0)
    except (OverflowError, ValueError):     # inf / nan
        return 0


_pool_lock = threading.Lock()
_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None


def _http_pool() -> concurrent.futures.ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=HTTP_WORKERS, thread_name_prefix="nim-http")
        return _pool


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
        self._http_lock = threading.Lock()

    def _send(self, url: str, headers: dict, body: dict, timeout: float, deadline: float):
        """Worker side of _http_post: stream the response, re-checking the
        wall-clock deadline on every chunk (a slow trickle never trips httpx's
        per-read timeout). The httpx per-phase timeouts stay in force so an
        abandoned worker ends on its own."""
        with self._http_lock:
            if self._http is None:
                self._http = httpx.Client()
            http = self._http
        if time.monotonic() >= deadline:    # sat in the queue past its deadline
            raise httpx.ReadTimeout("deadline passed before the request started")
        with http.stream("POST", url, headers=headers, json=body,
                         timeout=httpx.Timeout(timeout, connect=min(5.0, timeout))) as resp:
            if time.monotonic() >= deadline:
                raise httpx.ReadTimeout("deadline exceeded waiting for headers")
            chunks = []
            for chunk in resp.iter_raw():
                chunks.append(chunk)
                if time.monotonic() >= deadline:
                    raise httpx.ReadTimeout("deadline exceeded reading body")
            # Rebuild an in-memory response from the raw bytes; the constructor
            # applies Content-Encoding decoding, which iter_raw() skipped.
            return httpx.Response(resp.status_code, headers=resp.headers,
                                  content=b"".join(chunks))

    def _http_post(self, url: str, headers: dict, body: dict, timeout: float):
        """POST with a hard total cap of `timeout` seconds (DNS, connect,
        headers and body together). The request runs on a small shared pool so
        the caller can stop waiting at the deadline even while a connect or DNS
        lookup is still hanging; that worker is abandoned and ends on its own."""
        deadline = time.monotonic() + timeout
        future = _http_pool().submit(self._send, url, headers, body, timeout, deadline)
        try:
            return future.result(timeout=max(deadline - time.monotonic(), 0.0))
        except concurrent.futures.TimeoutError:
            future.cancel()                 # frees the slot if it never started
            raise httpx.ReadTimeout(f"hard deadline of {timeout:g} s exceeded") from None

    def complete(
        self,
        spec: NimSpec,
        system: str,
        user: str,
        schema_keys: Optional[set] = None,
        shadow: bool = False,
    ) -> NimResult:
        """One non-streaming chat completion. Never raises, never retries.

        shadow=True marks the observation-only structural call: it is admitted
        only under spare capacity (see NimGuard.acquire), and its outcomes never
        feed the per-model breaker (a 429 still sets the global cooldown)."""
        t0 = time.monotonic()
        try:
            return self._complete(spec, system, user, schema_keys, shadow)
        except Exception as exc:  # last line of defence: nothing may reach the trading path
            try:
                self.guard.record_other_failure(spec.model, "error", shadow=shadow, got_response=False)
            except Exception:
                pass
            ms = int((time.monotonic() - t0) * 1000)
            log.warning("[nim] %s failed (error: %s) after %d ms -- falling through",
                        spec.model, type(exc).__name__, ms)
            return NimResult(spec.model, latency_ms=ms, error="error")

    def _complete(self, spec: NimSpec, system: str, user: str,
                  schema_keys: Optional[set], shadow: bool) -> NimResult:
        model = spec.model
        skipped = self.guard.acquire(model, shadow=True) if shadow else self.guard.acquire(model)
        if skipped:
            # A refused shadow is routine capacity management, not news.
            log.log(logging.DEBUG if shadow else logging.WARNING,
                    "[nim] %s skipped (%s) -- falling through", model, skipped)
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
            self.guard.record_failure(model, "timeout", shadow=shadow)
            return fail("timeout")
        except httpx.TransportError as exc:
            self.guard.record_failure(model, "connection", shadow=shadow)
            return fail("connection", type(exc).__name__)
        except Exception as exc:  # anything unexpected must not reach the trading path
            self.guard.record_other_failure(model, "error", shadow=shadow, got_response=False)
            return fail("error", type(exc).__name__)

        status = resp.status_code
        if status == 429:
            wait = self.guard.record_rate_limited(model, _retry_after(resp), shadow=shadow)
            log.warning("[nim] HTTP 429 -- NIM unavailable for %.0f s", wait)
            return fail("429")
        if status in BREAKER_STATUSES:
            self.guard.record_failure(model, "5xx", shadow=shadow)
            return fail("5xx", f"HTTP {status}")
        if status >= 500:                   # some other 5xx: a response, not an outage
            self.guard.record_other_failure(model, "http_5xx", shadow=shadow)
            return fail("http_5xx", f"HTTP {status}")
        if status != 200:
            self.guard.record_other_failure(model, "http_4xx", shadow=shadow)
            return fail("http_4xx", f"HTTP {status}")

        # HTTP 200: the model is reachable whatever the content turns out to be.
        if not shadow:
            self.guard.record_success(model)
        latency_ms = int((time.monotonic() - t0) * 1000)
        try:
            data = resp.json()
            content = data["choices"][0]["message"].get("content")
            usage = data.get("usage")
            if not isinstance(usage, dict):
                usage = {}
            in_tok = _tokens(usage.get("prompt_tokens"))
            out_tok = _tokens(usage.get("completion_tokens"))
        except Exception:
            self.guard.record_other_failure(model, "parse", shadow=shadow)
            return fail("parse", "bad response envelope")

        # Only `content` counts. Reasoning models put their working in
        # `reasoning` / `reasoning_content`; that is never an answer.
        if not isinstance(content, str) or not content.strip():
            self.guard.record_other_failure(model, "empty", shadow=shadow)
            return fail("empty")
        raw = _clean_content(content)
        parsed = parse_json_strict(raw) if "<think>" not in raw.lower() else None
        if parsed is None:
            self.guard.record_other_failure(model, "parse", shadow=shadow)
            res = fail("parse")
            res.raw, res.in_tok, res.out_tok = raw[:2000], in_tok, out_tok
            return res
        if not isinstance(parsed, dict):    # valid JSON but not an object
            self.guard.record_other_failure(model, "schema", shadow=shadow)
            res = fail("schema", f"JSON is {type(parsed).__name__}, not an object")
            res.raw, res.in_tok, res.out_tok = raw[:2000], in_tok, out_tok
            return res
        missing = (schema_keys or set()) - parsed.keys()
        if missing:
            self.guard.record_other_failure(model, "schema", shadow=shadow)
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
    """The API key, or "" when there is none or it is unusable. Never raises."""
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        try:
            key = KEY_FILE.read_text().strip()
        except FileNotFoundError:
            return ""
        except (OSError, ValueError) as exc:    # unreadable, or not valid text (UnicodeDecodeError)
            log.warning("[nim] cannot read %s (%s) -- NIM disabled", KEY_FILE, type(exc).__name__)
            return ""
    if key and not (key.isascii() and key.isprintable() and " " not in key):
        # Would make every request fail while building the Authorization header.
        log.warning("[nim] API key has non-ASCII / control / space characters -- NIM disabled")
        return ""
    return key


def get_client() -> Optional[NimClient]:
    """The shared client, or None when NIM is disabled (no usable key). Resolved
    once per process: a missing or unreadable key is never retried per call."""
    global _client, _resolved
    with _client_lock:
        if not _resolved:
            try:
                key = _load_key()
            except Exception as exc:            # _load_key is meant not to raise; never retry if it does
                log.warning("[nim] key loading failed (%s) -- NIM disabled", type(exc).__name__)
                key = ""
            _client = NimClient(key) if key else None
            _resolved = True
            if _client is None:
                log.info("[nim] disabled (no usable %s / %s)", KEY_ENV, KEY_FILE)
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
    refused for the same reason. Never raises.
    """
    try:
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
    except Exception as exc:   # NIM must never take down a leg: the caller falls through to its other providers
        log.warning("[nim] chain failed (%s: %s) -- falling through to the next provider",
                    type(exc).__name__, str(exc)[:120])
        return None
