"""
tests/test_nim.py
=================
NVIDIA NIM provider: the shared guard (limiter, 429 cooldown, circuit breaker),
response handling, the per-leg provider chains, and the structural shadow call.

All HTTP is faked and the clock is injected; nothing here touches the network or
sleeps. tests/conftest.py disables NIM for every other test.
"""
from __future__ import annotations

import gzip
import http.server
import json
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.llm import nim
from src.llm.base import CostTracker
from src.llm.nim import NimClient, NimGuard, NimSpec

KEY = "nvapi-SECRET-test-key-0123456789"

RISK_GOOD = {"approved": True, "violations": [], "override_action": None, "reasoning": "ok"}
EXEC_GOOD = {"action": "open_long", "stop_price": 39700.0,
             "trailing_stop_atr_multiple": 2.0, "reasoning": "break"}
STRUCT_GOOD = {"trend": "up", "last_confirmed_hh": 40000.0, "last_confirmed_hl": 39500.0,
               "pattern_intact": True, "structural_signal": "new_hh_break",
               "confidence_0_to_1": 0.8, "reasoning": "uptrend"}
STRUCT_SHADOW = dict(STRUCT_GOOD, trend="down", structural_signal="pattern_broken")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, sec: float) -> None:
        self.t += sec


def ok(payload, *, usage=(100, 20)) -> httpx.Response:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
    })


def status(code: int, **headers) -> httpx.Response:
    return httpx.Response(code, headers=headers, text="err")


class FakePost:
    """post(url, headers, body, timeout) with a per-model script.

    plans[model] is a list of httpx.Response / Exception items consumed in order;
    the last item repeats. A model with no plan fails like a refused connection.
    """

    def __init__(self, plans=None) -> None:
        self.plans = {m: list(v) for m, v in (plans or {}).items()}
        self.calls: list = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "body": body, "timeout": timeout})
        plan = self.plans.get(body["model"])
        if not plan:
            raise httpx.ConnectError("no plan")
        item = plan.pop(0) if len(plan) > 1 else plan[0]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def models(self) -> list:
        return [c["body"]["model"] for c in self.calls]


def install(plans=None, clock=None):
    """Install a NIM client with a faked transport and an injectable clock."""
    clock = clock or FakeClock()
    post = FakePost(plans)
    client = NimClient(KEY, guard=NimGuard(clock=clock), post=post)
    nim.set_client(client)
    return client, post, clock


def fill_window(client, n=30):
    for _ in range(n):
        assert client.guard.acquire("openai/gpt-oss-20b") is None


GPT = nim.GPT_OSS_20B
GLM = nim.GLM_FLASH
NEMO = nim.NEMOTRON_SUPER
SPEC = NimSpec(GPT, 20.0)
RISK_KEYS = {"approved", "violations", "reasoning"}


class RecordingDB:
    def __init__(self) -> None:
        self.rows: list = []

    def insert_llm_call(self, row) -> int:
        self.rows.append(dict(row))
        return len(self.rows)


# ---------------------------------------------------------------------------
# Limiter
# ---------------------------------------------------------------------------

class TestLimiter:
    def test_31st_call_in_60s_is_skipped_without_a_request(self):
        client, post, clock = install({GPT: [ok(RISK_GOOD)]})
        for i in range(30):
            assert client.complete(SPEC, "s", "u", RISK_KEYS).ok, i
            clock.advance(1.0)  # 30 calls spread over 30 s
        assert len(post.calls) == 30

        res = client.complete(SPEC, "s", "u", RISK_KEYS)
        assert res.skipped == "ratelimit" and not res.ok
        assert len(post.calls) == 30, "a limited call must not reach the network"
        assert client.guard.stats()["skipped_ratelimit"] == 1

    def test_window_slides(self):
        client, post, clock = install({GPT: [ok(RISK_GOOD)]})
        fill_window(client)
        assert client.guard.acquire(GPT) == "ratelimit"
        clock.advance(59.9)
        assert client.guard.acquire(GPT) == "ratelimit"
        clock.advance(0.2)  # the first 30 stamps are now >= 60 s old
        assert client.guard.acquire(GPT) is None

    def test_limit_is_shared_across_models(self):
        client, post, clock = install({GPT: [ok(RISK_GOOD)], GLM: [ok(RISK_GOOD)]})
        for _ in range(15):
            assert client.complete(NimSpec(GPT, 20), "s", "u").ok
            assert client.complete(NimSpec(GLM, 25), "s", "u").ok
        assert client.complete(NimSpec(GPT, 20), "s", "u").skipped == "ratelimit"
        assert client.complete(NimSpec(GLM, 25), "s", "u").skipped == "ratelimit"

    def test_limiter_is_thread_safe(self):
        guard = NimGuard(clock=FakeClock())
        granted = []

        def worker():
            for _ in range(20):
                if guard.acquire("m") is None:
                    granted.append(1)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(granted) == nim.NIM_MAX_REQUESTS

    def test_run_chain_skips_second_nim_model_when_limited(self):
        client, post, clock = install({GPT: [ok(RISK_GOOD)], GLM: [ok(RISK_GOOD)]})
        fill_window(client)
        res = nim.run_chain((NimSpec(GPT, 20), NimSpec(GLM, 25)), "s", "u", RISK_KEYS)
        assert res is None
        assert post.calls == []
        # Only the first model is even asked: the 2nd would be refused for the same reason.
        assert client.guard.stats()["skipped_ratelimit"] == 1


# ---------------------------------------------------------------------------
# 429 cooldown
# ---------------------------------------------------------------------------

class TestRateLimitCooldown:
    def test_429_blocks_all_models_for_30s(self):
        client, post, clock = install({GPT: [status(429), ok(RISK_GOOD)], GLM: [ok(RISK_GOOD)]})
        res = client.complete(SPEC, "s", "u")
        assert res.error == "429"

        # Every model is blocked, not just the one that was limited.
        assert client.complete(NimSpec(GLM, 25), "s", "u").skipped == "cooldown"
        clock.advance(29.9)
        assert client.complete(SPEC, "s", "u").skipped == "cooldown"
        assert len(post.calls) == 1

        clock.advance(0.2)
        assert client.complete(SPEC, "s", "u").ok

    @pytest.mark.parametrize("retry_after,blocked_until", [
        ("5", 30.0),        # smaller than the floor: 30 s wins
        ("50", 50.0),       # larger: honoured
        ("9999", 120.0),    # capped
        ("soon", 30.0),     # unparseable: floor
    ])
    def test_retry_after(self, retry_after, blocked_until):
        client, post, clock = install({GPT: [status(429, **{"Retry-After": retry_after}), ok(RISK_GOOD)]})
        client.complete(SPEC, "s", "u")
        clock.advance(blocked_until - 0.5)
        assert client.guard.acquire(GPT) == "cooldown"
        clock.advance(1.0)
        assert client.guard.acquire(GPT) is None

    def test_429_does_not_count_towards_breaker(self):
        client, post, clock = install({GPT: [status(429)]})
        for _ in range(5):
            client.complete(SPEC, "s", "u")
            clock.advance(121)
        assert client.guard.acquire(GPT) is None  # breaker never opened

    def test_run_chain_does_not_try_second_model_after_429(self):
        client, post, clock = install({GPT: [status(429)], GLM: [ok(RISK_GOOD)]})
        assert nim.run_chain((NimSpec(GPT, 20), NimSpec(GLM, 25)), "s", "u", RISK_KEYS) is None
        assert post.models == [GPT]


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class TestBreaker:
    def test_opens_after_two_timeouts_and_half_opens_after_15_min(self, caplog):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t"), httpx.ReadTimeout("t"), ok(RISK_GOOD)]})
        with caplog.at_level(logging.INFO, logger="src.llm.nim"):
            assert client.complete(SPEC, "s", "u").error == "timeout"
            assert client.complete(SPEC, "s", "u").error == "timeout"
            assert any("breaker OPEN" in r.getMessage() and GPT in r.getMessage()
                       for r in caplog.records if r.levelno == logging.INFO)

            res = client.complete(SPEC, "s", "u")
            assert res.skipped == "breaker"
            assert len(post.calls) == 2

            clock.advance(15 * 60 - 1)
            assert client.complete(SPEC, "s", "u").skipped == "breaker"
            assert len(post.calls) == 2

            clock.advance(2)  # half-open: one probe is allowed
            assert client.complete(SPEC, "s", "u").ok
            assert len(post.calls) == 3
            assert any("breaker CLOSED" in r.getMessage() for r in caplog.records)

        # closed again: traffic flows
        assert client.complete(SPEC, "s", "u").ok

    def test_only_one_probe_while_half_open(self):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t")]})
        client.complete(SPEC, "s", "u")
        client.complete(SPEC, "s", "u")
        clock.advance(15 * 60 + 1)
        assert client.guard.acquire(GPT) is None       # the probe
        assert client.guard.acquire(GPT) == "breaker"  # nobody else until it reports back

    def test_failed_probe_reopens_for_another_15_min(self):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t")]})
        client.complete(SPEC, "s", "u")
        client.complete(SPEC, "s", "u")
        clock.advance(15 * 60 + 1)
        assert client.complete(SPEC, "s", "u").error == "timeout"   # probe fails
        assert len(post.calls) == 3
        assert client.complete(SPEC, "s", "u").skipped == "breaker"
        clock.advance(15 * 60 - 1)
        assert client.complete(SPEC, "s", "u").skipped == "breaker"
        clock.advance(2)
        assert client.complete(SPEC, "s", "u").error == "timeout"   # next probe
        assert len(post.calls) == 4

    @pytest.mark.parametrize("failure,kind", [
        (status(502), "5xx"),
        (status(504), "5xx"),
        (status(529), "5xx"),
        (httpx.ConnectError("refused"), "connection"),
        (httpx.RemoteProtocolError("closed"), "connection"),
        (httpx.ConnectTimeout("t"), "timeout"),
    ])
    def test_failure_kinds_that_trip(self, failure, kind):
        client, post, clock = install({GPT: [failure]})
        assert client.complete(SPEC, "s", "u").error == kind
        assert client.complete(SPEC, "s", "u").error == kind
        assert client.complete(SPEC, "s", "u").skipped == "breaker"

    def test_breaker_is_per_model(self):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t")], GLM: [ok(RISK_GOOD)]})
        client.complete(SPEC, "s", "u")
        client.complete(SPEC, "s", "u")
        assert client.complete(SPEC, "s", "u").skipped == "breaker"
        assert client.complete(NimSpec(GLM, 25), "s", "u").ok

    def test_run_chain_moves_to_second_model_when_first_breaker_is_open(self):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t")], GLM: [ok(RISK_GOOD)]})
        client.complete(SPEC, "s", "u")
        client.complete(SPEC, "s", "u")
        res = nim.run_chain((NimSpec(GPT, 20), NimSpec(GLM, 25)), "s", "u", RISK_KEYS)
        assert res is not None and res.model == GLM
        assert post.models == [GPT, GPT, GLM]

    def test_non_consecutive_failures_do_not_trip(self):
        client, post, clock = install(
            {GPT: [httpx.ReadTimeout("t"), ok(RISK_GOOD), httpx.ReadTimeout("t"), ok(RISK_GOOD)]})
        assert client.complete(SPEC, "s", "u").error == "timeout"
        assert client.complete(SPEC, "s", "u").ok
        assert client.complete(SPEC, "s", "u").error == "timeout"
        assert client.complete(SPEC, "s", "u").ok

    @pytest.mark.parametrize("answer,kind", [
        (ok({"approved": True}), "schema"),             # valid JSON, missing keys
        (ok("this is not json"), "parse"),
        (ok(""), "empty"),
        (ok("   "), "empty"),
        (status(400), "http_4xx"),
        (status(404), "http_4xx"),
    ])
    def test_schema_and_parse_failures_do_not_trip(self, answer, kind, caplog):
        client, post, clock = install({GPT: [answer]})
        with caplog.at_level(logging.WARNING, logger="src.llm.nim"):
            for _ in range(6):
                res = client.complete(SPEC, "s", "u", RISK_KEYS)
                assert res.error == kind and res.skipped is None
        assert len(post.calls) == 6, "the breaker must stay closed"
        assert any(kind in r.getMessage() for r in caplog.records), "failures are still logged"
        assert client.guard.stats()["failures"][kind] == 6

    def test_probe_answered_with_bad_schema_still_closes_breaker(self):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t"), httpx.ReadTimeout("t"), ok({"approved": True})]})
        client.complete(SPEC, "s", "u")
        client.complete(SPEC, "s", "u")
        clock.advance(15 * 60 + 1)
        assert client.complete(SPEC, "s", "u", RISK_KEYS).error == "schema"  # reachable, bad answer
        assert client.guard.acquire(GPT) is None                              # so it is closed

    def test_429_on_probe_leaves_next_call_as_probe(self):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t"), httpx.ReadTimeout("t"), status(429), ok(RISK_GOOD)]})
        client.complete(SPEC, "s", "u")
        client.complete(SPEC, "s", "u")
        clock.advance(15 * 60 + 1)
        assert client.complete(SPEC, "s", "u").error == "429"
        clock.advance(31)
        assert client.complete(SPEC, "s", "u").ok


# ---------------------------------------------------------------------------
# Request / response handling
# ---------------------------------------------------------------------------

class TestRequestAndResponse:
    def test_request_settings_per_model(self):
        client, post, clock = install({GPT: [ok(RISK_GOOD)], GLM: [ok(RISK_GOOD)], NEMO: [ok(RISK_GOOD)]})
        client.complete(NimSpec(GPT, 20), "SYS", "USR")
        client.complete(NimSpec(GLM, 25), "SYS", "USR")
        client.complete(NimSpec(NEMO, 8), "SYS", "USR")
        gpt, glm, nemo = (c["body"] for c in post.calls)
        for body in (gpt, glm, nemo):
            assert body["temperature"] == 0
            assert body["response_format"] == {"type": "json_object"}
            assert body["stream"] is False
            assert body["messages"] == [{"role": "system", "content": "SYS"},
                                        {"role": "user", "content": "USR"}]
        assert (gpt["max_tokens"], gpt["reasoning_effort"]) == (1024, "low")
        assert glm["max_tokens"] == 2048 and "reasoning_effort" not in glm
        assert nemo["max_tokens"] == 1024
        assert nemo["chat_template_kwargs"] == {"thinking": False, "enable_thinking": False}
        assert [c["timeout"] for c in post.calls] == [20, 25, 8]
        assert all(c["url"] == "https://integrate.api.nvidia.com/v1/chat/completions" for c in post.calls)
        assert post.calls[0]["headers"]["Authorization"] == f"Bearer {KEY}"

    def test_success_returns_cleaned_content_and_tokens(self):
        client, post, clock = install({GPT: [ok(RISK_GOOD, usage=(111, 22))]})
        res = client.complete(SPEC, "s", "u", RISK_KEYS)
        assert res.ok and json.loads(res.raw) == RISK_GOOD
        assert (res.in_tok, res.out_tok) == (111, 22)

    def test_strips_fences_and_think_blocks(self):
        wrapped = "<think>maybe {\"approved\": false}</think>\n```json\n" + json.dumps(RISK_GOOD) + "\n```"
        client, post, clock = install({GPT: [ok(wrapped)]})
        res = client.complete(SPEC, "s", "u", RISK_KEYS)
        assert res.ok and json.loads(res.raw) == RISK_GOOD

    def test_unclosed_think_is_a_parse_failure(self):
        client, post, clock = install({GPT: [ok("<think>still going {\"approved\": true, \"violations\": [], \"reasoning\": \"x\"}")]})
        assert client.complete(SPEC, "s", "u", RISK_KEYS).error == "parse"

    def test_reasoning_field_is_never_used(self):
        resp = httpx.Response(200, json={"choices": [{"message": {
            "content": None, "reasoning": json.dumps(RISK_GOOD), "reasoning_content": json.dumps(RISK_GOOD)}}]})
        client, post, clock = install({GPT: [resp]})
        res = client.complete(SPEC, "s", "u", RISK_KEYS)
        assert res.error == "empty" and res.raw == ""

    def test_malformed_envelope_is_a_parse_failure(self):
        client, post, clock = install({GPT: [httpx.Response(200, json={"choices": []})]})
        assert client.complete(SPEC, "s", "u").error == "parse"
        client, post, clock = install({GPT: [httpx.Response(200, text="<html>")]})
        assert client.complete(SPEC, "s", "u").error == "parse"

    def test_unexpected_exception_never_propagates(self):
        client, post, clock = install({GPT: [RuntimeError("boom")]})
        res = client.complete(SPEC, "s", "u")
        assert res.error == "error"
        for _ in range(5):  # and it does not trip the breaker
            client.complete(SPEC, "s", "u")
        assert client.guard.acquire(GPT) is None

    def test_no_retries(self):
        client, post, clock = install({GPT: [status(502), ok(RISK_GOOD)]})
        assert client.complete(SPEC, "s", "u").error == "5xx"
        assert len(post.calls) == 1

    def test_api_key_never_logged(self, caplog):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t"), status(500), status(429),
                                              ok("not json"), ok({"approved": True}), RuntimeError("x")]})
        with caplog.at_level(logging.DEBUG):
            for _ in range(6):
                client.complete(SPEC, "s", "u", RISK_KEYS)
                clock.advance(1000)
        assert caplog.records
        text = "\n".join(r.getMessage() + str(r.exc_info) for r in caplog.records)
        assert KEY not in text and "nvapi" not in text

    def test_fall_through_warning_names_model_and_reason(self, caplog):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t")]})
        with caplog.at_level(logging.WARNING, logger="src.llm.nim"):
            client.complete(SPEC, "s", "u")
        warn = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warn) == 1 and GPT in warn[0].getMessage() and "timeout" in warn[0].getMessage()


class TestStats:
    def test_stats_and_summary(self):
        client, post, clock = install({GPT: [ok(RISK_GOOD), httpx.ReadTimeout("t"), status(429)]})
        client.complete(SPEC, "s", "u")
        client.complete(SPEC, "s", "u")
        client.complete(SPEC, "s", "u")
        client.complete(SPEC, "s", "u")  # cooldown
        s = client.guard.stats()
        assert s["calls"] == 3 and s["skipped_cooldown"] == 1 and s["shadow_skipped"] == 0
        assert s["failures"] == {"timeout": 1, "429": 1}
        assert "calls=3" in client.guard.summary_line() and "timeout=1" in client.guard.summary_line()
        assert "shadow_skipped=0" in client.guard.summary_line()
        nim.GUARD.reset_stats()  # module-level helper zeroes the shared guard
        assert nim.summary_and_reset().startswith("NIM stats: calls=0")


# ---------------------------------------------------------------------------
# Enabled / disabled
# ---------------------------------------------------------------------------

class TestKeyResolution:
    def test_disabled_without_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv(nim.KEY_ENV, raising=False)
        monkeypatch.setattr(nim, "KEY_FILE", tmp_path / "missing")
        nim.reset()
        assert nim.get_client() is None
        assert nim.run_chain((SPEC,), "s", "u", RISK_KEYS) is None

    def test_env_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv(nim.KEY_ENV, KEY)
        monkeypatch.setattr(nim, "KEY_FILE", tmp_path / "missing")
        nim.reset()
        assert nim.get_client() is not None

    def test_file_key_and_env_wins(self, monkeypatch, tmp_path):
        f = tmp_path / "key"
        f.write_text("file-key\n")
        monkeypatch.delenv(nim.KEY_ENV, raising=False)
        monkeypatch.setattr(nim, "KEY_FILE", f)
        nim.reset()
        assert nim.get_client()._headers["Authorization"] == "Bearer file-key"
        monkeypatch.setenv(nim.KEY_ENV, "env-key")
        nim.reset()
        assert nim.get_client()._headers["Authorization"] == "Bearer env-key"

    def test_blank_key_file_means_disabled(self, monkeypatch, tmp_path):
        f = tmp_path / "key"
        f.write_text("  \n")
        monkeypatch.delenv(nim.KEY_ENV, raising=False)
        monkeypatch.setattr(nim, "KEY_FILE", f)
        nim.reset()
        assert nim.get_client() is None

    def test_legs_behave_as_before_when_disabled(self):
        """No key: no NIM request, the chain is exactly the pre-NIM one."""
        from src.llm.deepseek_risk import CF_MODELS, DeepSeekRisk
        nim.set_client(None)
        client = DeepSeekRisk()
        client._cf_client = MagicMock()
        client._cf_client.chat_completion = MagicMock(return_value=_cf_resp(RISK_GOOD))
        result = client.evaluate_raw("s", "u", bar_ts=1)
        assert result.model_used == CF_MODELS[0]
        assert nim.stats()["calls"] == 0


# ---------------------------------------------------------------------------
# Leg chains
# ---------------------------------------------------------------------------

def _cf_resp(data) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=json.dumps(data)))]
    resp.usage = MagicMock(prompt_tokens=60, completion_tokens=30)
    return resp


def _gemini_resp(data) -> MagicMock:
    resp = MagicMock()
    resp.text = json.dumps(data)
    resp.usage_metadata.prompt_token_count = 80
    resp.usage_metadata.candidates_token_count = 40
    return resp


class TestRiskChain:
    def _client(self, tracker=None):
        from src.llm.deepseek_risk import DeepSeekRisk
        c = DeepSeekRisk(tracker=tracker)
        c._cf_client = MagicMock()
        c._db = RecordingDB()
        return c

    def test_first_nim_model_answers(self):
        _, post, _ = install({GPT: [ok(RISK_GOOD)]})
        tracker = CostTracker()
        c = self._client(tracker)
        r = c.evaluate_raw("s", "u", bar_ts=7)
        assert r.error is None and r.parsed["approved"] is True
        assert r.model_used == "nim:openai/gpt-oss-20b"
        assert r.cost_usd == 0.0 and tracker.total_usd == 0.0
        assert post.models == [GPT] and post.calls[0]["timeout"] == 20
        c._cf_client.chat_completion.assert_not_called()
        assert c._db.rows[0]["model"] == "nim:openai/gpt-oss-20b"
        assert c._db.rows[0]["cost_usd"] == 0.0

    def test_nim_failure_falls_to_second_nim_then_cloudflare_in_order(self):
        _, post, _ = install({GPT: [httpx.ReadTimeout("t")], GLM: [status(502)]})
        c = self._client()
        cf_models = []

        def cf(*a, **kw):
            cf_models.append(kw["model"])
            if len(cf_models) == 1:
                raise Exception("503")
            return _cf_resp(RISK_GOOD)

        c._cf_client.chat_completion = MagicMock(side_effect=cf)
        r = c.evaluate_raw("s", "u", bar_ts=7)
        assert post.models == [GPT, GLM]
        assert [x["timeout"] for x in post.calls] == [20, 25]
        assert cf_models == ["@cf/meta/llama-4-scout-17b-16e-instruct",
                             "@cf/mistralai/mistral-small-3.1-24b-instruct"]
        assert r.model_used == "@cf/mistralai/mistral-small-3.1-24b-instruct"
        assert r.cost_usd == 0.0

    def test_schema_failure_on_first_falls_to_glm(self):
        _, post, _ = install({GPT: [ok({"approved": True})], GLM: [ok(RISK_GOOD)]})
        c = self._client()
        r = c.evaluate_raw("s", "u", bar_ts=7)
        assert post.models == [GPT, GLM]
        assert r.model_used == "nim:z-ai/glm-5.3-flash" and r.error is None
        c._cf_client.chat_completion.assert_not_called()

    def test_limiter_full_goes_straight_to_cloudflare(self):
        client, post, _ = install({GPT: [ok(RISK_GOOD)], GLM: [ok(RISK_GOOD)]})
        fill_window(client)
        c = self._client()
        c._cf_client.chat_completion = MagicMock(return_value=_cf_resp(RISK_GOOD))
        r = c.evaluate_raw("s", "u", bar_ts=7)
        assert post.calls == []
        assert r.model_used == "@cf/meta/llama-4-scout-17b-16e-instruct"

    def test_everything_down_returns_safe_default(self):
        install({GPT: [httpx.ReadTimeout("t")], GLM: [httpx.ReadTimeout("t")]})
        c = self._client()
        c._cf_client = None
        r = c.evaluate_raw("s", "u", bar_ts=7)
        assert r.used_fallback and "llm_unavailable" in r.parsed["violations"]

    def test_runner_wiring_has_no_hf_key(self):
        """DeepSeekRisk is built from (tracker, db) only; there is no HF client."""
        import inspect
        from src.llm.deepseek_risk import DeepSeekRisk
        assert list(inspect.signature(DeepSeekRisk.__init__).parameters)[1:] == [
            "tracker", "db", "prompt_path"]


class TestExecutionChain:
    def _client(self, tracker=None, cf=True):
        from src.llm.gemini_execution import GeminiExecution
        c = GeminiExecution(api_key="test-key-redacted", tracker=tracker)
        c._cf_client = MagicMock() if cf else None
        c._db = RecordingDB()
        return c

    def test_glm_answers_first_and_gemini_is_not_called(self):
        _, post, _ = install({GLM: [ok(EXEC_GOOD)]})
        tracker = CostTracker()
        c = self._client(tracker)
        with patch.object(c._client.models, "generate_content") as gen:
            r = c.evaluate_raw("s", "u", bar_ts=7)
            gen.assert_not_called()
        assert r.model_used == "nim:z-ai/glm-5.3-flash" and r.parsed["action"] == "open_long"
        assert post.calls[0]["timeout"] == 40
        assert r.cost_usd == 0.0 and tracker.total_usd == 0.0
        assert c._db.rows[0]["model"] == "nim:z-ai/glm-5.3-flash"

    def test_glm_then_nemotron_then_gemini_then_cloudflare(self):
        _, post, _ = install({GLM: [httpx.ReadTimeout("t")], NEMO: [status(503)]})
        c = self._client()
        order = []

        def gemini(*a, **kw):
            order.append(("gemini", kw["model"]))
            raise Exception("429 quota")

        c._cf_client.chat_completion = MagicMock(side_effect=lambda *a, **kw: (order.append(("cf", kw["model"])), _cf_resp(EXEC_GOOD))[1])
        with patch.object(c._client.models, "generate_content", side_effect=gemini):
            r = c.evaluate_raw("s", "u", bar_ts=7)
        assert post.models == [GLM, NEMO]
        assert [x["timeout"] for x in post.calls] == [40, 8]
        from src.llm.gemini_execution import GEMINI_MODELS
        assert order == [("gemini", m) for m in GEMINI_MODELS] + [("cf", "@cf/mistralai/mistral-small-3.1-24b-instruct")]
        assert r.model_used == "@cf/mistralai/mistral-small-3.1-24b-instruct"
        assert r.cost_usd == 0.0

    def test_nemotron_answers_when_glm_fails_schema(self):
        _, post, _ = install({GLM: [ok({"action": "hold"})], NEMO: [ok(EXEC_GOOD)]})
        c = self._client()
        with patch.object(c._client.models, "generate_content") as gen:
            r = c.evaluate_raw("s", "u", bar_ts=7)
            gen.assert_not_called()
        assert r.model_used == "nim:nvidia/nemotron-3-super-120b-a12b"

    def test_gemini_answers_when_nim_is_limited(self):
        client, post, _ = install({GLM: [ok(EXEC_GOOD)]})
        fill_window(client)
        c = self._client()
        with patch.object(c._client.models, "generate_content", return_value=_gemini_resp(EXEC_GOOD)) as gen:
            r = c.evaluate_raw("s", "u", bar_ts=7)
        assert post.calls == [] and gen.call_count == 1
        assert r.model_used == "gemini-3.6-flash" and r.cost_usd > 0

    def test_all_failed_without_cloudflare_is_safe_default(self):
        install({GLM: [httpx.ReadTimeout("t")], NEMO: [httpx.ReadTimeout("t")]})
        c = self._client(cf=False)
        with patch.object(c._client.models, "generate_content", side_effect=Exception("boom")):
            r = c.evaluate_raw("s", "u", bar_ts=7)
        assert r.used_fallback and r.parsed["action"] == "hold"

    def test_cloudflare_failure_keeps_original_error(self):
        install({})
        c = self._client()
        c._cf_client.chat_completion = MagicMock(side_effect=Exception("cf down"))
        with patch.object(c._client.models, "generate_content", side_effect=Exception("gemini down")):
            r = c.evaluate_raw("s", "u", bar_ts=7)
        assert r.used_fallback and "gemini down" in r.error

    def test_cost_only_free_for_free_providers(self):
        c = self._client()
        for label in ("nim:z-ai/glm-5.3-flash", "@cf/mistralai/mistral-small-3.1-24b-instruct"):
            c._last_model = label
            assert c._actual_cost_usd(1000, 1000) == 0.0
        c._last_model = "gemini-3.6-flash"
        assert c._actual_cost_usd(1000, 1000) > 0.0
        # The pre-call budget check must not hold back a chain that starts free.
        for label in ("nim:z-ai/glm-5.3-flash", "gemini-3.6-flash"):
            c._last_model = label
            assert c._estimated_cost_usd(4000) == 0.0
        assert c._estimated_cost_usd(10_000_000) == 0.0

    def test_exhausted_budget_does_not_block_the_free_chain(self):
        """cap 0 and nothing spent: the NIM call still goes out (estimate is 0)."""
        _, post, _ = install({GLM: [ok(EXEC_GOOD)]})
        c = self._client(CostTracker(cap_usd=0.0))
        r = c.evaluate_raw("s", "u", bar_ts=7)
        assert r.error is None and r.model_used == "nim:" + GLM and r.cost_usd == 0.0
        assert len(post.calls) == 1

    def test_budget_already_overspent_still_blocks(self):
        _, post, _ = install({GLM: [ok(EXEC_GOOD)]})
        tracker = CostTracker(cap_usd=0.0)
        tracker.record(0.01)
        r = self._client(tracker).evaluate_raw("s", "u", bar_ts=7)
        assert r.error == "budget_exceeded" and post.calls == []

    def test_risk_leg_estimate_is_already_zero(self):
        from src.llm.deepseek_risk import DeepSeekRisk
        assert DeepSeekRisk()._estimated_cost_usd(10_000_000) == 0.0


# ---------------------------------------------------------------------------
# Structural shadow
# ---------------------------------------------------------------------------

def _snapshot():
    return SimpleNamespace(bars=[], swings=[], sma200=39000.0, atr14=50.0, current_price=40000.0)


class TestStructuralShadow:
    SHADOW_MODEL = "nim-shadow:openai/gpt-oss-20b"

    def _client(self, tracker=None):
        from src.llm.haiku_structural import HaikuStructural
        c = HaikuStructural(api_key="test-key-redacted", tracker=tracker)
        c._db = RecordingDB()
        return c

    def _run(self, c, haiku_payload=STRUCT_GOOD, order=None):
        def create(*a, **kw):
            if order is not None:
                order.append("production")
            msg = MagicMock()
            msg.content = [MagicMock(type="text", text=json.dumps(haiku_payload))]
            msg.usage.input_tokens, msg.usage.output_tokens, msg.usage.cache_read_input_tokens = 100, 50, 0
            return msg

        with patch("src.llm.haiku_structural.sdk_available", return_value=False), \
                patch.object(c._client.messages, "create", side_effect=create):
            return c.evaluate(_snapshot(), bar_ts=4242)

    def test_shadow_row_logged_and_decision_unaffected(self):
        _, post, _ = install({GPT: [ok(STRUCT_SHADOW)]})
        tracker = CostTracker()
        c = self._client(tracker)
        order = []
        orig = post.__call__
        client = nim.get_client()
        client._post = lambda *a: (order.append("shadow"), orig(*a))[1]
        r = self._run(c, order=order)

        # production answer, untouched by the (disagreeing) shadow
        assert r.parsed == STRUCT_GOOD and r.model_used == "claude-haiku-5-5" and r.error is None
        assert order == ["production", "shadow"], "shadow runs after the production call"
        prod, shadow = c._db.rows
        assert prod["model"] == "claude-haiku-5-5"
        assert shadow["model"] == self.SHADOW_MODEL
        assert shadow["bar_ts"] == prod["bar_ts"] == "4242"
        assert shadow["prompt_hash"] == prod["prompt_hash"]
        assert json.loads(shadow["parsed_json"]) == STRUCT_SHADOW
        assert shadow["raw_response"] and shadow["error"] is None
        assert shadow["cost_usd"] == 0.0 and shadow["latency_ms"] >= 0
        assert post.calls[0]["timeout"] == 15
        # shadow spend is zero: only the production cost reached the tracker
        assert tracker.total_usd == pytest.approx(prod["cost_usd"])

    def test_shadow_failure_is_logged_with_error_and_does_not_affect_result(self):
        _, post, _ = install({GPT: [httpx.ReadTimeout("t")]})
        c = self._client()
        r = self._run(c)
        assert r.parsed == STRUCT_GOOD
        shadow = c._db.rows[1]
        assert shadow["model"] == self.SHADOW_MODEL
        assert shadow["error"] == "timeout" and shadow["parsed_json"] is None

    def test_shadow_schema_failure_keeps_raw_for_comparison(self):
        install({GPT: [ok({"trend": "up"})]})
        c = self._client()
        r = self._run(c)
        assert r.parsed == STRUCT_GOOD
        shadow = c._db.rows[1]
        assert shadow["error"] == "schema" and "trend" in shadow["raw_response"]
        assert shadow["parsed_json"] is None

    def test_shadow_never_raises_into_the_trading_path(self):
        client, post, _ = install({GPT: [ok(STRUCT_SHADOW)]})
        c = self._client()
        with patch.object(client, "complete", side_effect=RuntimeError("boom")):
            r = self._run(c)
        assert r.parsed == STRUCT_GOOD
        assert len(c._db.rows) == 1  # production row only

        # DB failure on the shadow insert
        class ExplodingShadowDB(RecordingDB):
            def insert_llm_call(self, row):
                if row["model"].startswith("nim-shadow"):
                    raise RuntimeError("db locked")
                return super().insert_llm_call(row)

        c = self._client()
        c._db = ExplodingShadowDB()
        r = self._run(c)
        assert r.parsed == STRUCT_GOOD

    @pytest.mark.parametrize("how", ["limiter", "cooldown", "breaker", "disabled"])
    def test_shadow_skipped_when_nim_unavailable(self, how):
        client, post, clock = install({GPT: [ok(STRUCT_SHADOW)]})
        if how == "limiter":
            fill_window(client)
        elif how == "cooldown":
            client.guard.record_rate_limited(GPT, None)
        elif how == "breaker":
            client.guard.acquire(GPT)
            client.guard.record_failure(GPT, "timeout")
            client.guard.acquire(GPT)
            client.guard.record_failure(GPT, "timeout")
        else:
            nim.set_client(None)
        c = self._client()
        r = self._run(c)
        assert r.parsed == STRUCT_GOOD
        assert post.calls == []
        assert [row["model"] for row in c._db.rows] == ["claude-haiku-5-5"]
        if how != "disabled":
            assert client.guard.stats()["shadow_skipped"] == 1

    def test_production_failure_still_returns_safe_default_and_shadow_still_logs(self):
        """The shadow is independent of the production outcome."""
        _, post, _ = install({GPT: [ok(STRUCT_SHADOW)]})
        c = self._client()
        with patch("src.llm.haiku_structural.sdk_available", return_value=False), \
                patch.object(c._client.messages, "create", side_effect=Exception("anthropic down")):
            r = c.evaluate(_snapshot(), bar_ts=4242)
        assert r.used_fallback and r.parsed["trend"] == "sideways"
        assert [row["model"] for row in c._db.rows] == [self.SHADOW_MODEL]

    def test_no_db_means_no_shadow_call(self):
        _, post, _ = install({GPT: [ok(STRUCT_SHADOW)]})
        c = self._client()
        c._db = None
        r = self._run(c)
        assert r.parsed == STRUCT_GOOD and post.calls == []

    def test_shadow_shares_the_rate_limit_but_leaves_ten_slots_free(self):
        client, post, clock = install({GPT: [ok(STRUCT_SHADOW)]})
        for _ in range(19):
            assert client.guard.acquire(GPT) is None
        c = self._client()
        self._run(c)
        assert len(post.calls) == 1            # 19 used -> admitted (20th slot)
        self._run(c)
        assert len(post.calls) == 2            # exactly 20 used = 10 free -> still admitted
        self._run(c)
        assert len(post.calls) == 2            # 21 used: the shadow stands down
        s = client.guard.stats()
        assert s["shadow_skipped"] == 1 and s["skipped_ratelimit"] == 0
        for _ in range(9):                     # and real calls still get the remaining 9 slots
            assert client.guard.acquire(GPT) is None
        assert client.guard.acquire(GPT) == "ratelimit"


# ---------------------------------------------------------------------------
# Fix 1: hard wall-clock deadline
# ---------------------------------------------------------------------------

class _Server:
    """Local HTTP server with a scripted misbehaviour. `delay` is the pause
    between single bytes (headers trickle or body trickle); 0 = reply at once."""

    BODY = json.dumps({"choices": [{"message": {"content": json.dumps(RISK_GOOD)}}],
                       "usage": {"prompt_tokens": 7, "completion_tokens": 3}}).encode()

    def __init__(self, mode: str = "ok", delay: float = 0.0) -> None:
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    outer.respond(self)
                except OSError:
                    pass                      # client gave up: expected

        self.mode, self.delay = mode, delay
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def respond(self, h) -> None:
        body = self.BODY
        if self.mode == "gzip":
            body = gzip.compress(body)
        head = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + (b"Content-Encoding: gzip\r\n" if self.mode == "gzip" else b"")
                + b"Content-Length: %d\r\n\r\n" % len(body))
        if self.mode == "trickle_headers":
            for b in b"HTTP/1.1 200 OK\r\nX-Pad: " + b"a" * 40:
                h.wfile.write(bytes([b])); h.wfile.flush(); time.sleep(self.delay)
            h.wfile.write(b"\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        elif self.mode == "trickle_body":
            h.wfile.write(head); h.wfile.flush()
            for b in body:
                h.wfile.write(bytes([b])); h.wfile.flush(); time.sleep(self.delay)
        else:
            h.wfile.write(head + body); h.wfile.flush()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    made = []

    def make(mode="ok", delay=0.0):
        srv = _Server(mode, delay)
        made.append(srv)
        return srv

    yield make
    for srv in made:
        srv.close()


def _real_client(monkeypatch, srv) -> NimClient:
    """A client on the real _http_post, pointed at the local server."""
    monkeypatch.setattr(nim, "NIM_URL", srv.url)
    client = NimClient(KEY, guard=NimGuard())
    nim.set_client(client)
    return client


class TestHardDeadline:
    @pytest.mark.parametrize("mode", ["trickle_body", "trickle_headers"])
    def test_slow_trickle_is_cut_off_at_the_deadline(self, monkeypatch, server, mode):
        """Every byte arrives well inside httpx's per-read timeout, so only a real
        total deadline stops this (it ran 76 s against a 2 s spec before)."""
        client = _real_client(monkeypatch, server(mode, delay=0.2))
        t0 = time.monotonic()
        res = client.complete(NimSpec(GPT, 0.5), "s", "u")
        wall = time.monotonic() - t0
        assert res.error == "timeout" and not res.ok
        assert 0.45 <= wall <= 0.5 + 0.5, wall
        assert client.guard.stats()["failures"] == {"timeout": 1}

    def test_deadline_counts_as_timeout_for_the_breaker(self, monkeypatch, server):
        client = _real_client(monkeypatch, server("trickle_body", delay=0.2))
        spec = NimSpec(GPT, 0.25)
        assert client.complete(spec, "s", "u").error == "timeout"
        assert client.complete(spec, "s", "u").error == "timeout"
        assert client.complete(spec, "s", "u").skipped == "breaker"

    def test_hung_connect_or_dns_is_bounded_by_the_caller_side_deadline(self, monkeypatch):
        """If the worker never comes back (DNS / connect hanging) the caller still
        returns at the deadline."""
        client = NimClient(KEY, guard=NimGuard())
        release = threading.Event()
        monkeypatch.setattr(client, "_send", lambda *a: release.wait(5))
        try:
            t0 = time.monotonic()
            res = client.complete(NimSpec(GPT, 0.4), "s", "u")
            wall = time.monotonic() - t0
        finally:
            release.set()
        assert res.error == "timeout" and 0.35 <= wall <= 0.9, wall

    def test_pool_is_capped_and_queued_requests_still_hit_their_deadline(self, monkeypatch):
        client = NimClient(KEY, guard=NimGuard())
        release = threading.Event()
        monkeypatch.setattr(client, "_send", lambda *a: release.wait(5))
        results, n = [], nim.HTTP_WORKERS + 3
        try:
            def one():
                t0 = time.monotonic()
                results.append((client.complete(NimSpec(GPT, 0.4), "s", "u").error,
                                time.monotonic() - t0))
            threads = [threading.Thread(target=one) for _ in range(n)]
            for t in threads: t.start()
            for t in threads: t.join(3)
            workers = [t for t in threading.enumerate() if t.name.startswith("nim-http")]
        finally:
            release.set()
        assert len(results) == n and all(e == "timeout" and w <= 0.9 for e, w in results), results
        assert len(workers) <= nim.HTTP_WORKERS

    def test_fast_response_and_gzip_still_work(self, monkeypatch, server):
        for mode in ("ok", "gzip"):
            client = _real_client(monkeypatch, server(mode))
            res = client.complete(NimSpec(GPT, 2.0), "s", "u", RISK_KEYS)
            assert res.ok and (res.in_tok, res.out_tok) == (7, 3), (mode, res)
            assert json.loads(res.raw)["approved"] is True

    def test_connection_refused_is_a_connection_failure(self, monkeypatch):
        monkeypatch.setattr(nim, "NIM_URL", "http://127.0.0.1:1/")
        client = NimClient(KEY, guard=NimGuard())
        assert client.complete(NimSpec(GPT, 2.0), "s", "u").error == "connection"

    def test_shadow_keeps_its_15s_deadline(self):
        from src.llm.haiku_structural import SHADOW_SPEC
        assert SHADOW_SPEC.timeout == 15.0


# ---------------------------------------------------------------------------
# Fix 2: complete() / run_chain() never raise
# ---------------------------------------------------------------------------

class TestNeverRaises:
    @pytest.mark.parametrize("usage,expected", [
        ([1, 2], (0, 0)),
        ("lots", (0, 0)),
        (None, (0, 0)),
        ({"prompt_tokens": "n/a"}, (0, 0)),
        ({"prompt_tokens": None, "completion_tokens": [3]}, (0, 0)),
        ({"prompt_tokens": {"x": 1}, "completion_tokens": True}, (0, 0)),
        ({"prompt_tokens": float("nan"), "completion_tokens": float("inf")}, (0, 0)),
        ({"prompt_tokens": -5, "completion_tokens": 2.9}, (0, 2)),
        ({"prompt_tokens": 12, "completion_tokens": 4}, (12, 4)),
    ])
    def test_malformed_usage_is_tolerated(self, usage, expected):
        payload = {"choices": [{"message": {"content": json.dumps(RISK_GOOD)}}], "usage": usage}
        resp = httpx.Response(200, content=json.dumps(payload).encode())   # allows NaN / Infinity
        client, post, clock = install({GPT: [resp]})
        res = client.complete(SPEC, "s", "u", RISK_KEYS)
        assert res.ok, res
        assert (res.in_tok, res.out_tok) == expected

    @pytest.mark.parametrize("content", ["[1, 2, 3]", "42", "\"text\"", "null", "true"])
    def test_non_object_json_is_a_failure_not_an_exception(self, content):
        client, post, clock = install({GPT: [ok(content)]})
        res = client.complete(SPEC, "s", "u", RISK_KEYS)
        assert not res.ok and res.error in ("parse", "schema")

    def test_parsed_non_object_is_a_schema_failure(self, monkeypatch):
        monkeypatch.setattr(nim, "parse_json_strict", lambda raw: [1, 2])
        client, post, clock = install({GPT: [ok(RISK_GOOD)]})
        res = client.complete(SPEC, "s", "u", RISK_KEYS)
        assert res.error == "schema" and not res.ok
        assert client.guard.stats()["failures"] == {"schema": 1}
        monkeypatch.setattr(nim, "parse_json_strict", lambda raw: "x")
        assert client.complete(SPEC, "s", "u").error == "schema"   # even with no required keys

    def test_anything_inside_complete_is_contained(self):
        client, post, clock = install({GPT: [ok(RISK_GOOD)]})
        with patch.object(client.guard, "acquire", side_effect=RuntimeError("guard exploded")):
            res = client.complete(SPEC, "s", "u")
        assert res.error == "error" and not res.ok
        resp = MagicMock(status_code=200)
        resp.json.side_effect = RuntimeError("boom")
        client._post = lambda *a: resp
        assert client.complete(SPEC, "s", "u").error == "parse"

    def test_run_chain_never_raises_and_logs_one_warning(self, caplog):
        client, post, clock = install({GPT: [ok(RISK_GOOD)]})
        with patch.object(client, "complete", side_effect=RuntimeError("kaboom")), \
                caplog.at_level(logging.WARNING, logger="src.llm.nim"):
            assert nim.run_chain((SPEC, NimSpec(GLM, 25)), "s", "u", RISK_KEYS) is None
        warn = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warn) == 1 and "kaboom" in warn[0].getMessage()
        with patch.object(nim, "get_client", side_effect=ValueError("no")):
            assert nim.run_chain((SPEC,), "s", "u") is None

    def test_risk_leg_falls_through_to_cloudflare_when_nim_blows_up(self):
        from src.llm.deepseek_risk import CF_MODELS, DeepSeekRisk
        client, post, clock = install({GPT: [ok(RISK_GOOD)]})
        c = DeepSeekRisk()
        c._cf_client = MagicMock()
        c._cf_client.chat_completion = MagicMock(return_value=_cf_resp(RISK_GOOD))
        with patch.object(client, "complete", side_effect=RuntimeError("kaboom")):
            r = c.evaluate_raw("s", "u", bar_ts=1)
        assert r.error is None and r.model_used == CF_MODELS[0]

    def test_execution_leg_falls_through_to_gemini_with_malformed_usage(self):
        from src.llm.gemini_execution import GeminiExecution
        resp = httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(EXEC_GOOD)}}], "usage": [1, 2]})
        install({GLM: [resp]})
        c = GeminiExecution(api_key="test-key-redacted", tracker=CostTracker())
        c._cf_client = None
        r = c.evaluate_raw("s", "u", bar_ts=1)
        assert r.error is None and r.model_used == "nim:" + GLM and r.input_tokens == 0


# ---------------------------------------------------------------------------
# Fix 3: key file loading
# ---------------------------------------------------------------------------

class TestKeyFileFailures:
    def _reset(self, monkeypatch, path):
        monkeypatch.delenv(nim.KEY_ENV, raising=False)
        monkeypatch.setattr(nim, "KEY_FILE", path)
        nim.reset()

    def test_non_utf8_key_file_disables_nim_once(self, monkeypatch, tmp_path, caplog):
        f = tmp_path / "key"
        f.write_bytes(b"\xff\xfe\x00bad")
        self._reset(monkeypatch, f)
        with caplog.at_level(logging.INFO, logger="src.llm.nim"):
            assert nim.get_client() is None
            for _ in range(5):
                assert nim.get_client() is None
                assert nim.run_chain((SPEC,), "s", "u") is None
        msgs = [r.getMessage() for r in caplog.records]
        assert sum("cannot read" in m for m in msgs) == 1, msgs
        assert sum("disabled" in m for m in msgs) == 2     # the read warning + the one "disabled" line

    def test_file_is_never_re_read_after_a_failure(self, monkeypatch, tmp_path):
        reads = []

        class BadFile:
            def read_text(self):
                reads.append(1)
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

            def __str__(self):
                return "/bad/key"

        monkeypatch.delenv(nim.KEY_ENV, raising=False)
        monkeypatch.setattr(nim, "KEY_FILE", BadFile())
        nim.reset()
        for _ in range(10):
            assert nim.get_client() is None
        assert len(reads) == 1

    def test_unreadable_key_file_is_disabled_not_fatal(self, monkeypatch, tmp_path):
        self._reset(monkeypatch, tmp_path)               # a directory: IsADirectoryError
        assert nim.get_client() is None

    @pytest.mark.parametrize("key", ["nvapi-k\u00e9y", "nvapi key", "nvapi\x00key"])
    def test_key_that_could_not_be_sent_in_a_header_disables_nim(self, monkeypatch, tmp_path, key):
        f = tmp_path / "key"
        f.write_text(key, encoding="utf-8")
        self._reset(monkeypatch, f)
        assert nim.get_client() is None

    def test_legs_still_work_with_a_bad_key_file(self, monkeypatch, tmp_path):
        from src.llm.deepseek_risk import CF_MODELS, DeepSeekRisk
        f = tmp_path / "key"
        f.write_bytes(b"\xff\xfe")
        self._reset(monkeypatch, f)
        c = DeepSeekRisk()
        c._cf_client = MagicMock()
        c._cf_client.chat_completion = MagicMock(return_value=_cf_resp(RISK_GOOD))
        r = c.evaluate_raw("s", "u", bar_ts=1)
        assert r.error is None and r.model_used == CF_MODELS[0]


# ---------------------------------------------------------------------------
# Fix 4: breaker semantics
# ---------------------------------------------------------------------------

class TestBreakerSemantics:
    @pytest.mark.parametrize("between", [
        status(429),
        status(400),
        status(401),
        status(404),
        status(500),
        status(503),
        ok(RISK_GOOD),
        ok({"approved": True}),          # 200, bad schema
        ok("not json"),
        ok(""),
    ], ids=["429", "400", "401", "404", "500", "503", "200-ok", "200-schema", "200-parse", "200-empty"])
    def test_any_response_between_two_timeouts_resets_the_count(self, between):
        client, post, clock = install(
            {GPT: [httpx.ReadTimeout("t"), between, httpx.ReadTimeout("t"), ok(RISK_GOOD)]})
        assert client.complete(SPEC, "s", "u", RISK_KEYS).error == "timeout"
        client.complete(SPEC, "s", "u", RISK_KEYS)
        clock.advance(60)                # clear a 429 cooldown
        assert client.complete(SPEC, "s", "u", RISK_KEYS).error == "timeout"
        assert client.guard.acquire(GPT) is None, "two timeouts were not consecutive"

    def test_two_timeouts_in_a_row_still_trip(self):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t")]})
        client.complete(SPEC, "s", "u")
        client.complete(SPEC, "s", "u")
        assert client.guard.acquire(GPT) == "breaker"

    @pytest.mark.parametrize("trip", [
        httpx.ReadTimeout("t"), httpx.ConnectTimeout("t"), httpx.ConnectError("x"),
        status(502), status(504), status(529),
    ], ids=["read-timeout", "connect-timeout", "connect-error", "502", "504", "529"])
    def test_mixed_tripping_failures_count_together(self, trip):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t"), trip]})
        client.complete(SPEC, "s", "u")
        client.complete(SPEC, "s", "u")
        assert client.guard.acquire(GPT) == "breaker"

    @pytest.mark.parametrize("code", [500, 501, 503, 505, 507, 598])
    def test_other_5xx_never_trips(self, code):
        client, post, clock = install({GPT: [status(code)]})
        for _ in range(6):
            res = client.complete(SPEC, "s", "u")
            assert res.error == "http_5xx" and res.skipped is None
        assert len(post.calls) == 6
        assert client.guard.stats()["failures"] == {"http_5xx": 6}

    def test_other_5xx_between_tripping_failures_resets(self):
        client, post, clock = install(
            {GPT: [status(502), status(503), status(504), status(529)]})
        for _ in range(3):
            client.complete(SPEC, "s", "u")
        assert client.guard.acquire(GPT) is None     # 502 / 503 / 504 are not consecutive
        client.guard.record_failure(GPT, "5xx")      # 504 + this one: now they are
        assert client.guard.acquire(GPT) == "breaker"

    def test_200_with_bad_schema_is_logged_as_schema_failure_with_no_breaker_effect(self, caplog):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t"), ok({"approved": True}),
                                              httpx.ReadTimeout("t")]})
        client.complete(SPEC, "s", "u", RISK_KEYS)
        with caplog.at_level(logging.WARNING, logger="src.llm.nim"):
            assert client.complete(SPEC, "s", "u", RISK_KEYS).error == "schema"
        assert any("schema" in r.getMessage() for r in caplog.records)
        assert client.guard.stats()["failures"] == {"timeout": 1, "schema": 1}
        client.complete(SPEC, "s", "u", RISK_KEYS)
        assert client.guard.acquire(GPT) is None

    def test_unexpected_local_error_is_not_a_response_and_does_not_reset(self):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t"), RuntimeError("x"),
                                              httpx.ReadTimeout("t")]})
        for _ in range(3):
            client.complete(SPEC, "s", "u")
        assert client.guard.acquire(GPT) == "breaker"

    def test_half_open_still_lets_exactly_one_probe_through(self):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t")]})
        client.complete(SPEC, "s", "u")
        client.complete(SPEC, "s", "u")
        clock.advance(15 * 60 - 1)
        assert client.guard.acquire(GPT) == "breaker"
        clock.advance(2)
        assert client.guard.acquire(GPT) is None
        for _ in range(3):
            assert client.guard.acquire(GPT) == "breaker"

    def test_non_tripping_answer_to_the_probe_keeps_it_half_open(self):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t"), httpx.ReadTimeout("t"),
                                              status(400), httpx.ReadTimeout("t")]})
        client.complete(SPEC, "s", "u")
        client.complete(SPEC, "s", "u")
        clock.advance(15 * 60 + 1)
        assert client.complete(SPEC, "s", "u").error == "http_4xx"    # probe answered, not "up"
        assert client.complete(SPEC, "s", "u").error == "timeout"     # so next call is the probe
        assert client.guard.acquire(GPT) == "breaker"                 # which failed: re-opened


# ---------------------------------------------------------------------------
# Fix 5: shadow isolation
# ---------------------------------------------------------------------------

class TestShadowIsolation:
    def test_admission_boundary_is_ten_free_slots(self):
        g = NimGuard(clock=FakeClock())
        for _ in range(20):
            assert g.acquire(GPT) is None
        assert g.acquire(GPT, shadow=True) is None           # 20 used -> 10 free -> admitted
        assert g.acquire(GPT, shadow=True) == "shadow"       # 21 used
        assert g.stats()["shadow_skipped"] == 1
        assert g.stats()["skipped_ratelimit"] == 0

    def test_window_slides_for_shadow_too(self):
        clock = FakeClock()
        g = NimGuard(clock=clock)
        for _ in range(25):
            g.acquire(GPT)
        assert g.acquire(GPT, shadow=True) == "shadow"
        clock.advance(61)
        assert g.acquire(GPT, shadow=True) is None

    def test_shadow_refused_in_cooldown_or_open_breaker_or_half_open(self):
        clock = FakeClock()
        g = NimGuard(clock=clock)
        g.record_rate_limited(GPT, None)
        assert g.acquire(GPT, shadow=True) == "shadow"
        clock.advance(31)
        assert g.acquire(GPT, shadow=True) is None
        for _ in range(2):
            g.acquire(GPT); g.record_failure(GPT, "timeout")
        assert g.acquire(GPT, shadow=True) == "shadow"        # open
        clock.advance(15 * 60 + 1)
        assert g.acquire(GPT, shadow=True) == "shadow"        # half-open: the probe is for real calls
        assert g.acquire(GPT) is None                         # ...and the probe is still available

    def test_shadow_failures_do_not_increment_the_breaker(self):
        client, post, clock = install({GPT: [httpx.ReadTimeout("t")]})
        for _ in range(5):
            assert client.complete(SPEC, "s", "u", shadow=True).error == "timeout"
        assert client.guard.acquire(GPT) is None              # real traffic untouched
        assert client.guard.stats()["failures"] == {"timeout": 5}
        client.guard.record_failure(GPT, "timeout")           # one real failure: still not tripped
        assert client.guard.acquire(GPT) is None

    @pytest.mark.parametrize("answer", [ok(STRUCT_SHADOW), ok({"trend": "up"}), status(400),
                                        status(503), ok("garbage")],
                             ids=["200", "schema", "400", "503", "parse"])
    def test_shadow_outcomes_do_not_reset_the_breaker(self, answer):
        client, post, clock = install({GPT: [answer]})
        client.guard.acquire(GPT)
        client.guard.record_failure(GPT, "timeout")           # real: 1 consecutive
        client.complete(SPEC, "s", "u", {"trend", "structural_signal"}, shadow=True)
        assert len(post.calls) == 1
        client.guard.acquire(GPT)
        client.guard.record_failure(GPT, "timeout")           # real: 2nd -> must trip
        assert client.guard.acquire(GPT) == "breaker"

    def test_shadow_success_does_not_close_an_open_breaker_or_use_the_probe(self):
        client, post, clock = install({GPT: [ok(STRUCT_SHADOW)]})
        for _ in range(2):
            client.guard.acquire(GPT); client.guard.record_failure(GPT, "timeout")
        clock.advance(15 * 60 + 1)
        assert client.complete(SPEC, "s", "u", shadow=True).skipped == "shadow"
        assert post.calls == []
        assert client.guard.acquire(GPT) is None              # the probe is still there

    def test_shadow_429_still_sets_the_global_cooldown(self):
        client, post, clock = install({GPT: [status(429, **{"retry-after": "50"})]})
        assert client.complete(SPEC, "s", "u", shadow=True).error == "429"
        assert client.guard.acquire(GLM) == "cooldown"
        clock.advance(49)
        assert client.guard.acquire(GLM) == "cooldown"
        clock.advance(2)
        assert client.guard.acquire(GLM) is None

    def test_shadow_429_does_not_touch_the_breaker_count(self):
        client, post, clock = install({GPT: [status(429)]})
        client.guard.acquire(GPT)
        client.guard.record_failure(GPT, "timeout")
        client.complete(SPEC, "s", "u", shadow=True)
        clock.advance(31)
        client.guard.acquire(GPT)
        client.guard.record_failure(GPT, "timeout")
        assert client.guard.acquire(GPT) == "breaker"

    def test_skipped_shadow_is_silent(self, caplog):
        client, post, clock = install({GPT: [ok(STRUCT_SHADOW)]})
        fill_window(client, 25)
        with caplog.at_level(logging.INFO, logger="src.llm.nim"):
            res = client.complete(SPEC, "s", "u", shadow=True)
        assert res.skipped == "shadow"
        assert not [r for r in caplog.records if r.levelno >= logging.INFO]

    def test_shadow_never_changes_the_real_chain_choice(self):
        """Window at 20: the shadow takes the 21st slot, times out, and is then
        refused. Real decisions keep the remaining 9 slots and gpt-oss stays the
        first choice (its breaker never saw the shadow's timeout)."""
        from src.llm.haiku_structural import HaikuStructural
        client, post, clock = install({GPT: [httpx.ReadTimeout("t")]})
        fill_window(client, 20)
        c = HaikuStructural(api_key="test-key-redacted")
        c._db = RecordingDB()
        with patch.object(c._client.messages, "create", side_effect=Exception("down")),                 patch("src.llm.haiku_structural.sdk_available", return_value=False):
            for _ in range(4):
                c.evaluate(_snapshot(), bar_ts=1)
        assert post.models == [GPT]                             # one shadow call, rest refused
        s = client.guard.stats()
        assert s["shadow_skipped"] == 3 and s["failures"] == {"timeout": 1}
        for _ in range(9):
            assert client.guard.acquire(GPT) is None            # breaker closed, slots left
        assert client.guard.acquire(GPT) == "ratelimit"


# ---------------------------------------------------------------------------
# Fix 6: timeouts on the non-NIM clients
# ---------------------------------------------------------------------------

class TestClientTimeouts:
    def test_cloudflare_client_gets_a_30s_timeout(self, monkeypatch):
        from src.llm import cf
        monkeypatch.setattr(cf, "CF_ACCOUNT_ID", "acct")
        monkeypatch.setattr(cf, "CF_API_TOKEN", "tok")
        with patch.object(cf, "InferenceClient") as ic:
            assert cf.make_cf_client("test") is ic.return_value
        assert ic.call_args.kwargs["timeout"] == 30.0

    def test_cloudflare_timeout_is_accepted_by_the_installed_signature(self):
        import inspect
        from huggingface_hub import InferenceClient
        assert "timeout" in inspect.signature(InferenceClient.__init__).parameters
        c = InferenceClient(base_url="http://x.invalid/v1", token="t", timeout=30.0)
        assert c.timeout == 30.0

    def test_gemini_client_gets_a_120s_http_timeout(self):
        from src.llm import gemini_execution as ge
        with patch.object(ge.genai, "Client") as client_cls:
            ge.GeminiExecution(api_key="test-key-redacted")
        opts = client_cls.call_args.kwargs["http_options"]
        assert opts.timeout == 120_000

    def test_gemini_http_options_reach_the_real_client(self):
        from src.llm.gemini_execution import GeminiExecution
        c = GeminiExecution(api_key="test-key-redacted")
        assert c._client._api_client._http_options.timeout == 120_000
