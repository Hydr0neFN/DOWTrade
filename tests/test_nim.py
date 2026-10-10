"""
tests/test_nim.py
=================
NVIDIA NIM provider: the shared guard (limiter, 429 cooldown, circuit breaker),
response handling, the per-leg provider chains, and the structural shadow call.

All HTTP is faked and the clock is injected; nothing here touches the network or
sleeps. tests/conftest.py disables NIM for every other test.
"""
from __future__ import annotations

import json
import logging
import threading
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
        client, post, clock = install({GPT: [status(503), ok(RISK_GOOD)]})
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
        assert s["calls"] == 3 and s["skipped_cooldown"] == 1
        assert s["failures"] == {"timeout": 1, "429": 1}
        assert "calls=3" in client.guard.summary_line() and "timeout=1" in client.guard.summary_line()
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
        c._last_model = "nim:z-ai/glm-5.3-flash"
        assert c._estimated_cost_usd(4000) > 0.0  # the pre-call budget check stays conservative


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

    def test_shadow_shares_the_rate_limit(self):
        client, post, clock = install({GPT: [ok(STRUCT_SHADOW)]})
        for _ in range(29):
            assert client.guard.acquire(GPT) is None
        c = self._client()
        self._run(c)
        assert len(post.calls) == 1            # 30th slot
        self._run(c)
        assert len(post.calls) == 1            # window full: second shadow skipped
        assert client.guard.stats()["skipped_ratelimit"] == 1
