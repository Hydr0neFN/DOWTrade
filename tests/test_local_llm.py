"""
tests/test_local_llm.py
=======================
The self-hosted llama-server provider (NimSpec provider="local"): routing to its
own URL / key / guard, how NIM and local skip each other independently, its
5-minute breaker, and where it sits in the risk and execution chains.

All HTTP is faked and the clock is injected; nothing here touches the network.
tests/conftest.py disables both NIM and the local endpoint for every test.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx

from src.llm import nim
from src.llm.base import CostTracker
from src.llm.nim import NimClient, NimGuard, NimSpec

NIM_KEY = "nvapi-SECRET-test-key-0123456789"
LOCAL_KEY = "local-SECRET-test-key-9876543210"
URL = "http://100.119.161.52:8097/v1/chat/completions"

RISK_GOOD = {"approved": True, "violations": [], "override_action": None, "reasoning": "ok"}
EXEC_GOOD = {"action": "open_long", "stop_price": 39700.0,
             "trailing_stop_atr_multiple": 2.0, "reasoning": "break"}
RISK_KEYS = {"approved", "violations", "reasoning"}

GPT = nim.GPT_OSS_20B
GLM = nim.GLM_FLASH
NEMO = nim.NEMOTRON_SUPER
LOC = nim.LOCAL_GPT_OSS_20B
NIM_SPEC = NimSpec(GPT, 20.0)
LOCAL_SPEC = NimSpec(LOC, 25.0, provider="local")


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, sec: float) -> None:
        self.t += sec


def ok(payload) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps(payload),
                                 "reasoning_content": "thinking, never read"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    })


def status(code: int, **headers) -> httpx.Response:
    return httpx.Response(code, headers=headers, text="err")


class FakePost:
    """post(url, headers, body, timeout) with a per-model script (last item repeats)."""

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


def install(nim_plans=None, local_plans=None, clock=None):
    """Install a NIM client and a local client, each with its own fake transport,
    guard and the shared injectable clock. Returns (nim_post, local_post, clock)."""
    clock = clock or FakeClock()
    nim_post, local_post = FakePost(nim_plans), FakePost(local_plans)
    nim.set_client(NimClient(NIM_KEY, guard=NimGuard(clock=clock), post=nim_post))
    nim.set_local_client(NimClient(
        LOCAL_KEY, post=local_post, url=URL,
        guard=NimGuard(clock=clock, max_requests=nim.LOCAL_MAX_REQUESTS,
                       breaker_open_sec=nim.LOCAL_BREAKER_OPEN_SEC)))
    return nim_post, local_post, clock


class RecordingDB:
    def __init__(self) -> None:
        self.rows: list = []

    def insert_llm_call(self, row) -> int:
        self.rows.append(dict(row))
        return len(self.rows)


def _cf_resp(data) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=json.dumps(data)))]
    resp.usage = MagicMock(prompt_tokens=60, completion_tokens=30)
    return resp


# ---------------------------------------------------------------------------
# Routing, request shape, key resolution
# ---------------------------------------------------------------------------

class TestRouting:
    def test_local_spec_uses_local_url_key_and_settings(self):
        nim_post, local_post, _ = install(local_plans={LOC: [ok(RISK_GOOD)]})
        res = nim.run_chain((LOCAL_SPEC,), "sys", "usr", RISK_KEYS)
        assert res is not None and res.ok and res.label == "local:gpt-oss-20b"
        assert nim_post.calls == []
        call = local_post.calls[0]
        assert call["url"] == URL
        assert call["headers"] == {"Authorization": "Bearer " + LOCAL_KEY}
        assert call["timeout"] == 25.0
        body = call["body"]
        assert body["model"] == LOC and body["temperature"] == 0
        assert body["reasoning_effort"] == "low"
        assert body["chat_template_kwargs"] == {"reasoning_effort": "low"}
        assert body["max_tokens"] == 1024
        assert body["response_format"] == {"type": "json_object"}
        assert json.loads(res.raw) == RISK_GOOD     # reasoning_content is ignored

    def test_nim_spec_still_uses_nim_url_and_key(self):
        nim_post, local_post, _ = install(nim_plans={GPT: [ok(RISK_GOOD)]})
        res = nim.run_chain((NIM_SPEC,), "s", "u", RISK_KEYS)
        assert res.label == "nim:openai/gpt-oss-20b"
        assert nim_post.calls[0]["headers"] == {"Authorization": "Bearer " + NIM_KEY}
        assert local_post.calls == []

    def test_default_client_url_is_nim_url(self):
        assert NimClient(NIM_KEY)._url is None          # resolved to nim.NIM_URL per request
        assert nim.LOCAL_URL == URL

    def test_local_calls_do_not_use_the_nim_window(self):
        _, _, _ = install(local_plans={LOC: [ok(RISK_GOOD)]})
        for _ in range(5):
            assert nim.run_chain((LOCAL_SPEC,), "s", "u", RISK_KEYS) is not None
        assert nim.GUARD.stats()["calls"] == 0
        assert nim.LOCAL_GUARD.stats()["calls"] == 0     # the installed client has its own guard
        assert nim.get_client().guard.stats()["calls"] == 0
        assert nim.get_local_client().guard.stats()["calls"] == 5


class TestKeyResolution:
    def test_disabled_without_key_and_nim_still_works(self, monkeypatch, tmp_path):
        monkeypatch.delenv(nim.LOCAL_KEY_ENV, raising=False)
        monkeypatch.setattr(nim, "LOCAL_KEY_FILE", tmp_path / "missing")
        nim_post = FakePost({GPT: [httpx.ReadTimeout("t")], GLM: [ok(RISK_GOOD)]})
        nim.reset()
        nim.set_client(NimClient(NIM_KEY, guard=NimGuard(), post=nim_post))
        assert nim.get_local_client() is None
        assert nim.run_chain((LOCAL_SPEC,), "s", "u", RISK_KEYS) is None
        res = nim.run_chain((NIM_SPEC, LOCAL_SPEC, NimSpec(GLM, 25.0)), "s", "u", RISK_KEYS)
        assert res.label == "nim:z-ai/glm-5.3-flash"
        assert nim_post.models == [GPT, GLM]

    def test_env_key_then_file_key(self, monkeypatch, tmp_path):
        f = tmp_path / "key"
        f.write_text("file-key\n")
        monkeypatch.setattr(nim, "LOCAL_KEY_FILE", f)
        monkeypatch.delenv(nim.LOCAL_KEY_ENV, raising=False)
        nim.reset()
        c = nim.get_local_client()
        assert c._headers["Authorization"] == "Bearer file-key"
        assert c._url == nim.LOCAL_URL and c.guard is nim.LOCAL_GUARD
        monkeypatch.setenv(nim.LOCAL_KEY_ENV, "env-key")
        nim.reset()
        assert nim.get_local_client()._headers["Authorization"] == "Bearer env-key"

    def test_bad_key_file_means_disabled(self, monkeypatch, tmp_path):
        f = tmp_path / "key"
        f.write_bytes(b"\xff\xfe\x00bad")
        monkeypatch.delenv(nim.LOCAL_KEY_ENV, raising=False)
        monkeypatch.setattr(nim, "LOCAL_KEY_FILE", f)
        nim.reset()
        assert nim.get_local_client() is None

    def test_nim_only_chain_never_resolves_local(self, monkeypatch):
        def boom():
            raise AssertionError("local resolved")
        monkeypatch.setattr(nim, "get_local_client", boom)
        nim.set_client(NimClient(NIM_KEY, guard=NimGuard(), post=FakePost({GPT: [ok(RISK_GOOD)]})))
        assert nim.run_chain((NIM_SPEC,), "s", "u", RISK_KEYS) is not None


# ---------------------------------------------------------------------------
# Providers skip independently
# ---------------------------------------------------------------------------

class TestIndependentSkips:
    CHAIN = (NIM_SPEC, LOCAL_SPEC, NimSpec(GLM, 25.0))

    def test_full_nim_window_skips_nim_specs_but_local_runs(self):
        nim_post, local_post, _ = install(local_plans={LOC: [ok(RISK_GOOD)]})
        client = nim.get_client()
        for _ in range(30):
            assert client.guard.acquire(GPT) is None
        res = nim.run_chain(self.CHAIN, "s", "u", RISK_KEYS)
        assert res.label == "local:gpt-oss-20b"
        assert nim_post.calls == [] and local_post.models == [LOC]

    def test_nim_429_skips_remaining_nim_but_local_runs(self):
        nim_post, local_post, _ = install(nim_plans={GPT: [status(429)], GLM: [ok(RISK_GOOD)]},
                                          local_plans={LOC: [ok(RISK_GOOD)]})
        res = nim.run_chain(self.CHAIN, "s", "u", RISK_KEYS)
        assert res.label == "local:gpt-oss-20b"
        assert nim_post.models == [GPT]                  # GLM never tried
        assert local_post.models == [LOC]

    def test_nim_429_and_local_failure_returns_none_without_trying_glm(self):
        nim_post, local_post, _ = install(nim_plans={GPT: [status(429)], GLM: [ok(RISK_GOOD)]},
                                          local_plans={LOC: [httpx.ReadTimeout("t")]})
        assert nim.run_chain(self.CHAIN, "s", "u", RISK_KEYS) is None
        assert nim_post.models == [GPT] and local_post.models == [LOC]

    def test_nim_cooldown_from_an_earlier_call_skips_nim_only(self):
        nim_post, local_post, _ = install(nim_plans={GPT: [status(429)]},
                                          local_plans={LOC: [ok(RISK_GOOD)]})
        nim.run_chain((NIM_SPEC,), "s", "u", RISK_KEYS)  # sets the NIM cooldown
        res = nim.run_chain(self.CHAIN, "s", "u", RISK_KEYS)
        assert res.label == "local:gpt-oss-20b"
        assert len(nim_post.calls) == 1

    def test_local_429_skips_remaining_local_but_nim_runs(self):
        other = NimSpec("other-local", 25.0, provider="local")
        chain = (LOCAL_SPEC, other, NimSpec(GLM, 25.0))
        nim_post, local_post, _ = install(nim_plans={GLM: [ok(RISK_GOOD)]},
                                          local_plans={LOC: [status(429)], "other-local": [ok(RISK_GOOD)]})
        res = nim.run_chain(chain, "s", "u", RISK_KEYS)
        assert res.label == "nim:z-ai/glm-5.3-flash"
        assert local_post.models == [LOC]                # "other-local" skipped
        # and the 429 on local set no NIM cooldown
        assert nim.get_client().guard.acquire(GLM) is None

    def test_local_down_degrades_to_next_nim_spec(self):
        nim_post, local_post, _ = install(nim_plans={GPT: [httpx.ReadTimeout("t")], GLM: [ok(RISK_GOOD)]},
                                          local_plans={LOC: [httpx.ConnectError("down")]})
        res = nim.run_chain(self.CHAIN, "s", "u", RISK_KEYS)
        assert res.label == "nim:z-ai/glm-5.3-flash"
        assert nim_post.models == [GPT, GLM] and local_post.models == [LOC]


class TestLocalBreaker:
    def test_opens_after_two_timeouts_and_reprobes_after_five_minutes(self):
        _, local_post, clock = install(local_plans={LOC: [httpx.ReadTimeout("t"), httpx.ReadTimeout("t"),
                                                          ok(RISK_GOOD)]})
        client = nim.get_local_client()
        assert client.complete(LOCAL_SPEC, "s", "u").error == "timeout"
        assert client.complete(LOCAL_SPEC, "s", "u").error == "timeout"
        assert len(local_post.calls) == 2
        res = client.complete(LOCAL_SPEC, "s", "u")
        assert res.skipped == "breaker" and len(local_post.calls) == 2
        clock.advance(299)
        assert client.complete(LOCAL_SPEC, "s", "u").skipped == "breaker"
        clock.advance(2)                                  # 5 min elapsed: one probe goes out
        res = client.complete(LOCAL_SPEC, "s", "u")
        assert res.ok and len(local_post.calls) == 3
        assert client.complete(LOCAL_SPEC, "s", "u").ok   # closed again

    def test_failed_probe_reopens_for_another_five_minutes(self):
        _, local_post, clock = install(local_plans={LOC: [httpx.ReadTimeout("t")]})
        client = nim.get_local_client()
        client.complete(LOCAL_SPEC, "s", "u")
        client.complete(LOCAL_SPEC, "s", "u")
        clock.advance(301)
        assert client.complete(LOCAL_SPEC, "s", "u").error == "timeout"     # the probe
        assert client.complete(LOCAL_SPEC, "s", "u").skipped == "breaker"
        clock.advance(301)
        assert client.complete(LOCAL_SPEC, "s", "u").error == "timeout"
        assert len(local_post.calls) == 4

    def test_nim_breaker_keeps_its_fifteen_minutes(self):
        nim_post, _, clock = install(nim_plans={GPT: [httpx.ReadTimeout("t")]})
        client = nim.get_client()
        client.complete(NIM_SPEC, "s", "u")
        client.complete(NIM_SPEC, "s", "u")
        clock.advance(301)
        assert client.complete(NIM_SPEC, "s", "u").skipped == "breaker"
        clock.advance(600)
        assert client.complete(NIM_SPEC, "s", "u").error == "timeout"       # probe after 15 min

    def test_guard_default_breaker_open_sec(self):
        assert NimGuard()._open_sec == nim.BREAKER_OPEN_SEC == 900.0
        assert nim.LOCAL_GUARD._open_sec == 300.0


class TestSummary:
    def test_local_stats_appear_when_local_is_enabled(self):
        install(local_plans={LOC: [ok(RISK_GOOD)]})
        nim.run_chain((LOCAL_SPEC,), "s", "u", RISK_KEYS)
        line = nim.summary_and_reset()
        assert line.startswith("NIM stats: calls=0") and "| local stats: calls=" in line

    def test_no_local_part_when_local_is_disabled(self):
        assert "local" not in nim.summary_and_reset()


# ---------------------------------------------------------------------------
# Leg chains
# ---------------------------------------------------------------------------

class TestRiskChainOrder:
    def _client(self):
        from src.llm.deepseek_risk import DeepSeekRisk
        c = DeepSeekRisk()
        c._cf_client = MagicMock()
        c._db = RecordingDB()
        return c

    def test_chain_definition(self):
        from src.llm.deepseek_risk import NIM_CHAIN
        assert [(s.provider, s.model, s.timeout) for s in NIM_CHAIN] == [
            ("nim", GPT, 20.0), ("local", LOC, 25.0), ("nim", GLM, 25.0)]

    def test_local_answers_between_the_two_nim_models(self):
        nim_post, local_post, _ = install(nim_plans={GPT: [httpx.ReadTimeout("t")], GLM: [ok(RISK_GOOD)]},
                                          local_plans={LOC: [ok(RISK_GOOD)]})
        c = self._client()
        r = c.evaluate_raw("s", "u", bar_ts=7)
        assert r.error is None and r.model_used == "local:gpt-oss-20b"
        assert nim_post.models == [GPT] and local_post.models == [LOC]
        assert r.cost_usd == 0.0
        assert c._db.rows[0]["model"] == "local:gpt-oss-20b"
        c._cf_client.chat_completion.assert_not_called()

    def test_order_nim_gpt_then_local_then_glm_then_cloudflare(self):
        nim_post, local_post, _ = install(
            nim_plans={GPT: [httpx.ReadTimeout("t")], GLM: [status(502)]},
            local_plans={LOC: [httpx.ConnectError("down")]})
        c = self._client()
        c._cf_client.chat_completion = MagicMock(return_value=_cf_resp(RISK_GOOD))
        r = c.evaluate_raw("s", "u", bar_ts=7)
        assert nim_post.models == [GPT, GLM] and local_post.models == [LOC]
        assert [x["timeout"] for x in local_post.calls] == [25.0]
        assert r.model_used == "@cf/meta/llama-4-scout-17b-16e-instruct"

    def test_first_nim_model_still_wins_and_local_is_untouched(self):
        _, local_post, _ = install(nim_plans={GPT: [ok(RISK_GOOD)]}, local_plans={LOC: [ok(RISK_GOOD)]})
        r = self._client().evaluate_raw("s", "u", bar_ts=7)
        assert r.model_used == "nim:openai/gpt-oss-20b" and local_post.calls == []

    def test_local_disabled_behaves_as_before(self):
        nim.set_local_client(None)
        nim_post = FakePost({GPT: [httpx.ReadTimeout("t")], GLM: [ok(RISK_GOOD)]})
        nim.set_client(NimClient(NIM_KEY, guard=NimGuard(), post=nim_post))
        r = self._client().evaluate_raw("s", "u", bar_ts=7)
        assert r.model_used == "nim:z-ai/glm-5.3-flash" and nim_post.models == [GPT, GLM]


class TestExecutionChainOrder:
    def _client(self, cf=True):
        from src.llm.gemini_execution import GeminiExecution
        c = GeminiExecution(api_key="test-key-redacted", tracker=CostTracker())
        c._cf_client = MagicMock() if cf else None
        c._db = RecordingDB()
        return c

    def test_chain_definition(self):
        from src.llm.gemini_execution import LOCAL_CHAIN, NIM_CHAIN
        assert [s.provider for s in NIM_CHAIN] == ["nim", "nim"]
        assert [(s.provider, s.model, s.timeout) for s in LOCAL_CHAIN] == [("local", LOC, 25.0)]

    def test_local_only_after_gemini_fails_and_before_cloudflare(self):
        nim_post, local_post, _ = install(
            nim_plans={GLM: [httpx.ReadTimeout("t")], NEMO: [status(503)]},
            local_plans={LOC: [ok(EXEC_GOOD)]})
        c = self._client()
        order = []

        def gemini(*a, **kw):
            order.append(("gemini", kw["model"]))
            assert local_post.calls == []                # local not tried while Gemini is still going
            raise Exception("429 quota")

        with patch.object(c._client.models, "generate_content", side_effect=gemini):
            r = c.evaluate_raw("s", "u", bar_ts=7)
        from src.llm.gemini_execution import GEMINI_MODELS
        assert order == [("gemini", m) for m in GEMINI_MODELS]
        assert nim_post.models == [GLM, NEMO] and local_post.models == [LOC]
        assert r.error is None and r.model_used == "local:gpt-oss-20b"
        assert r.parsed["action"] == "open_long" and r.cost_usd == 0.0
        assert c._db.rows[0]["model"] == "local:gpt-oss-20b"
        c._cf_client.chat_completion.assert_not_called()

    def test_gemini_answering_means_local_is_never_called(self):
        _, local_post, _ = install(local_plans={LOC: [ok(EXEC_GOOD)]})
        c = self._client()
        resp = MagicMock()
        resp.text = json.dumps(EXEC_GOOD)
        resp.usage_metadata.prompt_token_count = 80
        resp.usage_metadata.candidates_token_count = 40
        with patch.object(c._client.models, "generate_content", return_value=resp):
            r = c.evaluate_raw("s", "u", bar_ts=7)
        assert r.model_used == "gemini-3.6-flash" and r.cost_usd > 0
        assert local_post.calls == []

    def test_local_failure_falls_to_cloudflare(self):
        _, local_post, _ = install(local_plans={LOC: [httpx.ReadTimeout("t")]})
        c = self._client()
        c._cf_client.chat_completion = MagicMock(return_value=_cf_resp(EXEC_GOOD))
        with patch.object(c._client.models, "generate_content", side_effect=Exception("gemini down")):
            r = c.evaluate_raw("s", "u", bar_ts=7)
        assert local_post.models == [LOC]
        assert r.model_used == "@cf/mistralai/mistral-small-3.1-24b-instruct"

    def test_local_disabled_and_no_cloudflare_keeps_original_error(self):
        nim.set_local_client(None)
        c = self._client(cf=False)
        with patch.object(c._client.models, "generate_content", side_effect=Exception("gemini down")):
            r = c.evaluate_raw("s", "u", bar_ts=7)
        assert r.used_fallback and "gemini down" in r.error

    def test_local_prefix_costs_nothing(self):
        c = self._client()
        for label in ("local:gpt-oss-20b", "nim:z-ai/glm-5.3-flash", "@cf/x"):
            c._last_model = label
            assert c._actual_cost_usd(1000, 1000) == 0.0
        c._last_model = "gemini-3.6-flash"
        assert c._actual_cost_usd(1000, 1000) > 0.0

    def test_dashboard_derives_a_clean_name_from_the_local_label(self):
        from src.dashboard.app import _clean_model_id
        assert _clean_model_id("local:gpt-oss-20b") == "gpt-oss-20b"
