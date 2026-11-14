**English** · [繁體中文](features.zh-TW.md)

# DOWTrade features

Provider chains, guard limits, sim-fills, dashboard and market-data detail.
The README holds the short versions.

## Three-LLM ensemble

- **Claude** (Anthropic) — structural read. **Sonnet** via the Claude Agent SDK
  on your Pro plan's included usage for crucial bars, **Haiku** API otherwise and
  as fallback.
- **Gemini** — execution fallback, via the Gemini **API**. Google retired the
  individual-tier `gemini-cli` on 2026-06-18, so it is off by default.
- **DeepSeek** — **advisory** risk audit.

Each call is logged. DeepSeek's verdict is recorded for the `/decisions` view
but no longer gates execution — the deterministic `final_check` rails are the
sole safety authority. With no subscription token everything runs on the metered
Haiku API.

## NVIDIA NIM first, free fallbacks behind it

The risk and execution legs try NVIDIA NIM (free tier) before anything else. The
provider that answered is stored in `llm_calls.model` with a `nim:` prefix
(cost 0, so it never counts toward `MAX_LLM_SPEND_USD`).

- Risk audit: NIM `openai/gpt-oss-20b` (20 s) -> self-hosted `gpt-oss-20b` (25 s,
  logged as `local:gpt-oss-20b`) -> NIM `z-ai/glm-5.3-flash` (25 s) ->
  Cloudflare `@cf/meta/llama-4-scout-17b-16e-instruct` ->
  Cloudflare `@cf/mistralai/mistral-small-3.1-24b-instruct`. On the risk-audit
  benchmark (synthetic violations plus real bars) both NIM models flagged 18/18
  true violations with no false positives, against 9/18 for Cloudflare
  mistral-small. The Hugging Face DeepSeek leg was removed: that account has no
  credit and returned 402 on every call.
- Execution: NIM `z-ai/glm-5.3-flash` (40 s) -> NIM
  `nvidia/nemotron-3-super-120b-a12b` (8 s) -> the Gemini API chain, unchanged ->
  self-hosted `gpt-oss-20b` (25 s, `local:gpt-oss-20b`) -> Cloudflare
  mistral-small as a last resort.
- Structural: unchanged (Sonnet SDK -> Haiku), because no NIM model matched it. A
  **shadow** call to `openai/gpt-oss-20b` (15 s) runs after each production call
  and is logged as `nim-shadow:openai/gpt-oss-20b` for later comparison; its
  answer is never used for a decision.
- One shared guard covers every NIM call (`src/llm/nim.py`): at most 30 requests
  per rolling 60 s (a full window sends the call straight to the next provider,
  no sleeping), a 30 s global cooldown on HTTP 429 (`Retry-After` honoured,
  capped at 120 s), and a per-model circuit breaker that skips a model for
  15 min after 2 consecutive timeouts / 5xx / connection errors, then lets one
  probe through. No retries: every failure falls through at once.
- The self-hosted model (a llama-server reached over Tailscale, same weights as
  NIM's `gpt-oss-20b`) is a fallback only. It has its own client and guard
  (`LOCAL_GUARD`: 60 requests per 60 s, 5 min breaker), so it never uses NIM's
  window and a NIM cooldown does not skip it. Zero cost; with no key it is
  disabled and the legs behave as before.
- Models that answer in `reasoning` and leave `content` empty cannot satisfy the
  JSON schema; `content` is the only field ever read.

## Hard safety layer

`final_check` enforces max daily loss (`MAX_DAILY_LOSS_USD`, $600), fixed risk
per trade (`FIXED_RISK_PER_TRADE_USD`, $250), max open contracts
(`MAX_OPEN_CONTRACTS`, 3), mandatory stop-loss, ATR-bounded stops, no averaging
down, flat-before-weekend. These are module constants in `src/config.py`, not
env vars — they are asserted at startup and nothing in the environment can move
them.

## Golden/death cross filter

Entries require SMA 20/50 cross agreement on both the 15m and 1hr timeframes
(trend-following discipline).

## Sim-fills

`SIM_FILLS=1` synthesizes fills locally at bar close and tracks the open lots
(an entry plus its pyramid adds, aggregated into one position before the LLMs
ever see it), persisted across restarts in `sim_state` / `sim_positions` tables
(the broker's cert sandbox cannot fill orders). Total size is bounded by
`MAX_OPEN_CONTRACTS`, not by a separate lot count.

## Dashboard and journal

- **Dashboard** — FastAPI + Jinja2 in **zh-TW and English** (switch at
  `/lang/{code}`; strings in `src/dashboard/i18n.py`, kept byte-identical with
  the trader dashboard): equity, day P&L, per-LLM reasoning cards, and a
  yfinance heartbeat to detect silent stalls. `/` and `/decisions` are the two
  real pages; `/equity`, `/trades`, `/journal` and `/disagreements` are kept as
  redirects so older links still resolve.
- **Daily journal** — APScheduler writes an end-of-day review.

## Market data and broker

- **Data:** yfinance 15m `^DJI` (Dow index) bars (polled every 60s), configurable
  via `YF_DATA_SYMBOL` / `YF_HYDRATE_PERIOD`. The index is roll-proof; the
  previous `MYM=F` micro-future continuous symbol went dark at the **2026-06-19
  quarterly futures roll** and stalled the bot, so the structural feed now reads
  the index (orders are still placed as **MYM** on the broker). dxLink streaming
  exists as a fallback but the cert account has no live-data entitlement.
- **Broker:** Tastytrade certification sandbox. Order submission is stubbed by
  sim-fills because the sandbox rejects order placement for this account.
