**English** · [繁體中文](README.zh-TW.md)

# DOWTrade

Paper-trading bot for Micro E-mini Dow futures (**MYM**). A three-LLM analysis
pipeline proposes trades; a hard-coded Python safety layer has the final word
before any order is placed.

> **Paper-only.** `PAPER_ONLY=True` and `BROKER_ENV="demo"` are hard-coded in
> `src/config.py` and asserted at startup — the process refuses to run outside a
> sandbox. No real money, no profitability claims. A learning/research scaffold.

## Pipeline (per 15-minute bar)

```
yfinance ^DJI bar ─▶ feature extraction (ATR-14, SMA-200, Donchian-20, swings)
            │
            ▼
   1. Claude (Anthropic)     — structural read. Sonnet via subscription SDK on
                               crucial bars (position open), else Haiku API
   2. Gemini (API)           — action: open_long | open_short | close | add_pyramid | hold
   3. Risk auditor (NVIDIA NIM / Workers AI) — advisory risk audit (recorded for /decisions; non-blocking)
            │
            ▼
   4. Python final_check     — HARD override (daily-loss, size, stop, ATR bounds)
   5. Golden/Death cross gate — SMA 20/50 on 15m AND 1hr must agree before entry
            │
            ▼
   sim-fill (paper) ─▶ SQLite ─▶ FastAPI dashboard
```

The Python layer is the single source of truth for safety. No LLM, env var, or
config file can override the rails in `src/config.py`.

## Key features

- **Three-LLM ensemble** — Claude (structural; **Sonnet** via the Claude Agent SDK
  on your Pro plan's included usage for crucial bars, **Haiku** API otherwise and as
  fallback), Gemini (execution fallback, via the Gemini **API** — Google retired the
  individual-tier `gemini-cli` on 2026-06-18, so it is off by default),
  DeepSeek (**advisory** risk audit). Each call is logged. DeepSeek's verdict is
  recorded for the `/decisions` view but no longer gates execution — the
  deterministic `final_check` rails are the sole safety authority. With no
  subscription token everything runs on the metered Haiku API.
- **NVIDIA NIM first, free fallbacks behind it.** The risk and execution legs try
  NVIDIA NIM (free tier) before anything else; the provider that answered is stored
  in `llm_calls.model` with a `nim:` prefix (cost 0, so it never counts toward
  `MAX_LLM_SPEND_USD`).
  - Risk audit: NIM `openai/gpt-oss-20b` (20 s) -> NIM `z-ai/glm-5.3-flash` (25 s) ->
    Cloudflare `@cf/meta/llama-4-scout-17b-16e-instruct` ->
    Cloudflare `@cf/mistralai/mistral-small-3.1-24b-instruct`. On the risk-audit
    benchmark (synthetic violations plus real bars) both NIM models flagged 18/18 true violations with no false positives,
    against 9/18 for Cloudflare mistral-small. The Hugging Face DeepSeek leg was
    removed: that account has no credit and returned 402 on every call.
  - Execution: NIM `z-ai/glm-5.3-flash` (40 s) -> NIM
    `nvidia/nemotron-3-super-120b-a12b` (8 s) -> the Gemini API chain, unchanged ->
    Cloudflare mistral-small as a last resort.
  - Structural: unchanged (Sonnet SDK -> Haiku), because no NIM model matched it. A
    **shadow** call to `openai/gpt-oss-20b` (15 s) runs after each production call
    and is logged as `nim-shadow:openai/gpt-oss-20b` for later comparison; its answer
    is never used for a decision.
  - One shared guard covers every NIM call (`src/llm/nim.py`): at most 30 requests per
    rolling 60 s (a full window sends the call straight to the next provider, no
    sleeping), a 30 s global cooldown on HTTP 429 (`Retry-After` honoured, capped at
    120 s), and a per-model circuit breaker that skips a model for 15 min after 2
    consecutive timeouts / 5xx / connection errors, then lets one probe through. No
    retries: every failure falls through at once.
  - Models that answer in `reasoning` and leave `content` empty cannot satisfy the
    JSON schema; `content` is the only field ever read.
- **Hard safety layer** — `final_check` enforces max daily loss
  (`MAX_DAILY_LOSS_USD`, $600), fixed risk per trade
  (`FIXED_RISK_PER_TRADE_USD`, $250), max open contracts
  (`MAX_OPEN_CONTRACTS`, 3), mandatory stop-loss, ATR-bounded stops, no
  averaging down, flat-before-weekend. These are module constants in
  `src/config.py`, not env vars — they are asserted at startup and nothing in
  the environment can move them.
- **Golden/death cross filter** — entries require SMA 20/50 cross agreement on both
  the 15m and 1hr timeframes (trend-following discipline).
- **Sim-fills** — `SIM_FILLS=1` synthesizes fills locally at bar close and tracks
  the open lots (an entry plus its pyramid adds, aggregated into one position
  before the LLMs ever see it), persisted across restarts in `sim_state` /
  `sim_positions` tables (the broker's cert sandbox cannot fill orders). Total
  size is bounded by `MAX_OPEN_CONTRACTS`, not by a separate lot count.
- **Dashboard** — FastAPI + Jinja2 in **zh-TW and English** (switch at
  `/lang/{code}`; strings in `src/dashboard/i18n.py`, kept byte-identical with
  the trader dashboard): equity, day P&L, per-LLM reasoning cards, and a
  yfinance heartbeat to detect silent stalls. `/` and `/decisions` are the two
  real pages; `/equity`, `/trades`, `/journal` and `/disagreements` are kept as
  redirects so older links still resolve.
- **Daily journal** — APScheduler writes an end-of-day review.

## Market data & broker

- **Data:** yfinance 15m `^DJI` (Dow index) bars (polled every 60s), configurable via
  `YF_DATA_SYMBOL` / `YF_HYDRATE_PERIOD`. The index is roll-proof; the previous
  `MYM=F` micro-future continuous symbol went dark at the **2026-06-19 quarterly
  futures roll** and stalled the bot, so the structural feed now reads the index
  (orders are still placed as **MYM** on the broker). dxLink streaming exists as a
  fallback but the cert account has no live-data entitlement.
- **Broker:** Tastytrade certification sandbox. Order submission is stubbed by
  sim-fills because the sandbox rejects order placement for this account.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env        # fill in API keys + cert credentials
```

Required in `.env`: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
and the `TASTYTRADE_CERT_*` sandbox credentials. See `.env.example`.

**Recommended — NVIDIA NIM.** Put a free NIM key (build.nvidia.com) in the
`NVIDIA_NIM_API_KEY` environment variable or in a mode-600 file `~/.nvidia-nim-key`
— deliberately not in `.env` or the repo. Without a key NIM is disabled and the
legs behave as they did before NIM (Gemini for execution, Cloudflare for the risk
audit).

**Recommended — Cloudflare Workers AI fallback.** Set `CLOUDFLARE_ACCOUNT_ID` and
`CLOUDFLARE_API_TOKEN` (the token needs only **Account > Workers AI > Read**) to give
the risk auditor a free fallback behind NIM and the execution judge a last resort
behind Gemini, on a separate billing rail. The `HF_RISK_MODEL` variable no longer
exists.

**Optional — Claude Sonnet via subscription.** To run the structural judge on Claude
**Sonnet** through the Claude Agent SDK (free ~$20/mo subscription credit instead of
metered Haiku API tokens), add `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`).
Tunables: `DOWTRADE_SDK_FOR` (`positions` [default — Sonnet only while holding a live
position] | `all` | `none`), `CLAUDE_SDK_MONTHLY_CAP_USD` (default `20`),
`CLAUDE_SDK_MODEL` (default `sonnet`). With no token the bot runs Haiku-only, as before.

## Run

```bash
python -m src.main          # starts LiveRunner + dashboard on :8000
```

## Tests

```bash
python -m pytest                                   # all tests
python -m pytest --cov=src --cov-report=term-missing
```

## Layout

```
src/
├── config.py            # HARD-CODED safety rails + env Settings
├── main.py              # entrypoint: LiveRunner thread + uvicorn dashboard
├── live/
│   ├── runner.py        # bar loop, LLM pipeline, sim-fills, cross gate
│   └── yfinance_poller.py
├── llm/                 # haiku_structural, gemini_execution, deepseek_risk
│   └── prompts/
├── data/                # bars, features, cross_filter
├── broker/              # tastytrade client, models
├── safety/              # guard layer
├── sizing/              # risk-unit position sizing
├── db/                  # schema.sql + repo (raw sqlite, no ORM)
├── dashboard/           # FastAPI + Jinja2, i18n.py, static/theme.css
├── journal/             # daily review (APScheduler)
└── backtest/            # harness: final_check, compute_size, _PositionState
tests/
```
