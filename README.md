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
   3. DeepSeek (HF)          — advisory risk audit (recorded for /disagreements; non-blocking)
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
  fallback), Gemini (execution, via the Gemini **API** — Google retired the
  individual-tier `gemini-cli` on 2026-06-18, so it is off by default),
  DeepSeek/Qwen (**advisory** risk audit). Each call is logged. DeepSeek's verdict is
  recorded for the `/disagreements` view but no longer gates execution — the
  deterministic `final_check` rails are the sole safety authority. With no
  subscription token everything runs on the metered Haiku API.
- **Hard safety layer** — `final_check` enforces max daily loss ($200), fixed risk
  per trade ($50), max open contracts, mandatory stop-loss, ATR-bounded stops,
  no averaging down, flat-before-weekend.
- **Golden/death cross filter** — entries require SMA 20/50 cross agreement on both
  the 15m and 1hr timeframes (trend-following discipline).
- **Sim-fills** — `SIM_FILLS=1` synthesizes fills locally at bar close and tracks up
  to 5 concurrent positions, persisted across restarts in `sim_state` /
  `sim_positions` tables (the broker's cert sandbox cannot fill orders).
- **Dashboard** — FastAPI + Jinja2: equity, day P&L, per-LLM reasoning cards, and a
  yfinance heartbeat to detect silent stalls.
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

Required in `.env`: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `HUGGINGFACE_API_KEY`,
and the `TASTYTRADE_CERT_*` sandbox credentials. See `.env.example`.

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
├── dashboard/           # FastAPI + Jinja2
├── journal/             # daily review (APScheduler)
└── backtest/            # harness: final_check, compute_size, _PositionState
tests/
```
