**English** · [繁體中文](README.zh-TW.md)

# DOWTrade

Paper-trading bot for Micro E-mini Dow futures (**MYM**). A three-LLM analysis
pipeline proposes trades; a hard-coded Python safety layer has the final word
before any order is placed.

> **Paper-only.** `PAPER_ONLY=True` and `BROKER_ENV="demo"` are hard-coded in
> `src/config.py` and asserted at startup — the process refuses to run outside a
> sandbox. No real money, no profitability claims. A learning/research scaffold.

## At a glance

| | |
|---|---|
| What I built | A 15-minute-bar trading bot. Three LLMs read the market; a Python safety layer can override them. |
| Result | Risk-audit benchmark: both NIM models flagged 18/18 true violations, 0 false positives; Cloudflare mistral-small 9/18. |
| Stack | Python 3.11+, FastAPI + Jinja2, SQLite, APScheduler, yfinance, Tastytrade cert sandbox; Claude, Gemini, NVIDIA NIM, Cloudflare Workers AI. |
| Status | Research scaffold. Paper only. |

## How it works

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

## The engineering story

- **Symptom.** The structural feed went quiet and the bot stalled after the
  **2026-06-19 quarterly futures roll**. The old feed read `MYM=F`, the
  micro-future continuous symbol, which went dark at the roll.
- **Fix.** The feed now reads the Dow index `^DJI` from yfinance. The index is
  roll-proof. Orders are still placed as **MYM** on the broker.
- **Measurement.** The risk-audit benchmark (synthetic violations plus real
  bars) showed both NIM models flagging 18/18 true violations with no false
  positives, against 9/18 for Cloudflare mistral-small.
- **Cause of the Hugging Face leg.** The Hugging Face DeepSeek account has no
  credit and returned 402 on every call. That leg was removed.
- **Authority change.** DeepSeek's verdict is recorded for `/decisions` but no
  longer gates execution. The deterministic `final_check` rails are the sole
  safety authority.
- **Open.** The cert account has no live-data entitlement (dxLink streaming is a
  fallback only). The broker sandbox rejects order placement, so fills are
  simulated.

## Key features

- **Three-LLM ensemble** — Claude (structural), Gemini (execution fallback) and DeepSeek
  (**advisory** risk audit). Each call is logged. No subscription token means
  everything runs on the metered Haiku API.
- **Hard safety layer** — `final_check` enforces max daily loss
  (`MAX_DAILY_LOSS_USD`, $600), fixed risk per trade
  (`FIXED_RISK_PER_TRADE_USD`, $250), max open contracts
  (`MAX_OPEN_CONTRACTS`, 3), mandatory stop-loss, ATR-bounded stops, no
  averaging down, flat-before-weekend.
- **Golden/death cross filter** — entries require SMA 20/50 cross agreement on
  both the 15m and 1hr timeframes.
- **Sim-fills**, **dashboard** (zh-TW and English) and a **daily journal**.

Full detail (provider chains, NIM guard, timing, sim-fill persistence, dashboard
pages): [docs/features.md](docs/features.md).

## Market data and broker

- **Data:** yfinance 15m `^DJI` bars (polled every 60s), configurable via
  `YF_DATA_SYMBOL` / `YF_HYDRATE_PERIOD`.
- **Broker:** Tastytrade certification sandbox. Sim-fills stand in for order
  submission.

Details: [docs/features.md](docs/features.md#market-data-and-broker).

## Build / run

```bash
pip install -e ".[dev]"
cp .env.example .env        # fill in API keys + cert credentials
python -m src.main          # starts LiveRunner + dashboard on :8000
```

Required in `.env`: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, and the
`TASTYTRADE_CERT_*` sandbox credentials. See `.env.example`.

NVIDIA NIM, the self-hosted LLM fallback, Cloudflare Workers AI and Claude
Sonnet via subscription are set up with optional keys and variables:
[docs/setup.md](docs/setup.md).

## Tests

```bash
python -m pytest                                   # all tests
python -m pytest --cov=src --cov-report=term-missing
```

## Repo layout

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

## Limits and open items

- The bot is paper only; it places no real orders.
- The cert account has no live-data entitlement.
- The broker sandbox rejects order placement for this account, so fills are
  simulated.

## License

MIT. See [LICENSE](LICENSE).
