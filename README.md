# DOWTrade Bot

Paper-trading bot for Micro E-mini Dow futures (MYM) on Tradovate demo.

## Purpose

Automates a disciplined trading process informed by the philosophy encoded in
`~/DOWTrade/交易修煉克服人性駕馭趨勢-01..12.png` (12 pages of trading wisdom).
Three LLMs independently analyze market data and vote on trade decisions; a
Python safety layer acts as the hard override before any order is submitted.

## Non-Objectives

- This system is **paper-only** — no real money, no live brokerage integration.
- No profitability claims are made; the bot is a learning and research scaffold.
- Does not attempt to predict markets or guarantee consistent returns.

## Philosophy Source

The 12 PDF images in `~/DOWTrade/` (filenames
`交易修煉克服人性駕馭趨勢-01.png` through `交易修煉克服人性駕馭趨勢-12.png`)
encode the trading philosophy: trend-following, strict risk management, and
overcoming psychological biases. All safety constants in `src/config.py` are
derived from these principles.

## Three-LLM + Python-Safety Pipeline

Each 15-minute bar triggers three independent LLM calls — Claude (Anthropic),
Gemini (Google), and a Hugging Face open model — each returning a structured
decision (direction, stop, size reasoning). A Python majority-vote aggregator
reconciles the three signals; the Python safety layer then enforces hard limits
on daily loss, position size, stop placement, and paper-only mode before any
order is sent to Tradovate demo.

## Build Phases

1. **Phase 1 — Scaffolding**: project skeleton, database schema, safety config.
2. **Phase 2 — Data pipeline**: 15-min bar ingestion and feature extraction.
3. **Phase 3 — Broker integration**: Tradovate demo auth, order placement, fills.
4. **Phase 4 — LLM decision layer**: prompt design and three-model voting.
5. **Phase 5 — Safety & sizing**: guard checks, ATR-based stops, risk-unit sizing.
6. **Phase 6 — Dashboard & journal**: FastAPI dashboard, trade journaling, review.

## Running Tests

```bash
cd ~/DOWTrade/trading-bot
python -m pytest
# or with coverage
python -m pytest --cov=src --cov-report=term-missing
```

## Directory Tree

```
trading-bot/
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── data/                   # SQLite DB (gitignored)
├── logs/                   # Runtime logs (gitignored)
├── journal/                # Trade journal markdown (gitignored)
├── src/
│   ├── config.py           # Safety rails + env settings (HARD-CODED)
│   ├── db/
│   │   ├── schema.sql      # SQLite schema
│   │   └── repo.py         # Repository pattern, no ORM
│   ├── broker/             # Tradovate demo API client
│   ├── data/               # Bar ingestion and feature extraction
│   ├── llm/                # Claude, Gemini, HF decision callers
│   │   └── prompts/        # Prompt templates
│   ├── safety/             # Hard-limit guard layer
│   ├── sizing/             # Risk-unit position sizing
│   ├── journal/            # Trade journaling logic
│   ├── dashboard/          # FastAPI + Jinja2 dashboard
│   │   ├── static/
│   │   └── templates/
│   └── backtest/           # Backtest harness
└── tests/
    └── conftest.py
```
