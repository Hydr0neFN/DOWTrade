[English](README.md) · **繁體中文**

# DOWTrade

用來模擬交易微型道瓊期貨（**MYM**）的機器人。三組 LLM 組成的分析流程負責提出交易；訂單送出前，最後一關是寫死在 Python 裡的安全層，由它說了算。

> **僅限模擬交易。** `src/config.py` 裡寫死了 `PAPER_ONLY=True` 和 `BROKER_ENV="demo"`，啟動時會斷言檢查 — 不在 sandbox 裡，程式就拒絕執行。不碰真錢，不講獲利。純粹是學習與研究用的底架。

## 一覽

| | |
|---|---|
| 做了什麼 | 以 15 分鐘 K 線運作的交易機器人。三組 LLM 判讀行情，Python 安全層可以推翻它們。 |
| 結果 | 風險審查基準測試：兩個 NIM 模型都抓出 18/18 個真實違規、0 誤報；Cloudflare mistral-small 只有 9/18。 |
| 技術棧 | Python 3.11+、FastAPI + Jinja2、SQLite、APScheduler、yfinance、Tastytrade cert sandbox；Claude、Gemini、NVIDIA NIM、Cloudflare Workers AI。 |
| 狀態 | 研究用底架。只跑模擬單。 |

## 運作方式

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

Python 這層負責安全，也只有它能改安全規則。任何 LLM、環境變數或設定檔都改不動 `src/config.py` 裡的防護。

## 遇到的問題與處理

- **症狀。** 2026-06-19 季度期貨轉倉之後，結構判讀的行情來源斷掉，機器人跟著停擺。舊來源用 `MYM=F`，也就是微型期貨的連續代碼，轉倉時就沒有報價了。
- **處理。** 行情來源改讀 yfinance 上的道瓊指數 `^DJI`，指數不受轉倉影響。下單時仍然用 **MYM** 送到券商。
- **量測。** 風險審查基準測試（合成違規加真實 K 線）中，兩個 NIM 模型都抓出 18/18 個真實違規、0 誤報；Cloudflare mistral-small 只有 9/18。
- **Hugging Face 那段的原因。** 該帳號沒有額度，每次呼叫都回 402，所以整段拿掉。
- **誰說了算。** DeepSeek 的裁決只記在 `/decisions`，已經不擋執行。真正把關的只有 `final_check` 這些固定規則。
- **還沒解決的。** cert 帳號沒有即時行情權限（dxLink 串流只能當備援）。券商 sandbox 不接受下單，所以成交由模擬產生。

## 主要特點

- **三 LLM 組合** — Claude（結構判讀）、Gemini（執行後備）、DeepSeek（**只給意見**的風險審查）。每次呼叫都有記錄。沒有訂閱 token 時，全部跑按量計費的 Haiku API。
- **強制安全層** — `final_check` 會擋下超過每日最大虧損（`MAX_DAILY_LOSS_USD`，$600）、每筆交易固定風險（`FIXED_RISK_PER_TRADE_USD`，$250）、最大持倉口數（`MAX_OPEN_CONTRACTS`，3）的交易，也強制停損、把停損距離限制在 ATR 範圍、不准向下攤平、週末前平倉。
- **黃金交叉／死亡交叉過濾** — 要在 15m 和 1hr 兩個週期上，SMA 20/50 的交叉方向一致才准進場。
- **模擬成交**、**儀表板**（繁中與英文）與**每日日誌**。

細節（供應商順序、NIM 防護、時間設定、模擬部位保存、儀表板頁面）：
[docs/features.zh-TW.md](docs/features.zh-TW.md)。

## 市場數據與券商

- **數據：** yfinance 15m `^DJI` K 線，每 60 秒拉一次，可用 `YF_DATA_SYMBOL` / `YF_HYDRATE_PERIOD` 調整。
- **券商：** Tastytrade certification sandbox。送單由模擬成交代替。

細節：[docs/features.zh-TW.md](docs/features.zh-TW.md#市場數據與券商)。

## 建置與執行

```bash
pip install -e ".[dev]"
cp .env.example .env        # fill in API keys + cert credentials
python -m src.main          # starts LiveRunner + dashboard on :8000
```

`.env` 必填：`ANTHROPIC_API_KEY`、`GOOGLE_API_KEY`，以及 `TASTYTRADE_CERT_*`
沙盒憑證。範例見 `.env.example`。

NVIDIA NIM、自架 LLM 後備、Cloudflare Workers AI、訂閱版 Claude Sonnet
都是選配金鑰與變數，設定方式見 [docs/setup.zh-TW.md](docs/setup.zh-TW.md)。

## 測試

```bash
python -m pytest                                   # all tests
python -m pytest --cov=src --cov-report=term-missing
```

## 專案結構

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

## 限制與待辦

- 只跑模擬單，不會下真單。
- cert 帳號沒有即時行情權限。
- 券商 sandbox 不接受這個帳號下單，所以成交都是模擬的。

## 授權

MIT，詳見 [LICENSE](LICENSE)。
