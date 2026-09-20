[English](README.md) · **繁體中文**

# DOWTrade

適用於微型道瓊期貨（**MYM**）的模擬交易機器人。由三組 LLM 組成的分析流程提出交易建議；在送出任何訂單之前，硬編碼的 Python 安全防護層擁有最終決定權。

> **僅限模擬交易。** `src/config.py` 中硬編碼了 `PAPER_ONLY=True` 與 `BROKER_ENV="demo"`，並在啟動時進行斷言檢查 — 該程序拒絕在沙盒環境外執行。不涉及真實資金，也不對獲利能力作任何主張。僅作為學習與研究鷹架。

## 流程（每 15 分鐘 K 線）

```
yfinance ^DJI bar ─▶ feature extraction (ATR-14, SMA-200, Donchian-20, swings)
            │
            ▼
   1. Claude (Anthropic)     — structural read. Sonnet via subscription SDK on
                                crucial bars (position open), else Haiku API
   2. Gemini (API)           — action: open_long | open_short | close | add_pyramid | hold
   3. DeepSeek (HF) / Workers AI (CF) — advisory risk audit (recorded for /decisions; non-blocking)
            │
            ▼
   4. Python final_check     — HARD override (daily-loss, size, stop, ATR bounds)
   5. Golden/Death cross gate — SMA 20/50 on 15m AND 1hr must agree before entry
            │
            ▼
   sim-fill (paper) ─▶ SQLite ─▶ FastAPI dashboard
```

Python 層是安全防護的唯一真實來源。任何 LLM、環境變數或設定檔都無法覆寫 `src/config.py` 中的防護機制。

## 主要特點

- **三 LLM 集成** — Claude（結構判讀；在關鍵 K 線 [開倉] 上透過 Claude Agent SDK 使用 Pro 方案內含額度的 **Sonnet**，其餘情況及作為備用方案時使用 **Haiku** API）、Gemini（執行，透過 Gemini **API** — Google 於 2026-06-18 停用了個人層級的 `gemini-cli`，因此預設為關閉）、DeepSeek（**諮詢性質**的風險審查）。每次呼叫皆會記錄。DeepSeek 的裁決會記錄在 `/decisions` 視圖中，但不再阻擋執行 — 決定性的 `final_check` 防護規則是唯一的安全決策權威。若未設定訂閱 token，全部將在按用量計費的 Haiku API 上執行。
- **風險審查橫跨兩家供應商** — Hugging Face 的 `deepseek-ai/DeepSeek-V4.1-Flash`，後備為 Cloudflare Workers AI 的 `@cf/mistralai/mistral-small-3.1-24b-instruct` 與 `@cf/meta/llama-4-scout-17b-16e-instruct`。後備刻意架在**不同的計費管道**上：先前的模型鏈四個全在 Hugging Face，額度一旦耗盡就同時回傳 402，這一段自 2026-09-15 起已靜默失效。Cloudflare 免費層約可支應每日 600 次風險審查。把答案放進 `reasoning` 而讓 `content` 空白的模型無法滿足 JSON schema，排除依據是實測。
- **強制安全層** — `final_check` 強制執行每日最大虧損（`MAX_DAILY_LOSS_USD`，$600）、每筆交易固定風險（`FIXED_RISK_PER_TRADE_USD`，$250）、最大未平倉合約數（`MAX_OPEN_CONTRACTS`，3 口）、強制停損、ATR 限制停損範圍、禁止向下攤平、週末前清倉平倉。這些是 `src/config.py` 中的模組常數而非環境變數 —— 啟動時會斷言檢查，環境中沒有任何設定能改動它們。
- **黃金交叉／死亡交叉過濾器** — 進場需要 15m 與 1hr 時間框架上的 SMA 20/50 交叉一致（順勢交易紀律）。
- **模擬成交** — `SIM_FILLS=1` 在 K 線收盤時於本地合成成交，並追蹤未平倉的各筆 lot（一次進場加上其金字塔加碼，在送進 LLM 之前就已彙總成單一部位），跨重啟持久化儲存於 `sim_state` / `sim_positions` 資料表（券商的 cert 沙盒無法撮合成交訂單）。總部位大小由 `MAX_OPEN_CONTRACTS` 限制，而非由另一個 lot 數量上限控制。
- **儀表板** — FastAPI + Jinja2，提供**繁體中文與英文**兩種語言（於 `/lang/{code}` 切換，字串集中在 `src/dashboard/i18n.py`，與 trader 儀表板保持 byte-identical）：淨值、當日損益、各 LLM 推理卡片，以及用於偵測無聲停頓的 yfinance 心跳檢測。實際頁面只有 `/` 與 `/decisions` 兩個；`/equity`、`/trades`、`/journal` 與 `/disagreements` 保留為轉址，讓舊連結仍然可用。
- **每日日誌** — APScheduler 撰寫每日收盤回顧。

## 市場數據與券商

- **數據：** yfinance 15m `^DJI`（道瓊指數）K 線（每 60 秒輪詢一次），可透過 `YF_DATA_SYMBOL` / `YF_HYDRATE_PERIOD` 進行設定。指數不受轉倉影響；先前的 `MYM=F` 微型期貨連續商品代碼在 **2026-06-19 季度期貨轉倉**時停止更新並導致機器人停擺，因此現在結構饋送改讀取指數（券商端下單仍為 **MYM**）。dxLink 串流可作為備用方案，但 cert 帳號沒有即時數據權限。
- **券商：** Tastytrade certification 沙盒。由於沙盒拒絕此帳號下單，因此訂單送出由模擬成交替代。

## 安裝設定

```bash
pip install -e ".[dev]"
cp .env.example .env        # fill in API keys + cert credentials
```

`.env` 中必要項目：`ANTHROPIC_API_KEY`、`GOOGLE_API_KEY`、`HUGGINGFACE_API_KEY`，以及 `TASTYTRADE_CERT_*` 沙盒憑證。詳見 `.env.example`。

**建議設定 — Cloudflare Workers AI 後備。** 設定 `CLOUDFLARE_ACCOUNT_ID` 與 `CLOUDFLARE_API_TOKEN`（token 只需要 **Account > Workers AI > Read** 權限），可為風險審查提供架在不同計費管道上的免費後備。未設定時該段只剩 Hugging Face，額度一旦耗盡整段會同時失效。`HF_RISK_MODEL` 可在不改程式碼的情況下覆寫主要模型。

**選用 — 透過訂閱使用 Claude Sonnet。** 若要透過 Claude Agent SDK 在 Claude **Sonnet** 上執行結構判讀（使用每月約 $20 的訂閱額度，而非按用量計費的 Haiku API token），請新增 `CLAUDE_CODE_OAUTH_TOKEN`（來自 `claude setup-token`）。可微調參數：`DOWTRADE_SDK_FOR`（`positions` [預設 — 僅在持有即時倉位時使用 Sonnet] | `all` | `none`）、`CLAUDE_SDK_MONTHLY_CAP_USD`（預設 `20`）、`CLAUDE_SDK_MODEL`（預設 `sonnet`）。若無 token，機器人將如以往般僅以 Haiku 執行。

## 執行

```bash
python -m src.main          # starts LiveRunner + dashboard on :8000
```

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
