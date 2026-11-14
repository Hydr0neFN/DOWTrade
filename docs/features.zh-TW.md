[English](features.md) · **繁體中文**

# DOWTrade 功能細節

供應商順序、防護層限制、模擬成交、儀表板與市場數據的細節。摘要放在 README。

## 三 LLM 組合

- **Claude**（Anthropic）— 結構判讀。關鍵 K 線（有持倉）時，用 Pro 方案內含的額度透過 Claude Agent SDK 跑 **Sonnet**；其他情況與備援用 **Haiku** API。
- **Gemini** — 執行後備，走 Gemini **API**。Google 在 2026-06-18 停掉個人版的 `gemini-cli`，所以預設關閉。
- **DeepSeek** — **只給意見**的風險審查。

每次呼叫都有記錄。DeepSeek 的裁決記在 `/decisions`，已經不擋執行 — 最後把關的只有 `final_check` 這些固定規則。沒有訂閱 token 時，全部跑按量計費的 Haiku API。

## NVIDIA NIM 優先，後面接免費後備

風險審查與執行兩段都先試 NVIDIA NIM（免費層）。回答的供應商會以 `nim:` 前綴寫進 `llm_calls.model`（成本 0，不計入 `MAX_LLM_SPEND_USD`）。

- 風險審查：NIM `openai/gpt-oss-20b`（20 秒）→ 自架 `gpt-oss-20b`（25 秒，記為 `local:gpt-oss-20b`）→ NIM `z-ai/glm-5.3-flash`（25 秒）→ Cloudflare `@cf/meta/llama-4-scout-17b-16e-instruct` → Cloudflare `@cf/mistralai/mistral-small-3.1-24b-instruct`。風險審查基準測試（合成違規加真實 K 線）中，兩個 NIM 模型都抓出 18/18 個真實違規、沒有誤報；Cloudflare mistral-small 只有 9/18。Hugging Face 的 DeepSeek 那段已移除：該帳號沒有額度，每次呼叫都回 402。
- 執行：NIM `z-ai/glm-5.3-flash`（40 秒）→ NIM `nvidia/nemotron-3-super-120b-a12b`（8 秒）→ 原本的 Gemini API 鏈 → 自架 `gpt-oss-20b`（25 秒，`local:gpt-oss-20b`）→ 最後才用 Cloudflare mistral-small。
- 結構判讀：維持不變（Sonnet SDK → Haiku），因為沒有 NIM 模型比得上。每次正式呼叫之後，另外拿 `openai/gpt-oss-20b`（15 秒）跑一次**影子呼叫**，記為 `nim-shadow:openai/gpt-oss-20b` 留著日後比對；它的答案從來不會拿來做決定。
- 所有 NIM 呼叫共用一個防護（`src/llm/nim.py`）：任一個滾動 60 秒內最多 30 次請求（滿了就立刻換下一個供應商，不等待）、遇到 HTTP 429 全域冷卻 30 秒（照 `Retry-After`，最多 120 秒）、每個模型有自己的斷路器，連續 2 次逾時 / 5xx / 連線錯誤就跳過 15 分鐘，之後放一次探測。不重試，一失敗就直接往下走。
- 自架模型（透過 Tailscale 連線的 llama-server，權重和 NIM 的 `gpt-oss-20b` 一樣）只當後備。它有自己的一套 client 與防護（`LOCAL_GUARD`：每 60 秒 60 次請求，斷路器 5 分鐘），不會吃掉 NIM 的額度，NIM 冷卻時還是照常可用。成本為 0；沒設金鑰就停用，各段行為跟以前一樣。
- 只在 `reasoning` 回答、`content` 留空的模型過不了 JSON schema；程式只讀 `content` 這個欄位。

## 強制安全層

`final_check` 會擋下超過每日最大虧損（`MAX_DAILY_LOSS_USD`，$600）、每筆交易固定風險（`FIXED_RISK_PER_TRADE_USD`，$250）、最大持倉口數（`MAX_OPEN_CONTRACTS`，3）的交易，也強制停損、把停損距離限制在 ATR 範圍、不准向下攤平、週末前平倉。這些都是 `src/config.py` 裡的模組常數，不是環境變數 — 啟動時會斷言檢查，環境裡怎麼改都動不了它們。

## 黃金交叉／死亡交叉過濾

要在 15m 和 1hr 兩個週期上，SMA 20/50 的交叉方向一致才准進場（順勢交易的紀律）。

## 模擬成交

`SIM_FILLS=1` 時，K 線收盤就在本機合成成交，並追蹤未平倉的每一筆 lot（一次進場加上它的金字塔加碼，在交給 LLM 之前就先併成同一個部位），重啟後仍保存在 `sim_state` / `sim_positions` 資料表（券商的 cert sandbox 無法成交下單）。總口數受 `MAX_OPEN_CONTRACTS` 限制，沒有另外的 lot 數上限。

## 儀表板與日誌

- **儀表板** — FastAPI + Jinja2，**繁體中文與英文**（在 `/lang/{code}` 切換；字串在 `src/dashboard/i18n.py`，和 trader 儀表板保持 byte-identical）：淨值、當日損益、各 LLM 的推理卡片，以及抓無聲停擺用的 yfinance 心跳。真正的頁面只有 `/` 和 `/decisions`；`/equity`、`/trades`、`/journal`、`/disagreements` 保留成轉址，舊連結照樣能開。
- **每日日誌** — 由 APScheduler 寫每日收盤回顧。

## 市場數據與券商

- **數據：** yfinance 15m `^DJI`（道瓊指數）K 線，每 60 秒拉一次，可用 `YF_DATA_SYMBOL` / `YF_HYDRATE_PERIOD` 調整。指數不受轉倉影響；原本用的 `MYM=F` 微型期貨連續代碼在 2026-06-19 季度期貨轉倉時沒有報價，機器人跟著停擺，所以結構判讀改讀指數（下單時仍然用 **MYM** 送到券商）。dxLink 串流有留著當備援，但 cert 帳號沒有即時行情權限。
- **券商：** Tastytrade certification sandbox。這個帳號在 sandbox 下不了單，所以送單由模擬成交代替。
