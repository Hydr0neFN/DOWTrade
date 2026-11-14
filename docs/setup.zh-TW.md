[English](setup.md) · **繁體中文**

# DOWTrade 安裝設定

安裝與執行看 README。選配的供應商在這裡設定。

```bash
pip install -e ".[dev]"
cp .env.example .env        # fill in API keys + cert credentials
```

`.env` 必填：`ANTHROPIC_API_KEY`、`GOOGLE_API_KEY`，以及 `TASTYTRADE_CERT_*`
沙盒憑證。範例見 `.env.example`。

## 建議：NVIDIA NIM

把免費的 NIM 金鑰（build.nvidia.com）放進環境變數 `NVIDIA_NIM_API_KEY`，或放在權限 600 的檔案 `~/.nvidia-nim-key` — 刻意不進 `.env`、也不進 repo。沒設金鑰就停用 NIM，各段維持 NIM 之前的老樣子（執行走 Gemini，風險審查走 Cloudflare）。

## 選配：自架 LLM 後備

把 llama-server 的金鑰放進 `LOCAL_LLM_API_KEY`，或放在權限 600 的檔案 `~/.local-llm-key`（網址預設是 `src/llm/nim.py` 裡的 Tailscale 位址，可以用 `LOCAL_LLM_URL` 換掉）。沒設金鑰就停用這個後備。

## 建議：Cloudflare Workers AI 後備

設好 `CLOUDFLARE_ACCOUNT_ID` 和 `CLOUDFLARE_API_TOKEN`（token 只要 **Account > Workers AI > Read** 權限），風險審查在 NIM 之後就有免費後備，執行判斷在 Gemini 之後也有最後一道，而且是另一條計費管道。`HF_RISK_MODEL` 這個變數已經沒有了。

## 選配：訂閱版 Claude Sonnet

想讓結構判讀透過 Claude Agent SDK 跑 Claude **Sonnet**（用每月約 $20 的訂閱額度，不必花按量計費的 Haiku API token），請加上 `CLAUDE_CODE_OAUTH_TOKEN`（用 `claude setup-token` 取得）。可調參數：`DOWTRADE_SDK_FOR`（`positions`［預設 — 只在有持倉時用 Sonnet］ | `all` | `none`）、`CLAUDE_SDK_MONTHLY_CAP_USD`（預設 `20`）、`CLAUDE_SDK_MODEL`（預設 `sonnet`）。沒有 token 時，機器人就跟以前一樣只跑 Haiku。

## 執行與測試

```bash
python -m src.main          # starts LiveRunner + dashboard on :8000
python -m pytest                                   # all tests
python -m pytest --cov=src --cov-report=term-missing
```
