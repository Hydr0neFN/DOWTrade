**English** · [繁體中文](setup.zh-TW.md)

# DOWTrade setup

Install and run are on the README. The optional providers are set up here.

```bash
pip install -e ".[dev]"
cp .env.example .env        # fill in API keys + cert credentials
```

Required in `.env`: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
and the `TASTYTRADE_CERT_*` sandbox credentials. See `.env.example`.

## Recommended — NVIDIA NIM

Put a free NIM key (build.nvidia.com) in the `NVIDIA_NIM_API_KEY` environment
variable or in a mode-600 file `~/.nvidia-nim-key` — deliberately not in `.env`
or the repo. Without a key NIM is disabled and the legs behave as they did
before NIM (Gemini for execution, Cloudflare for the risk audit).

## Optional — self-hosted LLM fallback

Put the llama-server key in `LOCAL_LLM_API_KEY` or a mode-600 file
`~/.local-llm-key` (the URL defaults to the Tailscale address in
`src/llm/nim.py`; override it with `LOCAL_LLM_URL`). Without a key the local
fallback is disabled.

## Recommended — Cloudflare Workers AI fallback

Set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` (the token needs only
**Account > Workers AI > Read**) to give the risk auditor a free fallback behind
NIM and the execution judge a last resort behind Gemini, on a separate billing
rail. The `HF_RISK_MODEL` variable no longer exists.

## Optional — Claude Sonnet via subscription

To run the structural judge on Claude **Sonnet** through the Claude Agent SDK
(free ~$20/mo subscription credit instead of metered Haiku API tokens), add
`CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`). Tunables: `DOWTRADE_SDK_FOR`
(`positions` [default — Sonnet only while holding a live position] | `all` |
`none`), `CLAUDE_SDK_MONTHLY_CAP_USD` (default `20`), `CLAUDE_SDK_MODEL` (default
`sonnet`). With no token the bot runs Haiku-only, as before.

## Run and test

```bash
python -m src.main          # starts LiveRunner + dashboard on :8000
python -m pytest                                   # all tests
python -m pytest --cov=src --cov-report=term-missing
```
