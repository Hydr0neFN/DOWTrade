"""
src/llm/claude_sdk.py
=====================
Claude **Sonnet** via the Claude Agent SDK / `claude` CLI, billed against a
Pro-subscription OAuth token (CLAUDE_CODE_OAUTH_TOKEN, produced once with
`claude setup-token`) instead of metered Anthropic API tokens.

`HaikuStructural` uses this as its primary path and falls back to the metered
Anthropic API (Haiku) whenever the SDK is disabled, no OAuth token is present,
the shared monthly credit cap is reached, or any SDK call errors. With no token
configured the bot behaves exactly as before (Haiku-only), so this is safe to
ship before the subscription login is completed.

The monthly-spend ledger is SHARED with the sibling bot (trader.py) via one
file (~/.claude_sdk_credit.json) so both draw down the same ~$20 subscription
pool rather than $20 each.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def _load_oauth_token() -> str:
    """Find the subscription OAuth token.

    Order: process env (systemd EnvironmentFile injects it) → the bot's own
    .env → ~/.env. Avoids a hard dependency on python-dotenv and works under
    manual runs as well as systemd.
    """
    tok = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if tok:
        return tok
    for env_path in (Path(__file__).resolve().parents[2] / ".env", Path.home() / ".env"):
        try:
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("CLAUDE_CODE_OAUTH_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            continue
    return ""


SDK_ENABLED = os.environ.get("CLAUDE_SDK_ENABLED", "1").lower() in ("1", "true", "yes")
OAUTH_TOKEN = _load_oauth_token()
SDK_MODEL   = os.environ.get("CLAUDE_SDK_MODEL", "sonnet")
CLI_PATH    = os.environ.get("CLAUDE_CLI_PATH", "claude")
SDK_TIMEOUT = int(os.environ.get("CLAUDE_SDK_TIMEOUT", "45"))
CAP_USD     = float(os.environ.get("CLAUDE_SDK_MONTHLY_CAP_USD", "20"))
CREDIT_FILE = Path(os.environ.get(
    "CLAUDE_SDK_CREDIT_FILE", str(Path.home() / ".claude_sdk_credit.json")))

# Sonnet 4.x list pricing (USD/token) — used to value subscription usage against
# the monthly cap when the CLI reports total_cost_usd == 0 (subscription mode).
SONNET_IN_USD, SONNET_OUT_USD = 3e-6, 15e-6


def _month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _spent() -> float:
    try:
        return float(json.loads(CREDIT_FILE.read_text()).get(_month(), 0.0))
    except Exception:
        return 0.0


def _add(cost_usd: float) -> None:
    month = _month()
    try:
        ledger = json.loads(CREDIT_FILE.read_text())
    except Exception:
        ledger = {}
    ledger[month] = round(float(ledger.get(month, 0.0)) + max(0.0, cost_usd), 6)
    try:  # atomic write so the concurrent (trader) writer can't read a half file
        tmp = CREDIT_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(ledger))
        os.replace(tmp, CREDIT_FILE)
    except Exception as exc:
        log.warning("[sdk] credit ledger persist failed: %s", exc)


def sdk_available() -> bool:
    """True when the Sonnet-via-subscription path should be attempted."""
    return SDK_ENABLED and bool(OAUTH_TOKEN) and _spent() < CAP_USD


def sdk_complete(system: str, user: str, max_tokens: int = 400) -> tuple:
    """One-shot Claude Sonnet call via the `claude` CLI (subscription auth).

    Runs from /tmp with ANTHROPIC_API_KEY stripped so the CLI uses the OAuth
    subscription token (the $20 credit) rather than metered API billing.

    Returns (raw_text, in_tokens, out_tokens, cost_usd). Raises on any error so
    the caller can fall back to the API.
    """
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["CLAUDE_CODE_OAUTH_TOKEN"] = OAUTH_TOKEN
    proc = subprocess.run(
        [CLI_PATH, "-p", user,
         "--model", SDK_MODEL,
         "--system-prompt", system,
         "--output-format", "json",
         "--max-turns", "1",
         "--allowedTools", "",
         "--no-session-persistence",
         "--permission-mode", "default"],
        capture_output=True, text=True, timeout=SDK_TIMEOUT, cwd="/tmp", env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude exit {proc.returncode}: {proc.stderr[-200:]}")
    payload = json.loads(proc.stdout)
    if payload.get("is_error") or payload.get("subtype") != "success":
        raise RuntimeError(f"claude error: {str(payload.get('result'))[:200]}")
    text = (payload.get("result") or "").strip()
    if not text:
        raise RuntimeError("claude empty result")
    usage = payload.get("usage", {}) or {}
    in_tok = usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    cost = float(payload.get("total_cost_usd") or 0.0)
    if cost <= 0:  # subscription mode often reports 0 — value it via token usage
        cost = in_tok * SONNET_IN_USD + out_tok * SONNET_OUT_USD
    _add(cost)
    return text, in_tok, out_tok, cost
