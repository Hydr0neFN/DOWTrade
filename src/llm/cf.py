"""
src/llm/cf.py
=============
Cloudflare Workers AI (free plan) as a last-resort provider.

Workers AI exposes the OpenAI chat-completions schema, so huggingface_hub's
InferenceClient works against it with a swapped base_url. Shared by the risk
auditor (second-tier fallback after NIM) and the execution judge (last resort
after Gemini). Without CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN there is no
client and the caller simply has no Cloudflare leg.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

from huggingface_hub import InferenceClient

log = logging.getLogger(__name__)

CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()

# Total per-request timeout (seconds) handed to huggingface_hub's InferenceClient.
# Without it a stalled Workers AI call has no bound at all.
CF_TIMEOUT_SEC = 30.0


def make_cf_client(owner: str) -> Optional[InferenceClient]:
    """A Workers AI client, or None (with a warning naming `owner`) if unconfigured."""
    if CF_ACCOUNT_ID and CF_API_TOKEN:
        return InferenceClient(
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1",
            token=CF_API_TOKEN,
            timeout=CF_TIMEOUT_SEC,
        )
    log.warning("[%s] Cloudflare fallback disabled "
                "(CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN unset)", owner)
    return None


def cf_chat(client, model: str, system: str, user: str) -> Tuple[str, int, int]:
    """One Workers AI completion -> (raw_text, in_tokens, out_tokens).

    Raises ValueError on empty content: reasoning-only models answer in
    `reasoning` and leave `content` empty, which is a failure for a JSON schema.
    """
    resp = client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=500,
        temperature=0.1,
    )
    raw = resp.choices[0].message.content or ""
    if not raw.strip():
        raise ValueError("empty content (reasoning-only response)")
    in_tok = getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0
    out_tok = getattr(getattr(resp, "usage", None), "completion_tokens", 0) or 0
    return raw, in_tok, out_tok
