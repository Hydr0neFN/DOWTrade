import os
import json
import re
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from src.dashboard import i18n
from fastapi.staticfiles import StaticFiles
from src.config import Settings
from src.db.repo import Database
from datetime import datetime
import time as _time

HEARTBEAT_PATH = "/tmp/dowtrade_yf_heartbeat"

def _read_heartbeat():
    try:
        with open(HEARTBEAT_PATH) as f:
            ts = int(f.read().strip())
            return {"last_poll_ts": ts, "age_sec": int(_time.time()) - ts}
    except Exception:
        return {"last_poll_ts": None, "age_sec": None}

def _parse_votes(raw):
    if not raw:
        return {}
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    return {}

def _order_ts_human(ts):
    """Order timestamps are ISO strings; show date + time, drop the seconds."""
    try:
        return datetime.fromisoformat(str(ts)).strftime("%m-%d %H:%M")
    except Exception:
        return str(ts)


def _direction(v):
    """Normalise an agent's vote to up / down / flat. None if it did not vote."""
    v = (v or "").lower()
    if not v:
        return None
    if "up" in v or "long" in v or "buy" in v:
        return "up"
    if "down" in v or "short" in v or "sell" in v:
        return "down"
    return "flat"


def _clean_model_id(raw) -> str:
    """Tidy a model id with rules only — never a name lookup, because a lookup
    table is exactly what went stale in the footer before.

    cli:gemini-3.1-pro-preview    → gemini-3.1-pro-preview
    deepseek-ai/DeepSeek-V3.2-Exp → DeepSeek-V3.2-Exp
    claude-haiku-4-5-20251001     → claude-haiku-4-5
    """
    s = str(raw or "").strip()
    s = re.sub(r"^[A-Za-z0-9_.-]+:", "", s)   # provider scheme, e.g. "cli:"
    s = s.split("/")[-1]                       # namespace, e.g. "deepseek-ai/"
    s = re.sub(r"\s*\([^)]*\)", "", s)          # mode suffix, e.g. " (High)"
    s = re.sub(r"[-_@]\d{8}$", "", s)           # snapshot date
    s = re.sub(r"[:-]latest$", "", s, flags=re.I)
    return s.strip()


def _active_models(db) -> list:
    """The models this bot actually called most recently.

    The footer used to carry a hand-written list, which drifted: the Gemini
    model is picked from a fallback chain at runtime and Claude answers on two
    tiers. Reading the last few hundred calls means it cannot go stale.
    """
    try:
        rows = db._conn.execute(
            "SELECT model FROM llm_calls ORDER BY rowid DESC LIMIT 300"
        ).fetchall()
    except Exception:
        return []
    out = []
    for r in rows:
        m = _clean_model_id(r["model"])
        if m and m not in out:
            out.append(m)
    return out[:4]


def _outcome(d):
    """Classify a decision into what actually happened to it.

    `safety_ok = 0` is written for two unrelated reasons and the dashboard used
    to render both as "SAFETY REJECTED", which is wrong for one of them:

      * the cross filter or guards.final_check refused the trade — it really was
        blocked; nothing executed;
      * DeepSeek disapproved — but DeepSeek has been **advisory** since it was
        demoted (`src/live/runner.py`, "it no longer gates execution"), so the
        trade may well have gone through anyway.

    Telling a reader a trade was rejected when it filled is worse than saying
    nothing, hence three outcomes instead of two.
    """
    if d.get("safety_ok") != 0:
        return "ok"
    notes = str(d.get("safety_notes") or "")
    # Blocks name their rule; DeepSeek writes a repr'd list of violations.
    if notes.startswith("[") or not notes:
        return "ds_flagged"
    return "blocked"


def _disagrees(d):
    """True when the agents genuinely pulled against each other on this bar.

    Two traps this avoids. The original version pooled all three votes into one
    set and asked whether it held more than one value — but Haiku and Gemini
    speak in directions while DeepSeek speaks in approved/veto, so every bar
    DeepSeek answered on scored as a disagreement (100 of the last 100).
    Counting flat as a third opinion is the other trap: Haiku reads the
    structural trend and Gemini decides the trade, so "trend down, no position"
    is two agents agreeing about different questions, not a conflict.

    So: only opposite committed directions count, plus a DeepSeek veto of a
    trade the other two actually proposed.
    """
    h = _direction((d.get("haiku") or {}).get("trend"))
    g = _direction((d.get("gemini") or {}).get("action"))

    if h and g and h != g and "flat" not in (h, g):
        return True

    ds = d.get("ds")
    if ds is not None and not ds.get("approved"):
        proposed = g or h
        return proposed is not None and proposed != "flat"
    return False


def _bar_ts_human(ts):
    try:
        return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(ts)


app = FastAPI(title="DOWTrade Dashboard")
settings = Settings()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── Language ──────────────────────────────────────────────────────────────────
# Default follows Accept-Language, which follows the OS/browser setting; the
# navbar toggle writes a cookie that overrides it.
def _lang(request: Request) -> str:
    return i18n.choose(
        cookie_val=request.cookies.get(i18n.COOKIE),
        query_val=request.query_params.get("lang"),
        accept_language=request.headers.get("accept-language"),
    )


def _asset_v(name: str) -> int:
    """mtime of a file in /static, appended to its URL as ?v=.

    /static is served with a long max-age, so without this a deploy leaves an
    already-open browser on the previous stylesheet — which is exactly what
    happened on 2026-09-09 and took a manual hard reload to clear.
    """
    try:
        return int((STATIC_DIR / name).stat().st_mtime)
    except OSError:
        return 0


def render(request: Request, name: str, context: dict):
    """TemplateResponse with the language helpers already in the context, so a
    page cannot accidentally be built monolingual."""
    lang = _lang(request)
    db = get_db()
    try:
        models = _active_models(db)
    finally:
        db.close()
    ctx = {"request": request, **i18n.make_helpers(lang),
           "js_i18n": json.dumps(i18n.js_table(lang)),
           "active_models": models,
           "theme_v": _asset_v("theme.css"), "app_v": _asset_v("app.js"),
           **context}
    return templates.TemplateResponse(request=request, name=name, context=ctx)


@app.get("/lang/{code}")
async def set_lang(code: str, request: Request):
    """Persist a language choice and return where the user came from."""
    lang = i18n.normalize(code) or i18n.DEFAULT_LANG
    nxt = request.query_params.get("next", "/")
    if not nxt.startswith("/") or nxt.startswith("//"):
        nxt = "/"
    resp = RedirectResponse(nxt, status_code=302)
    resp.set_cookie(i18n.COOKIE, lang, max_age=60 * 60 * 24 * 365,
                    samesite="lax", path="/")
    return resp

def get_db():
    return Database(settings.db_path)


def _snapshot(db):
    """(account, position, unrealized) from the latest equity row and all fills.
    Shared by the index page and /api/refresh, which previously carried two
    copies of this SQL that had to be kept in step by hand."""
    eq = db._conn.execute("SELECT * FROM equity ORDER BY date DESC LIMIT 1").fetchone()
    unrealized = eq["unrealized_pnl"] if eq else 0.0
    account = {
        "equity":       round(eq["end_equity"], 2)   if eq else 0.0,
        "start_equity": round(eq["start_equity"], 2) if eq else 0.0,
        "day_pnl":      round((eq["realized_pnl"] or 0.0) + (eq["unrealized_pnl"] or 0.0), 2) if eq else 0.0,
        "commission":   round(eq["commission"], 2)   if eq else 0.0,
        "trades":       int(eq["trade_count"])       if eq else 0,
        "as_of":        eq["date"]                   if eq else "\u2014",
    }

    pos_cur = db._conn.execute('''
        SELECT sum(CASE WHEN o.side = 'BUY' THEN f.qty ELSE -f.qty END) as qty,
               sum(f.price * f.qty) / sum(f.qty) as avg_price
        FROM fills f JOIN orders o ON f.order_id = o.id
    ''').fetchone()

    qty = pos_cur["qty"] if pos_cur and pos_cur["qty"] else 0
    side = "LONG" if qty > 0 else ("SHORT" if qty < 0 else "FLAT")
    avg_price = pos_cur["avg_price"] if pos_cur and pos_cur["avg_price"] else 0.0

    position = {
        "qty": abs(qty),
        "side": side,
        "avg_price": round(avg_price, 2) if avg_price else 0.0,
        "unrealized": round(unrealized, 2),
    }
    realized = eq["realized_pnl"] if eq else 0.0
    return account, position, realized

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    db = get_db()
    account, position, today_realized = _snapshot(db)

    # Recent orders — this was a standalone /trades page whose rows were built
    # by string concatenation in app.js. Rendered here with everything else.
    trades = [
        {
            "ts_human":   _order_ts_human(r["ts"]),
            "symbol":     r["symbol"],
            "side":       r["side"],
            "qty":        r["qty"],
            "status":     r["status"],
            "fill_price": r["fill_price"],
        }
        for r in db._conn.execute('''
            SELECT o.ts, o.symbol, o.side, o.qty, o.status, f.price AS fill_price
            FROM orders o
            LEFT JOIN fills f ON o.id = f.order_id
            ORDER BY o.ts DESC
            LIMIT 25
        ''').fetchall()
    ]
    db.close()

    return render(request, "index.html",
                  {"position": position, "today_realized": today_realized,
                   "trades": trades, "account": account})


# The equity curve, the order list and the journal are now sections of the two
# remaining pages. Keep the old URLs working for anything bookmarked.
@app.get("/equity")
async def equity_redirect():
    return RedirectResponse("/", status_code=307)


@app.get("/trades")
async def trades_redirect():
    return RedirectResponse("/", status_code=307)


@app.get("/journal")
async def journal_redirect():
    return RedirectResponse("/decisions", status_code=307)


@app.get("/disagreements")
async def disagreements_redirect():
    """Renamed: the page shows the whole chain, not only the disagreements."""
    return RedirectResponse("/decisions", status_code=307)


@app.get("/decisions", response_class=HTMLResponse)
async def decisions(request: Request):
    db = get_db()
    rows = db._conn.execute("SELECT * FROM decisions ORDER BY bar_ts DESC LIMIT 100").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        votes = _parse_votes(d.get("raw_votes"))
        d["haiku"]  = votes.get("haiku")  if isinstance(votes, dict) else None
        d["gemini"] = votes.get("gemini") if isinstance(votes, dict) else None
        d["ds"]     = votes.get("ds")     if isinstance(votes, dict) else None
        d["bar_ts_human"] = _bar_ts_human(d.get("bar_ts"))
        # Precomputed so the client-side filter is a single attribute check.
        d["disagree"] = _disagrees(d)
        d["outcome"] = _outcome(d)
        out.append(d)
    journals = db._conn.execute("SELECT date FROM journal ORDER BY date DESC").fetchall()
    db.close()
    return render(request, "decisions.html",
                  {"decisions": out, "journals": journals})


@app.get("/api/refresh")
async def api_refresh():
    db = get_db()
    account, position, today_realized = _snapshot(db)
    last_d = db._conn.execute(
        "SELECT direction, bar_ts FROM decisions ORDER BY bar_ts DESC LIMIT 1"
    ).fetchone()
    db.close()

    return {
        "position": position,
        "today_realized": today_realized,
        "last_decision": dict(last_d) if last_d else None,
        "account": account,
        "heartbeat": _read_heartbeat(),
    }


@app.get("/api/equity")
async def api_equity():
    db = get_db()
    rows = db._conn.execute("SELECT date, end_equity FROM equity ORDER BY date DESC LIMIT 30").fetchall()
    db.close()
    return [{"ts": r["date"], "balance": r["end_equity"]} for r in reversed(rows)]

@app.get("/api/trades")
async def api_trades(page: int = 1):
    db = get_db()
    limit = 50
    offset = (page - 1) * limit
    rows = db._conn.execute(f'''
        SELECT o.id, o.ts, o.symbol, o.side, o.qty, o.status, 
               f.price as fill_price, d.id as decision_id, d.raw_votes
        FROM orders o
        LEFT JOIN fills f ON o.id = f.order_id
        LEFT JOIN decisions d ON o.decision_id = d.id
        ORDER BY o.ts DESC
        LIMIT {limit} OFFSET {offset}
    ''').fetchall()
    
    out = []
    for r in rows:
        d = dict(r)
        d["llm_calls"] = []
        if d["decision_id"]:
            bar_ts = db._conn.execute("SELECT bar_ts FROM decisions WHERE id=?", (d["decision_id"],)).fetchone()
            if bar_ts:
                lc = db._conn.execute("SELECT model, raw_response FROM llm_calls WHERE bar_ts=?", (bar_ts["bar_ts"],)).fetchall()
                d["llm_calls"] = [dict(c) for c in lc]
        out.append(d)
    db.close()
    return {"trades": out}

@app.get("/api/journal/{date}")
async def api_journal(date: str):
    db = get_db()
    row = db._conn.execute("SELECT body FROM journal WHERE date=?", (date,)).fetchone()
    db.close()
    if not row:
        return {"markdown": ""}
    return {"markdown": row["body"]}
