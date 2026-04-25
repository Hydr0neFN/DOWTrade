import os
import json
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.config import Settings
from src.db.repo import Database

app = FastAPI(title="DOWTrade Dashboard")
settings = Settings()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

def get_db():
    return Database(settings.db_path)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    db = get_db()
    
    eq = db._conn.execute("SELECT * FROM equity ORDER BY date DESC LIMIT 1").fetchone()
    today_realized = eq["realized_pnl"] if eq else 0.0
    unrealized = eq["unrealized_pnl"] if eq else 0.0
    
    pos_cur = db._conn.execute('''
        SELECT sum(CASE WHEN o.side = 'BUY' THEN f.qty ELSE -f.qty END) as qty,
               sum(f.price * f.qty) / sum(f.qty) as avg_price
        FROM fills f JOIN orders o ON f.order_id = o.id
    ''').fetchone()
    
    qty = pos_cur["qty"] if pos_cur and pos_cur["qty"] else 0
    side = "FLAT"
    if qty > 0: side = "LONG"
    elif qty < 0: side = "SHORT"
    avg_price = pos_cur["avg_price"] if pos_cur and pos_cur["avg_price"] else 0.0

    position = {
        "qty": abs(qty),
        "side": side,
        "avg_price": round(avg_price, 2) if avg_price else 0.0,
        "unrealized": round(unrealized, 2)
    }
    
    last_decisions = db._conn.execute('''
        SELECT d.*, lc.raw_response, lc.model 
        FROM decisions d
        LEFT JOIN llm_calls lc ON d.bar_ts = lc.bar_ts
        ORDER BY d.bar_ts DESC LIMIT 9
    ''').fetchall()
    
    d_map = {}
    for row in last_decisions:
        did = row["id"]
        if did not in d_map:
            d_map[did] = dict(row)
            d_map[did]["llm_calls"] = []
        if row["model"]:
            d_map[did]["llm_calls"].append({"model": row["model"], "raw_response": row["raw_response"]})
    
    decisions_list = list(d_map.values())[:3]
    db.close()
    
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request, "position": position, "today_realized": today_realized, "decisions": decisions_list})

@app.get("/equity", response_class=HTMLResponse)
async def equity(request: Request):
    return templates.TemplateResponse(request=request, name="equity.html", context={"request": request})

@app.get("/trades", response_class=HTMLResponse)
async def trades(request: Request):
    return templates.TemplateResponse(request=request, name="trades.html", context={"request": request})

@app.get("/journal", response_class=HTMLResponse)
async def journal(request: Request):
    db = get_db()
    rows = db._conn.execute("SELECT * FROM journal ORDER BY date DESC").fetchall()
    db.close()
    return templates.TemplateResponse(request=request, name="journal.html", context={"request": request, "journals": rows})

@app.get("/disagreements", response_class=HTMLResponse)
async def disagreements(request: Request):
    db = get_db()
    rows = db._conn.execute("SELECT * FROM decisions ORDER BY bar_ts DESC LIMIT 100").fetchall()
    disagreements_list = []
    for r in rows:
        votes = json.loads(r["raw_votes"]) if r["raw_votes"] else []
        dirs = [v.get("direction", v.get("trend", "FLAT")) for v in votes] if isinstance(votes, list) else []
        if len(set(dirs)) > 1:
            disagreements_list.append(r)
    db.close()
    return templates.TemplateResponse(request=request, name="disagreements.html", context={"request": request, "decisions": disagreements_list})

@app.get("/api/refresh")
async def api_refresh():
    db = get_db()
    eq = db._conn.execute("SELECT * FROM equity ORDER BY date DESC LIMIT 1").fetchone()
    today_realized = eq["realized_pnl"] if eq else 0.0
    unrealized = eq["unrealized_pnl"] if eq else 0.0
    
    pos_cur = db._conn.execute('''
        SELECT sum(CASE WHEN o.side = 'BUY' THEN f.qty ELSE -f.qty END) as qty,
               sum(f.price * f.qty) / sum(f.qty) as avg_price
        FROM fills f JOIN orders o ON f.order_id = o.id
    ''').fetchone()
    
    qty = pos_cur["qty"] if pos_cur and pos_cur["qty"] else 0
    side = "FLAT"
    if qty > 0: side = "LONG"
    elif qty < 0: side = "SHORT"
    avg_price = pos_cur["avg_price"] if pos_cur and pos_cur["avg_price"] else 0.0
    
    last_d = db._conn.execute("SELECT direction, bar_ts FROM decisions ORDER BY bar_ts DESC LIMIT 1").fetchone()
    db.close()
    
    return {
        "position": {
            "qty": abs(qty),
            "side": side,
            "avg_price": round(avg_price, 2) if avg_price else 0.0,
            "unrealized": round(unrealized, 2)
        },
        "today_realized": today_realized,
        "last_decision": dict(last_d) if last_d else None
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
