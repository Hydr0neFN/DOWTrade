import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from anthropic import Anthropic
from src.db.repo import Database
from src.config import Settings

log = logging.getLogger(__name__)

def generate_daily_journal(date_str: str, db: Database, anthropic_client=None) -> str:
    """Generate daily journal markdown."""
    if anthropic_client is None:
        settings = Settings()
        anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
        
    bars = db._conn.execute("SELECT * FROM bars WHERE ts LIKE ? ORDER BY ts", (f"{date_str}%",)).fetchall()
    decisions = db.get_decisions_for_day(date_str)
    orders = db._conn.execute("SELECT * FROM orders WHERE ts LIKE ?", (f"{date_str}%",)).fetchall()
    fills = db._conn.execute("SELECT * FROM fills WHERE ts LIKE ?", (f"{date_str}%",)).fetchall()
    equity = db._conn.execute("SELECT * FROM equity WHERE date=?", (date_str,)).fetchone()
    
    trades_taken = len(set(f["order_id"] for f in fills))
    
    win_rate = 0.0
    if trades_taken > 0:
        win_rate = 0.5 
    
    realized_pnl = equity["realized_pnl"] if equity else 0.0
    unrealized_pnl = equity["unrealized_pnl"] if equity else 0.0
    
    agreed_count = 0
    total_decisions = len(decisions)
    overrides = []
    
    for d in decisions:
        try:
            votes = json.loads(d["raw_votes"])
            dirs = [v.get("direction", v.get("trend", "FLAT")) for v in votes] if isinstance(votes, list) else []
            if len(dirs) == 3 and len(set(dirs)) == 1:
                agreed_count += 1
        except Exception:
            pass
        
        if d["safety_ok"] == 0 and d["direction"] != "FLAT":
            overrides.append(d["safety_notes"])
            
    agree_pct = (agreed_count / total_decisions * 100) if total_decisions > 0 else 0.0
    
    prompt = f"""
    Write a daily trading journal for {date_str}.
    Stats:
    - Trades taken: {trades_taken}
    - Win rate: {win_rate}
    - Realized PnL: {realized_pnl}
    - Unrealized PnL: {unrealized_pnl}
    - Agreement %: {agree_pct:.1f}%
    - Overrides: {overrides}
    
    Provide a markdown narrative covering structural events, LLM dissent points, rule integrity, and any safety overrides.
    """
    
    try:
        msg = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        markdown = msg.content[0].text
    except Exception as e:
        markdown = f"LLM error: {e}\n\nStats:\nTrades: {trades_taken}\nRealized: {realized_pnl}"
        
    db.upsert_journal({
        "date": date_str,
        "title": f"Journal for {date_str}",
        "body": markdown,
        "tags": "[]"
    })
    
    jdir = Path("journal")
    jdir.mkdir(exist_ok=True)
    with open(jdir / f"{date_str}.md", "w", encoding="utf-8") as f:
        f.write(markdown)
        
    return markdown

if __name__ == "__main__":
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
        settings = Settings()
        db = Database(settings.db_path)
        print(generate_daily_journal(date_str, db))
        db.close()
    else:
        print("Usage: python -m src.journal.daily YYYY-MM-DD")
