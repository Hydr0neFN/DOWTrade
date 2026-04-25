import pytest
import tempfile
from unittest.mock import MagicMock
from src.db.repo import Database
from src.journal.daily import generate_daily_journal

def test_generate_daily_journal():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
        
    db = Database(path)
    db._conn.execute("INSERT INTO bars (ts, open, high, low, close) VALUES ('2026-04-25T10:00:00Z', 100, 105, 95, 102)")
    db._conn.execute("INSERT INTO decisions (bar_ts, direction, confidence, raw_votes, safety_ok) VALUES ('2026-04-25T10:00:00Z', 'LONG', 0.9, '[{\"direction\": \"LONG\"}, {\"direction\": \"LONG\"}, {\"direction\": \"LONG\"}]', 1)")
    db._conn.execute("INSERT INTO orders (id, ts, decision_id, symbol, side, qty, order_type, status) VALUES (1, '2026-04-25T10:01:00Z', 1, 'MYM', 'BUY', 1, 'MARKET', 'FILLED')")
    db._conn.execute("INSERT INTO fills (order_id, ts, qty, price) VALUES (1, '2026-04-25T10:01:05Z', 1, 102.5)")
    db._conn.execute("INSERT INTO equity (date, start_equity, end_equity, realized_pnl, unrealized_pnl) VALUES ('2026-04-25', 10000, 10100, 100, 0)")
    db.close()

    db = Database(path)
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text="Mocked markdown response")]
    mock_client.messages.create.return_value = mock_msg

    markdown = generate_daily_journal("2026-04-25", db, anthropic_client=mock_client)
    
    assert "Mocked markdown response" in markdown
    
    row = db._conn.execute("SELECT * FROM journal WHERE date='2026-04-25'").fetchone()
    assert row is not None
    assert row["body"] == "Mocked markdown response"
    db.close()
