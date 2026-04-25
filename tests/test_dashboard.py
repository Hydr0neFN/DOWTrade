import pytest
from fastapi.testclient import TestClient
from src.dashboard.app import app
from src.config import Settings
from src.db.repo import Database
import tempfile

@pytest.fixture
def mock_db_path(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    
    db = Database(path)
    db._conn.execute("INSERT INTO equity (date, start_equity, end_equity, realized_pnl, unrealized_pnl) VALUES ('2026-04-25', 10000, 10100, 100, 0)")
    db._conn.execute("INSERT INTO bars (ts, open, high, low, close) VALUES ('2026-04-25T10:00:00Z', 100, 105, 95, 102)")
    db._conn.execute("INSERT INTO decisions (bar_ts, direction, confidence, raw_votes, safety_ok) VALUES ('2026-04-25T10:00:00Z', 'LONG', 0.9, '[{\"direction\": \"LONG\"}, {\"direction\": \"LONG\"}, {\"direction\": \"LONG\"}]', 1)")
    db._conn.execute("INSERT INTO orders (id, ts, decision_id, symbol, side, qty, order_type, status) VALUES (1, '2026-04-25T10:01:00Z', 1, 'MYM', 'BUY', 1, 'MARKET', 'FILLED')")
    db._conn.execute("INSERT INTO fills (order_id, ts, qty, price) VALUES (1, '2026-04-25T10:01:05Z', 1, 102.5)")
    db._conn.execute("INSERT INTO llm_calls (bar_ts, model, prompt_hash, raw_response, parsed_json) VALUES ('2026-04-25T10:00:00Z', 'haiku', 'hash', 'I am long', '{}')")
    db.close()

    monkeypatch.setattr("src.dashboard.app.settings.db_path", path)
    return path

client = TestClient(app)

def test_index(mock_db_path):
    response = client.get("/")
    assert response.status_code == 200
    assert "Current Position" in response.text

def test_equity(mock_db_path):
    response = client.get("/equity")
    assert response.status_code == 200

def test_trades(mock_db_path):
    response = client.get("/trades")
    assert response.status_code == 200

def test_journal(mock_db_path):
    response = client.get("/journal")
    assert response.status_code == 200

def test_disagreements(mock_db_path):
    response = client.get("/disagreements")
    assert response.status_code == 200

def test_api_refresh(mock_db_path):
    response = client.get("/api/refresh")
    assert response.status_code == 200
    data = response.json()
    assert "position" in data
    assert data["position"]["qty"] == 1
    assert data["position"]["side"] == "LONG"
    assert data["today_realized"] == 100.0

def test_api_trades(mock_db_path):
    response = client.get("/api/trades?page=1")
    assert response.status_code == 200
    assert len(response.json()["trades"]) == 1
