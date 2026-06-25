"""
Repository layer for DOWTrade Bot.

Thin SQLite wrapper -- no ORM.  All SQL is parameterized.
Accepts plain dicts (or any Mapping) for inserts; returns sqlite3.Row objects
for reads so callers can use both positional and column-name access.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


# ---------------------------------------------------------------------------
# Module-level convenience helper
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> "Database":
    """Open (or create) the database and apply the schema. Returns Database."""
    return Database(db_path)


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    """Manages a single SQLite connection with WAL mode and Row factory."""

    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        # The runner thread and the dashboard's per-request connections write to
        # the same file. WAL allows concurrent readers but only one writer; with
        # the default busy_timeout=0 a second writer hits SQLITE_BUSY immediately
        # ("database is locked"), which the runner's broad except swallows
        # (dropping that bar's persistence). Wait up to 5s for the lock instead.
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._apply_schema()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _apply_schema(self) -> None:
        """Execute schema.sql idempotently (all statements use IF NOT EXISTS)."""
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        # sqlite3 executescript handles multi-statement SQL
        self._conn.executescript(sql)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute(
        self,
        sql: str,
        params: Sequence[Any] | Mapping[str, Any] = (),
    ) -> sqlite3.Cursor:
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return cur

    # ------------------------------------------------------------------
    # bars
    # ------------------------------------------------------------------

    def insert_bar(self, bar: Mapping[str, Any]) -> int:
        """Insert a bar row. Returns the new rowid."""
        sql = """
            INSERT OR IGNORE INTO bars (ts, open, high, low, close, volume)
            VALUES (:ts, :open, :high, :low, :close, :volume)
        """
        cur = self._execute(sql, bar)
        return cur.lastrowid  # type: ignore[return-value]

    def get_latest_bars(self, n: int = 50) -> list[sqlite3.Row]:
        """Return the n most-recent bars ordered oldest-first."""
        sql = """
            SELECT * FROM bars
            ORDER BY ts DESC
            LIMIT ?
        """
        rows = self._conn.execute(sql, (n,)).fetchall()
        # Reverse so caller gets chronological order
        return list(reversed(rows))

    # ------------------------------------------------------------------
    # features
    # ------------------------------------------------------------------

    def insert_features(self, features: Mapping[str, Any]) -> int:
        sql = """
            INSERT OR REPLACE INTO features
                (bar_ts, atr14, ema20, ema50, rsi14, vwap, extra_json)
            VALUES
                (:bar_ts, :atr14, :ema20, :ema50, :rsi14, :vwap, :extra_json)
        """
        cur = self._execute(sql, features)
        return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # llm_calls
    # ------------------------------------------------------------------

    def insert_llm_call(self, call: Mapping[str, Any]) -> int:
        sql = """
            INSERT INTO llm_calls
                (bar_ts, model, prompt_hash, raw_response, parsed_json,
                 latency_ms, cost_usd, error)
            VALUES
                (:bar_ts, :model, :prompt_hash, :raw_response, :parsed_json,
                 :latency_ms, :cost_usd, :error)
        """
        cur = self._execute(sql, call)
        return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # decisions
    # ------------------------------------------------------------------

    def insert_decision(self, decision: Mapping[str, Any]) -> int:
        sql = """
            INSERT OR REPLACE INTO decisions
                (bar_ts, direction, confidence, stop_price, entry_price,
                 raw_votes, safety_ok, safety_notes)
            VALUES
                (:bar_ts, :direction, :confidence, :stop_price, :entry_price,
                 :raw_votes, :safety_ok, :safety_notes)
        """
        cur = self._execute(sql, decision)
        return cur.lastrowid  # type: ignore[return-value]

    def get_decisions_for_day(self, date_str: str) -> list[sqlite3.Row]:
        """Return all decisions on the ET calendar day date_str. bar_ts is stored
        as UNIX-epoch-second strings (Bar.ts is epoch), so range-scan the day's
        epoch window rather than LIKE-matching a date prefix that never appears."""
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        day = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=ZoneInfo("America/New_York"))
        lo, hi = str(int(day.timestamp())), str(int((day + timedelta(days=1)).timestamp()))
        sql = "SELECT * FROM decisions WHERE bar_ts >= ? AND bar_ts < ? ORDER BY bar_ts ASC"
        return self._conn.execute(sql, (lo, hi)).fetchall()

    # ------------------------------------------------------------------
    # orders
    # ------------------------------------------------------------------

    def insert_order(self, order: Mapping[str, Any]) -> int:
        sql = """
            INSERT INTO orders
                (ts, decision_id, broker_id, symbol, side, qty,
                 order_type, limit_price, stop_price, status, raw_response)
            VALUES
                (:ts, :decision_id, :broker_id, :symbol, :side, :qty,
                 :order_type, :limit_price, :stop_price, :status, :raw_response)
        """
        cur = self._execute(sql, order)
        return cur.lastrowid  # type: ignore[return-value]

    def update_order_status(
        self,
        order_id: int,
        status: str,
        broker_id: Optional[str] = None,
        raw_response: Optional[str] = None,
    ) -> None:
        sql = """
            UPDATE orders
            SET status       = ?,
                broker_id    = COALESCE(?, broker_id),
                raw_response = COALESCE(?, raw_response),
                updated_at   = strftime('%Y-%m-%dT%H:%M:%SZ','now')
            WHERE id = ?
        """
        self._execute(sql, (status, broker_id, raw_response, order_id))

    # ------------------------------------------------------------------
    # fills
    # ------------------------------------------------------------------

    def insert_fill(self, fill: Mapping[str, Any]) -> int:
        sql = """
            INSERT INTO fills
                (order_id, broker_fill_id, ts, qty, price, commission)
            VALUES
                (:order_id, :broker_fill_id, :ts, :qty, :price, :commission)
        """
        cur = self._execute(sql, fill)
        return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # equity
    # ------------------------------------------------------------------

    def insert_equity(self, equity: Mapping[str, Any]) -> int:
        sql = """
            INSERT OR REPLACE INTO equity
                (date, start_equity, end_equity, realized_pnl,
                 unrealized_pnl, commission, trade_count)
            VALUES
                (:date, :start_equity, :end_equity, :realized_pnl,
                 :unrealized_pnl, :commission, :trade_count)
        """
        cur = self._execute(sql, equity)
        return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # journal
    # ------------------------------------------------------------------

    def upsert_journal(self, entry: Mapping[str, Any]) -> int:
        """Insert or replace a journal entry keyed on date."""
        sql = """
            INSERT INTO journal (date, title, body, tags, updated_at)
            VALUES (:date, :title, :body, :tags, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            ON CONFLICT(date) DO UPDATE SET
                title      = excluded.title,
                body       = excluded.body,
                tags       = excluded.tags,
                updated_at = excluded.updated_at
        """
        cur = self._execute(sql, entry)
        return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Commit any pending transaction and close the connection."""
        self._conn.commit()
        self._conn.close()
