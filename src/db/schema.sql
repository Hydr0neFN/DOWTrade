PRAGMA journal_mode=WAL;

-- ---------------------------------------------------------------
-- bars: raw OHLCV bars from the broker feed
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bars (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT    NOT NULL UNIQUE,   -- ISO-8601 UTC bar open time
    open      REAL    NOT NULL,
    high      REAL    NOT NULL,
    low       REAL    NOT NULL,
    close     REAL    NOT NULL,
    volume    INTEGER NOT NULL DEFAULT 0
);

-- ---------------------------------------------------------------
-- features: derived indicators computed on each bar
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_ts    TEXT    NOT NULL UNIQUE REFERENCES bars(ts),
    atr14     REAL,
    ema20     REAL,
    ema50     REAL,
    rsi14     REAL,
    vwap      REAL,
    extra_json TEXT    -- JSON blob for additional indicators
);

-- ---------------------------------------------------------------
-- llm_calls: raw request/response log for each LLM call
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_calls (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_ts       TEXT    NOT NULL REFERENCES bars(ts),
    model        TEXT    NOT NULL,   -- e.g. "claude-3-5-sonnet", "gemini-pro"
    prompt_hash  TEXT    NOT NULL,   -- sha256 of the prompt for dedup / audit
    raw_response TEXT    NOT NULL,
    parsed_json  TEXT,               -- cleaned structured output (JSON)
    latency_ms   INTEGER,
    cost_usd     REAL,
    error        TEXT,               -- NULL if successful
    created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ---------------------------------------------------------------
-- decisions: aggregated (voted) trade decision per bar
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS decisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    bar_ts       TEXT    NOT NULL UNIQUE REFERENCES bars(ts),
    direction    TEXT    NOT NULL CHECK (direction IN ('LONG','SHORT','FLAT')),
    confidence   REAL,               -- 0.0-1.0 aggregate confidence
    stop_price   REAL,
    entry_price  REAL,
    raw_votes    TEXT    NOT NULL,   -- JSON array of the three LLM votes
    safety_ok    INTEGER NOT NULL DEFAULT 0,  -- 1 if safety layer approved
    safety_notes TEXT,
    created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ---------------------------------------------------------------
-- orders: paper orders sent to Tradovate demo
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    decision_id  INTEGER REFERENCES decisions(id),
    broker_id    TEXT,               -- Tradovate order id (may be null before ack)
    symbol       TEXT    NOT NULL DEFAULT 'MYM',
    side         TEXT    NOT NULL CHECK (side IN ('BUY','SELL')),
    qty          INTEGER NOT NULL,
    order_type   TEXT    NOT NULL,   -- 'LIMIT', 'MARKET', 'STOP'
    limit_price  REAL,
    stop_price   REAL,
    status       TEXT    NOT NULL DEFAULT 'PENDING',
    -- PENDING | ACCEPTED | FILLED | PARTIALLY_FILLED | CANCELLED | REJECTED
    raw_response TEXT,
    updated_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ---------------------------------------------------------------
-- fills: individual fill events for orders
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fills (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id     INTEGER NOT NULL REFERENCES orders(id),
    broker_fill_id TEXT,
    ts           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    qty          INTEGER NOT NULL,
    price        REAL    NOT NULL,
    commission   REAL    NOT NULL DEFAULT 0.0
);

-- ---------------------------------------------------------------
-- equity: daily equity snapshots
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS equity (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         TEXT    NOT NULL UNIQUE,  -- YYYY-MM-DD
    start_equity REAL    NOT NULL,
    end_equity   REAL    NOT NULL,
    realized_pnl REAL    NOT NULL DEFAULT 0.0,
    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
    commission   REAL    NOT NULL DEFAULT 0.0,
    trade_count  INTEGER NOT NULL DEFAULT 0
);

-- ---------------------------------------------------------------
-- journal: one markdown-style note per trade or day
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS journal (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         TEXT    NOT NULL UNIQUE,  -- YYYY-MM-DD (natural key)
    title        TEXT    NOT NULL DEFAULT '',
    body         TEXT    NOT NULL DEFAULT '',
    tags         TEXT    NOT NULL DEFAULT '[]',   -- JSON array of strings
    updated_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ---------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_llm_calls_bar_ts   ON llm_calls(bar_ts);
CREATE INDEX IF NOT EXISTS idx_decisions_bar_ts   ON decisions(bar_ts);
CREATE INDEX IF NOT EXISTS idx_orders_ts          ON orders(ts);
CREATE INDEX IF NOT EXISTS idx_fills_order_id     ON fills(order_id);
