import asyncio
import logging
import math
import os
from datetime import datetime
from typing import Optional

import yfinance as yf

from src.config import Settings
from src.db.repo import Database
from src.broker.base import Bar
from src.data.bars import Bar as DataBar
from src.broker.models import Order, Position, AccountState
from src.broker.tastytrade import TastytradeBroker, dxfeed_symbol
from src.live.dxlink import DxLinkStreamer
from src.live.yfinance_poller import YFinancePoller

from src.data.bars import BarWindow
from src.data.features import MarketSnapshot, build_snapshot
from src.data.cross_filter import CrossFilter
from src.llm.base import CostTracker, CostBudgetExceeded, LLMCallResult
from src.llm.haiku_structural import HaikuStructural
from src.llm.gemini_execution import GeminiExecution
from src.llm.deepseek_risk import DeepSeekRisk
from src.backtest.harness import final_check, compute_size, _PositionState
from src.broker.models import AccountState, Position
from zoneinfo import ZoneInfo
import json
import uuid

SIM_FILLS = os.environ.get("SIM_FILLS", "1") == "1"
MAX_POSITIONS = int(os.environ.get("MAX_POSITIONS", "5"))
ET = ZoneInfo("America/New_York")


log = logging.getLogger(__name__)

class LiveRunner:
    # Use yfinance for market data when dxLink cert has no live data subscription.
    # Default ON. Set USE_YFINANCE=0 in .env to revert to dxLink streaming.
    USE_YFINANCE = os.environ.get("USE_YFINANCE", "1") == "1"

    # Minimum bars in window before we run the LLM pipeline. SMA-200 is the
    # binding indicator. With yfinance hydration this is met immediately on boot.
    MIN_WARMUP_BARS = 200

    def __init__(self):
        self.settings = Settings()
        self.db = Database(self.settings.db_path)
        self.broker = TastytradeBroker(self.settings)
        self.tracker = CostTracker()
        
        self.haiku = HaikuStructural(self.settings.anthropic_api_key, self.tracker, db=self.db)
        self.gemini = GeminiExecution(self.settings.google_api_key, self.tracker, db=self.db)
        self.deepseek = DeepSeekRisk(self.settings.huggingface_api_key, self.tracker, db=self.db)
        
        self.window = BarWindow(maxlen=500)
        self.streamer: Optional[DxLinkStreamer] = None
        self._budget_exceeded = False
        self._last_day = None
        
        self.candle_queue = asyncio.Queue()

        # Sim-fill state (active when SIM_FILLS=1)
        self._positions: list[_PositionState] = []
        self._cash: float = 0.0
        self._start_equity_today: float = 0.0
        self._realized_pnl_today: float = 0.0
        self._trade_count: int = 0
        self._cross = CrossFilter()

    def _on_candle(self, symbol: str, candle: dict):
        self.candle_queue.put_nowait(candle)

    async def _hydrate_window(self):
        if self.USE_YFINANCE:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None, lambda: yf.download("MYM=F", period="5d", interval="15m", progress=False)
            )
            if not df.empty:
                if hasattr(df.columns, "get_level_values"):
                    df.columns = df.columns.get_level_values(0)
                for row in df.itertuples():
                    try:
                        o = float(row.Open); h = float(row.High)
                        lo = float(row.Low); c = float(row.Close)
                        if any(math.isnan(x) for x in (o, h, lo, c)):
                            continue
                        v_f = float(row.Volume or 0)
                        v = 0 if math.isnan(v_f) else int(v_f)
                        ts = int(row.Index.timestamp())
                        bar = DataBar(ts=ts, o=o, h=h, l=lo, c=c, v=v)
                        self.window.append(bar)
                        self.db.insert_bar({"ts": str(ts), "open": o, "high": h, "low": lo, "close": c, "volume": v})
                    except Exception:
                        continue
        else:
            end_ts = int(datetime.now().timestamp())
            start_ts = end_ts - 5 * 24 * 3600
            bars = self.broker.fetch_historical_bars("MYM", start_ts, end_ts)
            for b in bars:
                self.window.append(DataBar(ts=b.ts_utc, o=b.open, h=b.high, l=b.low, c=b.close, v=int(b.volume)))
        log.info("Hydrated BarWindow with %d bars", len(self.window))
        self._cross.bulk_load(self.window.as_list())

    def _ensure_sim_table(self):
        """Create sim_state table if not exists."""
        import sqlite3
        conn = sqlite3.connect(self.settings.db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS sim_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            cash REAL NOT NULL,
            pos_side TEXT,
            pos_qty INTEGER DEFAULT 0,
            pos_avg_price REAL DEFAULT 0,
            pos_stop REAL DEFAULT 0,
            pos_entry_ts INTEGER DEFAULT 0,
            realized_pnl_today REAL DEFAULT 0,
            trade_count INTEGER DEFAULT 0,
            start_equity_today REAL DEFAULT 0,
            updated_at TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS sim_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            side TEXT NOT NULL,
            qty INTEGER NOT NULL,
            avg_price REAL NOT NULL,
            current_stop REAL NOT NULL,
            entry_ts INTEGER NOT NULL
        )""")
        conn.commit()
        conn.close()

    def _save_sim_state(self):
        """Persist sim state + all positions to DB."""
        import sqlite3
        conn = sqlite3.connect(self.settings.db_path)
        conn.execute("""INSERT OR REPLACE INTO sim_state
            (id, cash, pos_side, pos_qty, pos_avg_price, pos_stop, pos_entry_ts,
             realized_pnl_today, trade_count, start_equity_today, updated_at)
            VALUES (1, ?, NULL, 0, 0, 0, 0, ?, ?, ?, datetime('now'))""",
            (self._cash, self._realized_pnl_today, self._trade_count,
             self._start_equity_today))
        conn.execute("DELETE FROM sim_positions")
        for pos in self._positions:
            conn.execute("""INSERT INTO sim_positions (side, qty, avg_price, current_stop, entry_ts)
                VALUES (?, ?, ?, ?, ?)""",
                (pos.side, pos.qty, pos.avg_price, pos.current_stop, pos.entry_ts))
        conn.commit()
        conn.close()

        conn.close()

    def _load_sim_state(self) -> bool:
        """Load sim state + positions from DB."""
        import sqlite3
        conn = sqlite3.connect(self.settings.db_path)
        row = conn.execute("SELECT * FROM sim_state WHERE id=1").fetchone()
        if row is None:
            conn.close()
            return False
        self._cash = row[1]
        self._start_equity_today = row[9]
        self._realized_pnl_today = row[7]
        self._trade_count = row[8]
        self._positions = []
        for pr in conn.execute("SELECT side, qty, avg_price, current_stop, entry_ts FROM sim_positions").fetchall():
            self._positions.append(_PositionState(
                side=pr[0], qty=pr[1], avg_price=pr[2],
                current_stop=pr[3], pyramid_adds_used=0, entry_ts=pr[4]))
        conn.close()
        log.info("Loaded sim state: cash=%.2f positions=%d realized=%.2f",
                 self._cash, len(self._positions), self._realized_pnl_today)
        for pos in self._positions:
            log.info("  pos: %s qty=%d entry=%.2f stop=%.2f", pos.side, pos.qty, pos.avg_price, pos.current_stop)
        return True


    def _reset_daily_state(self):
        now = datetime.now()
        day_str = now.strftime("%Y-%m-%d")
        if self._last_day != day_str:
            if now.hour >= 17 or self._last_day is None:
                self._budget_exceeded = False
                self._last_day = day_str
                self._realized_pnl_today = 0.0
                self._trade_count = 0
                self._start_equity_today = self._cash
                log.info("Reset daily state for %s (start_equity=%.2f)", day_str, self._start_equity_today)

    async def _process_loop(self):
        while True:
            candle = await self.candle_queue.get()
            try:
                self._reset_daily_state()
                
                ts = candle.get("time", 0) // 1000

                vol_raw = candle.get("volume", 0)
                try:
                    vol_f = float(vol_raw)
                    import math
                    vol_int = 0 if math.isnan(vol_f) else int(vol_f)
                except (ValueError, TypeError):
                    vol_int = 0
                    
                bar = DataBar(
                    ts=ts,
                    o=candle.get("open", 0.0),
                    h=candle.get("high", 0.0),
                    l=candle.get("low", 0.0),
                    c=candle.get("close", 0.0),
                    v=vol_int
                )
                
                if len(self.window) > 0 and bar.ts <= self.window.as_list()[-1].ts:
                    continue
                    
                self.window.append(bar)
                self._cross.update(bar)
                self.db.insert_bar({"ts": str(bar.ts), "open": bar.o, "high": bar.h, "low": bar.l, "close": bar.c, "volume": bar.v})

                # Sim stop-check on every bar (all positions)
                if SIM_FILLS and self._positions:
                    stopped = []
                    for i, pos in enumerate(self._positions):
                        hit = False
                        if pos.side == "long" and bar.l <= pos.current_stop:
                            hit = True
                        elif pos.side == "short" and bar.h >= pos.current_stop:
                            hit = True
                        if hit:
                            # Model a gap-through: if the bar opened beyond the
                            # stop, fill at the open (the realistic price), not the
                            # stop, so paper P&L isn't optimistically biased.
                            if pos.side == "long":
                                fill_price = min(pos.current_stop, bar.o)
                            else:
                                fill_price = max(pos.current_stop, bar.o)
                            pnl = pos.unrealized_pnl(fill_price)
                            self._cash += pnl
                            self._realized_pnl_today += pnl
                            # fills.order_id is NOT NULL REFERENCES orders(id) with
                            # foreign_keys=ON, so order_id=0 raised IntegrityError and
                            # the stop exit was silently dropped from the DB (dashboard
                            # kept showing the closed position as open). Insert a
                            # closing order first and reference its id.
                            stop_db_side = "BUY" if pos.side == "short" else "SELL"
                            try:
                                stop_order_id = self.db.insert_order({
                                    "ts": str(bar.ts), "decision_id": None, "broker_id": "sim",
                                    "symbol": "MYM", "side": stop_db_side, "qty": pos.qty,
                                    "order_type": "market", "limit_price": 0.0,
                                    "stop_price": float(pos.current_stop),
                                    "status": "filled", "raw_response": "sim-stop exit",
                                })
                                self.db.insert_fill({
                                    "order_id": stop_order_id,
                                    "broker_fill_id": "sim-stop-" + str(uuid.uuid4())[:8],
                                    "ts": str(bar.ts),
                                    "qty": pos.qty,
                                    "price": fill_price,
                                    "commission": 0.0,
                                })
                            except Exception as exc:
                                log.warning("sim stop-fill insert failed: %s", exc)
                            log.info("sim STOP HIT %s qty=%d fill=%.2f pnl=%.2f cash=%.2f",
                                     pos.side, pos.qty, fill_price, pnl, self._cash)
                            stopped.append(i)
                            self._trade_count += 1
                    if stopped:
                        self._positions = [p for i, p in enumerate(self._positions) if i not in stopped]
                        self._save_sim_state()

                # Build AccountState the LLMs / final_check will see.
                if SIM_FILLS:
                    sim_unreal = sum(p.unrealized_pnl(bar.c) for p in self._positions)
                    # Aggregate the open lots into a single Position for the LLMs
                    # and final_check. Position has fields (side, qty, avg_price,
                    # unrealized_pnl, pyramid_adds_used) ONLY — no symbol/market_value.
                    # Use GROSS qty so _check_max_contracts can't be fooled by
                    # opposite-side lots netting out.
                    if self._positions:
                        net = sum(p.qty if p.side == "long" else -p.qty for p in self._positions)
                        agg_side = "long" if net > 0 else ("short" if net < 0 else "flat")
                        agg_pos = Position(
                            side=agg_side,
                            qty=sum(p.qty for p in self._positions),
                            avg_price=self._positions[0].avg_price,
                            unrealized_pnl=sim_unreal,
                            pyramid_adds_used=sum(p.pyramid_adds_used for p in self._positions),
                        )
                    else:
                        agg_pos = Position(
                            side="flat", qty=0, avg_price=0.0,
                            unrealized_pnl=0.0, pyramid_adds_used=0,
                        )
                    state = AccountState(
                        equity=self._cash + sim_unreal,
                        realized_pnl_today=self._realized_pnl_today,
                        unrealized_pnl=sim_unreal,
                        position=agg_pos,
                        now_et=datetime.fromtimestamp(bar.ts, tz=ET),
                    )
                    self.db.insert_equity({
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "start_equity": self._start_equity_today,
                        "end_equity": state.equity,
                        "realized_pnl": self._realized_pnl_today,
                        "unrealized_pnl": sim_unreal,
                        "commission": 0.0,
                        "trade_count": self._trade_count,
                    })
                else:
                    state = self.broker.get_account_state()
                    self.db.insert_equity({
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "start_equity": state.equity,
                        "end_equity": state.equity,
                        "realized_pnl": getattr(state, "realized_pnl_today", 0.0),
                        "unrealized_pnl": getattr(state, "unrealized_pnl", 0.0),
                        "commission": 0.0,
                        "trade_count": 0,
                    })

                if self._budget_exceeded:
                    continue

                if len(self.window) < self.MIN_WARMUP_BARS:
                    continue
                    
                snapshot = build_snapshot(self.window.as_list())
                atr14 = snapshot.atr14 or 1.0

                haiku_result = self.haiku.evaluate(snapshot, bar_ts=bar.ts)
                haiku_res = haiku_result.parsed
                if haiku_res is None:
                    haiku_res = dict(self.haiku.safe_default)
                    
                gemini_result = self.gemini.evaluate(haiku_res, snapshot, state.position, state.equity, bar_ts=bar.ts)
                gemini_res = gemini_result.parsed or {"action": "hold", "stop_price": 0.0, "trailing_stop_atr_multiple": 2.0}
                
                action = gemini_res.get("action", "hold")
                gem_stop = gemini_res.get("stop_price") or 0.0
                mark_price = bar.c
                if gem_stop:
                    order_stop = gem_stop
                else:
                    # Side-aware fallback stop: a short's protective stop sits
                    # ABOVE entry. The old code always used entry-2*ATR, so shorts
                    # with no Gemini stop were always rejected by final_check.
                    if action == "open_short":
                        order_stop = mark_price + atr14 * 2.0
                    else:
                        order_stop = mark_price - atr14 * 2.0
                sz = compute_size(entry=mark_price, stop=order_stop)
                # exec_qty is the risk-disciplined size: 0 means "skip — one
                # contract would risk more than the fixed budget". pqty (min 1) is
                # only an advisory quantity for the DeepSeek prompt.
                exec_qty = sz.contracts if sz.contracts > 0 else 0
                pqty = max(exec_qty, 1)

                ds_result = self.deepseek.evaluate(gemini_res, pqty, state, atr14, bar_ts=bar.ts, mark_price=mark_price)
                ds_res = ds_result.parsed or {"approved": False, "violations": ["PARSE_ERROR"], "override_action": "hold"}

                _dir_map = {"open_long": "LONG", "open_short": "SHORT"}
                direction = _dir_map.get(action, "FLAT")
                raw_votes = json.dumps({"haiku": haiku_res, "gemini": gemini_res, "ds": ds_res})
                decision = {
                    "bar_ts": bar.ts,
                    "action": action,
                    "reasoning": f"Haiku {haiku_res.get('trend')}, Gemini {action}, DS approved={ds_res.get('approved')}",
                    "disagreement_flags": {"haiku": haiku_res, "gemini": gemini_res, "ds": ds_res}
                }

                self.db.insert_decision({
                    "bar_ts": str(bar.ts),
                    "direction": direction,
                    "confidence": float(haiku_res.get("confidence_0_to_1", 0.5) or 0.5),
                    "stop_price": float(gem_stop or 0.0),
                    "entry_price": float(bar.c),
                    "raw_votes": raw_votes,
                    "safety_ok": 1 if ds_res.get("approved") else 0,
                    "safety_notes": str(ds_res.get("violations", [])),
})
                
                # We only execute if DeepSeek approved and action is an open/close
                # For simplicity, we just check open_long/open_short as in original runner
                mapped = None
                if action == "open_long": mapped = ("long", "open")
                elif action == "open_short": mapped = ("short", "open")
                elif action == "close": mapped = ("close", "close")
                
                if mapped and ds_res.get("approved"):
                    # ???? Golden/Death cross filter (dad's rule: ??????????? ????
                    cross_ok, cross_reason = self._cross.allows(action)
                    if not cross_ok:
                        log.info("Cross filter blocked %s: %s", action, cross_reason)
                        self.db.insert_decision({
                            "bar_ts": str(bar.ts),
                            "direction": "FLAT",
                            "confidence": 0.0,
                            "stop_price": 0.0,
                            "entry_price": float(bar.c),
                            "raw_votes": raw_votes,
                            "safety_ok": 0,
                            "safety_notes": cross_reason,
                        })
                    else:
                        log.info("Cross filter passed: %s", cross_reason)

                # --- close handler (no cross-filter, no final_check needed) ---
                if action == "close" and ds_res.get("approved") and SIM_FILLS and self._positions:
                    for pos in list(self._positions):
                        fill_price = bar.c
                        pnl = pos.unrealized_pnl(fill_price)
                        self._cash += pnl
                        self._realized_pnl_today += pnl
                        self._trade_count += 1
                        db_close_side = "BUY" if pos.side == "short" else "SELL"
                        try:
                            close_order_id = self.db.insert_order({
                                "ts": str(bar.ts), "decision_id": None, "broker_id": "sim",
                                "symbol": "MYM", "side": db_close_side, "qty": pos.qty,
                                "order_type": "market", "limit_price": 0.0,
                                "stop_price": 0.0,
                                "status": "filled", "raw_response": "sim-close at bar.c",
                            })
                            self.db.insert_fill({
                                "order_id": close_order_id,
                                "broker_fill_id": "sim-close-" + str(uuid.uuid4())[:8],
                                "ts": str(bar.ts),
                                "qty": pos.qty,
                                "price": fill_price,
                                "commission": 0.0,
                            })
                        except Exception as exc:
                            log.error("sim close order/fill insert failed: %s", exc)
                        log.info("sim CLOSE %s qty=%d fill=%.2f pnl=%.2f cash=%.2f",
                                 pos.side, pos.qty, fill_price, pnl, self._cash)
                    self._positions = []
                    self._save_sim_state()

                # --- add_pyramid handler ---
                if action == "add_pyramid" and ds_res.get("approved") and SIM_FILLS \
                        and self._positions and len(self._positions) < MAX_POSITIONS:
                    # A pyramid always adds to the existing position's side (this
                    # block only runs for action=="add_pyramid", so the old
                    # open_long/open_short arms were dead code).
                    pyramid_side = self._positions[0].side
                    if pyramid_side == self._positions[0].side:
                        cross_ok_pyr, cross_reason_pyr = self._cross.allows(
                            "open_long" if pyramid_side == "long" else "open_short"
                        )
                        if not cross_ok_pyr:
                            log.info("add_pyramid blocked by cross filter: %s", cross_reason_pyr)
                        else:
                            fill_price = bar.c
                            pyr_stop = float(gem_stop or 0.0)
                            pyr_db_side = "BUY" if pyramid_side == "long" else "SELL"
                            try:
                                pyr_order_id = self.db.insert_order({
                                    "ts": str(bar.ts), "decision_id": None, "broker_id": "sim",
                                    "symbol": "MYM", "side": pyr_db_side, "qty": pqty,
                                    "order_type": "market", "limit_price": 0.0,
                                    "stop_price": pyr_stop,
                                    "status": "filled", "raw_response": "sim-pyramid at bar.c",
                                })
                                self.db.insert_fill({
                                    "order_id": pyr_order_id,
                                    "broker_fill_id": "sim-pyr-" + str(uuid.uuid4())[:8],
                                    "ts": str(bar.ts),
                                    "qty": pqty,
                                    "price": fill_price,
                                    "commission": 0.0,
                                })
                            except Exception as exc:
                                log.error("sim pyramid order/fill insert failed: %s", exc)
                            self._positions.append(_PositionState(
                                side=pyramid_side,
                                qty=pqty,
                                avg_price=fill_price,
                                current_stop=pyr_stop,
                                pyramid_adds_used=0,
                                entry_ts=bar.ts,
                            ))
                            log.info("sim PYRAMID %s qty=%d entry=%.2f stop=%.2f positions=%d",
                                     pyr_db_side, pqty, fill_price, pyr_stop, len(self._positions))
                            self._save_sim_state()
                    else:
                        log.info("add_pyramid skipped: existing side=%s would conflict",
                                 self._positions[0].side)

                # Only the OPEN path builds an Order/final_check here; close and
                # add_pyramid are fully handled by their own blocks above. Without
                # this guard, action=="close" (mapped=("close","close")) fell
                # through and built Order(side="close"), inserting a phantom SELL
                # and a garbage side="close" position.
                if mapped and mapped[1] == "open" and ds_res.get("approved") and self._cross.allows(action)[0]:
                    side, act = mapped
                    if exec_qty < 1:
                        log.info("skip %s: risk-unit size is 0 (stop too wide for $ budget)", action)
                        continue
                    order = Order(
                        order_id="",
                        symbol="MYM",
                        side=side,
                        action=act,
                        qty=exec_qty,
                        entry_price=bar.c,
                        stop_price=order_stop,
                        atr=atr14,
                        status="pending"
                    )
                    
                    guard = final_check(order, state)
                    if guard.approved:
                        db_side = "BUY" if order.side == "long" else "SELL"
                        if SIM_FILLS:
                            if len(self._positions) >= MAX_POSITIONS:
                                log.info("sim fill skipped: max positions reached (%d/%d)",
                                         len(self._positions), MAX_POSITIONS)
                            else:
                                fill_price = bar.c
                                try:
                                    order_id = self.db.insert_order({
                                        "ts": str(bar.ts), "decision_id": None, "broker_id": "sim",
                                        "symbol": order.symbol, "side": db_side, "qty": order.qty,
                                        "order_type": "market", "limit_price": 0.0,
                                        "stop_price": float(order.stop_price or 0.0),
                                        "status": "filled", "raw_response": "sim-fill at bar.c",
                                    })
                                    self.db.insert_fill({
                                        "order_id": order_id,
                                        "broker_fill_id": "sim-" + str(uuid.uuid4())[:8],
                                        "ts": str(bar.ts),
                                        "qty": order.qty,
                                        "price": fill_price,
                                        "commission": 0.0,
                                    })
                                except Exception as exc:
                                    log.error("sim order/fill insert failed: %s", exc)
                                self._positions.append(_PositionState(
                                    side=order.side,
                                    qty=order.qty,
                                    avg_price=fill_price,
                                    current_stop=float(order.stop_price or 0.0),
                                    pyramid_adds_used=0,
                                    entry_ts=bar.ts,
                                ))
                                log.info("sim FILL %s %s qty=%d entry=%.2f stop=%.2f",
                                         db_side, order.symbol, order.qty, fill_price,
                                         float(order.stop_price or 0.0))
                                self._save_sim_state()
                        else:
                            try:
                                self.broker.submit_bracket_order(order)
                                self.db.insert_order({
                                    "ts": str(bar.ts), "decision_id": None, "broker_id": "",
                                    "symbol": order.symbol, "side": db_side, "qty": order.qty,
                                    "order_type": "bracket", "limit_price": 0.0,
                                    "stop_price": float(order.stop_price or 0.0),
                                    "status": "submitted", "raw_response": "",
                                })
                            except Exception as e:
                                log.error(f"Order submission failed: {e}")
                                self.db.insert_order({
                                    "ts": str(bar.ts), "decision_id": None, "broker_id": "",
                                    "symbol": order.symbol, "side": db_side, "qty": order.qty,
                                    "order_type": "bracket", "limit_price": 0.0,
                                    "stop_price": float(order.stop_price or 0.0),
                                    "status": "rejected", "raw_response": str(e)[:500],
                                })
                    else:
                        log.info("final_check rejected the order")
                        self.db.insert_decision({
                            "bar_ts": str(bar.ts),
                            "direction": "FLAT",
                            "confidence": 0.0,
                            "stop_price": 0.0,
                            "entry_price": float(bar.c),
                            "raw_votes": raw_votes,
                            "safety_ok": 0,
                            "safety_notes": f"final_check rejected: {guard.reason}",
                        })

            except CostBudgetExceeded:
                log.warning("CostBudgetExceeded. Stopping orders for the day.")
                self._budget_exceeded = True
            except Exception as e:
                log.error(f"Error in process loop: {e}", exc_info=True)

    async def start(self):
        log.info("LiveRunner starting...")
        self.broker.connect()
        self._ensure_sim_table()
        if SIM_FILLS and self._load_sim_state():
            log.info("Sim state restored from DB (cash=%.2f)", self._cash)
        else:
            try:
                initial_state = self.broker.get_account_state()
                self._cash = float(initial_state.equity)
                self._start_equity_today = self._cash
                log.info("Sim-fill seed: cash=%.2f from broker", self._cash)
            except Exception as exc:
                log.warning("Could not seed sim cash: %s -- defaulting to 1,000,000", exc)
                self._cash = 1_000_000.0
                self._start_equity_today = self._cash
            self._save_sim_state()
        await self._hydrate_window()
        
        if self.USE_YFINANCE:
            self.streamer = YFinancePoller("MYM=F", self._on_candle)
            await self.streamer.connect()
            await self.streamer.subscribe_candles("MYM=F", "15m", from_time_ms=0)
        else:
            token_data = self.broker.get_dxlink_token()
            self.streamer = DxLinkStreamer(
                token_data["dxlink-url"],
                token_data["token"],
                self._on_candle,
                token_refresh_fn=lambda: self.broker.get_dxlink_token()["token"],
            )
            await self.streamer.connect()
            target_sym = dxfeed_symbol("MYM") + "{=15m}"
            from_ms = int(datetime.now().timestamp() * 1000)
            await self.streamer.subscribe_candles(target_sym, "15m", from_ms)
        
        asyncio.create_task(self.streamer.run())
        asyncio.create_task(self._process_loop())
        log.info("LiveRunner started")
        
        while True:
            await asyncio.sleep(3600)

    async def stop(self):
        log.info("Stopping LiveRunner...")
        if self.streamer:
            await self.streamer.close()
        self.broker.disconnect()
