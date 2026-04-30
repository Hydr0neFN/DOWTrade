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
from src.llm.base import CostTracker, CostBudgetExceeded, LLMCallResult
from src.llm.haiku_structural import HaikuStructural
from src.llm.gemini_execution import GeminiExecution
from src.llm.deepseek_risk import DeepSeekRisk
from src.backtest.harness import final_check, compute_size
import json

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

    def _reset_daily_state(self):
        now = datetime.now()
        day_str = now.strftime("%Y-%m-%d")
        if self._last_day != day_str:
            if now.hour >= 17 or self._last_day is None:
                self._budget_exceeded = False
                self._last_day = day_str
                log.info(f"Reset daily state for {day_str}")

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
                self.db.insert_bar({"ts": str(bar.ts), "open": bar.o, "high": bar.h, "low": bar.l, "close": bar.c, "volume": bar.v})
                
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
                
                gem_stop = gemini_res.get("stop_price") or 0.0
                mark_price = bar.c
                if gem_stop:
                    sz = compute_size(entry=mark_price, stop=gem_stop)
                else:
                    fb = mark_price - atr14 * 2.0
                    sz = compute_size(entry=mark_price, stop=fb)
                pqty = max(sz.contracts if sz.contracts > 0 else 0, 1)

                ds_result = self.deepseek.evaluate(gemini_res, pqty, state, atr14, bar_ts=bar.ts)
                ds_res = ds_result.parsed or {"approved": False, "violations": ["PARSE_ERROR"], "override_action": "hold"}
                
                action = gemini_res.get("action", "hold")
                _dir_map = {"open_long": "LONG", "open_short": "SHORT"}
                direction = _dir_map.get(action, "FLAT")
                raw_votes = json.dumps({"haiku": haiku_res, "gemini": gemini_res, "ds": ds_res})
                decision = {
                    "bar_ts": bar.ts,
                    "action": action,
                    "reasoning": f"Haiku {haiku_res.get('regime')}, Gemini {action}, DS approved={ds_res.get('approved')}",
                    "disagreement_flags": {"haiku": haiku_res, "gemini": gemini_res, "ds": ds_res}
                }

                self.db.insert_decision({
                    "bar_ts": str(bar.ts),
                    "direction": direction,
                    "confidence": float(haiku_res.get("confidence", 0.5) or 0.5),
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
                
                if mapped and ds_res.get("approved"):
                    side, act = mapped
                    order = Order(
                        order_id="",
                        symbol="MYM",
                        side=side,
                        action=act,
                        qty=pqty,
                        entry_price=bar.c,
                        stop_price=gem_stop,
                        atr=atr14,
                        status="pending"
                    )
                    
                    guard = final_check(order, state)
                    if guard.approved:
                        # orders.side CHECK requires BUY/SELL; map from internal long/short.
                        db_side = "BUY" if order.side == "long" else "SELL"
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
