import asyncio
import logging
from datetime import datetime
from typing import Optional

from src.config import Settings
from src.db.repo import Database
from src.broker.base import Bar
from src.data.bars import Bar as DataBar
from src.broker.models import Order, Position, AccountState
from src.broker.tastytrade import TastytradeBroker, dxfeed_symbol
from src.live.dxlink import DxLinkStreamer

from src.data.bars import BarWindow
from src.data.features import MarketSnapshot, build_snapshot
from src.llm.base import CostTracker, CostBudgetExceeded, LLMCallResult
from src.llm.haiku_structural import HaikuStructural
from src.llm.gemini_execution import GeminiExecution
from src.llm.deepseek_risk import DeepSeekRisk
from src.backtest.harness import final_check, compute_size

log = logging.getLogger(__name__)

class LiveRunner:
    # Minimum bars in window before we run the LLM pipeline. SMA-200 is the
    # binding indicator. Cert sandbox does not provide historical Candle replay
    # over dxLink, so this accumulates from the first live bar (~50h at 15m).
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
        end_ts = int(datetime.now().timestamp())
        start_ts = end_ts - 5 * 24 * 3600
        bars = self.broker.fetch_historical_bars("MYM", start_ts, end_ts)
        for b in bars:
            self.window.append(DataBar(ts=b.ts_utc, o=b.open, h=b.high, l=b.low, c=b.close, v=int(b.volume)))
        log.info(f"Hydrated BarWindow with {len(self.window)} bars")

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
                self.db.insert_bars([bar])
                
                state = self.broker.get_account_state()
                self.db.insert_equity_record(bar.ts, state.equity, state.realized_pnl_today, state.unrealized_pnl)
                
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
                decision = {
                    "bar_ts": bar.ts,
                    "action": action,
                    "reasoning": f"Haiku {haiku_res.get('regime')}, Gemini {action}, DS approved={ds_res.get('approved')}",
                    "disagreement_flags": {"haiku": haiku_res, "gemini": gemini_res, "ds": ds_res}
                }
                
                self.db.insert_decision(
                    bar_ts=bar.ts,
                    action=action,
                    reasoning=decision["reasoning"],
                    disagreement_flags=decision["disagreement_flags"],
                    cost_usd=self.tracker.total_usd
                )
                
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
                        entry_price=0.0,
                        stop_price=gem_stop,
                        atr=atr14,
                        status="pending"
                    )
                    
                    guard = final_check(order, state)
                    if guard.approved:
                        try:
                            self.broker.submit_bracket_order(order)
                            self.db.insert_order(order, bar_ts=bar.ts)
                        except Exception as e:
                            log.error(f"Order submission failed: {e}")
                            order.status = "rejected"
                            self.db.insert_order(order, bar_ts=bar.ts)
                    else:
                        log.info("final_check rejected the order")
                        decision["disagreement_flags"]["final_check"] = guard.reason
                        self.db.insert_decision(
                            bar_ts=bar.ts,
                            action="none",
                            reasoning=f"final_check rejected: {guard.reason}",
                            disagreement_flags=decision["disagreement_flags"],
                            cost_usd=self.tracker.total_usd
                        )

            except CostBudgetExceeded:
                log.warning("CostBudgetExceeded. Stopping orders for the day.")
                self._budget_exceeded = True
            except Exception as e:
                log.error(f"Error in process loop: {e}", exc_info=True)

    async def start(self):
        log.info("LiveRunner starting...")
        self.broker.connect()
        await self._hydrate_window()
        
        token_data = self.broker.get_dxlink_token()
        self.streamer = DxLinkStreamer(token_data["dxlink-url"], token_data["token"], self._on_candle)
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
