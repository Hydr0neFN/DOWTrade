"""
src/backtest/harness.py
=======================
Backtest harness: replays historical 15-min MYM bars through the full
decision pipeline.  Supports stub and live-llm mode (real LLM calls).
"""
from __future__ import annotations

import json
import time as _time_mod
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Literal, Optional, Tuple
from zoneinfo import ZoneInfo

from src.backtest.stubs import StubDeepSeek, StubGemini, StubHaiku
from src.backtest.synthetic import read_csv
from src.broker.models import AccountState, Position, ProposedOrder
from src.config import FIXED_RISK_PER_TRADE_USD, MAX_OPEN_CONTRACTS, POINT_VALUE_USD
from src.data.bars import Bar, BarWindow
from src.data.features import MarketSnapshot, build_snapshot
from src.db.repo import init_db
from src.safety.guards import final_check
from src.sizing.risk_unit import compute_size

ET = ZoneInfo("America/New_York")
_SESSION_RESET_HOUR = 17   # 17:00 ET = CME daily session boundary


# ---------------------------------------------------------------------------
# Public config / result types
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    csv_path: str
    initial_equity: float = 10_000.0
    db_path: str = ":memory:"
    mode: Literal["stub", "live-llm"] = "stub"
    warmup_bars: int = 200
    write_equity_png: Optional[str] = None
    write_trades_csv: Optional[str] = None


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity_curve: List[Tuple[int, float]]
    trades: List[dict]
    stats: dict
    safety_overrides: int
    decisions_count: int


# ---------------------------------------------------------------------------
# Internal position tracker
# ---------------------------------------------------------------------------

@dataclass
class _PositionState:
    side: Literal["long", "short", "flat"] = "flat"
    qty: int = 0
    avg_price: float = 0.0
    current_stop: float = 0.0
    pyramid_adds_used: int = 0
    entry_ts: int = 0

    def is_flat(self) -> bool:
        return self.side == "flat"

    def unrealized_pnl(self, price: float) -> float:
        if self.is_flat():
            return 0.0
        direction = 1 if self.side == "long" else -1
        return (price - self.avg_price) * self.qty * POINT_VALUE_USD * direction

    def to_broker_position(self, mark_price: float) -> Position:
        return Position(
            side=self.side,
            qty=self.qty,
            avg_price=self.avg_price,
            unrealized_pnl=self.unrealized_pnl(mark_price),
            pyramid_adds_used=self.pyramid_adds_used,
        )


# ---------------------------------------------------------------------------
# Session boundary helper
# ---------------------------------------------------------------------------

def _session_date_key(ts: int) -> str:
    dt_et = datetime.fromtimestamp(ts, tz=ET)
    if dt_et.hour >= _SESSION_RESET_HOUR:
        session_close_day = (dt_et + timedelta(days=1)).date()
    else:
        session_close_day = dt_et.date()
    return session_close_day.isoformat()


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(cfg: BacktestConfig) -> BacktestResult:
    t0 = _time_mod.monotonic()

    all_bars: List[Bar] = read_csv(cfg.csv_path)
    if not all_bars:
        raise ValueError(f"No bars loaded from {cfg.csv_path}")

    db = init_db(cfg.db_path)

    window = BarWindow(maxlen=500)
    pos = _PositionState()
    equity: float = cfg.initial_equity
    realized_pnl_today: float = 0.0
    current_session_key: str = ""

    equity_curve: List[Tuple[int, float]] = []
    trades: List[dict] = []
    safety_overrides: int = 0
    decisions_count: int = 0

    haiku_stub = StubHaiku()
    gemini_stub = StubGemini()
    deepseek_stub = StubDeepSeek()
    _decision_seq: int = 0
    _total_llm_cost_usd: float = 0.0
    _budget_aborted: bool = False

    # Live-LLM client initialisation (only for live-llm mode)
    _live_haiku = None
    _live_gemini = None
    _live_deepseek = None
    _live_tracker = None
    if cfg.mode == "live-llm":
        from src.config import Settings
        from src.llm.base import CostBudgetExceeded as _CostBudgetExceeded
        from src.llm.base import CostTracker as _CostTracker
        from src.llm.deepseek_risk import DeepSeekRisk as _DeepSeekRisk
        from src.llm.gemini_execution import GeminiExecution as _GeminiExecution
        from src.llm.haiku_structural import HaikuStructural as _HaikuStructural
        _settings = Settings()
        _live_tracker = _CostTracker()
        _live_haiku = _HaikuStructural(_settings.anthropic_api_key, _live_tracker, db=db)
        _live_gemini = _GeminiExecution(_settings.google_api_key, _live_tracker, db=db)
        _live_deepseek = _DeepSeekRisk(_settings.huggingface_api_key, _live_tracker, db=db)

    for bar in all_bars:

        # --- session boundary ---
        sess_key = _session_date_key(bar.ts)
        if sess_key != current_session_key:
            realized_pnl_today = 0.0
            current_session_key = sess_key

        # --- insert bar ---
        try:
            db._execute(
                "INSERT OR IGNORE INTO bars (ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?)",
                (str(bar.ts), bar.o, bar.h, bar.l, bar.c, bar.v),
            )
        except Exception:
            pass

        window.append(bar)

        # --- STOP CHECK before decision ---
        if not pos.is_flat():
            stop_hit = False
            fill_price = pos.current_stop
            if pos.side == "long" and bar.l <= pos.current_stop:
                stop_hit = True
            elif pos.side == "short" and bar.h >= pos.current_stop:
                stop_hit = True

            if stop_hit:
                pnl = pos.unrealized_pnl(fill_price)
                realized_pnl_today += pnl
                equity += pnl
                trades.append({
                    "entry_ts": pos.entry_ts,
                    "exit_ts": bar.ts,
                    "side": pos.side,
                    "qty": pos.qty,
                    "avg_price": pos.avg_price,
                    "exit_price": fill_price,
                    "realized_pnl": pnl,
                    "exit_reason": "stop_hit",
                    "pyramid_adds": pos.pyramid_adds_used,
                })
                try:
                    db._execute(
                        "INSERT INTO fills (order_id, broker_fill_id, ts, qty, price, commission) VALUES (?,?,?,?,?,?)",
                        (0, str(uuid.uuid4()), str(bar.ts), pos.qty, fill_price, 0.0),
                    )
                except Exception:
                    pass
                pos = _PositionState()

        # --- mark-to-market ---
        mark_price = bar.c
        unrealized = pos.unrealized_pnl(mark_price)
        total_equity = equity + unrealized
        equity_curve.append((bar.ts, total_equity))

        # --- warmup skip ---
        if len(window) < cfg.warmup_bars:
            continue

        # --- snapshot + state ---
        snapshot: MarketSnapshot = build_snapshot(window.as_list())
        dt_et = datetime.fromtimestamp(bar.ts, tz=ET)
        state = AccountState(
            equity=equity,
            realized_pnl_today=realized_pnl_today,
            unrealized_pnl=unrealized,
            position=pos.to_broker_position(mark_price),
            now_et=dt_et,
        )

        # --- pipeline dispatch: stub or live-llm ---
        atr14 = snapshot.atr14 or 1.0

        if cfg.mode == "live-llm":
            # Budget-aborted: skip remaining bars entirely
            if _budget_aborted:
                continue
            try:
                haiku_result = _live_haiku.evaluate(snapshot, bar_ts=bar.ts)
                _total_llm_cost_usd = _live_tracker.total_usd
                haiku = haiku_result.parsed
                if haiku is None:
                    haiku = dict(_live_haiku.safe_default)
                # Haiku fallback -> hold, skip Gemini/DeepSeek (cost saving)
                if haiku_result.used_fallback:
                    gemini = {
                        "action": "hold",
                        "stop_price": 0.0,
                        "trailing_stop_atr_multiple": 2.0,
                        "reasoning": "haiku_fallback",
                    }
                    deepseek = {
                        "approved": False,
                        "violations": ["HAIKU_FALLBACK"],
                        "override_action": "hold",
                        "reasoning": "Haiku used safe_default; holding bar.",
                    }
                else:
                    gemini_result = _live_gemini.evaluate(
                        haiku, snapshot, state.position, state.equity, bar_ts=bar.ts
                    )
                    _total_llm_cost_usd = _live_tracker.total_usd
                    gemini = gemini_result.parsed or {
                        "action": "hold", "stop_price": 0.0,
                        "trailing_stop_atr_multiple": 2.0, "reasoning": "parse_error",
                    }
                    _gem_stop = gemini.get("stop_price") or 0.0
                    if _gem_stop:
                        _sz = compute_size(entry=mark_price, stop=_gem_stop)
                    else:
                        _fb = mark_price - atr14 * 2.0
                        if _fb <= 0:
                            _fb = mark_price * 0.99
                        _sz = compute_size(entry=mark_price, stop=_fb)
                    _pqty = _sz.contracts if _sz.contracts > 0 else 0
                    deepseek_result = _live_deepseek.evaluate(
                        gemini, max(_pqty, 1), state, atr14, bar_ts=bar.ts
                    )
                    _total_llm_cost_usd = _live_tracker.total_usd
                    deepseek = deepseek_result.parsed or {
                        "approved": False, "violations": ["PARSE_ERROR"],
                        "override_action": "hold", "reasoning": "parse_error",
                    }
            except _CostBudgetExceeded:
                _budget_aborted = True
                _total_llm_cost_usd = _live_tracker.total_usd
                continue
        else:
            # --- stubs ---
            haiku = haiku_stub.evaluate(snapshot)
            gemini = gemini_stub.evaluate(haiku, snapshot, state.position, state.equity)
            deepseek = None  # computed below after sizing

        gemini_stop = gemini.get("stop_price") or 0.0

        if gemini_stop:
            sizing = compute_size(entry=mark_price, stop=gemini_stop)
        else:
            fallback_stop = mark_price - atr14 * 2.0
            if fallback_stop <= 0:
                fallback_stop = mark_price * 0.99
            sizing = compute_size(entry=mark_price, stop=fallback_stop)

        proposed_qty = sizing.contracts if sizing.contracts > 0 else 0

        if cfg.mode != "live-llm":
            deepseek = deepseek_stub.evaluate(
                gemini, max(proposed_qty, 1), state, atr14
            )

        gemini_action = gemini.get("action", "hold")

        action_map = {
            "open_long":   ("long",  "open"),
            "open_short":  ("short", "open"),
            "add_pyramid": (pos.side if not pos.is_flat() else "long", "add_pyramid"),
            "close":       (pos.side if not pos.is_flat() else "long", "close"),
            "hold":        None,
        }
        mapped = action_map.get(gemini_action)

        _decision_seq += 1
        decisions_count += 1

        # Cap pyramid qty to remaining capacity so guard doesn't always reject
        if mapped is not None and mapped[1] == "add_pyramid" and not pos.is_flat():
            remaining = MAX_OPEN_CONTRACTS - pos.qty
            proposed_qty = min(proposed_qty, remaining)

        raw_votes = json.dumps({"haiku": haiku, "gemini": gemini, "deepseek": deepseek})

        proposed: Optional[ProposedOrder] = None
        guard_approved = False
        safety_notes = "hold - no action"

        if mapped is not None and proposed_qty > 0:
            order_side, order_action = mapped
            if order_action == "close":
                stop_for_guard = gemini_stop if gemini_stop else (
                    mark_price - atr14 if order_side == "long" else mark_price + atr14
                )
                # Ensure stop is on the correct side for close
                if order_side == "long" and stop_for_guard >= mark_price:
                    stop_for_guard = mark_price - atr14
                elif order_side == "short" and stop_for_guard <= mark_price:
                    stop_for_guard = mark_price + atr14
            else:
                stop_for_guard = gemini_stop

            if stop_for_guard:
                proposed = ProposedOrder(
                    side=order_side,
                    action=order_action,
                    entry_price=mark_price,
                    stop_price=stop_for_guard,
                    qty=proposed_qty,
                    atr=atr14,
                )
                guard = final_check(proposed, state)
                guard_approved = guard.approved
                safety_notes = guard.reason
                if not guard_approved:
                    safety_overrides += 1

        # --- insert decision ---
        direction_map = {
            "open_long":   "LONG",
            "open_short":  "SHORT",
            "add_pyramid": "LONG" if (pos.side == "long" or pos.is_flat()) else "SHORT",
            "close":       "FLAT",
            "hold":        "FLAT",
        }
        direction = direction_map.get(gemini_action, "FLAT")
        try:
            db._execute(
                """INSERT OR REPLACE INTO decisions
                   (bar_ts, direction, confidence, stop_price, entry_price,
                    raw_votes, safety_ok, safety_notes)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    str(bar.ts), direction,
                    haiku.get("confidence_0_to_1", 0.5),
                    gemini_stop, mark_price, raw_votes,
                    1 if guard_approved else 0,
                    safety_notes,
                ),
            )
        except Exception:
            pass

        # --- execute ---
        if guard_approved and proposed is not None:
            oa = proposed.action
            if oa == "open" and pos.is_flat():
                pos.side = proposed.side
                pos.qty = proposed.qty
                pos.avg_price = mark_price
                pos.current_stop = proposed.stop_price
                pos.pyramid_adds_used = 0
                pos.entry_ts = bar.ts
                try:
                    db_side = "BUY" if proposed.side == "long" else "SELL"
                    db._execute(
                        """INSERT INTO orders (ts, decision_id, broker_id, symbol, side, qty,
                           order_type, limit_price, stop_price, status, raw_response)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (str(bar.ts), _decision_seq, str(uuid.uuid4()),
                         "MYM", db_side, proposed.qty,
                         "MARKET", mark_price, proposed.stop_price,
                         "FILLED", None),
                    )
                except Exception:
                    pass

            elif oa == "add_pyramid" and not pos.is_flat() and pos.side == proposed.side:
                total_qty = pos.qty + proposed.qty
                pos.avg_price = (pos.avg_price * pos.qty + mark_price * proposed.qty) / total_qty
                pos.qty = total_qty
                pos.current_stop = proposed.stop_price
                pos.pyramid_adds_used += 1

            elif oa == "close" and not pos.is_flat():
                pnl = pos.unrealized_pnl(mark_price)
                realized_pnl_today += pnl
                equity += pnl
                trades.append({
                    "entry_ts": pos.entry_ts,
                    "exit_ts": bar.ts,
                    "side": pos.side,
                    "qty": pos.qty,
                    "avg_price": pos.avg_price,
                    "exit_price": mark_price,
                    "realized_pnl": pnl,
                    "exit_reason": "decision_close",
                    "pyramid_adds": pos.pyramid_adds_used,
                })
                try:
                    db._execute(
                        "INSERT INTO fills (order_id, broker_fill_id, ts, qty, price, commission) VALUES (?,?,?,?,?,?)",
                        (0, str(uuid.uuid4()), str(bar.ts), pos.qty, mark_price, 0.0),
                    )
                except Exception:
                    pass
                pos = _PositionState()

        # --- trailing stop ---
        _update_trailing_stop(pos, snapshot, atr14)

    # --- close open position at end ---
    if not pos.is_flat() and all_bars:
        last_bar = all_bars[-1]
        fill_price = last_bar.c
        pnl = pos.unrealized_pnl(fill_price)
        realized_pnl_today += pnl
        equity += pnl
        trades.append({
            "entry_ts": pos.entry_ts,
            "exit_ts": last_bar.ts,
            "side": pos.side,
            "qty": pos.qty,
            "avg_price": pos.avg_price,
            "exit_price": fill_price,
            "realized_pnl": pnl,
            "exit_reason": "end_of_backtest",
            "pyramid_adds": pos.pyramid_adds_used,
        })

    # --- stats ---
    elapsed = _time_mod.monotonic() - t0
    start_equity = cfg.initial_equity
    end_equity = equity

    num_trades = len(trades)
    wins = [t for t in trades if t["realized_pnl"] > 0]
    num_wins = len(wins)
    win_rate = (num_wins / num_trades * 100.0) if num_trades > 0 else 0.0
    total_r = sum(t["realized_pnl"] / FIXED_RISK_PER_TRADE_USD for t in trades)
    avg_r = (total_r / num_trades) if num_trades > 0 else 0.0

    peak = start_equity
    max_dd_pct = 0.0
    for _, eq in equity_curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak * 100.0
            if dd > max_dd_pct:
                max_dd_pct = dd

    if cfg.mode == "live-llm" and _live_tracker is not None:
        _total_llm_cost_usd = _live_tracker.total_usd

    stats = {
        "start_equity": start_equity,
        "end_equity": end_equity,
        "return_pct": (end_equity - start_equity) / start_equity * 100.0,
        "num_trades": num_trades,
        "num_wins": num_wins,
        "win_rate_pct": win_rate,
        "avg_r": avg_r,
        "max_drawdown_pct": max_dd_pct,
        "bars_processed": len(all_bars),
        "elapsed_sec": elapsed,
        "total_llm_cost_usd": _total_llm_cost_usd,
    }
    if _budget_aborted:
        stats["aborted_reason"] = "budget_exceeded"

    if cfg.write_equity_png:
        from src.backtest.reports import write_equity_curve_png
        write_equity_curve_png(equity_curve, cfg.write_equity_png)

    if cfg.write_trades_csv:
        from src.backtest.reports import write_trades_csv
        write_trades_csv(trades, cfg.write_trades_csv)

    db.close()

    return BacktestResult(
        config=cfg,
        equity_curve=equity_curve,
        trades=trades,
        stats=stats,
        safety_overrides=safety_overrides,
        decisions_count=decisions_count,
    )


# ---------------------------------------------------------------------------
# Trailing stop logic
# ---------------------------------------------------------------------------

def _update_trailing_stop(
    pos: _PositionState,
    snapshot: MarketSnapshot,
    atr14: float,
) -> None:
    if pos.is_flat() or atr14 <= 0:
        return

    price = snapshot.current_price
    swings = snapshot.swings

    if pos.side == "long":
        hl_swings = [s for s in swings if s.kind == "HL"]
        if not hl_swings:
            return
        new_stop = hl_swings[-1].price
        if new_stop <= pos.current_stop:
            return
        if price - new_stop < 2.0 * atr14:
            return
        pos.current_stop = new_stop

    elif pos.side == "short":
        lh_swings = [s for s in swings if s.kind == "LH"]
        if not lh_swings:
            return
        new_stop = lh_swings[-1].price
        if new_stop >= pos.current_stop:
            return
        if new_stop - price < 2.0 * atr14:
            return
        pos.current_stop = new_stop
