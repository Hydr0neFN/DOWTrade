"""Deterministic control arm for DOWTrade.

Seventy trades cannot settle whether an LLM ensemble has an edge in absolute
terms -- the same statistical-power wall `~/trader` hit. What a small sample
CAN settle is a *relative* question: does the ensemble beat a dumb mechanical
rule fed the identical bars, under identical risk, stops, session rules and
execution costs? That is what this replays.

Baseline rule (deliberately boring, and the one most widely used on this kind
of intraday index-futures series):
  * EMA(9) / EMA(21) cross on the same 15m bars
  * entry on the cross bar's close, adverse slippage, same commission
  * initial stop 2 x ATR(14), the same band the live bot's stops are bounded to
  * size = FIXED_RISK_PER_TRADE_USD / (stop distance x POINT_VALUE), capped at
    MAX_OPEN_CONTRACTS
  * exit on stop, on the opposite cross, or at the Friday 15:45 ET flat
  * one position at a time, no pyramiding

Read-only against bot.db unless --write is passed, in which case it stores its
round trips in `baseline_round_trips` (its own table -- it never touches the
live bot's tables).
"""
from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, "/root/DOWTrade/trading-bot")
from src.config import (POINT_VALUE_USD, FIXED_RISK_PER_TRADE_USD, MAX_OPEN_CONTRACTS,
                        COMMISSION_PER_CONTRACT_USD, SLIPPAGE_TICKS, TICK_SIZE_POINTS,
                        WEEKEND_FLAT_DAY, REPORTING_CAPITAL_USD)

ET = ZoneInfo("America/New_York")
DB = "/root/DOWTrade/trading-bot/data/bot.db"
FAST, SLOW, ATR_N, STOP_MULT = 9, 21, 14, 2.0


def ema_series(vals, n):
    k, out, cur = 2.0 / (n + 1), [], None
    for v in vals:
        cur = v if cur is None else v * k + cur * (1 - k)
        out.append(cur)
    return out


def atr_series(bars, n):
    trs, out, cur = [], [], None
    prev_c = None
    for b in bars:
        tr = b["high"] - b["low"] if prev_c is None else max(
            b["high"] - b["low"], abs(b["high"] - prev_c), abs(b["low"] - prev_c))
        trs.append(tr)
        cur = tr if cur is None else (cur * (n - 1) + tr) / n
        out.append(cur)
        prev_c = b["close"]
    return out


def fill_px(raw, is_buy):
    slip = SLIPPAGE_TICKS * TICK_SIZE_POINTS
    return raw + slip if is_buy else raw - slip


def run(write: bool):
    db = sqlite3.connect(DB); db.row_factory = sqlite3.Row
    bars = db.execute("select ts, open, high, low, close from bars order by ts").fetchall()
    if len(bars) < SLOW + ATR_N:
        print("not enough bars"); return
    closes = [b["close"] for b in bars]
    ef, es, atr = ema_series(closes, FAST), ema_series(closes, SLOW), atr_series(bars, ATR_N)

    pos = None          # dict(side, qty, entry, stop, entry_ts, risk)
    trips = []

    def close_pos(ts, raw_px, reason):
        nonlocal pos
        is_buy = pos["side"] == "short"
        px = fill_px(raw_px, is_buy)
        gross = (px - pos["entry"]) * pos["qty"] * POINT_VALUE_USD * (1 if pos["side"] == "long" else -1)
        comm = COMMISSION_PER_CONTRACT_USD * pos["qty"]          # exit side
        net = gross - comm
        trips.append({
            "entry_ts": pos["entry_ts"], "exit_ts": ts, "side": pos["side"],
            "qty": pos["qty"], "entry_price": pos["entry"], "exit_price": px,
            "initial_stop": pos["stop0"], "initial_risk_usd": pos["risk"],
            "gross_pnl_usd": gross, "commission_usd": comm + pos["entry_comm"],
            "net_pnl_usd": net - pos["entry_comm"],
            "r_multiple": (net - pos["entry_comm"]) / pos["risk"] if pos["risk"] > 0 else None,
            "hold_minutes": (ts - pos["entry_ts"]) / 60.0,
            "reason": reason,
        })
        pos = None

    for i in range(SLOW, len(bars)):
        b = bars[i]
        ts = int(b["ts"])
        et = datetime.fromtimestamp(ts, ET)
        prev_up = ef[i - 1] > es[i - 1]
        now_up = ef[i] > es[i]
        crossed = prev_up != now_up

        if pos:  # 1) stop, checked against the bar's own range
            if pos["side"] == "long" and b["low"] <= pos["stop"]:
                close_pos(ts, min(pos["stop"], b["open"]), "stop")
            elif pos["side"] == "short" and b["high"] >= pos["stop"]:
                close_pos(ts, max(pos["stop"], b["open"]), "stop")
        if pos and et.weekday() == WEEKEND_FLAT_DAY and (et.hour, et.minute) >= (15, 45):
            close_pos(ts, b["close"], "weekend-flat"); continue
        if pos and crossed:
            close_pos(ts, b["close"], "opposite-cross")

        late_friday = (et.weekday() == WEEKEND_FLAT_DAY
                       and (et.hour, et.minute) >= (15, 45))
        if pos is None and crossed and not late_friday:
            side = "long" if now_up else "short"
            a = atr[i] or 0.0
            dist = a * STOP_MULT
            if dist <= 0:
                continue
            qty = int(FIXED_RISK_PER_TRADE_USD // (dist * POINT_VALUE_USD))
            qty = max(0, min(qty, MAX_OPEN_CONTRACTS))
            if qty < 1:
                continue
            entry = fill_px(b["close"], is_buy=(side == "long"))
            stop0 = entry - dist if side == "long" else entry + dist
            pos = {"side": side, "qty": qty, "entry": entry, "stop": stop0,
                   "stop0": stop0, "entry_ts": ts,
                   "risk": abs(entry - stop0) * qty * POINT_VALUE_USD,
                   "entry_comm": COMMISSION_PER_CONTRACT_USD * qty}

    # ---- summary -----------------------------------------------------------
    if not trips:
        print("baseline produced no trades"); return
    nets = [t["net_pnl_usd"] for t in trips]
    rs = [t["r_multiple"] for t in trips if t["r_multiple"] is not None]
    wins = [x for x in nets if x > 0]
    span = f"{datetime.fromtimestamp(trips[0]['entry_ts'], ET):%Y-%m-%d} -> {datetime.fromtimestamp(trips[-1]['exit_ts'], ET):%Y-%m-%d}"
    print(f"EMA({FAST}/{SLOW}) baseline, {span}, {len(bars)} bars")
    print(f"  trades        {len(trips)}")
    print(f"  net P&L       ${sum(nets):,.2f}   (commissions ${sum(t['commission_usd'] for t in trips):,.2f})")
    print(f"  win rate      {len(wins)/len(trips)*100:.1f}%")
    print(f"  avg R         {statistics.mean(rs):+.3f}" if rs else "  avg R  n/a")
    print(f"  expectancy    ${statistics.mean(nets):+,.2f} per trade")
    print(f"  vs capital    {sum(nets)/REPORTING_CAPITAL_USD*100:+.2f}% of ${REPORTING_CAPITAL_USD:,.0f}")

    # ---- the live bot, re-priced under the SAME cost model -----------------
    # Its historical fills were booked at bar.c with zero commission, so a raw
    # comparison would flatter it. Replay them FIFO and charge the same
    # slippage and commission the baseline just paid.
    lf = db.execute("""select f.id, f.ts, o.side, f.qty, f.price
                       from fills f join orders o on o.id = f.order_id
                       order by f.ts asc, f.id asc""").fetchall()
    lot, live_trips, live_comm = [], [], 0.0
    for r in lf:
        side, q, raw = r["side"], r["qty"], r["price"]
        px = fill_px(raw, is_buy=(side == "BUY"))
        live_comm += COMMISSION_PER_CONTRACT_USD * q
        while q > 0 and lot and lot[0][0] != side:
            s0, pq, pp = lot[0]
            m = min(pq, q)
            gross = (px - pp) * m * POINT_VALUE_USD * (1 if s0 == "BUY" else -1)
            live_trips.append(gross)
            pq -= m; q -= m
            lot.pop(0) if pq == 0 else lot[0].__setitem__(1, pq)
        if q > 0:
            lot.append([side, q, px])
    live_gross = sum(live_trips)
    live_net = live_gross - live_comm
    print(f"\nDOWTrade (LLM ensemble), same bars, SAME cost model:")
    print(f"  round trips   {len(live_trips)}")
    print(f"  net P&L       ${live_net:,.2f}   (commissions ${live_comm:,.2f},"
          f" slippage already in the fill prices)")
    print(f"  expectancy    ${live_net/len(live_trips):+,.2f} per round trip"
          if live_trips else "  expectancy    n/a")
    print(f"  vs capital    {live_net/REPORTING_CAPITAL_USD*100:+.2f}% of ${REPORTING_CAPITAL_USD:,.0f}")
    print(f"\nRELATIVE: LLM ensemble - EMA baseline = ${live_net - sum(nets):+,.2f}"
          f"  ({live_net/len(live_trips) - statistics.mean(nets):+,.2f} per trade)"
          if live_trips else "")
    print("Both sides are gross of the fact that neither has enough trades to be")
    print("significant -- this is a direction, not a proof.")
    try:
        rt = db.execute("select count(*), coalesce(sum(net_pnl_usd),0), coalesce(avg(r_multiple),0) "
                        "from round_trips").fetchone()
        print(f"\nround_trips table (live, R-multiples, populated from 2026-09-09): "
              f"{rt[0]} rows, net ${rt[1]:,.2f}, avg R {rt[2]:+.3f}")
    except sqlite3.OperationalError:
        print("\nround_trips table not created yet (the service creates it at startup).")

    if not write:
        print("\nread-only. pass --write to store the baseline trips.")
        return
    with db:
        db.execute("""CREATE TABLE IF NOT EXISTS baseline_round_trips (
            id INTEGER PRIMARY KEY AUTOINCREMENT, exit_ts INTEGER, entry_ts INTEGER,
            side TEXT, qty INTEGER, entry_price REAL, exit_price REAL,
            initial_stop REAL, initial_risk_usd REAL, gross_pnl_usd REAL,
            commission_usd REAL, net_pnl_usd REAL, r_multiple REAL,
            hold_minutes REAL, reason TEXT)""")
        db.execute("DELETE FROM baseline_round_trips")
        db.executemany("""INSERT INTO baseline_round_trips
            (exit_ts, entry_ts, side, qty, entry_price, exit_price, initial_stop,
             initial_risk_usd, gross_pnl_usd, commission_usd, net_pnl_usd,
             r_multiple, hold_minutes, reason)
            VALUES (:exit_ts,:entry_ts,:side,:qty,:entry_price,:exit_price,:initial_stop,
                    :initial_risk_usd,:gross_pnl_usd,:commission_usd,:net_pnl_usd,
                    :r_multiple,:hold_minutes,:reason)""", trips)
        db.execute("""CREATE TABLE IF NOT EXISTS baseline_summary (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            generated_at TEXT, bars INTEGER,
            baseline_trades INTEGER, baseline_net_usd REAL,
            baseline_win_rate REAL, baseline_avg_r REAL,
            live_round_trips INTEGER, live_gross_usd REAL,
            live_net_cost_adjusted_usd REAL, live_commission_usd REAL,
            reporting_capital_usd REAL)""")
        db.execute("""INSERT OR REPLACE INTO baseline_summary
            (id, generated_at, bars, baseline_trades, baseline_net_usd,
             baseline_win_rate, baseline_avg_r, live_round_trips, live_gross_usd,
             live_net_cost_adjusted_usd, live_commission_usd, reporting_capital_usd)
            VALUES (1,?,?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now(ET).isoformat(timespec="seconds"), len(bars),
             len(trips), sum(nets), len(wins) / len(trips),
             statistics.mean(rs) if rs else None,
             len(live_trips), live_gross, live_net, live_comm,
             REPORTING_CAPITAL_USD))
    print(f"\nwrote {len(trips)} rows to baseline_round_trips + baseline_summary")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    run(ap.parse_args().write)
