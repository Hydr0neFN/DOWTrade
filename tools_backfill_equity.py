"""Rebuild DOWTrade's `equity` table from `fills`, which is the ground truth.

The bot used to write its equity row before each bar's own exits were booked,
so an exit on the last bar of a session never reached realized_pnl. Eight days
are wrong; the fills that prove it were always correct. Also corrects
sim_state.cash, which drifted +$350.83 above what the fills account for.

Read-only unless --apply is passed.
"""
import argparse, sqlite3, sys
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
DB = "/root/DOWTrade/trading-bot/data/bot.db"
POINT_VALUE = 0.50
START_CASH = 1_000_000.0

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
args = ap.parse_args()

db = sqlite3.connect(DB); db.row_factory = sqlite3.Row
et_day = lambda ts: datetime.fromtimestamp(int(ts), ET).strftime("%Y-%m-%d")

fills = db.execute("""select f.id fid, f.ts, o.side, f.qty, f.price
                      from fills f join orders o on o.id = f.order_id
                      order by f.ts asc, f.id asc""").fetchall()

# last bar close per ET day -- the mark for any position carried into the night
last_close = {}
for ts, c in db.execute("select ts, close from bars order by ts"):
    last_close[et_day(ts)] = c

# Replay FIFO, bucketing realized P&L and closing-fill counts by ET day.
pos, realized, closes = [], defaultdict(float), defaultdict(int)
open_after = {}          # ET day -> lots still open at its end
for r in fills:
    d, q, price, side = et_day(r["ts"]), r["qty"], r["price"], r["side"]
    while q > 0 and pos and pos[0][0] != side:
        s, pq, pp = pos[0]
        m = min(pq, q)
        realized[d] += (price - pp) * m * POINT_VALUE * (1 if s == "BUY" else -1)
        closes[d] += 1
        pq -= m; q -= m
        pos.pop(0) if pq == 0 else pos[0].__setitem__(1, pq)
    if q > 0:
        pos.append([side, q, price])
    open_after[d] = [list(p) for p in pos]

leftover = pos
rows = db.execute("select * from equity order by date").fetchall()
dates = [r["date"] for r in rows]

def mtm(lots, mark):
    if not lots or not mark:
        return 0.0
    return sum((mark - pp) * pq * POINT_VALUE * (1 if s == "BUY" else -1) for s, pq, pp in lots)

# Walk the existing dates in order, carrying cash and the open book forward.
cash, carried, changes = START_CASH, [], []
for d in dates:
    cash += realized.get(d, 0.0)
    if d in open_after:
        carried = open_after[d]
    unreal = mtm(carried, last_close.get(d))
    new = {
        "date": d,
        "start_equity": None,          # filled below from the previous end
        "end_equity": cash + unreal,
        "realized_pnl": realized.get(d, 0.0),
        "unrealized_pnl": unreal,
        "commission": 0.0,
        "trade_count": closes.get(d, 0),
    }
    changes.append(new)
prev_end = START_CASH
for c in changes:
    c["start_equity"] = prev_end
    prev_end = c["end_equity"]

old = {r["date"]: r for r in rows}
n_changed = 0
print(f"{'date':<12}{'realized old':>14}{'realized new':>14}{'end old':>15}{'end new':>15}  ")
for c in changes:
    o = old[c["date"]]
    dirty = (abs(o["realized_pnl"] - c["realized_pnl"]) > 0.005
             or abs(o["end_equity"] - c["end_equity"]) > 0.005
             or abs(o["start_equity"] - c["start_equity"]) > 0.005
             or o["trade_count"] != c["trade_count"])
    if dirty:
        n_changed += 1
        print(f"{c['date']:<12}{o['realized_pnl']:>14.2f}{c['realized_pnl']:>14.2f}"
              f"{o['end_equity']:>15.2f}{c['end_equity']:>15.2f}")

tot_realized = sum(realized.values())
print(f"\nrows {len(changes)}, changed {n_changed}")
print(f"realized total from fills : {tot_realized:>12.2f}")
print(f"equity table realized sum : {sum(r['realized_pnl'] for r in rows):>12.2f}")
print(f"cash: {db.execute('select cash from sim_state').fetchone()[0]:.2f} -> {START_CASH + tot_realized:.2f}"
      f"  (drift removed {db.execute('select cash from sim_state').fetchone()[0] - (START_CASH + tot_realized):+.2f})")
print(f"lots still open at end of file: {leftover}")

if not args.apply:
    print("\nDRY RUN -- nothing written. Re-run with --apply.")
    sys.exit(0)

with db:
    for c in changes:
        db.execute("""update equity set start_equity=?, end_equity=?, realized_pnl=?,
                      unrealized_pnl=?, commission=?, trade_count=? where date=?""",
                   (c["start_equity"], c["end_equity"], c["realized_pnl"],
                    c["unrealized_pnl"], c["commission"], c["trade_count"], c["date"]))
    db.execute("update sim_state set cash=? where id=1", (START_CASH + tot_realized,))
print("\nAPPLIED.")
