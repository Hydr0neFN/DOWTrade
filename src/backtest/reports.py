"""
src/backtest/reports.py
=======================
Report generation: equity-curve PNG and trades CSV.
"""
from __future__ import annotations

import csv
from datetime import datetime
from zoneinfo import ZoneInfo

try:
    import matplotlib
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")          # non-interactive backend
except ImportError:
    pass

ET = ZoneInfo("America/New_York")


def write_equity_curve_png(
    equity_curve: list[tuple[int, float]],
    path: str,
) -> None:
    """
    Write a single-panel equity curve plot to *path* (PNG).
    x-axis = datetime (UTC -> ET), y-axis = equity USD.
    """

    if not equity_curve:
        return

    timestamps = [datetime.fromtimestamp(ts, tz=ET) for ts, _ in equity_curve]
    equities = [eq for _, eq in equity_curve]

    end_equity = equities[-1]
    start_equity = equities[0]
    ret_pct = (end_equity - start_equity) / start_equity * 100.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps, equities, linewidth=1.2, color="steelblue")
    ax.set_title(
        f"Equity Curve  |  End: ${end_equity:,.2f}  |  Return: {ret_pct:+.2f}%",
        fontsize=13,
    )
    ax.set_xlabel("Date (ET)")
    ax.set_ylabel("Equity (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d", tz=ET))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3, tz=ET))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_trades_csv(trades: list[dict], path: str) -> None:
    """Write closed round-trip trades to a CSV file."""
    if not trades:
        # Write header-only file
        fieldnames = [
            "entry_ts", "exit_ts", "side", "qty", "avg_price",
            "exit_price", "realized_pnl", "exit_reason", "pyramid_adds",
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
        return

    fieldnames = list(trades[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)
