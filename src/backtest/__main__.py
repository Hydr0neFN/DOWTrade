"""
src/backtest/__main__.py
========================
CLI entry point for the backtest harness.

Usage:
    python -m src.backtest --csv data/mym_synthetic_30d.csv --mode stub \
        --db data/backtest.db --png reports/equity.png \
        --trades reports/trades.csv
"""
from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DOWTrade backtest harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--mode", default="stub", choices=["stub", "live-llm"],
                        help="Pipeline mode")
    parser.add_argument("--db", default=":memory:", help="SQLite DB path")
    parser.add_argument("--png", default=None, help="Output equity-curve PNG path")
    parser.add_argument("--trades", default=None, help="Output trades CSV path")
    parser.add_argument("--equity", type=float, default=10_000.0,
                        help="Initial equity USD")
    parser.add_argument("--warmup", type=int, default=200,
                        help="Warmup bars before trading")

    args = parser.parse_args()

    # Create reports/ dir if writing files
    for fpath in [args.png, args.trades]:
        if fpath:
            dirpath = os.path.dirname(fpath)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)

    from src.backtest.harness import BacktestConfig, run_backtest

    cfg = BacktestConfig(
        csv_path=args.csv,
        initial_equity=args.equity,
        db_path=args.db,
        mode=args.mode,
        warmup_bars=args.warmup,
        write_equity_png=args.png,
        write_trades_csv=args.trades,
    )

    print(f"Running backtest: csv={args.csv} mode={args.mode} warmup={args.warmup}")
    result = run_backtest(cfg)

    print("\n=== BACKTEST STATS ===")
    for k, v in result.stats.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    print(f"\n  {'safety_overrides':30s}: {result.safety_overrides}")
    print(f"  {'decisions_count':30s}: {result.decisions_count}")

    if args.png:
        print(f"\nEquity PNG written: {os.path.abspath(args.png)}")
    if args.trades:
        print(f"Trades CSV written: {os.path.abspath(args.trades)}")


if __name__ == "__main__":
    main()
