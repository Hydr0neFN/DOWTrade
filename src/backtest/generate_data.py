"""
CLI entry point: generate synthetic MYM 15-min bars and write to CSV.

Usage:
    python -m src.backtest.generate_data
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from src.backtest.synthetic import SyntheticConfig, generate_bars, write_csv

OUTPUT_PATH = "data/mym_synthetic_30d.csv"


def main() -> None:
    cfg = SyntheticConfig(
        start_date=date(2025, 1, 6),   # Monday 2025-01-06
        num_days=30,
        seed=42,
    )
    bars = generate_bars(cfg)

    write_csv(bars, OUTPUT_PATH)

    first_ts = datetime.fromtimestamp(bars[0].ts, tz=timezone.utc).isoformat()
    last_ts  = datetime.fromtimestamp(bars[-1].ts, tz=timezone.utc).isoformat()
    min_close = min(b.c for b in bars)
    max_close = max(b.c for b in bars)

    print(f"Bars generated : {len(bars)}")
    print(f"First bar ts   : {first_ts}  (unix={bars[0].ts})")
    print(f"Last  bar ts   : {last_ts}  (unix={bars[-1].ts})")
    print(f"Close range    : {min_close:.2f} – {max_close:.2f}")
    print(f"Written to     : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
