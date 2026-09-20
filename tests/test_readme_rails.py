"""The README must not quote a safety rail that src/config.py has moved on from.

Both READMEs claimed a $200 daily-loss limit and $50 fixed risk per trade for
long enough that nobody noticed; the real constants are $600 and $250. Nothing
caught it because a number in prose is not executable. This makes it executable:
the value is read from the module that defines it and looked for in the text, so
changing a rail without touching the docs fails here instead of misleading a
reader.

Deliberately a *check*, not a generator -- the surrounding prose is worth writing
by hand. It only asserts the figures agree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config import (
    FIXED_RISK_PER_TRADE_USD,
    MAX_DAILY_LOSS_USD,
    MAX_LLM_SPEND_USD,
    MAX_OPEN_CONTRACTS,
    REPORTING_CAPITAL_USD,
)

ROOT = Path(__file__).resolve().parent.parent
READMES = [ROOT / "README.md", ROOT / "README.zh-TW.md"]

# Constants the READMEs quote next to their own name, as `NAME`, $value / NAME，$value.
DOCUMENTED = [
    ("MAX_DAILY_LOSS_USD", MAX_DAILY_LOSS_USD, "$"),
    ("FIXED_RISK_PER_TRADE_USD", FIXED_RISK_PER_TRADE_USD, "$"),
    ("MAX_OPEN_CONTRACTS", MAX_OPEN_CONTRACTS, ""),
]


def _fmt(value, prefix: str) -> str:
    n = int(value) if float(value) == int(value) else value
    return f"{prefix}{n:,}" if prefix else str(n)


@pytest.mark.parametrize("path", READMES, ids=lambda p: p.name)
@pytest.mark.parametrize("name,value,prefix", DOCUMENTED, ids=lambda x: x if isinstance(x, str) else "")
def test_readme_quotes_current_rail(path: Path, name: str, value, prefix: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert f"`{name}`" in text, f"{path.name} does not mention {name} at all"

    # The value must appear within the same clause as the constant's name, so a
    # figure that happens to occur elsewhere in the document cannot satisfy this.
    expected = _fmt(value, prefix)
    window = re.search(re.escape(f"`{name}`") + r".{0,40}", text, re.S)
    assert window and expected in window.group(0), (
        f"{path.name}: {name} is {expected} in src/config.py, "
        f"but the README says {window.group(0) if window else '<not found>'!r}"
    )


@pytest.mark.parametrize("path", READMES, ids=lambda p: p.name)
def test_readme_has_no_superseded_rail_figures(path: Path, ) -> None:
    """Guard the specific wrong numbers that shipped, so they cannot come back."""
    text = path.read_text(encoding="utf-8")
    for stale in ("max daily loss ($200)", "每日最大虧損（$200）",
                  "fixed risk\n  per trade ($50)", "固定風險（$50）"):
        assert stale not in text, f"{path.name} still carries the superseded figure: {stale!r}"


def test_reporting_capital_matches_profile_publisher() -> None:
    """The GitHub profile publisher mirrors this constant and must stay in step.

    /root/Hydr0neFN/update_stats.py carries its own copy of this number to quote
    DOWTrade's return against risk capital rather than the $1M sim balance. It is
    outside this repo, so this only pins the value the comment there refers to.
    """
    assert REPORTING_CAPITAL_USD == 25_000.0
    assert MAX_LLM_SPEND_USD > 0
