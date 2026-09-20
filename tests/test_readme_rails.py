"""No document may quote a safety rail that src/config.py has moved on from.

Both READMEs claimed a $200 daily-loss limit and $50 fixed risk per trade for
long enough that nobody noticed; the real constants are $600 and $250. Nothing
caught it because a number in prose is not executable. This makes it executable:
the value is read from the module that defines it and looked for in the text, so
changing a rail without touching the docs fails here instead of misleading a
reader.

Deliberately a *check*, not a generator -- the prose around these figures is
worth writing by hand. Only the figures have to be mechanically true.

The same applies to prompts: a prompt that restates a threshold is another copy
of it, and risk_audit.txt is where the wrong $200 had been living. It now
interpolates the rails, and the last test here proves the rendered text carries
the live numbers.
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
    MAX_PYRAMID_ADDS,
    REPORTING_CAPITAL_USD,
    STOP_ATR_MAX_MULT,
    STOP_ATR_MIN_MULT,
)

ROOT = Path(__file__).resolve().parent.parent
READMES = [ROOT / "README.md", ROOT / "README.zh-TW.md"]

# Constants the READMEs quote next to their own backticked name.
DOCUMENTED = [
    ("MAX_DAILY_LOSS_USD", MAX_DAILY_LOSS_USD, "$"),
    ("FIXED_RISK_PER_TRADE_USD", FIXED_RISK_PER_TRADE_USD, "$"),
    ("MAX_OPEN_CONTRACTS", MAX_OPEN_CONTRACTS, ""),
]

# How far after the constant's name the figure may sit. Wide enough for a short
# parenthetical in either language, tight enough that an unrelated number
# further down the paragraph cannot satisfy the assertion.
WINDOW = 60


def _plain(value) -> str:
    """The number as a document would write it: 600.0 -> "600", 1.0 -> "1"."""
    return str(int(value)) if float(value) == int(value) else str(value)


@pytest.mark.parametrize("path", READMES, ids=lambda p: p.name)
@pytest.mark.parametrize("name,value,prefix", DOCUMENTED, ids=[d[0] for d in DOCUMENTED])
def test_readme_quotes_current_rail(path: Path, name: str, value, prefix: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert f"`{name}`" in text, f"{path.name} does not mention {name} at all"

    # Digit boundaries on both sides: without them "3" is satisfied by "30" and
    # "$60" is satisfied by "$600", so a halved rail would pass unnoticed.
    figure = re.escape(prefix) + r"(?<!\d)" + re.escape(_plain(value)) + r"(?!\d)(?!\.\d)"

    # Every mention is a candidate, not just the first -- an introductory
    # mention with no figure must not mask the one that carries it.
    windows = [m.group(0) for m in
               re.finditer(re.escape(f"`{name}`") + r".{0," + str(WINDOW) + r"}", text, re.S)]
    assert windows, f"{path.name}: no window around `{name}`"
    assert any(re.search(figure, w) for w in windows), (
        f"{path.name}: {name} is {prefix}{_plain(value)} in src/config.py, but no "
        f"mention quotes it. Windows searched: {windows!r}"
    )


@pytest.mark.parametrize("path", READMES, ids=lambda p: p.name)
def test_readme_has_no_superseded_rail_figures(path: Path) -> None:
    """Pin the specific wrong numbers that shipped, so they cannot come back.

    Matched with \\s+ between words rather than as literal strings: these live in
    hand-wrapped paragraphs, and a reflow would otherwise silently disarm the
    guard while the stale figure sat there.
    """
    text = path.read_text(encoding="utf-8")
    stale = [
        r"max\s+daily\s+loss\s*\(\$200\)",
        r"每日最大虧損（\$200）",
        r"fixed\s+risk\s+per\s+trade\s*\(\$50\)",
        r"固定風險（\$50）",
        r"up\s+to\s+5\s+concurrent\s+positions",
        r"最多\s*5\s*個同時持有的倉位",
    ]
    for pattern in stale:
        assert not re.search(pattern, text), (
            f"{path.name} carries a superseded figure matching {pattern!r}"
        )


def test_prompt_rails_are_interpolated_not_restated() -> None:
    """The advisory auditor must be told the rails the bot actually enforces."""
    from src.llm.base import render_prompt

    template = ROOT / "src" / "llm" / "prompts" / "risk_audit.txt"
    raw = template.read_text(encoding="utf-8")

    # The template itself must not spell any rail out.
    assert "-$200" not in raw, "risk_audit.txt has gone back to a hardcoded daily-loss limit"
    for placeholder in ("{max_daily_loss_usd}", "{max_open_contracts}",
                        "{max_pyramid_adds}", "{stop_atr_min_mult}", "{stop_atr_max_mult}"):
        assert placeholder in raw, f"risk_audit.txt no longer interpolates {placeholder}"

    # render_prompt supplies the rails itself, so a caller that knows nothing
    # about them still gets a correct prompt. That is the property worth having:
    # it is what keeps every existing call site and fixture working.
    system, _ = render_prompt(
        template,
        action="open_long", stop_price=39850.0, trailing_stop_atr_multiple=2.0,
        gemini_reasoning="", proposed_qty=1, position_side="flat", position_qty=0,
        avg_price=0.0, unrealized_pnl=0.0, pyramid_adds_used=0, equity=10000.0,
        realized_pnl_today=0.0, atr14=150.0, mark_price=40000.0,
    )
    assert f"-${_plain(MAX_DAILY_LOSS_USD)}" in system
    assert f"{_plain(MAX_OPEN_CONTRACTS)} contracts" in system
    assert f"fewer than {_plain(MAX_PYRAMID_ADDS)} prior adds" in system
    assert (f"between {_plain(STOP_ATR_MIN_MULT)}x and "
            f"{_plain(STOP_ATR_MAX_MULT)}x ATR") in system

    # The JSON schema in the same half is escaped with doubled braces; if that
    # ever regressed, .format() would raise rather than reach this line -- but
    # assert the output shape too, since a silently mangled schema would teach
    # the auditor to emit unparseable JSON.
    assert '{"approved"' in system


def test_reporting_capital_matches_profile_publisher() -> None:
    """The GitHub profile publisher mirrors this constant and must stay in step.

    /root/Hydr0neFN/update_stats.py carries its own copy to quote DOWTrade's
    return against risk capital rather than the $1M sim balance. It lives
    outside this repo, so this only pins the value its comment refers to.
    """
    assert REPORTING_CAPITAL_USD == 25_000.0
    assert MAX_LLM_SPEND_USD > 0
