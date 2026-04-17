"""
Project-wide configuration and HARD-CODED safety rails.

These constants are the single source of truth for the safety layer.
They MUST NOT be overridable by env vars, CLI flags, config files, or LLMs.
The startup assertions at the bottom refuse to run the process if the
safety posture has been tampered with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =====================================================================
# HARD-CODED SAFETY RAILS (PDF p.9 rules + project non-objectives)
# No LLM, env var, or config file may override any of these.
# =====================================================================

PAPER_ONLY: bool = True
BROKER_ENV: str = "demo"
SYMBOL: str = "MYM"
POINT_VALUE_USD: float = 0.50           # MYM: $0.50 per index point

MAX_DAILY_LOSS_USD: float = 200.0
MAX_OPEN_CONTRACTS: int = 3
MAX_PYRAMID_ADDS: int = 2
FIXED_RISK_PER_TRADE_USD: float = 50.0

MANDATORY_STOP_LOSS: bool = True
NO_AVERAGING_DOWN: bool = True
FLAT_BEFORE_WEEKEND: bool = True

LLM_TIMEOUT_SEC: int = 15
TIMEFRAME_MINUTES: int = 15
TRADING_HOURS_ET: Tuple[str, str] = ("18:00", "17:00")  # Sun 6pm -> Fri 5pm ET

WEEKEND_FLAT_DAY: int = 4                # 0=Mon ... 4=Fri
WEEKEND_FLAT_TIME_ET: str = "16:45"

# ATR-bounded stop: Gemini's proposed stop distance must fall in this range.
STOP_ATR_MIN_MULT: float = 1.0
STOP_ATR_MAX_MULT: float = 3.0

# Budget cap on real-LLM backtests / live paper runs.
# Derived from the user's ~$30/mo estimate in the project brief.
MAX_LLM_SPEND_USD: float = 30.0


# =====================================================================
# Env-backed settings (credentials, URLs, paths). Never safety values.
# =====================================================================

class Settings(BaseSettings):
    """Environment-backed config. Credentials and endpoints only."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Tradovate demo credentials
    tradovate_demo_username: str = Field(default="")
    tradovate_demo_password: str = Field(default="")
    tradovate_demo_app_id: str = Field(default="")
    tradovate_demo_cid: str = Field(default="")
    tradovate_demo_secret: str = Field(default="")
    tradovate_base_url: str = Field(
        default="https://demo.tradovateapi.com/v1",
        description="Must contain 'demo'. Asserted at startup.",
    )

    # LLM API keys
    anthropic_api_key: str = Field(default="")
    google_api_key: str = Field(default="")
    huggingface_api_key: str = Field(default="")

    # Storage / logs
    db_path: str = Field(default="./data/bot.db")
    log_level: str = Field(default="INFO")


def assert_safety_posture(settings: Settings) -> None:
    """
    Startup assertion. Called from main.py and any entry point that may
    place a real order. Refuses to continue outside paper-mode.
    """
    assert PAPER_ONLY is True, "Refusing to start: PAPER_ONLY must be True."
    assert BROKER_ENV == "demo", "Refusing to start: BROKER_ENV must be 'demo'."
    assert SYMBOL == "MYM", "Refusing to start: SYMBOL locked to MYM."
    assert "demo" in settings.tradovate_base_url.lower(), (
        f"Refusing to start: Tradovate URL is not demo "
        f"({settings.tradovate_base_url!r})."
    )
    assert MANDATORY_STOP_LOSS is True
    assert NO_AVERAGING_DOWN is True
    assert MAX_DAILY_LOSS_USD > 0
    assert FIXED_RISK_PER_TRADE_USD > 0
    assert MAX_OPEN_CONTRACTS >= 1
    assert 1 <= MAX_PYRAMID_ADDS <= 5
    assert POINT_VALUE_USD > 0


PROJECT_ROOT = Path(__file__).resolve().parent.parent
