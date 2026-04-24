"""
src/backtest/stubs.py
=====================
Rule-based stubs that impersonate the three LLMs used in the live pipeline.
No real LLM calls are made -- all logic is encoded from the PDF-philosophy
rules so backtests run deterministically and cheaply.

Output dicts match the exact JSON schemas from the project brief section 6.

Note on "new_hh_break" for down-trends:
    The schema keeps a single signal name for schema consistency. In a down-
    trend context "new_hh_break" means "new structural continuation break in
    the current trend direction" -- i.e. a close below the last confirmed LL.
"""

from __future__ import annotations

from datetime import time
from typing import Callable, List, Optional
from zoneinfo import ZoneInfo

from src.broker.models import AccountState, Position
from src.config import (
    FIXED_RISK_PER_TRADE_USD,
    MAX_DAILY_LOSS_USD,
    MAX_OPEN_CONTRACTS,
    MAX_PYRAMID_ADDS,
    POINT_VALUE_USD,
    STOP_ATR_MAX_MULT,
    STOP_ATR_MIN_MULT,
    TRADING_HOURS_ET,
    WEEKEND_FLAT_DAY,
    WEEKEND_FLAT_TIME_ET,
)
from src.data.features import MarketSnapshot, SwingPoint

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_time(t: str) -> time:
    h, m = t.split(":")
    return time(int(h), int(m))


def _swings_of_kind(swings: List[SwingPoint], *kinds: str) -> List[SwingPoint]:
    return [s for s in swings if s.kind in kinds]


# ---------------------------------------------------------------------------
# StubHaiku -- structural judge (PDF p.8)
# ---------------------------------------------------------------------------

class StubHaiku:
    """
    Structural judge.

    Reads swing-point labels (HH/HL/LH/LL) and SMA-200 direction to
    classify trend, identify the last confirmed swing levels, assess
    pattern integrity, and emit a structural signal for downstream use.
    """

    def evaluate(self, snapshot: MarketSnapshot) -> dict:
        swings = snapshot.swings
        price = snapshot.current_price

        # ---- trend classification ----------------------------------------
        if snapshot.sma200 is None:
            trend = "sideways"
        else:
            sma_rising = snapshot.current_price > snapshot.sma200

            highs = _swings_of_kind(swings, "HH", "LH")
            lows  = _swings_of_kind(swings, "HL", "LL")

            last2_highs = highs[-2:] if len(highs) >= 2 else []
            last2_lows  = lows[-2:]  if len(lows) >= 2 else []

            up_highs = len(last2_highs) == 2 and all(s.kind == "HH" for s in last2_highs)
            up_lows  = len(last2_lows)  == 2 and all(s.kind == "HL" for s in last2_lows)
            dn_highs = len(last2_highs) == 2 and all(s.kind == "LH" for s in last2_highs)
            dn_lows  = len(last2_lows)  == 2 and all(s.kind == "LL" for s in last2_lows)

            if sma_rising and up_highs and up_lows:
                trend = "up"
            elif (not sma_rising) and dn_highs and dn_lows:
                trend = "down"
            else:
                trend = "sideways"

        # ---- last confirmed levels ----------------------------------------
        hh_swings = _swings_of_kind(swings, "HH")
        hl_swings = _swings_of_kind(swings, "HL")
        ll_swings = _swings_of_kind(swings, "LL")

        last_confirmed_hh: Optional[float] = hh_swings[-1].price if hh_swings else None
        last_confirmed_hl: Optional[float] = hl_swings[-1].price if hl_swings else None
        last_confirmed_ll: Optional[float] = ll_swings[-1].price if ll_swings else None

        # ---- pattern intact -----------------------------------------------
        # Checks the last 4 swings for contradictory structure:
        #   - LL appearing in a sequence that also has HH = up-trend reversal
        #   - HH appearing in a sequence that also has LH/LL = down-trend reversal
        # Evaluated independently of current trend so it fires at the moment
        # the break happens (which itself may reclassify the trend).
        pattern_intact = True
        if len(swings) >= 2:
            last4 = swings[-4:]
            kinds4 = {s.kind for s in last4}
            if "HH" in kinds4 and "LL" in kinds4:
                # LL in an up-structure context -> broken
                pattern_intact = False
            elif "LH" in kinds4 and "HH" in kinds4:
                # HH appearing after down-structure -> broken
                pattern_intact = False

        # ---- structural signal --------------------------------------------
        structural_signal = "none"

        if trend == "up":
            if last_confirmed_hh is not None and price > last_confirmed_hh:
                structural_signal = "new_hh_break"
            elif (
                swings
                and swings[-1].kind == "HL"
                and last_confirmed_hl is not None
                and price > last_confirmed_hl
            ):
                structural_signal = "new_hl_hold"
        elif trend == "down":
            if last_confirmed_ll is not None and price < last_confirmed_ll:
                structural_signal = "new_hh_break"

        # Pattern broken overrides other signals
        if not pattern_intact:
            structural_signal = "pattern_broken"

        # ---- confidence --------------------------------------------------
        _conf_map = {
            "new_hh_break":   0.8,
            "new_hl_hold":    0.8,
            "pattern_broken": 0.9,
            "none":           0.4,
        }
        if trend == "sideways" and structural_signal == "none":
            confidence = 0.5
        else:
            confidence = _conf_map.get(structural_signal, 0.4)

        # ---- reasoning ---------------------------------------------------
        reasoning = (
            "Trend=" + trend + "; signal=" + structural_signal + "; "
            "last_HH=" + str(last_confirmed_hh) + "; last_HL=" + str(last_confirmed_hl) + "; "
            "pattern_intact=" + str(pattern_intact) + "."
        )

        return {
            "trend": trend,
            "last_confirmed_hh": last_confirmed_hh,
            "last_confirmed_hl": last_confirmed_hl,
            "pattern_intact": pattern_intact,
            "structural_signal": structural_signal,
            "confidence_0_to_1": confidence,
            "reasoning": reasoning,
        }


# ---------------------------------------------------------------------------
# StubGemini -- execution judge (brief section 6.2)
# ---------------------------------------------------------------------------

class StubGemini:
    """
    Execution judge.

    Translates Haiku structural signal into a concrete trade action,
    computing stop prices but never position sizes (Python layer does that).
    """

    def evaluate(
        self,
        haiku: dict,
        snapshot: MarketSnapshot,
        position: Position,
        equity: float,
    ) -> dict:
        trend  = haiku["trend"]
        signal = haiku["structural_signal"]
        atr14  = snapshot.atr14
        price  = snapshot.current_price

        # Insufficient data guard
        if atr14 is None:
            return {
                "action": "hold",
                "stop_price": 0.0,
                "trailing_stop_atr_multiple": 2.0,
                "reasoning": "ATR unavailable -- insufficient bars for entry decision.",
            }


        # Adaptive stop multiplier: pick a value in [STOP_ATR_MIN_MULT,
        # STOP_ATR_MAX_MULT] x ATR such that risk per single contract stays
        # within FIXED_RISK_PER_TRADE_USD. Prefer 2x ATR; tighten when ATR
        # is too wide for the risk budget.
        if atr14 > 0:
            max_mult_for_one_contract = FIXED_RISK_PER_TRADE_USD / (atr14 * POINT_VALUE_USD)
        else:
            max_mult_for_one_contract = STOP_ATR_MIN_MULT
        preferred_mult = 2.0
        stop_mult = max(
            STOP_ATR_MIN_MULT,
            min(STOP_ATR_MAX_MULT, min(preferred_mult, max_mult_for_one_contract)),
        )

        is_flat  = position.side == "flat"
        is_long  = position.side == "long"
        is_short = position.side == "short"

        # 1. Pattern broken with open position -> close
        if signal == "pattern_broken" and not is_flat:
            return {
                "action": "close",
                "stop_price": 0.0,
                "trailing_stop_atr_multiple": 2.0,
                "reasoning": "Pattern broken -- closing position to protect capital.",
            }

        # 2. Flat position: look for entry
        if is_flat:
            if trend == "up" and signal in ("new_hh_break", "new_hl_hold"):
                stop = price - stop_mult * atr14
                return {
                    "action": "open_long",
                    "stop_price": round(stop, 2),
                    "trailing_stop_atr_multiple": stop_mult,
                    "reasoning": (
                        "Uptrend confirmed with " + signal + "; opening long "
                        "with stop at " + str(round(stop, 2)) + "."
                    ),
                }
            if trend == "down" and signal == "new_hh_break":
                stop = price + stop_mult * atr14
                return {
                    "action": "open_short",
                    "stop_price": round(stop, 2),
                    "trailing_stop_atr_multiple": stop_mult,
                    "reasoning": (
                        "Downtrend continuation break; opening short "
                        "with stop at " + str(round(stop, 2)) + "."
                    ),
                }
            return {
                "action": "hold",
                "stop_price": 0.0,
                "trailing_stop_atr_multiple": 2.0,
                "reasoning": "No actionable signal for entry.",
            }

        # 3. Open position -- check for pyramid add or hold
        # Never propose counter-trend action
        position_aligned = (trend == "up" and is_long) or (trend == "down" and is_short)

        if position_aligned:
            profitable = position.unrealized_pnl > 0
            below_max  = position.pyramid_adds_used < MAX_PYRAMID_ADDS

            if signal == "new_hh_break" and profitable and below_max:
                if is_long:
                    stop = price - stop_mult * atr14
                else:
                    stop = price + stop_mult * atr14
                return {
                    "action": "add_pyramid",
                    "stop_price": round(stop, 2),
                    "trailing_stop_atr_multiple": stop_mult,
                    "reasoning": (
                        "Adding pyramid #" + str(position.pyramid_adds_used + 1)
                        + " on " + signal + "; profitable position."
                    ),
                }

        return {
            "action": "hold",
            "stop_price": 0.0,
            "trailing_stop_atr_multiple": 2.0,
            "reasoning": "Holding -- no pyramid criteria met or counter-trend position.",
        }


# ---------------------------------------------------------------------------
# StubDeepSeek -- risk auditor (brief section 6.3)
# ---------------------------------------------------------------------------

class StubDeepSeek:
    """
    Risk auditor.

    Runs a deterministic checklist against the proposed action and account
    state. Any violation flips approved=False and sets override_action.
    """

    def evaluate(
        self,
        gemini: dict,
        proposed_qty: int,
        state: AccountState,
        atr14: float,
    ) -> dict:
        action     = gemini["action"]
        stop_price = gemini.get("stop_price") or 0.0
        pos        = state.position
        violations: list = []

        is_size_action = action in ("open_long", "open_short", "add_pyramid")
        is_close_hold  = action in ("close", "hold")

        # 1. MISSING_STOP
        if is_size_action and (stop_price is None or stop_price == 0.0):
            violations.append("MISSING_STOP")

        # 2. STOP_ATR_OUT_OF_BOUNDS -- for open/pyramid with an existing position
        if is_size_action and stop_price and atr14 and atr14 > 0:
            if pos.side != "flat":
                ref_price = pos.avg_price
                distance = abs(ref_price - stop_price)
                min_dist = STOP_ATR_MIN_MULT * atr14
                max_dist = STOP_ATR_MAX_MULT * atr14
                if not (min_dist <= distance <= max_dist):
                    violations.append("STOP_ATR_OUT_OF_BOUNDS")

        # 3. AVERAGING_DOWN -- adding size to a losing same-side position
        if is_size_action and pos.side != "flat" and pos.unrealized_pnl < 0:
            action_side = None
            if action == "open_long":
                action_side = "long"
            elif action == "open_short":
                action_side = "short"
            elif action == "add_pyramid":
                action_side = pos.side
            if action_side == pos.side:
                violations.append("AVERAGING_DOWN")

        # 4. PYRAMID_VIOLATION
        if action == "add_pyramid":
            if pos.side == "flat":
                violations.append("PYRAMID_VIOLATION")
            elif pos.unrealized_pnl <= 0:
                violations.append("PYRAMID_VIOLATION")
            elif pos.pyramid_adds_used >= MAX_PYRAMID_ADDS:
                violations.append("PYRAMID_VIOLATION")

        # 5. DAILY_LOSS_LIMIT
        total_pnl = state.realized_pnl_today + state.unrealized_pnl
        if total_pnl <= -MAX_DAILY_LOSS_USD:
            violations.append("DAILY_LOSS_LIMIT")

        # 6. MAX_CONTRACTS
        if is_size_action:
            current_qty = pos.qty if pos.side != "flat" else 0
            if current_qty + proposed_qty > MAX_OPEN_CONTRACTS:
                violations.append("MAX_CONTRACTS")

        # 7. OUT_OF_HOURS
        if self._check_out_of_hours(state):
            violations.append("OUT_OF_HOURS")

        # ---- build result ------------------------------------------------
        if violations:
            override_action = None if is_close_hold else "hold"
            reasoning = "Violations: " + "; ".join(violations) + "."
            return {
                "approved": False,
                "violations": violations,
                "override_action": override_action,
                "reasoning": reasoning,
            }

        return {
            "approved": True,
            "violations": [],
            "override_action": None,
            "reasoning": "All checks passed.",
        }

    @staticmethod
    def _check_out_of_hours(state: AccountState) -> bool:
        """Return True if the current time is outside valid trading hours."""
        now = state.now_et
        if now.tzinfo is None:
            return True
        now_et = now.astimezone(ET)

        dow = now_et.weekday()   # 0=Mon ... 6=Sun
        t   = now_et.time().replace(second=0, microsecond=0)

        session_open  = _parse_time(TRADING_HOURS_ET[0])   # 18:00
        session_close = _parse_time(TRADING_HOURS_ET[1])   # 17:00
        flat_time     = _parse_time(WEEKEND_FLAT_TIME_ET)  # 16:45

        # Saturday: market closed
        if dow == 5:
            return True

        # Sunday: only open after 18:00
        if dow == 6:
            return t < session_open

        # Daily maintenance 17:00-18:00 (Mon-Fri)
        if session_close <= t < session_open:
            return True

        # Friday after flat-time: new opens/adds are out-of-hours
        if dow == WEEKEND_FLAT_DAY and t >= flat_time:
            return True

        return False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_stub_pipeline(
    snapshot: MarketSnapshot,
    state: AccountState,
    proposed_qty_fn: Callable[[float, float], int],
) -> tuple:
    """
    Run Haiku -> Gemini -> DeepSeek in order.

    Args:
        snapshot:        Current market snapshot.
        state:           Current account/position state.
        proposed_qty_fn: Callable(risk_usd, atr) -> int that computes
                         contract qty (e.g. wraps compute_size from
                         src/sizing/risk_unit.py).

    Returns:
        (haiku_dict, gemini_dict, deepseek_dict) -- three JSON-compatible
        dicts matching the schemas from brief section 6.
    """
    haiku_result  = StubHaiku().evaluate(snapshot)
    gemini_result = StubGemini().evaluate(
        haiku_result, snapshot, state.position, state.equity
    )

    atr14 = snapshot.atr14 or 0.0
    proposed_qty = proposed_qty_fn(FIXED_RISK_PER_TRADE_USD, atr14) if atr14 > 0 else 1

    deepseek_result = StubDeepSeek().evaluate(
        gemini_result, proposed_qty, state, atr14
    )

    return haiku_result, gemini_result, deepseek_result
