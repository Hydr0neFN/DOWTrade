"""
Exhaustive tests for src/sizing/risk_unit.py — targets >= 95% branch coverage.

Run from the trading-bot root:
    python3 -m pytest tests/test_sizing.py -v
"""

from __future__ import annotations

import math
import pytest

from src.sizing.risk_unit import SizingResult, compute_size


# ======================================================================
# Happy-path: correct contract counts
# ======================================================================

class TestHappyPath:
    """Named cases from the PDF p.10 spec."""

    def test_mym_50pt_stop_2_contracts(self):
        # 50 pts * $0.50 = $25/contract, $50 risk → floor(50/25) = 2
        # risk_usd = 2 * $25 = $50
        result = compute_size(42000, 41950)
        assert result.contracts == 2
        assert result.skip_reason is None
        assert math.isclose(result.stop_distance_points, 50.0)
        assert math.isclose(result.risk_usd, 50.0)  # 2 * $25

    def test_mym_20pt_stop_capped_at_3(self):
        # 20 pts * $0.50 = $10/contract, $50 risk → floor(50/10) = 5, capped to 3
        result = compute_size(42000, 41980)
        assert result.contracts == 3
        assert result.skip_reason is None
        assert math.isclose(result.stop_distance_points, 20.0)
        assert math.isclose(result.risk_usd, 30.0)  # 3 * $10

    def test_short_side_same_math(self):
        # entry < stop (short): 50 pts * $0.50 = $25/contract → 2 contracts
        result = compute_size(42000, 42050)
        assert result.contracts == 2
        assert result.skip_reason is None
        assert math.isclose(result.stop_distance_points, 50.0)

    def test_exact_fit_1_contract(self):
        # 100 pts * $0.50 = $50/contract, $50 risk → floor(50/50) = 1 exactly
        result = compute_size(42000, 41900)
        assert result.contracts == 1
        assert result.skip_reason is None
        assert math.isclose(result.stop_distance_points, 100.0)
        assert math.isclose(result.risk_usd, 50.0)


# ======================================================================
# Skip path: returns 0 contracts with reason string
# ======================================================================

class TestSkipPath:

    def test_stop_too_wide_returns_skip(self):
        # 1000 pts * $0.50 = $500/contract, $50 risk → floor(50/500) = 0 → skip
        result = compute_size(42000, 41000)
        assert result.contracts == 0
        assert result.risk_usd == 0.0
        assert result.skip_reason == "stop too wide for risk unit"
        assert math.isclose(result.stop_distance_points, 1000.0)

    def test_zero_stop_distance_returns_skip(self):
        result = compute_size(42000, 42000)
        assert result.contracts == 0
        assert result.risk_usd == 0.0
        assert result.skip_reason == "zero stop distance"
        assert result.stop_distance_points == 0.0


# ======================================================================
# Error path: ValueError on non-positive inputs
# ======================================================================

class TestErrors:

    def test_negative_entry_raises(self):
        with pytest.raises(ValueError, match="entry"):
            compute_size(-100, 42000)

    def test_zero_entry_raises(self):
        with pytest.raises(ValueError, match="entry"):
            compute_size(0.0, 42000)

    def test_negative_stop_raises(self):
        with pytest.raises(ValueError, match="stop"):
            compute_size(42000, -1)

    def test_zero_stop_raises(self):
        with pytest.raises(ValueError, match="stop"):
            compute_size(42000, 0.0)

    def test_zero_fixed_risk_raises(self):
        with pytest.raises(ValueError, match="fixed_risk_usd"):
            compute_size(42000, 41950, fixed_risk_usd=0.0)

    def test_negative_fixed_risk_raises(self):
        with pytest.raises(ValueError, match="fixed_risk_usd"):
            compute_size(42000, 41950, fixed_risk_usd=-50.0)

    def test_zero_point_value_raises(self):
        with pytest.raises(ValueError, match="point_value_usd"):
            compute_size(42000, 41950, point_value_usd=0.0)

    def test_negative_point_value_raises(self):
        with pytest.raises(ValueError, match="point_value_usd"):
            compute_size(42000, 41950, point_value_usd=-0.50)

    def test_zero_max_contracts_raises(self):
        with pytest.raises(ValueError, match="max_contracts"):
            compute_size(42000, 41950, max_contracts=0)

    def test_negative_max_contracts_raises(self):
        with pytest.raises(ValueError, match="max_contracts"):
            compute_size(42000, 41950, max_contracts=-1)


# ======================================================================
# Edge / parametric cases
# ======================================================================

@pytest.mark.parametrize(
    "entry, stop, expected_contracts",
    [
        # Floor truncates: 40 pts * $0.50 = $20/c, $50/$20 = 2.5 → floor = 2
        (42000, 41960, 2),
        # Minimum 1-contract: 100 pts * $0.50 = $50/c, exactly 1
        (42000, 41900, 1),
        # Capped to max_contracts (3): 10 pts * $0.50 = $5/c, $50/$5 = 10 → cap 3
        (42000, 41990, 3),
        # Short side: entry < stop, 100 pts * $0.50 = $50/c → 1
        (41900, 42000, 1),
        # Very tight stop: 1 pt * $0.50 = $0.50/c, $50/$0.50 = 100 → capped 3
        (1000, 999, 3),
        # Revisit exact-fit long
        (42000, 41900, 1),
    ],
)
def test_parametric_grid(entry, stop, expected_contracts):
    result = compute_size(entry, stop)
    assert result.contracts == expected_contracts, (
        f"entry={entry}, stop={stop}: expected {expected_contracts} contracts, "
        f"got {result.contracts} (stop_dist={result.stop_distance_points})"
    )


def test_risk_usd_never_exceeds_fixed_risk():
    """risk_usd must be <= fixed_risk_usd whenever contracts > 0."""
    test_cases = [
        (42000, 41960),
        (42000, 41950),
        (42000, 41980),
        (42000, 41900),
        (42000, 41990),
    ]
    for entry, stop in test_cases:
        result = compute_size(entry, stop)
        if result.contracts > 0:
            assert result.risk_usd <= 50.0 + 1e-9, (
                f"risk_usd {result.risk_usd} exceeds fixed_risk_usd 50.0 "
                f"for entry={entry}, stop={stop}"
            )


def test_sizing_result_is_frozen():
    """SizingResult is a frozen dataclass; mutation must raise FrozenInstanceError."""
    result = compute_size(42000, 41950)
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError (or AttributeError)
        result.contracts = 99  # type: ignore[misc]


def test_custom_risk_and_point_value():
    """Overriding defaults produces correct results."""
    # 50 pts * $1.00 = $50/contract, $100 risk → floor(100/50) = 2 contracts
    result = compute_size(42000, 41950, fixed_risk_usd=100.0, point_value_usd=1.0)
    assert result.contracts == 2
    assert math.isclose(result.risk_usd, 100.0)


def test_custom_max_contracts_respected():
    """max_contracts override caps correctly at a lower value."""
    # Without cap: floor($50 / (20pts * $0.50)) = 5; cap to 2
    result = compute_size(42000, 41980, max_contracts=2)
    assert result.contracts == 2


def test_skip_reason_is_none_on_valid_trade():
    result = compute_size(42000, 41950)
    assert result.skip_reason is None


def test_stop_distance_points_always_positive_on_skip():
    """stop_distance_points reflects the actual distance even on a skip."""
    result = compute_size(42000, 41000)   # too wide → skip
    assert result.stop_distance_points == 1000.0
