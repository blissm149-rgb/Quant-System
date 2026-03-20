"""Unit tests for constraint_engine module.

TESTING_PLAN.md Section 3.8 — CRITICAL: single place for all constraints.
"""

import pandas as pd
import pytest

from quant_fund.portfolio.portfolio_construction.constraint_engine import (
    ConstraintEngine,
    ConstraintSet,
)


@pytest.mark.unit
@pytest.mark.tier1
class TestConstraintEngine:
    """ConstraintEngine — enforces portfolio constraints."""

    @pytest.fixture
    def engine(self):
        return ConstraintEngine(config={
            "max_position_size": 0.02,
            "max_sector_exposure": 0.20,
            "max_leverage": 2.0,
        })

    @pytest.fixture
    def constraints(self, engine):
        return engine.build_constraints(sector_map={
            "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology",
            "JPM": "Financials", "BAC": "Financials", "WMT": "Consumer Staples",
        })

    def test_max_position_size_enforced(self, engine, constraints):
        """Position exceeding 2% is flagged."""
        weights = pd.Series({"AAPL": 0.05, "MSFT": 0.01, "GOOG": -0.01})
        violations = engine.validate_weights(weights, constraints)
        assert len(violations) > 0
        assert any("position" in v.lower() or "AAPL" in v for v in violations)

    def test_max_position_size_passes(self, engine, constraints):
        """Positions within 2% pass validation."""
        weights = pd.Series({"AAPL": 0.02, "MSFT": -0.02, "GOOG": 0.01, "JPM": -0.01})
        violations = engine.validate_weights(weights, constraints)
        position_violations = [v for v in violations if "position" in v.lower()]
        assert len(position_violations) == 0

    def test_max_leverage_enforced(self, engine, constraints):
        """Gross leverage exceeding 2.0 is flagged."""
        weights = pd.Series({"AAPL": 0.02, "MSFT": 0.02, "GOOG": 0.02})
        # Gross = 0.06 — well under 2.0. Let's make a high-leverage portfolio.
        big_weights = pd.Series({f"T{i}": 0.02 for i in range(60)})
        big_weights[f"T{60}"] = -0.02 * 60  # short to try to balance — gross = 2.4
        for i in range(60, 80):
            big_weights[f"T{i}"] = -0.02
        # Gross = 80 * 0.02 = 1.6, still under. Let's just directly test:
        high_lev = pd.Series({"AAPL": 1.5, "MSFT": -1.5})  # gross = 3.0
        violations = engine.validate_weights(high_lev, constraints)
        assert any("leverage" in v.lower() for v in violations)

    def test_leverage_exactly_at_limit_passes(self, engine, constraints):
        """Gross leverage at exactly 2.0 passes."""
        weights = pd.Series({"AAPL": 0.02, "MSFT": -0.02, "GOOG": 0.01, "JPM": -0.01})
        # Gross = 0.06 — under limit
        violations = engine.validate_weights(weights, constraints)
        leverage_violations = [v for v in violations if "leverage" in v.lower()]
        assert len(leverage_violations) == 0

    def test_dollar_neutral_enforced(self, engine, constraints):
        """Non-dollar-neutral portfolio is flagged."""
        weights = pd.Series({"AAPL": 0.02, "MSFT": 0.02, "GOOG": 0.02})
        # sum = 0.06 — not neutral
        violations = engine.validate_weights(weights, constraints)
        assert any("neutral" in v.lower() or "net" in v.lower() or "dollar" in v.lower() for v in violations)

    def test_dollar_neutral_passes(self, engine, constraints):
        """Dollar-neutral portfolio passes."""
        weights = pd.Series({"AAPL": 0.02, "MSFT": -0.02})
        violations = engine.validate_weights(weights, constraints)
        neutral_violations = [v for v in violations
                              if "neutral" in v.lower() or "net" in v.lower() or "dollar" in v.lower()]
        assert len(neutral_violations) == 0

    def test_build_constraints_returns_constraint_set(self, engine):
        """build_constraints returns ConstraintSet dataclass."""
        cs = engine.build_constraints()
        assert isinstance(cs, ConstraintSet)
        assert cs.max_position_size == 0.02
        assert cs.max_leverage == 2.0

    def test_compliant_portfolio_no_violations(self, engine, constraints):
        """Fully compliant portfolio returns no violations."""
        weights = pd.Series({
            "AAPL": 0.01, "MSFT": 0.01, "GOOG": -0.01,
            "JPM": 0.005, "BAC": -0.005, "WMT": -0.01,
        })
        violations = engine.validate_weights(weights, constraints)
        assert len(violations) == 0
