"""Test robustness of portfolio constraints under perturbation.

Validates that the constraint engine correctly limits positions, leverage,
and sector exposure even when alpha scores are extreme or adversarial.
"""

import pytest
import pandas as pd
import numpy as np

from quant_fund.portfolio.portfolio_construction.constraint_engine import (
    ConstraintEngine,
    ConstraintSet,
)
from tests.conftest import STANDARD_TICKERS, STANDARD_SECTORS


pytestmark = [pytest.mark.validation]


class TestConstraintRobustness:
    """Verify constraint engine enforces limits under adversarial inputs."""

    @pytest.fixture
    def engine(self):
        return ConstraintEngine()

    @pytest.fixture
    def constraints(self, engine):
        return engine.build_constraints(sector_map=STANDARD_SECTORS)

    def test_extreme_weights_detected(self, engine, constraints):
        """Weights that violate max_position_size must be flagged."""
        tickers = STANDARD_TICKERS[:10]
        # Create weights with one massive position
        weights = pd.Series(0.0, index=tickers)
        weights.iloc[0] = 0.50  # 50% in single stock
        weights.iloc[1:] = -0.50 / (len(tickers) - 1)

        violations = engine.validate_weights(weights, constraints)
        assert len(violations) > 0, (
            "Constraint engine should detect 50% position vs 2% limit"
        )
        has_position_violation = any("position" in v.lower() for v in violations)
        assert has_position_violation, (
            f"Expected position size violation, got: {violations}"
        )

    def test_leverage_constraint_detected(self, engine, constraints):
        """Gross leverage exceeding max_leverage must be flagged."""
        tickers = STANDARD_TICKERS[:10]
        # 3x leverage: long 1.5, short 1.5
        weights = pd.Series(0.0, index=tickers)
        weights.iloc[:5] = 0.60   # 3.0 long
        weights.iloc[5:] = -0.60  # 3.0 short -> 6.0 gross

        violations = engine.validate_weights(weights, constraints)
        has_leverage_violation = any("leverage" in v.lower() for v in violations)
        assert has_leverage_violation, (
            f"Expected leverage violation for 6x gross, got: {violations}"
        )

    def test_dollar_neutral_constraint_detected(self, engine, constraints):
        """Net exposure significantly different from zero must be flagged
        when dollar_neutral=True."""
        tickers = STANDARD_TICKERS[:10]
        # Pure long portfolio -> not dollar neutral
        weights = pd.Series(0.10, index=tickers)

        violations = engine.validate_weights(weights, constraints)
        has_neutral_violation = any(
            "neutral" in v.lower() or "net" in v.lower() for v in violations
        )
        assert has_neutral_violation, (
            f"Expected dollar-neutral violation for long-only portfolio, "
            f"got: {violations}"
        )
