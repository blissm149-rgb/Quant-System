"""Unit tests for config_validator module.

TESTING_PLAN.md Section 3.1 — Config Subsystem.
"""

import pytest

from quant_fund.config.config_validator import (
    BoundSpec,
    ConfigurationError,
    ConfigValidator,
    EXECUTION_BOUNDS,
    RISK_BOUNDS,
    SYSTEM_BOUNDS,
    TRADING_BOUNDS,
    validate_config,
)


@pytest.mark.unit
@pytest.mark.tier1
class TestConfigValidator:
    """ConfigValidator — validates YAML config files."""

    def test_valid_config_no_errors(self):
        """Valid config produces no errors."""
        config = {"max_leverage": 2.0, "max_position_size": 0.02}
        bounds = [
            BoundSpec(name="max_leverage", min_val=0, max_val=10),
            BoundSpec(name="max_position_size", min_val=0, max_val=1),
        ]
        errors = validate_config(config, bounds)
        assert errors == []

    def test_missing_required_key_raises(self):
        """Missing required keys produce errors."""
        config = {}
        bounds = [BoundSpec(name="max_leverage", required=True)]
        errors = validate_config(config, bounds)
        assert len(errors) > 0
        assert any("max_leverage" in e for e in errors)

    def test_invalid_type_rejected(self):
        """Non-numeric value for numeric bound produces error."""
        config = {"max_leverage": "not_a_number"}
        bounds = [BoundSpec(name="max_leverage", min_val=0, max_val=10, expected_type=float)]
        errors = validate_config(config, bounds)
        assert len(errors) > 0

    def test_value_below_min_rejected(self):
        """Value below minimum bound produces error."""
        config = {"max_leverage": -1.0}
        bounds = [BoundSpec(name="max_leverage", min_val=0, max_val=10)]
        errors = validate_config(config, bounds)
        assert len(errors) > 0

    def test_value_above_max_rejected(self):
        """Value above maximum bound produces error."""
        config = {"max_leverage": 100.0}
        bounds = [BoundSpec(name="max_leverage", min_val=0, max_val=10)]
        errors = validate_config(config, bounds)
        assert len(errors) > 0

    def test_exclusive_vs_inclusive_bounds(self):
        """Exclusive bounds reject boundary values, inclusive accept them."""
        config = {"val": 0.0}
        # min_exclusive=True means > 0 required
        exclusive = [BoundSpec(name="val", min_val=0, min_exclusive=True)]
        errors = validate_config(config, exclusive)
        assert len(errors) > 0

        # min_exclusive=False means >= 0 is OK
        inclusive = [BoundSpec(name="val", min_val=0, min_exclusive=False)]
        errors = validate_config(config, inclusive)
        assert errors == []

    def test_nested_config_value_found(self):
        """Values in nested dicts are found by dotted key path."""
        config = {"risk": {"drawdown_limit": 0.20}}
        bounds = [BoundSpec(name="drawdown_limit", min_val=0, max_val=1)]
        errors = validate_config(config, bounds)
        assert errors == []

    def test_validate_all_raises_on_errors(self):
        """validate_all raises ConfigurationError when any config is invalid."""
        validator = ConfigValidator()
        validator.add_bounds("trading", TRADING_BOUNDS)
        with pytest.raises(ConfigurationError):
            validator.validate_all({"trading": {"max_leverage": -5}})

    def test_validate_single_returns_error_list(self):
        """validate_single returns list of error strings, does not raise."""
        validator = ConfigValidator()
        errors = validator.validate_single(
            {"max_leverage": -1},
            "trading",
            bounds=TRADING_BOUNDS,
        )
        assert isinstance(errors, list)
        assert len(errors) > 0

    def test_builtin_risk_bounds_exist(self):
        """RISK_BOUNDS contains expected parameters."""
        names = {b.name for b in RISK_BOUNDS}
        assert "drawdown_limit" in names

    def test_builtin_trading_bounds_exist(self):
        """TRADING_BOUNDS contains expected parameters."""
        names = {b.name for b in TRADING_BOUNDS}
        assert "max_leverage" in names

    def test_multiple_errors_accumulated(self):
        """All errors collected, not just the first."""
        config = {"a": -1, "b": -1}
        bounds = [
            BoundSpec(name="a", min_val=0),
            BoundSpec(name="b", min_val=0),
        ]
        errors = validate_config(config, bounds)
        assert len(errors) >= 2
