"""Unit tests for base_feature_generator module.

TESTING_PLAN.md Section 3.3 — Feature Factory.
"""

import pytest

from quant_fund.feature_factory.base_feature_generator import BaseFeatureGenerator


@pytest.mark.unit
@pytest.mark.tier1
class TestBaseFeatureGenerator:
    """BaseFeatureGenerator — abstract interface."""

    def test_cannot_instantiate_directly(self):
        """Abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseFeatureGenerator("test", 20, "daily")

    def test_subclass_must_implement_compute(self):
        """Subclass without compute() raises TypeError."""
        class Incomplete(BaseFeatureGenerator):
            def validate(self, feature_output):
                return True

        with pytest.raises(TypeError):
            Incomplete("test", 20, "daily")

    def test_subclass_must_implement_validate(self):
        """Subclass without validate() raises TypeError."""
        import pandas as pd

        class Incomplete(BaseFeatureGenerator):
            def compute(self, data, as_of):
                return pd.Series()

        with pytest.raises(TypeError):
            Incomplete("test", 20, "daily")

    def test_complete_subclass_instantiates(self):
        """Subclass with all abstract methods can be instantiated."""
        import pandas as pd

        class Complete(BaseFeatureGenerator):
            def compute(self, data, as_of):
                return pd.Series()
            def validate(self, feature_output):
                return True

        gen = Complete("test_feature", lookback_days=20, recompute_frequency="daily")
        assert gen.feature_name == "test_feature"
        assert gen.lookback_days == 20

    def test_get_metadata(self):
        """get_metadata returns correct attributes."""
        import pandas as pd

        class Complete(BaseFeatureGenerator):
            def compute(self, data, as_of):
                return pd.Series()
            def validate(self, feature_output):
                return True

        gen = Complete("my_feature", lookback_days=60, recompute_frequency="daily")
        meta = gen.get_metadata()
        assert meta["feature_name"] == "my_feature"
        assert meta["lookback_days"] == 60
        assert meta["recompute_frequency"] == "daily"
