"""Validation framework shared fixtures.

Extends the root tests/conftest.py with validation-specific fixtures
for backtest harness, multi-regime data, and factor generator collections.
"""

import pytest
import pandas as pd
import numpy as np

from tests.conftest import (
    STANDARD_TICKERS,
    STANDARD_SECTORS,
    STANDARD_MARKET_DATA,
    make_ohlcv,
    make_returns,
    make_factor_returns,
    make_cointegrated_pair,
)
from tests.generators.market_regime_simulator import MarketRegimeSimulator
from tests.validation.shared.backtest_harness import BacktestHarness


@pytest.fixture
def backtest_harness():
    """Returns a configured BacktestHarness for running strategies over synthetic data."""
    return BacktestHarness(seed=42, initial_nav=1_000_000.0)


@pytest.fixture
def multi_regime_data():
    """252-day dataset with labeled regime transitions.

    Sequence: bull(60) -> crash(5) -> bear(30) -> recovery(60) -> sideways(97)
    """
    sim = MarketRegimeSimulator(tickers=STANDARD_TICKERS[:5], seed=42)
    return sim.regime_sequence(
        [
            ("bull", {"days": 60}),
            ("crash", {"days": 5}),
            ("bear", {"days": 30}),
            ("recovery", {"days": 60}),
            ("sideways", {"days": 97}),
        ]
    )


@pytest.fixture
def all_factor_generators():
    """Instantiates all 5 factor models + 2 mean reversion strategies."""
    from quant_fund.research_algorithms.factor_models.momentum_factor import (
        MomentumFactor,
    )
    from quant_fund.research_algorithms.factor_models.value_factor import ValueFactor
    from quant_fund.research_algorithms.factor_models.quality_factor import (
        QualityFactor,
    )
    from quant_fund.research_algorithms.factor_models.low_volatility_factor import (
        LowVolatilityFactor,
    )
    from quant_fund.research_algorithms.factor_models.size_factor import SizeFactor
    from quant_fund.research_algorithms.mean_reversion.zscore_reversion_strategy import (
        ZScoreReversionStrategy,
    )
    from quant_fund.research_algorithms.mean_reversion.short_term_reversal_strategy import (
        ShortTermReversalStrategy,
    )

    return [
        MomentumFactor(),
        ValueFactor(),
        QualityFactor(),
        LowVolatilityFactor(),
        SizeFactor(),
        ZScoreReversionStrategy(),
        ShortTermReversalStrategy(),
    ]


@pytest.fixture
def validation_ohlcv():
    """Standard 500-day OHLCV dataset for validation tests."""
    return make_ohlcv(tickers=STANDARD_TICKERS[:5], periods=500, seed=42)


@pytest.fixture
def validation_returns():
    """Standard 500-day returns dataset for validation tests."""
    return make_returns(n_dates=500, tickers=STANDARD_TICKERS[:5], seed=42)
