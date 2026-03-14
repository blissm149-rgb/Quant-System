"""Stress test engine running portfolio against historical scenarios.

Runs offline (not in live trading loop). Reports estimated drawdown and
factor P&L attribution under each scenario.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StressTestResult:
    """Result of a single stress test scenario."""

    scenario_name: str
    portfolio_return: float
    portfolio_drawdown: float
    factor_pnl: Dict[str, float]
    description: str


# Pre-defined stress scenario factor shocks (annualised, applied as daily equivalent)
DEFAULT_SCENARIOS = {
    "2008_financial_crisis": {
        "description": "2008 GFC: broad market crash with credit/liquidity shock",
        "shocks": {
            "market": -0.40,
            "momentum": -0.30,
            "value": -0.25,
            "quality": 0.05,
            "low_vol": -0.10,
            "size": -0.20,
        },
    },
    "2020_covid_crash": {
        "description": "2020 COVID: rapid sell-off with factor rotation",
        "shocks": {
            "market": -0.35,
            "momentum": -0.40,
            "value": -0.15,
            "quality": 0.10,
            "low_vol": 0.05,
            "size": -0.25,
        },
    },
    "2022_rate_shock": {
        "description": "2022 rate shock: duration and growth factor drawdown",
        "shocks": {
            "market": -0.20,
            "momentum": -0.15,
            "value": 0.10,
            "quality": -0.05,
            "low_vol": 0.05,
            "size": -0.10,
        },
    },
}


class StressTestEngine:
    """Runs portfolio against historical stress scenarios.

    Reports estimated portfolio drawdown and factor P&L attribution
    under each scenario.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        scenario_names = cfg.get("stress_tests", list(DEFAULT_SCENARIOS.keys()))
        self._scenarios = {
            name: DEFAULT_SCENARIOS[name]
            for name in scenario_names
            if name in DEFAULT_SCENARIOS
        }

    def run_all(
        self,
        weights: pd.Series,
        factor_exposures: pd.DataFrame,
        idiosyncratic_variance: Optional[pd.Series] = None,
    ) -> List[StressTestResult]:
        """Run all configured stress scenarios.

        Args:
            weights: Portfolio weights indexed by ticker.
            factor_exposures: DataFrame (tickers x factors).
            idiosyncratic_variance: Optional per-stock idio variance.

        Returns:
            List of StressTestResult, one per scenario.
        """
        results = []
        for name, scenario in self._scenarios.items():
            result = self.run_scenario(
                name, scenario, weights, factor_exposures, idiosyncratic_variance
            )
            results.append(result)
        return results

    def run_scenario(
        self,
        name: str,
        scenario: dict,
        weights: pd.Series,
        factor_exposures: pd.DataFrame,
        idiosyncratic_variance: Optional[pd.Series] = None,
    ) -> StressTestResult:
        """Run a single stress scenario.

        Portfolio return under stress = sum of factor contributions + idio noise.
        Factor contribution = portfolio_factor_exposure * factor_shock.
        """
        shocks = scenario["shocks"]
        common_tickers = weights.index.intersection(factor_exposures.index)
        w = weights.reindex(common_tickers, fill_value=0.0)
        B = factor_exposures.reindex(common_tickers, fill_value=0.0)

        # Portfolio factor exposure
        portfolio_factor_exp = {}
        for factor in B.columns:
            portfolio_factor_exp[factor] = float((w * B[factor]).sum())

        # Factor P&L contribution
        factor_pnl = {}
        total_return = 0.0
        for factor, exposure in portfolio_factor_exp.items():
            shock = shocks.get(factor, 0.0)
            pnl = exposure * shock
            factor_pnl[factor] = pnl
            total_return += pnl

        # Add idiosyncratic risk estimate (conservative: assume 1 std move)
        if idiosyncratic_variance is not None:
            idio = idiosyncratic_variance.reindex(common_tickers, fill_value=0.0)
            idio_contribution = -float((w.abs() * np.sqrt(idio)).sum()) * 0.5
            factor_pnl["idiosyncratic"] = idio_contribution
            total_return += idio_contribution

        return StressTestResult(
            scenario_name=name,
            portfolio_return=total_return,
            portfolio_drawdown=abs(min(0, total_return)),
            factor_pnl=factor_pnl,
            description=scenario.get("description", ""),
        )
