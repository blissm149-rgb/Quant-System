"""Cointegration engine for pairs trading.

Identifies cointegrated pairs within sectors using Engle-Granger test.
Refreshed monthly. Only pairs with p < 0.05 are considered tradeable.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class CointegrationResult:
    """Result of a cointegration test between two tickers."""

    ticker_a: str
    ticker_b: str
    p_value: float
    hedge_ratio: float
    half_life: float
    is_cointegrated: bool


class CointegrationEngine:
    """Identifies cointegrated pairs for statistical arbitrage.

    Uses OLS-based Engle-Granger test with ADF on residuals.
    Pairs are tested within the same sector to increase economic rationale.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._p_value_threshold = cfg.get("cointegration_p_threshold", 0.05)
        self._min_observations = cfg.get("cointegration_min_obs", 252)
        self._max_half_life = cfg.get("cointegration_max_half_life", 60)

    @classmethod
    def from_config_file(cls, config_path: str) -> "CointegrationEngine":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config=config)

    def find_pairs(
        self,
        price_data: pd.DataFrame,
        tickers: List[str],
        sector_groups: Optional[Dict[str, List[str]]] = None,
    ) -> List[CointegrationResult]:
        """Find cointegrated pairs from the given universe.

        Args:
            price_data: DataFrame with MultiIndex (date, ticker) and 'close' column.
            tickers: List of tickers to consider.
            sector_groups: Optional dict mapping sector -> list of tickers.
                If provided, only tests pairs within the same sector.

        Returns:
            List of CointegrationResult for pairs that pass the threshold.
        """
        if sector_groups:
            pairs_to_test = self._generate_sector_pairs(sector_groups)
        else:
            pairs_to_test = [
                (tickers[i], tickers[j])
                for i in range(len(tickers))
                for j in range(i + 1, len(tickers))
            ]

        results = []
        for ticker_a, ticker_b in pairs_to_test:
            result = self.test_pair(price_data, ticker_a, ticker_b)
            if result is not None and result.is_cointegrated:
                results.append(result)

        results.sort(key=lambda r: r.p_value)
        return results

    def test_pair(
        self,
        price_data: pd.DataFrame,
        ticker_a: str,
        ticker_b: str,
    ) -> Optional[CointegrationResult]:
        """Test cointegration between two tickers using Engle-Granger.

        Args:
            price_data: DataFrame with MultiIndex (date, ticker) and 'close' column.
            ticker_a: First ticker.
            ticker_b: Second ticker.

        Returns:
            CointegrationResult, or None if insufficient data.
        """
        try:
            prices_a = price_data.xs(ticker_a, level="ticker")["close"].sort_index()
            prices_b = price_data.xs(ticker_b, level="ticker")["close"].sort_index()
        except KeyError:
            return None

        # Align dates
        common_idx = prices_a.index.intersection(prices_b.index)
        if len(common_idx) < self._min_observations:
            return None

        y = prices_a.loc[common_idx].values
        x = prices_b.loc[common_idx].values

        # OLS regression: y = beta * x + alpha + epsilon
        hedge_ratio, intercept = self._ols_regression(x, y)
        spread = y - hedge_ratio * x - intercept

        # ADF test on spread
        p_value = self._adf_test(spread)

        # Compute half-life of mean reversion
        half_life = self._compute_half_life(spread)

        is_cointegrated = (
            p_value < self._p_value_threshold
            and 0 < half_life < self._max_half_life
        )

        return CointegrationResult(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            p_value=p_value,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            is_cointegrated=is_cointegrated,
        )

    def compute_spread(
        self,
        price_data: pd.DataFrame,
        ticker_a: str,
        ticker_b: str,
        hedge_ratio: float,
    ) -> pd.Series:
        """Compute the spread between two tickers given a hedge ratio.

        Args:
            price_data: DataFrame with MultiIndex (date, ticker).
            ticker_a: Long leg.
            ticker_b: Short leg.
            hedge_ratio: Number of units of ticker_b to short per unit of ticker_a.

        Returns:
            Spread series indexed by date.
        """
        prices_a = price_data.xs(ticker_a, level="ticker")["close"].sort_index()
        prices_b = price_data.xs(ticker_b, level="ticker")["close"].sort_index()
        common_idx = prices_a.index.intersection(prices_b.index)
        return prices_a.loc[common_idx] - hedge_ratio * prices_b.loc[common_idx]

    def _generate_sector_pairs(
        self, sector_groups: Dict[str, List[str]]
    ) -> List[Tuple[str, str]]:
        """Generate all pairs within each sector."""
        pairs = []
        for tickers in sector_groups.values():
            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    pairs.append((tickers[i], tickers[j]))
        return pairs

    @staticmethod
    def _ols_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Simple OLS: y = beta * x + alpha."""
        X = np.column_stack([x, np.ones(len(x))])
        beta, alpha = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(beta), float(alpha)

    @staticmethod
    def _adf_test(series: np.ndarray) -> float:
        """Simplified ADF test. Returns approximate p-value.

        Uses the Dickey-Fuller regression: delta_y = rho * y_lag + error
        and compares the t-statistic against critical values.
        """
        y = series[1:]
        y_lag = series[:-1]
        delta_y = y - y_lag

        # Regression: delta_y = rho * y_lag
        X = np.column_stack([y_lag, np.ones(len(y_lag))])
        coeffs = np.linalg.lstsq(X, delta_y, rcond=None)[0]
        rho = coeffs[0]

        residuals = delta_y - X @ coeffs
        se = np.sqrt(np.sum(residuals ** 2) / (len(residuals) - 2))
        se_rho = se / np.sqrt(np.sum((y_lag - y_lag.mean()) ** 2))

        if se_rho == 0:
            return 1.0

        t_stat = rho / se_rho

        # Approximate p-value from ADF critical values (n > 250)
        # 1%: -3.43, 5%: -2.86, 10%: -2.57
        if t_stat < -3.43:
            return 0.01
        elif t_stat < -2.86:
            return 0.05
        elif t_stat < -2.57:
            return 0.10
        else:
            return 0.50

    @staticmethod
    def _compute_half_life(spread: np.ndarray) -> float:
        """Compute half-life of mean reversion via AR(1) regression."""
        y = spread[1:]
        y_lag = spread[:-1]
        delta_y = y - y_lag

        if len(y_lag) < 2:
            return float("inf")

        X = np.column_stack([y_lag, np.ones(len(y_lag))])
        coeffs = np.linalg.lstsq(X, delta_y, rcond=None)[0]
        phi = coeffs[0]

        if phi >= 0:
            return float("inf")

        half_life = -np.log(2) / np.log(1 + phi) if (1 + phi) > 0 else float("inf")
        return float(half_life)
