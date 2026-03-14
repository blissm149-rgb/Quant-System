"""Pairs trading strategy combining cointegration and Kalman filter.

Orchestrates pair selection via CointegrationEngine and signal generation
via KalmanSpreadModel.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from quant_fund.research_algorithms.statistical_arbitrage.cointegration_engine import (
    CointegrationEngine,
    CointegrationResult,
)
from quant_fund.research_algorithms.statistical_arbitrage.kalman_spread_model import (
    KalmanSpreadModel,
)

logger = logging.getLogger(__name__)


@dataclass
class PairSignal:
    """Trading signal for a single pair."""

    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    z_score: float
    signal: float  # 1, -1, or 0


class PairsTradingStrategy:
    """Pairs trading strategy using cointegration-based pair selection
    and Kalman filter-based signal generation.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._config = cfg
        self._cointegration_engine = CointegrationEngine(cfg)
        self._kalman_model = KalmanSpreadModel(cfg)
        self._max_pairs = cfg.get("max_pairs", 20)
        self._active_pairs: List[CointegrationResult] = []

    def select_pairs(
        self,
        price_data: pd.DataFrame,
        tickers: List[str],
        sector_groups: Optional[Dict[str, List[str]]] = None,
    ) -> List[CointegrationResult]:
        """Select cointegrated pairs for trading."""
        pairs = self._cointegration_engine.find_pairs(
            price_data, tickers, sector_groups
        )
        self._active_pairs = pairs[: self._max_pairs]
        logger.info("Selected %d cointegrated pairs", len(self._active_pairs))
        return self._active_pairs

    def generate_signals(
        self,
        price_data: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> List[PairSignal]:
        """Generate trading signals for all active pairs.

        Args:
            price_data: DataFrame with MultiIndex (date, ticker) and 'close'.
            as_of: Point-in-time boundary.

        Returns:
            List of PairSignal for active pairs with non-zero signals.
        """
        signals = []
        for pair in self._active_pairs:
            try:
                prices_a = price_data.xs(pair.ticker_a, level="ticker")["close"].sort_index()
                prices_b = price_data.xs(pair.ticker_b, level="ticker")["close"].sort_index()
            except KeyError:
                continue

            # Filter to before as_of
            prices_a = prices_a[prices_a.index < as_of]
            prices_b = prices_b[prices_b.index < as_of]

            result_df = self._kalman_model.compute_pair_signal(prices_a, prices_b)
            if result_df.empty:
                continue

            latest = result_df.iloc[-1]
            if latest["signal"] != 0:
                signals.append(
                    PairSignal(
                        ticker_a=pair.ticker_a,
                        ticker_b=pair.ticker_b,
                        hedge_ratio=latest["hedge_ratio"],
                        z_score=latest["z_score"],
                        signal=latest["signal"],
                    )
                )

        return signals

    def signals_to_weights(
        self, signals: List[PairSignal], tickers: List[str]
    ) -> pd.Series:
        """Convert pair signals to per-ticker weights.

        Args:
            signals: Active pair signals.
            tickers: Full universe of tickers.

        Returns:
            Series of weights indexed by ticker.
        """
        weights = pd.Series(0.0, index=tickers)
        if not signals:
            return weights

        weight_per_pair = 1.0 / len(signals)
        for sig in signals:
            if sig.ticker_a in weights.index:
                weights[sig.ticker_a] += sig.signal * weight_per_pair
            if sig.ticker_b in weights.index:
                weights[sig.ticker_b] -= sig.signal * sig.hedge_ratio * weight_per_pair

        return weights
