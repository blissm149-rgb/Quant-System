"""Research runner — daily research cycle entry point.

Orchestrates the full research pipeline:
1. Load and validate data
2. Compute features (point-in-time)
3. Generate alpha scores
4. Evaluate signals
5. Update alpha monitoring
6. Log results

This is the first entry point to build and test.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ResearchConfig:
    """Configuration for the research runner."""

    universe: List[str] = field(default_factory=list)
    start_date: str = "2019-01-01"
    end_date: str = "2024-01-01"
    lookback_days: int = 252
    config_path: str = ""


@dataclass
class ResearchResult:
    """Result of a single research cycle."""

    as_of: pd.Timestamp
    alpha_scores: Optional[pd.Series] = None
    feature_matrix: Optional[pd.DataFrame] = None
    signal_metrics: Optional[dict] = None
    validation_flags: List[str] = field(default_factory=list)
    status: str = "completed"
    error_message: str = ""


class ResearchRunner:
    """Orchestrates the daily research cycle.

    Follows the pipeline pseudocode from HANDOFF.md S17:
    1. Update data → validate
    2. Compute features (point-in-time via data_alignment_engine)
    3. Normalise features
    4. Compute alpha scores via research algorithms
    5. Evaluate signal quality
    6. Update alpha monitoring
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._universe = cfg.get("universe", [])
        self._lookback_days = cfg.get("lookback_days", 252)

        # Component references — injected after construction
        self._data_loader = None
        self._data_validator = None
        self._feature_generators = []
        self._feature_normalizer = None
        self._signal_ranking = None
        self._alpha_monitor = None
        self._ic_monitor = None

        self._results: List[ResearchResult] = []

    def inject_components(
        self,
        data_loader=None,
        data_validator=None,
        feature_generators=None,
        feature_normalizer=None,
        signal_ranking=None,
        alpha_monitor=None,
        ic_monitor=None,
    ) -> None:
        """Inject pipeline components.

        All components are optional to support partial testing.
        """
        if data_loader is not None:
            self._data_loader = data_loader
        if data_validator is not None:
            self._data_validator = data_validator
        if feature_generators is not None:
            self._feature_generators = feature_generators
        if feature_normalizer is not None:
            self._feature_normalizer = feature_normalizer
        if signal_ranking is not None:
            self._signal_ranking = signal_ranking
        if alpha_monitor is not None:
            self._alpha_monitor = alpha_monitor
        if ic_monitor is not None:
            self._ic_monitor = ic_monitor

    def run_cycle(
        self,
        as_of: pd.Timestamp,
        market_data: Optional[pd.DataFrame] = None,
    ) -> ResearchResult:
        """Run a single research cycle for the given date.

        Parameters
        ----------
        as_of : pd.Timestamp
            Point-in-time date. All computations use only
            data available strictly before this timestamp.
        market_data : pd.DataFrame, optional
            Pre-loaded market data (for testing). If None,
            loads via data_loader.

        Returns
        -------
        ResearchResult
        """
        result = ResearchResult(as_of=as_of)

        try:
            # Step 1: Load data
            data = market_data
            if data is None and self._data_loader is not None:
                data = self._data_loader.load(
                    tickers=self._universe, as_of=as_of
                )

            if data is None or data.empty:
                result.status = "no_data"
                result.error_message = "No market data available"
                return result

            # Step 2: Validate data
            if self._data_validator is not None:
                validation = self._data_validator.validate(data, as_of=as_of)
                if hasattr(validation, "flags"):
                    result.validation_flags = validation.flags
                if hasattr(validation, "is_valid") and not validation.is_valid:
                    logger.warning(
                        "Data validation failed for %s: %s",
                        as_of, result.validation_flags,
                    )

            # Step 3: Compute features
            features = {}
            for gen in self._feature_generators:
                try:
                    feature_name = getattr(gen, "feature_name", str(gen))
                    feature_values = gen.compute(data, as_of=as_of)
                    features[feature_name] = feature_values
                except Exception as e:
                    logger.warning("Feature %s failed: %s", gen, e)

            if features:
                feature_matrix = pd.DataFrame(features)
                result.feature_matrix = feature_matrix

                # Step 4: Normalise
                if self._feature_normalizer is not None:
                    feature_matrix = self._feature_normalizer.normalize(
                        feature_matrix
                    )

                # Step 5: Compute alpha scores
                if self._signal_ranking is not None:
                    alpha_scores = self._signal_ranking.combine(
                        feature_matrix
                    )
                    result.alpha_scores = alpha_scores
            else:
                logger.info("No features computed for %s", as_of)

            # Step 6: Update monitoring
            if self._alpha_monitor is not None and result.alpha_scores is not None:
                self._alpha_monitor.update(result.alpha_scores)

            result.status = "completed"

        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            logger.error("Research cycle failed for %s: %s", as_of, e)

        self._results.append(result)
        return result

    def run_backtest(
        self,
        dates: List[pd.Timestamp],
        market_data: Optional[pd.DataFrame] = None,
    ) -> List[ResearchResult]:
        """Run research cycles over a list of dates.

        Parameters
        ----------
        dates : list of pd.Timestamp
            Trading dates to run research on.
        market_data : pd.DataFrame, optional
            Full historical data.

        Returns
        -------
        list of ResearchResult
        """
        results = []
        for dt in dates:
            # Filter data up to as_of for point-in-time compliance
            data = None
            if market_data is not None:
                if isinstance(market_data.index, pd.DatetimeIndex):
                    data = market_data.loc[market_data.index < dt]
                else:
                    data = market_data
            result = self.run_cycle(as_of=dt, market_data=data)
            results.append(result)
        return results

    @property
    def results(self) -> List[ResearchResult]:
        return list(self._results)
