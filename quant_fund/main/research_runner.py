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
        self._signal_decay_detector = None
        self._strategy_retirement_manager = None
        self._feature_selector = None
        self._model_store = None
        self._ml_models = []
        self._retrain_frequency = cfg.get("retrain_frequency", 21)
        self._cycles_since_retrain = 0
        self._oos_metrics: List[dict] = []

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
        signal_decay_detector=None,
        strategy_retirement_manager=None,
        feature_selector=None,
        model_store=None,
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
        if signal_decay_detector is not None:
            self._signal_decay_detector = signal_decay_detector
        if strategy_retirement_manager is not None:
            self._strategy_retirement_manager = strategy_retirement_manager
        if feature_selector is not None:
            self._feature_selector = feature_selector
        if model_store is not None:
            self._model_store = model_store

    def inject_ml_models(self, models: list) -> None:
        """Inject ML models that support train_model() for periodic retraining."""
        self._ml_models = models

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

                # Step 4b: Feature selection (if selector and target available)
                if self._feature_selector is not None:
                    target = self._get_selection_target(data, feature_matrix)
                    if target is not None:
                        ranking = self._feature_selector.select(
                            feature_matrix, target
                        )
                        if ranking.selected_features:
                            feature_matrix = feature_matrix[ranking.selected_features]
                            logger.info(
                                "Feature selection: kept %d/%d features",
                                len(ranking.selected_features),
                                len(ranking.selected_features) + len(ranking.removed_features),
                            )

                # Step 5: Compute alpha scores
                if self._signal_ranking is not None:
                    alpha_scores = self._signal_ranking.combine(
                        feature_matrix
                    )
                    result.alpha_scores = alpha_scores
            else:
                logger.info("No features computed for %s", as_of)

            # Step 5b: Update monitoring
            if self._alpha_monitor is not None and result.alpha_scores is not None:
                self._alpha_monitor.update(result.alpha_scores)

            # Step 5c: IC monitoring and signal decay detection
            if result.alpha_scores is not None and data is not None:
                fwd_ret = self._get_selection_target(data, feature_matrix) if features else None
                if fwd_ret is not None:
                    # Track IC
                    if self._ic_monitor is not None:
                        alerts = self._ic_monitor.update(
                            signal_name="composite_alpha",
                            signal_values=result.alpha_scores,
                            forward_returns=fwd_ret,
                            date=as_of,
                        )
                        for a in alerts:
                            logger.warning("IC degradation alert: %s", a.message)

                    # Track signal decay
                    if self._signal_decay_detector is not None and self._ic_monitor is not None:
                        ic_hist = getattr(self._ic_monitor, "_ic_history", {})
                        ic_vals = ic_hist.get("composite_alpha", [])
                        if ic_vals:
                            self._signal_decay_detector.update(
                                "composite_alpha", ic_vals[-1], as_of,
                            )
                            decay = self._signal_decay_detector.detect("composite_alpha")
                            if decay is not None:
                                result.signal_metrics = result.signal_metrics or {}
                                result.signal_metrics["decay_half_life"] = decay.ic_half_life_days
                                result.signal_metrics["decay_action"] = decay.action.value

            # Step 5d: Log retraining signals for the offline pipeline.
            # Retraining NEVER happens in the live loop. The offline
            # pipeline reads health events to prioritize models.
            if self._ml_models and features:
                self._cycles_since_retrain += 1
                if (
                    result.signal_metrics
                    and result.signal_metrics.get("decay_action") == "review"
                ):
                    logger.info(
                        "Signal decay detected -- offline pipeline will "
                        "prioritize retraining on next scheduled run"
                    )

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

    def _get_selection_target(
        self,
        data: pd.DataFrame,
        feature_matrix: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """Extract a target variable for feature selection.

        Uses the 'close' column to compute forward returns if available.
        Returns None if no suitable target can be derived.
        """
        if "close" not in data.columns:
            return None

        try:
            if isinstance(data.index, pd.MultiIndex):
                # Multi-index with ticker level: use cross-sectional returns
                close = data["close"].unstack(level="ticker")
                returns = close.pct_change(5).iloc[-1]
                common = feature_matrix.index.intersection(returns.index)
                if len(common) < 30:
                    return None
                return returns[common]
            else:
                returns = data["close"].pct_change(5)
                common = feature_matrix.index.intersection(returns.index)
                if len(common) < 30:
                    return None
                return returns[common]
        except Exception:
            return None

    def _retrain_models(
        self,
        feature_matrix: pd.DataFrame,
        data: pd.DataFrame,
    ) -> None:
        """Retrain all ML models on current expanding window of data."""
        target = self._get_selection_target(data, feature_matrix)
        if target is None:
            return

        for model in self._ml_models:
            if hasattr(model, "train_model"):
                try:
                    metrics = model.train_model(feature_matrix, target)
                    self._track_oos_performance(model, metrics)

                    # Persist trained model via ModelStore
                    model_name = getattr(model, "feature_name", str(model))
                    if (
                        self._model_store is not None
                        and hasattr(model, "_model")
                        and model._model is not None
                    ):
                        feature_names = list(feature_matrix.columns)
                        train_start = ""
                        train_end = ""
                        if hasattr(data.index, "min"):
                            train_start = str(data.index.min())
                            train_end = str(data.index.max())
                        version_id = self._model_store.save_sklearn_model(
                            model=model._model,
                            model_name=model_name,
                            train_start_date=train_start,
                            train_end_date=train_end,
                            feature_names=feature_names,
                            metrics=metrics,
                        )
                        beats, details = self._model_store.check_challenger_beats_champion(
                            model_name, version_id, metric_name="oos_ic",
                        )
                        if beats:
                            self._model_store.promote_to_champion(model_name, version_id)
                            logger.info(
                                "Model '%s' v%s promoted to champion (improvement: %.4f)",
                                model_name, version_id,
                                details.get("improvement", 0),
                            )
                except Exception as e:
                    logger.warning("Model retrain failed: %s", e)

    def _track_oos_performance(self, model, metrics: dict) -> None:
        """Record OOS metrics from a model training run."""
        model_name = getattr(model, "feature_name", str(model))
        record = {"model_name": model_name}
        record.update(metrics)
        self._oos_metrics.append(record)

    def load_champion_models(self) -> int:
        """Load champion model weights from ModelStore for each ML model.

        Returns the number of models successfully loaded.
        """
        if self._model_store is None:
            return 0
        loaded = 0
        for model in self._ml_models:
            model_name = getattr(model, "feature_name", str(model))
            try:
                fitted = self._model_store.load_sklearn_model(model_name)
                model._model = fitted
                loaded += 1
                logger.info("Loaded champion model for '%s'", model_name)
            except FileNotFoundError:
                logger.info(
                    "No champion model found for '%s' — will train from scratch",
                    model_name,
                )
            except Exception as e:
                logger.warning("Failed to load model '%s': %s", model_name, e)
        return loaded

    @property
    def oos_metrics(self) -> List[dict]:
        return list(self._oos_metrics)

    @property
    def retrain_count(self) -> int:
        return len(self._oos_metrics)

    @property
    def results(self) -> List[ResearchResult]:
        return list(self._results)
