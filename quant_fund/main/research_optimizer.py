"""Research → optimization pipeline.

Extracted from TradingEngine._run_research_and_optimize() to reduce
the monolithic orchestrator to a thin dispatcher.

Runs the full research → factor model → optimization → target weights chain:
1. Run research cycle to produce alpha scores
2. Estimate factor exposures and covariance
3. Build constraints
4. Optimize portfolio
5. Return target weights for convergence loop
"""

import logging
from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd

from quant_fund.infrastructure.event_bus import (
    Event,
    EventBus,
    EventPriority,
    EventType,
)

logger = logging.getLogger(__name__)


def run_research_and_optimize(
    *,
    research_runner,
    portfolio_optimizer=None,
    constraint_engine=None,
    broker=None,
    event_bus: EventBus,
    factor_exposure_estimator=None,
    factor_covariance_estimator=None,
    stock_returns: Optional[pd.DataFrame] = None,
    factor_returns: Optional[pd.DataFrame] = None,
    factor_exposures: Optional[pd.DataFrame] = None,
    sector_map: Optional[Dict[str, str]] = None,
    historical_data: Optional[pd.DataFrame] = None,
    live_data_adapter=None,
) -> tuple:
    """Run research → factor model → optimization → target weights.

    Returns
    -------
    (target_weights, alpha_scores, factor_exposures) : tuple
        target_weights and alpha_scores may be None if pipeline fails.
        factor_exposures is the updated estimate (or the input if unchanged).
    """
    if research_runner is None:
        return None, None, factor_exposures

    as_of = pd.Timestamp.now(tz="America/New_York")

    # Get market data for research
    market_data = historical_data
    if market_data is None and live_data_adapter is not None:
        try:
            market_data = live_data_adapter.get_latest_bar()
        except Exception:
            logger.warning("Failed to get latest market data for research")

    # Step 1: Research cycle → alpha scores
    try:
        research_result = research_runner.run_cycle(
            as_of=as_of, market_data=market_data,
        )
        alpha_scores = research_result.alpha_scores
    except Exception:
        logger.exception("Research cycle failed")
        return None, None, factor_exposures

    if alpha_scores is None or alpha_scores.empty:
        logger.info("Research produced no alpha scores")
        return None, None, factor_exposures

    logger.info("Research produced alpha scores for %d tickers", len(alpha_scores))

    # Publish signal event
    event_bus.publish(
        Event(
            event_type=EventType.SIGNAL_GENERATED,
            payload={
                "alpha_scores": alpha_scores.to_dict(),
                "as_of": str(as_of),
            },
            source="research_runner",
            priority=EventPriority.NORMAL,
        )
    )

    # Step 2: Factor model estimation
    updated_exposures = factor_exposures
    factor_covariance = None

    if (
        factor_exposure_estimator is not None
        and factor_covariance_estimator is not None
        and stock_returns is not None
        and factor_returns is not None
    ):
        try:
            updated_exposures = factor_exposure_estimator.estimate(
                returns=stock_returns,
                factor_returns=factor_returns,
                as_of=as_of,
                sector_map=sector_map,
            )

            factor_covariance = factor_covariance_estimator.estimate(
                factor_returns=factor_returns,
                as_of=as_of,
            )
        except Exception:
            logger.exception("Factor model estimation failed")

    # Step 3: Portfolio optimization
    if portfolio_optimizer is not None:
        try:
            constraints = None
            if constraint_engine is not None:
                constraints = constraint_engine.build_constraints(
                    sector_map=sector_map,
                )

            current_positions = pd.Series(dtype=float)
            if broker is not None:
                current_positions = broker.get_positions()

            target_weights = portfolio_optimizer.optimize(
                alpha_scores=alpha_scores,
                factor_covariance=factor_covariance,
                factor_exposures=updated_exposures,
                constraints=constraints,
                current_positions=current_positions,
            )

            logger.info(
                "Optimization produced weights for %d tickers (gross=%.3f)",
                len(target_weights),
                target_weights.abs().sum(),
            )
            return target_weights, alpha_scores, updated_exposures
        except Exception:
            logger.exception("Portfolio optimization failed")
            return None, alpha_scores, updated_exposures
    else:
        # No optimizer — use alpha scores as simple weights
        total = alpha_scores.abs().sum()
        if total > 0:
            target_weights = alpha_scores / total * 0.5
            logger.info("No optimizer — using normalized alpha scores as weights")
            return target_weights, alpha_scores, updated_exposures

    return None, alpha_scores, updated_exposures
