"""Engine builder — construct a fully wired TradingEngine for paper trading.

Extracted from main_run.py to keep the CLI wrapper thin.
"""

import logging
import os
from typing import List

import pandas as pd

from quant_fund.config.typed_config import (
    BrokerConfig,
    ConstraintConfig,
    DrawdownConfig,
    EngineConfig,
    ExposureConfig,
    HealthMonitorConfig,
    KillSwitchConfig,
    LeverageConfig,
    LiveDataConfig,
    ModelHealthConfig,
)

from quant_fund.alpha_discovery.signal_ranking_engine import SignalRankingEngine
from quant_fund.alpha_monitoring.alpha_performance_tracker import (
    AlphaPerformanceTracker,
)
from quant_fund.alpha_monitoring.information_coefficient_monitor import (
    InformationCoefficientMonitor,
)
from quant_fund.alpha_monitoring.signal_decay_detector import SignalDecayDetector
from quant_fund.alpha_monitoring.strategy_retirement_manager import (
    StrategyRetirementManager,
)
from quant_fund.broker_interface.simulation_broker import SimulationBroker
from quant_fund.data_layer.connectors.yfinance_feed_provider import (
    YFinanceFeedProvider,
)
from quant_fund.data_layer.data_validator import DataValidator
from quant_fund.data_layer.live_data_stream_adapter import LiveDataStreamAdapter
from quant_fund.data_layer.market_data_utils import bars_to_market_data
from quant_fund.execution.order_management.market_hours_enforcer import (
    MarketHoursEnforcer,
)
from quant_fund.execution.order_management.order_generator import OrderGenerator
from quant_fund.execution.order_management.order_router import OrderRouter
from quant_fund.feature_factory.feature_normalizer import FeatureNormalizer
from quant_fund.feature_factory.technical_indicator_engine import (
    TechnicalIndicatorEngine,
)
from quant_fund.infrastructure.model_loader import ModelRegistry
from quant_fund.infrastructure.model_store import ModelStore
from quant_fund.infrastructure.trade_recorder import TradeRecorder
from quant_fund.main.research_runner import ResearchRunner
from quant_fund.main.trading_engine import TradingEngine
from quant_fund.monitoring.alerting_system import AlertingSystem
from quant_fund.monitoring.execution_quality_monitor import ExecutionQualityMonitor
from quant_fund.monitoring.model_health_monitor import ModelHealthMonitor
from quant_fund.monitoring.pnl_dashboard import PnLDashboard
from quant_fund.monitoring.system_health_monitor import SystemHealthMonitor
from quant_fund.portfolio.capacity_model.market_impact_model import MarketImpactModel
from quant_fund.portfolio.factor_risk_model.factor_covariance_estimator import (
    FactorCovarianceEstimator,
)
from quant_fund.portfolio.factor_risk_model.factor_exposure_estimator import (
    FactorExposureEstimator,
)
from quant_fund.portfolio.portfolio_construction.constraint_engine import (
    ConstraintEngine,
)
from quant_fund.portfolio.portfolio_construction.portfolio_optimizer import (
    PortfolioOptimizer,
)
from quant_fund.research_algorithms.machine_learning.gradient_boosted_tree_model import (
    GradientBoostedTreeModel,
)
from quant_fund.risk_engine.drawdown_monitor import DrawdownMonitor
from quant_fund.risk_engine.exposure_monitor import ExposureMonitor
from quant_fund.risk_engine.leverage_controller import LeverageController
from quant_fund.risk_engine.portfolio_kill_switch import KillSwitch
from quant_fund.risk_engine.risk_cascade_coordinator import RiskCascadeCoordinator

logger = logging.getLogger(__name__)

# Project root for model paths
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def build_engine(
    tickers: List[str],
    initial_cash: float = 1_000_000.0,
    research_interval_s: float = 3600.0,
    price_poll_interval_s: float = 60.0,
    db_path: str = "quantfund_paper.db",
):
    """Construct a fully wired TradingEngine for paper trading.

    Returns (engine, broker, dashboard, health, live_adapter).
    """
    # Broker
    broker = SimulationBroker(config={
        "initial_cash": initial_cash,
        "half_spread_bps": 5.0,
        "market_impact_bps": 2.0,
    })

    # Live data feed
    feed_provider = YFinanceFeedProvider()
    live_adapter = LiveDataStreamAdapter(
        config={"polling_interval_sec": price_poll_interval_s, "buffer_size": 1000},
        feed_provider=feed_provider,
    )

    # Monitoring
    dashboard = PnLDashboard(config={"initial_nav": initial_cash})
    health = SystemHealthMonitor(config={"max_data_lag_s": price_poll_interval_s * 5})
    alerting = AlertingSystem()
    eq_monitor = ExecutionQualityMonitor()

    # Risk
    kill_switch = KillSwitch(config={
        "drawdown_limit": 0.15,
        "initial_nav": initial_cash,
    })
    drawdown_monitor = DrawdownMonitor(config={
        "initial_nav": initial_cash,
        "drawdown_warning": 0.05,
        "drawdown_alert": 0.10,
        "drawdown_limit": 0.15,
    })
    leverage_controller = LeverageController(config={"max_leverage": 1.5})
    capacity_model = MarketImpactModel()

    risk_cascade = RiskCascadeCoordinator(
        kill_switch=kill_switch,
        drawdown_monitor=drawdown_monitor,
        leverage_controller=leverage_controller,
    )

    # Compliance audit trail
    trades_db = db_path.replace(".db", "_trades.db") if db_path != ":memory:" else ":memory:"
    trade_recorder = TradeRecorder(db_path=trades_db)

    # Market hours
    market_hours = MarketHoursEnforcer()

    # Research pipeline
    tech_engine = TechnicalIndicatorEngine(config={
        "technical_indicators": {
            "return_windows": [5, 20, 60],
            "volatility_windows": [20],
            "rsi_window": 14,
            "bollinger_window": 20,
            "relative_volume_window": 20,
        }
    })

    # ML model for alpha generation
    gbt_model = GradientBoostedTreeModel(config={
        "gbt_n_estimators": 200,
        "gbt_max_depth": 5,
        "gbt_learning_rate": 0.05,
        "gbt_min_train_days": 126,
        "random_seed": 42,
    })

    # Model persistence
    model_store = ModelStore(config={"model_dir": "./models"})

    # Model registry
    model_registry_path = os.path.join(_PROJECT_ROOT, "models", "registry")
    model_registry = None
    if os.path.isdir(model_registry_path):
        model_registry = ModelRegistry(
            registry_path=model_registry_path,
            auto_reload=True,
        )

    # Model health monitor
    model_health_monitor = ModelHealthMonitor(
        event_log_path=os.path.join(_PROJECT_ROOT, "logs", "health_events.jsonl"),
        prediction_window=500,
        drift_z_threshold=2.5,
        staleness_hours=48,
    )

    research_runner = ResearchRunner(config={
        "universe": tickers,
        "lookback_days": 252,
        "retrain_frequency": 24,
    })
    research_runner.inject_components(
        feature_generators=tech_engine.generators + [gbt_model],
        feature_normalizer=FeatureNormalizer(),
        signal_ranking=SignalRankingEngine(config={
            "min_ic": 0.0,
            "min_ic_tstat": 0.0,
            "min_eval_days": 0,
        }),
        data_validator=DataValidator(),
        alpha_monitor=AlphaPerformanceTracker(),
        ic_monitor=InformationCoefficientMonitor(),
        signal_decay_detector=SignalDecayDetector(),
        strategy_retirement_manager=StrategyRetirementManager(),
        model_store=model_store,
    )
    research_runner.inject_ml_models([gbt_model])

    # Portfolio optimizer and constraints
    portfolio_optimizer = PortfolioOptimizer(config={"risk_aversion": 1.0})
    constraint_engine = ConstraintEngine(config={
        "position_limits": {
            "max_position_size": 0.10,
            "max_sector_exposure": 0.40,
            "max_leverage": 1.0,
        }
    })

    # Exposure monitor and factor risk model
    exposure_monitor = ExposureMonitor(config={
        "max_leverage": 1.0,
        "max_sector_exposure": 0.40,
        "max_single_name_exposure": 0.10,
    })
    factor_exposure_estimator = FactorExposureEstimator()
    factor_covariance_estimator = FactorCovarianceEstimator()

    # Engine
    engine = TradingEngine(config={
        "tick_interval_s": 1.0,
        "convergence_interval_s": price_poll_interval_s,
        "research_interval_s": research_interval_s,
        "snapshot_interval_s": 300.0,
        "reconciliation_interval_s": 600.0,
        "health_check_interval_s": 60.0,
        "state_db_path": db_path,
    })

    # Inject all components
    engine.inject_components(
        broker=broker,
        order_generator=OrderGenerator(),
        order_router=OrderRouter(broker),
        kill_switch=kill_switch,
        drawdown_monitor=drawdown_monitor,
        leverage_controller=leverage_controller,
        pnl_dashboard=dashboard,
        alerting=alerting,
        health_monitor=health,
        live_data_adapter=live_adapter,
        risk_cascade=risk_cascade,
        execution_quality_monitor=eq_monitor,
        capacity_model=capacity_model,
        market_hours_enforcer=market_hours,
        sector_map=None,  # set by caller
        research_runner=research_runner,
        portfolio_optimizer=portfolio_optimizer,
        constraint_engine=constraint_engine,
        trade_recorder=trade_recorder,
        exposure_monitor=exposure_monitor,
        factor_exposure_estimator=factor_exposure_estimator,
        factor_covariance_estimator=factor_covariance_estimator,
        model_registry=model_registry,
        model_health_monitor=model_health_monitor,
    )

    # Subscribe to tickers
    live_adapter.subscribe(tickers)

    # Register a bar callback to update broker market data
    def on_new_bars(bars: pd.DataFrame):
        md = bars_to_market_data(bars)
        if md:
            broker.set_market_data(md)
            nav = broker.get_account_value()
            dashboard.update(nav)
            health.record_data_timestamp("yfinance", pd.Timestamp.now())
            logger.debug("Market data updated: %d tickers", len(md))
        else:
            logger.warning("Received empty market data -- prices may be stale")

    live_adapter.on_bar(on_new_bars)

    return engine, broker, dashboard, health, live_adapter
