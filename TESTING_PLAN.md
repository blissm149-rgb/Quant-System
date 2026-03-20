TESTING_PLAN.md — QuantFund V8 Comprehensive Testing & Validation Framework
Table of Contents
Test Architecture
Directory Structure
Unit Tests per Module
Module Invariant Tests
Integration Tests
End-to-End Tests
Regression Tests
Performance Benchmarks
Stress Tests & Fault Injection
Synthetic Data Generators
Simulation Environments
Mocking/Stubbing Strategy
Deterministic Reproducibility
Long-Duration Stability Tests
Requirement → Module → Test Traceability Matrix
CI/CD Integration Plan
Test Coverage Targets
Quantitative Pass/Fail Metrics
Section 1: Test Architecture
Testing Pyramid
Layer	Proportion	Count (Current)	Count (Target)	Execution Time
Unit Tests	60%	~500	~720	< 2 min
Integration Tests	25%	~150	~300	< 10 min
End-to-End Tests	10%	~49	~120	< 30 min
Stress + Performance	5%	~0	~60	< 60 min
Total	100%	~699	~1200	
Pytest Markers
@pytest.mark.unit          # Fast, isolated, no I/O
@pytest.mark.integration   # Multi-module interaction
@pytest.mark.e2e           # Full pipeline
@pytest.mark.slow          # > 10s execution
@pytest.mark.stress        # Fault injection, extreme load
@pytest.mark.benchmark     # Performance measurement
@pytest.mark.tier1         # Pre-commit (< 30s)
@pytest.mark.tier2         # PR gate (< 5 min)
@pytest.mark.tier3         # Post-merge (< 15 min)
@pytest.mark.tier4         # Nightly (< 60 min)
@pytest.mark.tier5         # Weekly (long-duration)

Execution Strategy
Parallel execution via pytest-xdist for unit tests (-n auto)
Sequential execution for integration and E2E tests (state-dependent)
Isolated fixtures per test — no shared mutable state between tests
Deterministic ordering via pytest-randomly with fixed seed for reproducibility
Section 2: Directory Structure
tests/
├── conftest.py                        # Shared fixtures (KEEP existing)
├── real_market_data.py                # Real historical data helpers (KEEP existing)
├── generators/                        # NEW: synthetic data generators
│   ├── __init__.py
│   ├── ohlcv_generator.py
│   ├── factor_returns_generator.py
│   ├── order_flow_generator.py
│   └── stress_scenario_generator.py
├── unit/                              # NEW: reorganized unit tests
│   ├── test_config_validator.py
│   ├── test_data_validator.py
│   ├── test_corporate_action_adjuster.py
│   ├── test_historical_data_loader.py
│   ├── test_data_storage_manager.py
│   ├── test_live_data_stream_adapter.py
│   ├── test_data_alignment_engine.py
│   ├── test_feature_normalizer.py
│   ├── test_technical_indicator_engine.py
│   ├── test_signal_ranking_engine.py
│   ├── test_signal_decay_detector.py
│   ├── test_strategy_retirement_manager.py
│   ├── test_constraint_engine.py
│   ├── test_portfolio_optimizer.py
│   ├── test_portfolio_kill_switch.py
│   ├── test_drawdown_monitor.py
│   ├── test_exposure_monitor.py
│   ├── test_leverage_controller.py
│   ├── test_order_generator.py
│   ├── test_order_router.py
│   ├── test_vwap_execution.py
│   ├── test_simulation_broker.py
│   ├── test_event_bus.py
│   ├── test_system_state_machine.py
│   ├── test_reconciliation_engine.py
│   ├── test_state_persistence_manager.py
│   ├── test_approval_workflow.py
│   ├── test_deployment_controller.py
│   ├── test_strategy_review_pipeline.py
│   ├── ... (one per production module, ~127 files total)
├── integration/                       # NEW: module interaction tests
│   ├── test_data_to_features.py
│   ├── test_features_to_alpha.py
│   ├── test_alpha_to_portfolio.py
│   ├── test_portfolio_to_execution.py
│   ├── test_risk_chain.py
│   ├── test_governance_pipeline.py
│   ├── test_event_driven_loop.py
│   └── test_broker_reconnection_flow.py
├── e2e/                               # NEW: full pipeline tests
│   ├── test_research_cycle.py
│   ├── test_paper_trading_day.py
│   ├── test_multi_day_simulation.py
│   ├── test_historical_scenarios.py   # (MOVE existing)
│   └── test_real_data_scenarios.py
├── regression/                        # NEW: golden-file regression tests
│   ├── test_deterministic_baselines.py
│   ├── golden_files/
│   └── test_signal_stability.py
├── stress/                            # NEW: stress and fault injection
│   ├── test_chaos_broker.py
│   ├── test_data_corruption.py
│   ├── test_kill_switch_extremes.py
│   ├── test_concurrent_events.py
│   └── test_memory_pressure.py
├── benchmarks/                        # NEW: performance benchmarks
│   ├── bench_feature_computation.py
│   ├── bench_portfolio_optimization.py
│   ├── bench_order_generation.py
│   └── bench_event_bus_throughput.py
└── stability/                         # NEW: long-duration tests
    ├── test_30day_paper_trading.py
    ├── test_memory_leak_detection.py
    └── test_state_accumulation.py

Section 3: Unit Tests per Module
3.1 Config Subsystem (1 module)
config_validator.py

Validates all 5 YAML config files (system, trading, data, execution, risk)
Tests:
Valid config loads without error
Missing required keys raise ConfigValidationError
Invalid types rejected (e.g., max_leverage: "abc")
Schema enforcement catches unknown keys
Each module receives only its config sub-section
def test_valid_config_loads():
    validator = ConfigValidator("configs/")
    assert validator.validate_all() is True

def test_missing_required_key_raises():
    bad_config = {"trading": {}}  # missing max_position_size
    with pytest.raises(ConfigValidationError):
        ConfigValidator.validate_trading(bad_config)

def test_invalid_type_rejected():
    bad_config = {"max_leverage": "not_a_number"}
    with pytest.raises(ConfigValidationError):
        ConfigValidator.validate_risk(bad_config)

3.2 Data Layer (8 modules)
data_validator.py

Tests:
Rejects future timestamps (timestamp > now)
Rejects NaN close prices
Rejects negative prices
Detects and removes duplicate rows
Validates MultiIndex(date, ticker) format
Survivor-bias check: delisted tickers included
def test_rejects_future_timestamps(make_ohlcv):
    df = make_ohlcv(tickers=["AAPL"], days=10)
    df.index = df.index.set_levels([df.index.levels[0] + pd.Timedelta(days=3650)], level=0)
    with pytest.raises(DataValidationError):
        DataValidator().validate(df)

def test_rejects_nan_prices(make_ohlcv):
    df = make_ohlcv(tickers=["AAPL"], days=10)
    df.loc[df.index[0], "close"] = np.nan
    with pytest.raises(DataValidationError):
        DataValidator().validate(df)

def test_rejects_negative_prices(make_ohlcv):
    df = make_ohlcv(tickers=["AAPL"], days=10)
    df.loc[df.index[0], "close"] = -1.0
    with pytest.raises(DataValidationError):
        DataValidator().validate(df)

corporate_action_adjuster.py

Tests:
2:1 split correctly halves pre-split prices
Dividend adjustment applied backward only
Raw data never modified (copy-on-write)
Spinoff correctly distributes value
Multiple adjustments chain correctly
historical_data_loader.py

Tests:
Returns MultiIndex(date, ticker) DataFrame
Point-in-time snapshot respects as_of parameter
Delisted tickers included in historical queries
5-year OHLCV load returns expected shape
data_storage_manager.py

Tests:
Parquet write/read roundtrip preserves schema
purge_old_partitions() removes only expired data
get_storage_stats() returns correct metrics
live_data_stream_adapter.py

Tests:
WebSocket connection established
Data normalized to standard OHLCV format
Reconnection on disconnect
data_layer/connectors/ (3 modules)

base_connector.py: Abstract interface compliance
csv_connector.py: CSV load, schema validation, missing column handling
yfinance_connector.py: API call mocking, rate limiting, error handling
3.3 Feature Factory (4 modules)
data_alignment_engine.py (CRITICAL — look-ahead bias enforcer)

Tests:
Strict mode raises on any data with timestamp ≥ as_of
Permissive mode silently removes future data
Edge case: as_of exactly on data boundary
No look-ahead in any feature pipeline routed through alignment engine
def test_strict_mode_raises_on_future_data():
    engine = DataAlignmentEngine(mode="strict")
    as_of = pd.Timestamp("2023-06-15")
    data = make_ohlcv(end_date="2023-06-20")  # has future data
    with pytest.raises(LookAheadError):
        engine.align(data, as_of=as_of)

def test_permissive_mode_removes_future_data():
    engine = DataAlignmentEngine(mode="permissive")
    as_of = pd.Timestamp("2023-06-15")
    data = make_ohlcv(end_date="2023-06-20")
    aligned = engine.align(data, as_of=as_of)
    assert aligned.index.get_level_values("date").max() < as_of

feature_normalizer.py

Tests:
Z-score output has mean ≈ 0, std ≈ 1
Winsorization clips at ±3σ
Rank normalization produces uniform distribution
Normalization uses only historical data (no look-ahead in mean/std)
technical_indicator_engine.py

Tests:
RSI bounded [0, 100]
MACD signal line computed correctly
Bollinger Bands symmetric around SMA
Returns computation matches manual calculation
Volatility uses correct lookback window
base_feature_generator.py

Tests:
Abstract interface enforces feature_name, lookback_days, recompute_frequency
Subclasses must implement compute() method
3.4 Alpha Discovery (5 modules)
signal_ranking_engine.py

Tests:
IC > 0.03 threshold correctly filters candidates
t-stat > 2.0 threshold enforced
Crowding penalty reduces score for correlated signals
Turnover penalty applied to high-turnover signals
Ranking is deterministic for same input
feature_combinator.py

Tests:
Linear combinations produce valid features
Ratio features handle zero denominators
Interaction terms computed correctly
genetic_algorithm_search.py

Tests:
GA converges with known fitness landscape
IC-based fitness function correctly evaluates
Population diversity maintained
ml_feature_selector.py

Tests:
SHAP importance computed for all features
Redundancy removal eliminates correlated features
Permutation importance consistent with SHAP
symbolic_regression_engine.py

Tests:
Discovers known mathematical relationships
Complexity penalization works
Output expressions are valid
3.5 Alpha Monitoring (4 modules)
signal_decay_detector.py

Tests:
IC half-life < 20 days triggers review flag
Stable signal (half-life > 60 days) passes
Edge case: monotonically declining IC
strategy_retirement_manager.py

Tests:
State transitions: ACTIVE → REVIEW → SUSPENDED → RETIRED
Invalid transitions rejected
Retirement criteria correctly evaluated
alpha_performance_tracker.py

Tests:
Per-signal IC tracking over rolling window
IR (Information Ratio) computation correct
PnL attribution by signal
information_coefficient_monitor.py

Tests:
Degradation detected when IC < 0.0
Alert triggered when t-stat < 1.5
Rolling window IC computation correct
3.6 Alternative Data (14 modules)
analyst_estimates/

analyst_data_ingestion.py: Loads with announcement dates; validates as_of
estimate_revision_features.py: SUE computation, revision momentum direction
options_data/

options_chain_ingestion.py: Loads chains with as_of timestamps
implied_volatility_surface_builder.py: SVI/SABR surface validity, no-arbitrage
options_feature_generation.py: IV rank bounded [0,1], put/call ratio positive
news_sentiment/

news_ingestion.py: Headlines loaded with timestamps
sentiment_model.py: Output bounded [-1, +1], FinBERT inference
sentiment_feature_generation.py: Volume-weighted aggregation sums correctly
text_cleaning.py: Tokenization preserves meaning, NLP pipeline stable
etf_flows/

etf_flow_ingestion.py: Flow data with timestamps, no future data
flow_feature_generation.py: Net flow % computed correctly
macro_data/

macro_data_ingestion.py: Macro indicators loaded with release dates
macro_feature_generation.py: Features use only released data
macro_regime_classifier.py: 4 regimes (growth×inflation) classified correctly
3.7 Research Algorithms (16 modules)
factor_models/ (5 modules)

momentum_factor.py: 12-1 month return (skip last month); sign correct
value_factor.py: Book-to-market ratio positive; composite scoring
quality_factor.py: ROE, earnings stability, low accruals composite
size_factor.py: Log market cap inverted (small-cap tilt)
low_volatility_factor.py: 252-day realized vol inverted
mean_reversion/ (2 modules)

short_term_reversal_strategy.py: 1-week reversal with volume filter
zscore_reversion_strategy.py: Cross-sectional z-score; long bottom decile, short top
statistical_arbitrage/ (3 modules)

cointegration_engine.py: Engle-Granger p < 0.05; pairs within sectors
kalman_spread_model.py: Dynamic hedge ratio; entry |z| > 2.0, exit < 0.5, stop > 3.5
pairs_trading_strategy.py: Cointegrated pairs identified and traded
machine_learning/ (3 modules)

gradient_boosted_tree_model.py: LightGBM quintile prediction; no look-ahead in features
neural_network_predictor.py: MLP on embeddings; consistent output shape
random_forest_model.py: Ensemble classifier; validation alongside GBT
regime_models/ (3 modules)

hidden_markov_regime_model.py: 2-4 hidden states; Gaussian HMM converges
volatility_regime_detector.py: VIX-based classification (low/high)
market_state_classifier.py: Combines HMM + macro + vol → single enum
3.8 Portfolio (11 modules)
portfolio_construction/

constraint_engine.py (CRITICAL — single place for all constraints)

Tests:
max_position_size = 0.02 enforced
max_sector_exposure = 0.20 enforced
max_leverage = 2.0 enforced
Dollar-neutral: |sum(weights)| < 0.02
Constraint violation raises exception, not silent clipping
def test_max_position_size():
    engine = ConstraintEngine(config)
    weights = pd.Series({"AAPL": 0.05, "MSFT": 0.01})
    with pytest.raises(ConstraintViolation):
        engine.validate(weights)

def test_dollar_neutral():
    engine = ConstraintEngine(config)
    weights = pd.Series({"AAPL": 0.5, "MSFT": -0.5})
    assert abs(weights.sum()) < 0.02
    engine.validate(weights)  # should pass

portfolio_optimizer.py

Tests:
Output weights satisfy all constraint_engine constraints
Objective: minimises λ·w^T·Σ·w - α^T·w
CVXPY + OSQP solver converges
Alpha scores correctly mapped to weights (higher alpha → higher weight)
factor_risk_model/

factor_covariance_estimator.py: Ledoit-Wolf shrinkage; PSD matrix output
factor_exposure_estimator.py: Cross-sectional regression; exposures bounded
risk_decomposition.py: Factor + idiosyncratic = total variance (within tolerance)
capital_allocation/

dynamic_strategy_allocator.py: Sharpe-weighted; correlation-penalized; regime-gated
strategy_correlation_matrix.py: Symmetric; diagonal = 1.0; bounded [-1, 1]
strategy_performance_tracker.py: Trailing Sharpe, Sortino computed correctly
capacity_model/

market_impact_model.py: impact_bps = σ × (order_size/ADV)^0.6; positive
liquidity_estimator.py: Available liquidity per security; positive
capacity_simulator.py: Max AUM estimate given signals + turnover
3.9 Risk Engine (5 modules)
portfolio_kill_switch.py (CRITICAL — simplest module, no dependencies)

Tests:
Fires at exactly 20% drawdown
update_peak() tracks high-water mark correctly
Does NOT fire at 19.9% drawdown
Fires at 20.0% drawdown (boundary)
No external dependencies (only needs current NAV + peak NAV)
def test_kill_switch_fires_at_20_percent():
    ks = PortfolioKillSwitch()
    ks.update_peak(1_000_000)
    assert ks.check(800_000) is True   # exactly 20% DD

def test_kill_switch_does_not_fire_below_threshold():
    ks = PortfolioKillSwitch()
    ks.update_peak(1_000_000)
    assert ks.check(800_001) is False  # just under 20%

def test_peak_tracking():
    ks = PortfolioKillSwitch()
    ks.update_peak(900_000)
    ks.update_peak(1_000_000)
    ks.update_peak(950_000)  # should not lower peak
    assert ks.peak_nav == 1_000_000

drawdown_monitor.py

Tests:
10% warning alert fires
15% critical alert fires
20% kill switch alert fires
Peak-to-trough NAV tracking correct
exposure_monitor.py

Tests:
Net exposure within limits
Gross exposure within limits
Per-sector exposure checked against 20% limit
Per-factor exposure checked against configured limits
leverage_controller.py

Tests:
Leverage ≤ 2.0 enforced by proportional scaling
Post-optimization weights rescaled correctly
Edge case: leverage exactly 2.0 passes
stress_test_engine.py

Tests:
2008 crisis scenario applied correctly
2020 COVID scenario applied correctly
2022 rate shock scenario applied correctly
Portfolio impact computed accurately
3.10 Execution (13 modules)
order_management/

order_generator.py

Tests:
Target weights → Order objects conversion correct
Min trade filter eliminates small orders
Order direction (BUY/SELL) matches weight delta
order_router.py

Tests:
Large orders routed to VWAP
Small urgent orders routed to market
Routing rules configurable via config
order_safety_validator.py

Tests:
Rejects negative quantities
Rejects prices outside reasonable bounds
Validates order against position limits
market_hours_enforcer.py

Tests:
Orders rejected outside market hours
Pre-market/after-hours handling configurable
execution_algorithms/

vwap_execution.py: 5% participation rate; U-shaped intraday volume profile
twap_execution.py: Equal-sized slices over horizon
liquidity_seeking_execution.py: Triggers on volume > 2x average
microstructure_models/ (5 modules)

bid_ask_spread_model.py: Spread estimation from vol/market-cap
fill_probability_model.py: Limit order fill probability positive and ≤ 1
adverse_selection_model.py: VPIN-based flagging
order_book_liquidity_model.py: Queue depth estimation
queue_position_estimator.py: Arrival time based estimation
reconciliation_engine.py

Tests:
Broker = source of truth for positions
Discrepancy detected and logged
Auto-correction applied when configured
3.11 Broker Interface (5 modules)
simulation_broker.py (CRITICAL — must be complete before live adapters)

Tests:
Fills at mid ± spread/2
Partial fills when insufficient liquidity
Slippage model applied correctly
Cash floor enforced (no negative cash)
Latency simulation working
broker_abstraction_layer.py

Tests:
All adapters implement the same interface
No direct adapter calls bypass abstraction layer
Order submission, cancellation, status query all routed
broker_reconnection_manager.py

Tests:
Exponential backoff on disconnect
Failover to secondary broker
Reconnection state preserved
interactive_brokers_adapter.py / alpaca_adapter.py

Tests (always mocked):
API calls correctly formatted
Response parsing handles edge cases
Error handling for API failures
3.12 Infrastructure (11 modules)
event_bus.py

Tests:
Pub/sub delivery to all subscribers
Idempotency: duplicate events deduplicated
Priority ordering respected
Replay from checkpoint works
system_state_machine.py

Tests:
Valid transitions only: INIT → DATA_READY → TRADING → HALT → SHUTDOWN
Invalid transitions raise InvalidTransitionError
Readiness checks gate transitions
state_persistence_manager.py / state_store.py

Tests:
Save/restore roundtrip preserves all state
Corruption detection on load
Snapshot versioning
trade_recorder.py: Persistent trade log; order history query
model_store.py: Model save/load; champion/challenger management
dataset_version_control.py: Data versioning; reproducibility
ingestion_scheduler.py: Cooperative scheduling; no conflicts
ci_cd_pipeline_manager.py: Pipeline orchestration
docker_environment_setup.py: Container config
disaster_recovery_manager.py: Backup/restore procedures

3.13 Monitoring (5 modules)
alerting_system.py: INFO/WARNING/CRITICAL dispatch; correct channel routing
execution_quality_monitor.py: Implementation shortfall computation; fill rate tracking
pnl_dashboard.py: NAV computation; daily return; attribution by strategy/factor/sector
risk_dashboard.py: Leverage display; sector limits; VaR computation
system_health_monitor.py: Data feed status; broker connectivity; latency tracking
3.14 Governance (3 modules)
approval_workflow.py

Tests:
State machine: SUBMITTED → UNDER_REVIEW → APPROVED → DEPLOYED
risk_team_approved = True required for APPROVED state
Invalid transitions rejected
deployment_controller.py (CRITICAL — only module that flips paper → live)

Tests:
Checks approval_workflow state before deployment
Refuses deployment without approval
Only this module can flip paper → live
strategy_review_pipeline.py

Tests:
OOS Sharpe ≥ 1.0 check
Max drawdown ≤ 25% check
≥ 252 trading days minimum evaluation
No look-ahead flags check
3.15 Main (4 modules)
trading_engine.py: Always-on event loop; convergence; snapshots; recovery
research_runner.py: Research pipeline entry point; signal normalization before optimizer
paper_trading_runner.py: Live data + simulated execution
live_trading_runner.py: Requires governance sign-off before start
3.16 Representation Learning (5 modules)
autoencoder_model.py: Bottleneck latent dim 16; reconstruction loss decreases
cross_asset_embedding_model.py: Pairwise similarity; joint embeddings
embedding_feature_store.py: (model_id, date, ticker) keyed storage roundtrip
temporal_model_lstm.py: 60-day feature windows; per-ticker latent vector
temporal_model_transformer.py: Longer contexts; consistent output shape
3.17 Research Cluster (3 modules)
distributed_backtest_runner.py: joblib/Ray parallelism; isolated worker state
experiment_scheduler.py: Job queue management; experiment_id + hash
parallel_signal_evaluator.py: Parallel IC evaluation; t-stat ranked output
Section 4: Module Invariant Tests
Cross-cutting invariants that must hold across ALL modules:

Data Format Invariants
MultiIndex(date, ticker): All OHLCV DataFrames use pd.MultiIndex with levels (date, ticker)
Feature output format: All feature outputs are pd.Series indexed by ticker
Timestamp type: All timestamps are pd.Timestamp, not datetime.datetime
Signal Flow Invariants
Normalization gate: All alpha_scores are cross-sectionally normalized before entering optimizer
No raw signals in optimizer: Raw signals must pass through feature_normalizer first
Kill switch before orders: Kill switch checked BEFORE every order batch generation
Architectural Invariants
Config-driven: No magic numbers in code — all parameters from YAML configs
Broker abstraction: All broker calls go through broker_abstraction_layer only
Constraint centralization: All portfolio constraints live in constraint_engine.py only
Deployment gate: Only deployment_controller.py flips paper → live
Look-ahead enforcement: All feature computation routes through data_alignment_engine
Test Implementation
class TestModuleInvariants:
    """Cross-cutting invariant tests run against all modules."""

    def test_no_magic_numbers(self):
        """Scan all source files for hardcoded numeric constants."""
        # Grep for bare numbers not in config loading context
        violations = scan_for_magic_numbers("quant_fund/")
        assert len(violations) == 0, f"Magic numbers found: {violations}"

    def test_all_ohlcv_multiindex(self, sample_pipeline_output):
        """Verify OHLCV data uses MultiIndex(date, ticker)."""
        assert isinstance(sample_pipeline_output.index, pd.MultiIndex)
        assert sample_pipeline_output.index.names == ["date", "ticker"]

    def test_features_are_ticker_indexed_series(self, sample_features):
        """All feature outputs must be pd.Series indexed by ticker."""
        for feature_name, feature_values in sample_features.items():
            assert isinstance(feature_values, pd.Series)
            assert feature_values.index.name == "ticker"

    def test_kill_switch_before_orders(self, trading_engine):
        """Kill switch must be checked before order generation."""
        # Instrument order_generator to track call order
        call_log = []
        trading_engine.run_one_cycle(call_log=call_log)
        ks_idx = call_log.index("kill_switch.check")
        og_idx = call_log.index("order_generator.generate")
        assert ks_idx < og_idx

Section 5: Integration Tests
Chain 1: Data → Features → Alpha
historical_data_loader → corporate_action_adjuster → data_validator
    → data_alignment_engine → technical_indicator_engine
    → feature_normalizer → signal_ranking_engine

Tests:

Raw OHLCV flows through full pipeline to ranked alpha scores
No look-ahead at any stage (mock future data injection)
Adjusted prices used for features (not raw)
Normalized features have mean ≈ 0, std ≈ 1
Signal ranking produces ordered list with IC > 0.03 filter
Chain 2: Alpha → Portfolio → Execution
alpha_scores → portfolio_optimizer → constraint_engine
    → risk checks (exposure, leverage, kill switch)
    → order_generator → order_router → broker_abstraction_layer

Tests:

Alpha scores correctly drive optimizer weights
Constraints satisfied in optimizer output
Risk checks pass before order generation
Orders routed to correct execution algorithm
Broker abstraction receives well-formed orders
Chain 3: Risk Chain
drawdown_monitor + exposure_monitor + leverage_controller
    + portfolio_kill_switch → HALT trading

Tests:

10% drawdown triggers warning
15% drawdown triggers alert
20% drawdown triggers kill switch → HALT state
Exposure breach detected and flagged
Leverage > 2.0 triggers proportional scaling
Chain 4: Governance Pipeline
strategy_review_pipeline → approval_workflow
    → deployment_controller → paper/live

Tests:

Strategy with Sharpe < 1.0 rejected at review
Strategy with DD > 25% rejected at review
Approved strategy can be deployed to paper
Paper → live requires risk_team_approved = True
Deployment without approval raises error
Chain 5: Always-On Architecture
EventBus → SystemStateMachine → TradingEngine
    → ReconciliationEngine → StatePersistenceManager

Tests:

Events published on bus reach all subscribers
State machine transitions drive engine behavior
Reconciliation detects position discrepancies
State snapshots saved and restored correctly
Recovery from crash restores last known good state
Chain 6: Broker Resilience
BrokerReconnectionManager → disconnect → reconnect
    → failover → reconnect

Tests:

Disconnect detected within timeout
Exponential backoff applied on reconnect attempts
Failover to secondary broker when primary unreachable
Orders in flight handled correctly during disconnect
State consistent after reconnection
Section 6: End-to-End Tests
E2E-1: Full Research Cycle
Complete pipeline from data loading to signal evaluation:

def test_full_research_cycle():
    """HANDOFF.md S17 pseudocode — full research pipeline."""
    config = load_config("configs/")
    data = HistoricalDataLoader(config).load(tickers=TEST_TICKERS, years=3)
    data = CorporateActionAdjuster().adjust(data)
    DataValidator().validate(data)

    features = TechnicalIndicatorEngine(config).compute(data)
    features = FeatureNormalizer().normalize(features)

    research = ResearchRunner(config)
    results = research.run(data, features)

    assert results.sharpe > 0  # sanity check
    assert results.max_drawdown < 0.50  # not catastrophic
    assert not results.look_ahead_flags  # no look-ahead detected

E2E-2: 30-Day Paper Trading Simulation
def test_30_day_paper_trading():
    """Simulate 30 trading days with paper trading."""
    engine = PaperTradingRunner(config)
    engine.run(days=30, tickers=TEST_TICKERS)

    assert engine.nav_history[-1] > 0  # still solvent
    assert len(engine.trade_log) > 0  # trades executed
    assert engine.reconciliation_errors == 0  # no discrepancies
    # PnL attribution sums to total return
    assert abs(sum(engine.pnl_attribution.values()) - engine.total_return) < 1e-6

E2E-3: Multi-Regime Scenario (Bull → Bear → Recovery)
def test_multi_regime_scenario():
    """Test system behavior across market regimes."""
    scenarios = [
        ("bull", make_bull_market(days=60)),
        ("bear", make_bear_market(days=30, drawdown=0.30)),
        ("recovery", make_recovery(days=60)),
    ]
    engine = ResearchRunner(config)
    for regime_name, data in scenarios:
        result = engine.run(data)
        if regime_name == "bear":
            assert result.kill_switch_triggered or result.max_drawdown < 0.20

E2E-4: Real Historical Data Scenarios
Using real_market_data.py generators:

2008 Financial Crisis
2020 COVID Crash
2022 Rate Shock
2021 Meme Stock Volatility
NEW: Flash Crash scenario
NEW: Sector rotation event
NEW: Low volatility regime
NEW: High correlation regime
E2E-5: Kill Switch Trigger and Recovery
def test_kill_switch_trigger_and_recovery():
    """Verify kill switch fires and system can recover."""
    engine = TradingEngine(config)
    engine.start()

    # Inject 25% crash
    engine.inject_market_shock(drawdown=0.25)

    assert engine.state == SystemState.HALT
    assert engine.kill_switch.triggered is True
    assert len(engine.pending_orders) == 0  # no new orders

    # Manual recovery
    engine.reset_kill_switch(new_peak=engine.current_nav)
    engine.transition_to(SystemState.TRADING)
    assert engine.state == SystemState.TRADING

Section 7: Regression Tests
7.1 Deterministic Baselines
Run full pipeline with seed=42, capture golden outputs:

def test_deterministic_baseline():
    """Same seed → same output, always."""
    rng = np.random.default_rng(seed=42)
    result = run_full_pipeline(rng=rng, tickers=BASELINE_TICKERS)

    golden = load_golden_file("golden_files/baseline_seed42.parquet")
    pd.testing.assert_frame_equal(result.weights, golden.weights, atol=1e-6)
    pd.testing.assert_series_equal(result.alpha_scores, golden.alpha_scores, atol=1e-6)

7.2 Signal Stability
def test_signal_stability():
    """Same inputs → same alpha_scores across code changes."""
    data = load_golden_file("golden_files/input_data.parquet")
    scores = compute_alpha_scores(data, seed=42)
    golden_scores = load_golden_file("golden_files/alpha_scores.parquet")
    pd.testing.assert_series_equal(scores, golden_scores, atol=1e-6)

7.3 Portfolio Weight Stability
def test_portfolio_weight_stability():
    """Same alpha → same weights (within floating-point tolerance)."""
    alpha = load_golden_file("golden_files/alpha_scores.parquet")
    weights = optimize_portfolio(alpha, seed=42)
    golden_weights = load_golden_file("golden_files/weights.parquet")
    pd.testing.assert_series_equal(weights, golden_weights, atol=1e-4)

7.4 NAV Trajectory
def test_nav_trajectory():
    """Known scenario → known NAV path."""
    scenario = load_golden_file("golden_files/scenario_data.parquet")
    nav_history = run_simulation(scenario, seed=42)
    golden_nav = load_golden_file("golden_files/nav_trajectory.parquet")
    np.testing.assert_allclose(nav_history, golden_nav, atol=1e-6)

Section 8: Performance Benchmarks
Benchmark	Input Size	Target	Fail Threshold
Feature computation	10 tickers × 252 days	< 100ms	> 500ms
Feature computation	400 tickers × 252 days	< 5s	> 15s
Portfolio optimization	10 tickers	< 200ms	> 1000ms
Portfolio optimization	400 tickers	< 500ms	> 2000ms
Order generation	100 trades	< 50ms	> 200ms
EventBus throughput	10,000 events	> 10K events/s	< 1K events/s
Full research cycle	10 tickers × 3 years	< 2s	> 10s
Full research cycle	400 tickers × 3 years	< 30s	> 120s
Memory usage	400 tickers × 10 years	< 2 GB	> 4 GB
Benchmark Implementation
@pytest.mark.benchmark
def test_feature_computation_performance(benchmark):
    data = make_ohlcv(tickers=10, days=252)
    result = benchmark(TechnicalIndicatorEngine(config).compute, data)
    assert benchmark.stats["mean"] < 0.1  # 100ms

@pytest.mark.benchmark
def test_portfolio_optimization_performance(benchmark):
    alpha = make_alpha_scores(n_tickers=400)
    cov = make_covariance_matrix(n_tickers=400)
    result = benchmark(PortfolioOptimizer(config).optimize, alpha, cov)
    assert benchmark.stats["mean"] < 0.5  # 500ms

@pytest.mark.benchmark
def test_event_bus_throughput(benchmark):
    bus = EventBus()
    events = [Event(f"test_{i}") for i in range(10_000)]
    def publish_all():
        for e in events:
            bus.publish(e)
    benchmark(publish_all)
    throughput = 10_000 / benchmark.stats["mean"]
    assert throughput > 10_000  # > 10K events/sec

Section 9: Stress Tests & Fault Injection
9.1 Broker Disconnect Mid-Order-Batch
@pytest.mark.stress
def test_broker_disconnect_mid_batch():
    broker = SimulationBroker(config, disconnect_after=5)
    orders = generate_orders(n=20)
    results = []
    for order in orders:
        try:
            results.append(broker.submit(order))
        except BrokerDisconnectedError:
            break
    # Verify partial fills handled correctly
    assert len(results) == 5
    assert all(r.status in ("FILLED", "PARTIAL") for r in results)

9.2 NaN Injection Into OHLCV Mid-Stream
@pytest.mark.stress
def test_nan_injection_caught():
    data = make_ohlcv(tickers=10, days=100)
    # Inject NaN at random positions
    for _ in range(50):
        idx = np.random.randint(len(data))
        data.iloc[idx, data.columns.get_loc("close")] = np.nan
    with pytest.raises(DataValidationError):
        DataValidator().validate(data)

9.3 1000-Ticker Universe (Scalability)
@pytest.mark.stress
def test_1000_ticker_universe():
    data = make_ohlcv(tickers=1000, days=252)
    features = TechnicalIndicatorEngine(config).compute(data)
    alpha = compute_alpha_scores(features)
    weights = PortfolioOptimizer(config).optimize(alpha)
    assert len(weights) == 1000
    assert abs(weights).max() <= 0.02 + 1e-4

9.4 10-Year Data Window (Memory Pressure)
@pytest.mark.stress
def test_10_year_memory_pressure():
    import tracemalloc
    tracemalloc.start()
    data = make_ohlcv(tickers=400, days=2520)  # ~10 years
    features = TechnicalIndicatorEngine(config).compute(data)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 2 * 1024**3  # < 2 GB

9.5 Kill Switch with Extreme Drawdown (90% Crash)
@pytest.mark.stress
def test_kill_switch_extreme_crash():
    ks = PortfolioKillSwitch()
    ks.update_peak(10_000_000)
    assert ks.check(1_000_000) is True  # 90% drawdown
    # Verify system halts cleanly, no overflow/underflow

9.6 Concurrent EventBus Publishing (Thread Safety)
@pytest.mark.stress
def test_concurrent_event_bus():
    bus = EventBus()
    received = []
    bus.subscribe("test", lambda e: received.append(e))

    threads = []
    for i in range(10):
        t = threading.Thread(target=lambda: [bus.publish(Event(f"t{i}_{j}")) for j in range(100)])
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(received) == 1000  # all events delivered

9.7 Config with Extreme Values
@pytest.mark.stress
def test_extreme_config_values():
    extreme_config = {
        "max_leverage": 10.0,
        "max_position_size": 0.5,
        "max_sector_exposure": 1.0,
    }
    # System should either reject or handle gracefully
    engine = ConstraintEngine(extreme_config)
    # Verify constraints still enforced (even if loose)

9.8 Rapid State Machine Transitions (Fuzzing)
@pytest.mark.stress
def test_state_machine_fuzzing():
    sm = SystemStateMachine()
    states = [SystemState.INIT, SystemState.DATA_READY, SystemState.TRADING,
              SystemState.HALT, SystemState.SHUTDOWN]
    for _ in range(10_000):
        target = random.choice(states)
        try:
            sm.transition_to(target)
        except InvalidTransitionError:
            pass  # expected for invalid transitions
    # No crashes, no corruption
    assert sm.current_state in states

Section 10: Synthetic Data Generators
10.1 Existing Generators (KEEP)
From tests/conftest.py:

make_ohlcv(tickers, days) — synthetic OHLCV with realistic price dynamics
make_returns(tickers, days) — log returns from geometric Brownian motion
make_factor_returns(factors, days) — synthetic factor return series
make_cointegrated_pair() — two price series with known cointegration
From tests/real_market_data.py:

make_real_ohlcv() — realistic OHLCV calibrated to actual market statistics
make_real_returns() — returns matching empirical distribution
make_real_factor_returns() — factor returns matching historical factor premiums
10.2 New Generators Needed
generators/order_flow_generator.py

Realistic order sequences for execution testing
Configurable: order size distribution, urgency levels, timing patterns
Supports: market, limit, VWAP, TWAP order types
generators/stress_scenario_generator.py

Configurable crash/recovery/regime scenarios
Parameters: drawdown magnitude, duration, recovery shape
Predefined: flash crash, slow bleed, V-recovery, L-shaped
generators/alt_data_generator.py

Synthetic analyst estimates with realistic revision patterns
News sentiment with configurable signal-to-noise ratio
Options chains with valid no-arbitrage IV surfaces
generators/config_fuzzer.py

Generates edge-case config combinations
Boundary values for all numeric parameters
Invalid type combinations for negative testing
Section 11: Simulation Environments
11.1 SimulationBroker Configurations
Config	Slippage	Latency	Partial Fills	Use Case
ideal	0 bps	0 ms	No	Unit tests, deterministic baselines
realistic	2-5 bps	10-50 ms	Yes (80% fill rate)	Integration tests
adverse	10-20 bps	100-500 ms	Yes (50% fill rate)	Stress tests
chaos	Random	Random	Random disconnects	Fault injection
11.2 Market Regime Simulator
class MarketRegimeSimulator:
    """Generates synthetic market data matching specific regimes."""

    def bull_market(self, days=60, annual_return=0.15, vol=0.12):
        """Low vol, positive drift."""

    def bear_market(self, days=30, drawdown=0.30, vol=0.35):
        """High vol, negative drift, fat tails."""

    def sideways(self, days=90, vol=0.08):
        """Very low vol, mean-reverting."""

    def crash(self, days=5, drawdown=0.15, vol=0.80):
        """Extreme vol, sharp decline."""

    def regime_sequence(self, regimes: list):
        """Chain multiple regimes together."""

11.3 Multi-Broker Failover Simulation
Primary broker: SimulationBroker with configurable failure probability
Secondary broker: SimulationBroker with different latency profile
Failover trigger: 3 consecutive failures or timeout > 5s
11.4 Data Feed Interruption Simulation
Gap injection: missing bars at random intervals
Stale data: repeated bars for N periods
Burst: rapid delivery of accumulated data after gap
Section 12: Mocking/Stubbing Strategy
Always Real (No Mocking)
These modules are simple, fast, and critical — always use real implementations:

Module	Reason
PortfolioKillSwitch	Simplest module; must be battle-tested with real logic
DataValidator	Critical safety gate; must validate with real logic
ConstraintEngine	Single source of truth for constraints; no shortcuts
FeatureNormalizer	Fast, pure math; mock would miss edge cases
DataAlignmentEngine	Look-ahead bias enforcer; must run real checks
Real for Integration, Mock for Unit
Module	Unit Test Mock	Integration Test
PortfolioOptimizer	Mock with known weight output	Real CVXPY solver
SimulationBroker	Mock with instant fills	Real fill simulation
ResearchRunner	SeededAlphaResearch stub	Full pipeline
TechnicalIndicatorEngine	Mock with synthetic features	Real computation
EventBus	Mock with direct delivery	Real pub/sub
Always Mock (External Dependencies)
External Dependency	Mock Strategy
Interactive Brokers API	MockIBAdapter — predefined responses
Alpaca API	MockAlpacaAdapter — predefined responses
Yahoo Finance API	MockYFinanceConnector — cached data
File system (DataStorageManager)	tmp_path fixture with in-memory parquet
Network calls	responses or httpx_mock library
Existing Test Doubles
SimulationBroker: Replaces live brokers in all non-live tests (already implemented)
SeededAlphaResearch: Replaces ResearchRunner for scenario tests (existing pattern in test files)
DataDrivenResearch: Replaces SeededAlpha for real-data tests (existing pattern)
Section 13: Deterministic Reproducibility
Random Number Generation
All RNG uses np.random.default_rng(seed=42) (existing codebase pattern):

# CORRECT — deterministic
rng = np.random.default_rng(seed=42)
samples = rng.standard_normal(100)

# WRONG — non-deterministic
samples = np.random.randn(100)  # uses global state

Floating-Point Tolerances
Quantity	Absolute Tolerance	Relative Tolerance
Prices	1e-6	1e-8
Portfolio weights	1e-4	1e-6
Sharpe ratio	1e-2	1e-3
IC values	1e-4	1e-5
NAV	1e-6	1e-8
Drawdown %	1e-4	1e-5
Platform Independence
Use pd.Timestamp not datetime.datetime
Use pathlib.Path not OS-dependent separators
Avoid dict ordering assumptions (use sorted() when needed)
Pin numpy/pandas versions for golden file generation
Golden Files
Format: compressed Parquet (.parquet.gz)
Location: tests/regression/golden_files/
Regeneration: pytest --regenerate-golden flag
Version tracking: git-committed with SHA in filename
Section 14: Long-Duration Stability Tests
14.1: 30-Day Paper Trading Simulation
@pytest.mark.tier5
@pytest.mark.stability
def test_30_day_stability():
    """Verify no memory growth or state accumulation over 30 days."""
    engine = PaperTradingRunner(config)
    initial_memory = get_memory_usage()

    for day in range(30):
        engine.run_day(day)

        # Memory check every 5 days
        if day % 5 == 0:
            current_memory = get_memory_usage()
            growth = (current_memory - initial_memory) / initial_memory
            assert growth < 0.10, f"Memory grew {growth:.1%} by day {day}"

    # State accumulation bounded
    assert len(engine.state_store.snapshots) <= 30  # one per day max

14.2: 252-Day Backtest (IC Window Stability)
@pytest.mark.tier5
def test_252_day_ic_stability():
    """Verify IC computation window remains stable over full year."""
    data = make_ohlcv(tickers=50, days=504)  # 2 years for 1-year OOS
    runner = ResearchRunner(config)
    results = runner.run_walk_forward(data, train_days=252, test_days=252)

    # IC should not degrade systematically
    ic_series = results.rolling_ic
    assert ic_series.mean() > 0.03
    assert ic_series.std() < 0.10  # reasonable stability

14.3: EventBus — 100,000 Events
@pytest.mark.tier5
def test_event_bus_100k_events():
    """Verify EventBus handles 100K events without backpressure failure."""
    bus = EventBus()
    received_count = 0
    def handler(event):
        nonlocal received_count
        received_count += 1

    bus.subscribe("stress_test", handler)
    for i in range(100_000):
        bus.publish(Event("stress_test", data={"seq": i}))

    assert received_count == 100_000

14.4: StateStore — 1,000 Snapshots
@pytest.mark.tier5
def test_state_store_1000_snapshots():
    """Verify StateStore handles 1000 snapshots without corruption."""
    store = StateStore(path=tmp_path)
    for i in range(1000):
        state = {"nav": 1_000_000 + i, "positions": {"AAPL": 100 + i}}
        store.save_snapshot(state, snapshot_id=i)

    # Verify last snapshot
    restored = store.load_snapshot(999)
    assert restored["nav"] == 1_000_999
    assert restored["positions"]["AAPL"] == 1099

    # Verify first snapshot still accessible
    restored = store.load_snapshot(0)
    assert restored["nav"] == 1_000_000

14.5: Broker Reconnection — 50 Cycles
@pytest.mark.tier5
def test_broker_50_reconnection_cycles():
    """Verify broker reconnection manager handles 50 disconnect/reconnect cycles."""
    manager = BrokerReconnectionManager(config)
    for i in range(50):
        manager.simulate_disconnect()
        assert manager.is_connected is False
        manager.reconnect()
        assert manager.is_connected is True
    # No leaked connections, no state corruption
    assert manager.connection_count == 1  # single active connection

Section 15: Requirement → Module → Test Traceability Matrix
Req ID	Requirement	Module(s)	Test ID(s)
R01	Point-in-time: features use only data < as_of	data_alignment_engine, all feature generators	test_group_b::test_filters_future_data_strict, test_group_b::test_no_lookahead_*, test_group_c::test_no_lookahead_in_factors
R02	No NaN prices, no negatives, no duplicates	data_validator	test_group_a::TestDataValidator (5 tests)
R03	max_position_size = 0.02	constraint_engine, portfolio_optimizer	test_group_f::test_constraint_engine, test_group_f::test_portfolio_optimizer
R04	max_sector_exposure = 0.20	constraint_engine, exposure_monitor	test_group_f::test_sector_constraint, test_group_g::test_exposure_monitor
R05	max_leverage = 2.0	constraint_engine, leverage_controller	test_group_f::test_leverage_constraint, test_group_g::test_leverage_controller
R06	Kill switch at 20% DD	portfolio_kill_switch	test_group_g::test_kill_switch_* (5 tests), test_phase1_safety
R07	OOS IC > 0.03 for 252+ days	signal_ranking_engine	test_group_e::TestSignalRankingEngine
R08	Order flow: generator → router → abstraction → adapter	order_generator, order_router, broker_abstraction_layer	test_group_h::TestOrderGenerator, TestOrderRouter
R09	Governance gate: Sharpe ≥ 1, DD ≤ 25%, 6mo paper, sign-off	approval_workflow, deployment_controller, strategy_review_pipeline	test_group_j::TestApprovalWorkflow, TestDeploymentController
R10	Config-driven: no magic numbers	config_validator, all YAML configs	test_group_a::TestConfigFiles
R11	Raw data immutability	corporate_action_adjuster	test_group_a::TestCorporateActionAdjuster
R12	Signals normalized before optimizer	feature_normalizer, research_runner	test_group_b::TestFeatureNormalizer, test_group_k::TestResearchRunner
R13	State machine: INIT → DATA_READY → TRADING → HALT → SHUTDOWN	system_state_machine	test_always_on::TestSystemStateMachine
R14	Reconciliation: broker = source of truth	reconciliation_engine	test_always_on::TestReconciliationEngine
R15	Broker resilience: reconnect + failover	broker_reconnection_manager	test_always_on::TestBrokerReconnection
R16	Dollar-neutral	constraint_engine	test_group_f::test_dollar_neutral
R17	Survivor-bias-free	historical_data_loader	test_group_a::TestHistoricalDataLoader
R18	IC half-life < 20 days → review flag	signal_decay_detector	test_group_e::TestSignalDecayDetector
R19	Strategy lifecycle: ACTIVE → REVIEW → SUSPENDED → RETIRED	strategy_retirement_manager	test_group_e::TestStrategyRetirementManager
R20	Drawdown alerts: 10% warning, 15% alert, 20% kill	drawdown_monitor	test_group_g::TestDrawdownMonitor
R21	Event bus idempotency + replay	event_bus	test_always_on::TestEventBus
R22	State persistence save/restore	state_persistence_manager	test_always_on::TestStatePersistence
R23	VWAP at 5% participation	vwap_execution	test_group_h::TestVWAPExecution
R24	SimulationBroker before live adapters	simulation_broker	test_group_h::TestSimulationBroker
R25	Feature normalizer: z-score + ±3σ winsorization	feature_normalizer	test_group_b::TestFeatureNormalizer
Section 16: CI/CD Integration Plan
Tier 1 — Pre-commit (< 30 seconds)
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: ruff-lint
      name: Ruff linting
      entry: ruff check quant_fund/ tests/
      language: system
    - id: ruff-format
      name: Ruff formatting
      entry: ruff format --check quant_fund/ tests/
      language: system
    - id: mypy
      name: Type checking
      entry: mypy quant_fund/ --ignore-missing-imports
      language: system
    - id: smoke-tests
      name: Unit test smoke
      entry: pytest tests/ -m "tier1" --timeout=30 -q
      language: system

Scope: ~100 fast unit tests, linting, type checks

Tier 2 — PR Gate (< 5 minutes)
# .github/workflows/pr-gate.yml
name: PR Gate
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/ -m "tier1 or tier2" --timeout=300 -q
        env:
          PYTHONPATH: .
      - run: pytest tests/ --cov=quant_fund --cov-fail-under=85

Scope: All unit tests + module invariant tests (~700 tests). Coverage gate: ≥ 85%

Tier 3 — Post-merge (< 15 minutes)
# .github/workflows/post-merge.yml
name: Post-merge Validation
on:
  push:
    branches: [main]
jobs:
  integration:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/ -m "tier1 or tier2 or tier3" --timeout=900 -q

Scope: Integration tests + E2E scenarios (~200 additional tests)

Tier 4 — Nightly (< 60 minutes)
# .github/workflows/nightly.yml
name: Nightly Full Suite
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC
jobs:
  full-suite:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/ -m "not tier5" --timeout=3600 -q
      - run: pytest tests/benchmarks/ --benchmark-json=bench.json
      - run: pytest tests/stress/ --timeout=1800
      - run: pytest tests/regression/ --timeout=600

Scope: Full suite + performance benchmarks + stress tests + regression

Tier 5 — Weekly (long-duration)
# .github/workflows/weekly.yml
name: Weekly Stability
on:
  schedule:
    - cron: '0 0 * * 0'  # Sunday midnight
jobs:
  stability:
    runs-on: ubuntu-latest
    timeout-minutes: 240
    steps:
      - run: pytest tests/stability/ -m "tier5" --timeout=14400 -v

Scope: 30-day simulation, memory leak detection, 252-day backtest

Required Pytest Markers
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "tier1: Pre-commit smoke tests (< 30s)",
    "tier2: PR gate tests (< 5min)",
    "tier3: Post-merge integration (< 15min)",
    "tier4: Nightly full suite (< 60min)",
    "tier5: Weekly stability (long-duration)",
    "unit: Isolated unit test",
    "integration: Multi-module interaction",
    "e2e: Full pipeline test",
    "stress: Fault injection and extreme load",
    "benchmark: Performance measurement",
    "slow: Execution time > 10s",
    "stability: Long-duration stability test",
]

Section 17: Test Coverage Targets
Subsystem	Current Est.	Target Line	Target Branch
config/	80%	95%	90%
data_layer/	85%	95%	90%
feature_factory/	95%	98%	95%
alternative_data/	60%	80%	70%
research_algorithms/	90%	95%	85%
alpha_discovery/	90%	95%	85%
alpha_monitoring/	90%	95%	85%
representation_learning/	85%	90%	80%
portfolio/	95%	98%	90%
risk_engine/	95%	98%	95%
execution/	90%	95%	90%
broker_interface/	85%	90%	85%
infrastructure/	90%	95%	90%
monitoring/	90%	95%	85%
governance/	90%	95%	90%
main/	85%	95%	90%
research_cluster/	85%	90%	80%
Overall	~87%	≥ 93%	≥ 87%
Coverage Enforcement
# Run with coverage
pytest tests/ --cov=quant_fund --cov-report=html --cov-report=term-missing

# Fail if below threshold
pytest tests/ --cov=quant_fund --cov-fail-under=85

# Per-subsystem coverage (custom script)
python scripts/coverage_by_subsystem.py --target=93

Section 18: Quantitative Pass/Fail Metrics
Signal Quality
Metric	Pass Threshold	Fail Threshold	Test Location
IC mean (252-day OOS)	≥ 0.03	< 0.01	test_group_e::TestSignalRankingEngine
IC t-statistic	≥ 2.0	< 1.5	test_group_e::TestSignalRankingEngine
IC half-life	≥ 20 days	< 10 days	test_group_e::TestSignalDecayDetector
Portfolio Constraints
Metric	Pass Threshold	Fail Threshold	Test Location
Max position weight	≤ 0.02 + 1e-4	> 0.021	test_group_f::test_constraint_engine
Max sector exposure	≤ 0.20 + 1e-4	> 0.201	test_group_f::test_sector_constraint
Max leverage	≤ 2.0 + 1e-4	> 2.001	test_group_f::test_leverage_constraint
Dollar neutrality	|sum(w)| < 0.02	|sum(w)| ≥ 0.05	test_group_f::test_dollar_neutral
Risk Thresholds
Metric	Pass Threshold	Fail Threshold	Test Location
Kill switch trigger	Fires at DD ≥ 20.0%	Fails to fire	test_group_g::test_kill_switch_*
Drawdown warning	Fires at DD ≥ 10.0%	Misses alert	test_group_g::TestDrawdownMonitor
Drawdown alert	Fires at DD ≥ 15.0%	Misses alert	test_group_g::TestDrawdownMonitor
Performance Targets
Metric	Pass Threshold	Fail Threshold	Test Location
Feature computation (10 tickers)	< 100ms	> 500ms	benchmarks/bench_feature_computation
Optimization (10 tickers)	< 200ms	> 1000ms	benchmarks/bench_portfolio_optimization
Optimization (400 tickers)	< 500ms	> 2000ms	benchmarks/bench_portfolio_optimization
Event bus throughput	> 10K events/s	< 1K events/s	benchmarks/bench_event_bus_throughput
Order generation (100 trades)	< 50ms	> 200ms	benchmarks/bench_order_generation
Stability Metrics
Metric	Pass Threshold	Fail Threshold	Test Location
30-day simulation memory growth	< 10%	> 50%	stability/test_30day_paper_trading
State machine transition errors	0	> 0	stability/test_state_accumulation
EventBus 100K delivery	100% delivered	< 99.9%	stability/test_30day_paper_trading
Data Quality
Metric	Pass Threshold	Fail Threshold	Test Location
Future timestamp ratio	0.0	> 0.0	test_group_a::TestDataValidator
NaN close price ratio	0.0	> 0.0	test_group_a::TestDataValidator
Negative price ratio	0.0	> 0.0	test_group_a::TestDataValidator
Reproducibility
Metric	Pass Threshold	Fail Threshold	Test Location
Cross-run NAV delta (same seed)	< 1e-6	> 1e-4	regression/test_deterministic_baselines
Cross-run weight delta (same seed)	< 1e-4	> 1e-2	regression/test_signal_stability
Cross-run alpha delta (same seed)	< 1e-6	> 1e-4	regression/test_signal_stability
Generated for QuantFund V8. Covers 127 production modules across 18 subsystems, targeting ~1200 tests with ≥93% line coverage.
