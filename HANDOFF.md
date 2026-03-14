# QuantFund V8 — Claude Code Handoff

## Read this entire file before writing any code.

-----

# PART 1: BRIEFING

## What this project is

**QuantFund V8** is an institutional multi-strategy quantitative trading platform
targeting 200–400 US equities at medium frequency. It is a research-driven
automated alpha discovery and execution platform.

**Primary goals:**

- Automated discovery and evaluation of candidate trading signals
- Factor, mean-reversion, stat-arb, ML, and regime-based strategies coexisting
- Portfolio construction with factor risk model and dynamic capital allocation
- Institutional-grade execution with microstructure-aware algorithms
- Full monitoring, governance, and deployment lifecycle

**Performance targets:**

- Sharpe ratio ≥ 1.5
- Max drawdown ≤ 20%
- Moderate turnover profile

**Trading parameters:**

- Portfolio update: hourly or daily
- Execution window: market hours
- Max position size: 2% of portfolio
- Sector exposure limit: 20%
- Max leverage: 2.0×

-----

## Repo layout and implementation status

```
quant_fund/
├── HANDOFF.md                          ← this file
│
├── config/
│   ├── system_config.yaml              ✅ system-level settings
│   ├── trading_config.yaml             ✅ universe, position limits, leverage
│   ├── data_config.yaml                ✅ data sources, schedules, vendors
│   ├── execution_config.yaml           ✅ algo params, broker routing
│   └── risk_config.yaml                ✅ drawdown limits, exposure limits
│
├── data_layer/
│   ├── historical_data_loader.py       ✅
│   ├── live_data_stream_adapter.py     ⬜
│   ├── corporate_action_adjuster.py    ✅
│   ├── data_validator.py               ✅
│   └── data_storage_manager.py         ✅
│
├── alternative_data/
│   ├── news_sentiment/
│   │   ├── news_ingestion.py           ✅
│   │   ├── text_cleaning.py            ✅
│   │   ├── sentiment_model.py          ✅
│   │   └── sentiment_feature_generation.py ✅
│   ├── analyst_estimates/
│   │   ├── analyst_data_ingestion.py   ✅
│   │   └── estimate_revision_features.py ✅
│   ├── options_data/
│   │   ├── options_chain_ingestion.py  ✅
│   │   ├── implied_volatility_surface_builder.py ✅
│   │   └── options_feature_generation.py ✅
│   ├── etf_flows/
│   │   ├── etf_flow_ingestion.py       ✅
│   │   └── flow_feature_generation.py  ✅
│   └── macro_data/
│       ├── macro_data_ingestion.py     ✅
│       ├── macro_feature_generation.py ✅
│       └── macro_regime_classifier.py  ✅
│
├── feature_factory/
│   ├── base_feature_generator.py       ✅
│   ├── technical_indicator_engine.py   ✅
│   ├── feature_normalizer.py           ✅
│   └── data_alignment_engine.py        ✅
│
├── representation_learning/
│   ├── autoencoder_model.py            ✅
│   ├── temporal_model_lstm.py          ✅
│   ├── temporal_model_transformer.py   ✅
│   ├── cross_asset_embedding_model.py  ✅
│   └── embedding_feature_store.py      ✅
│
├── alpha_discovery/
│   ├── feature_combinator.py           ✅
│   ├── symbolic_regression_engine.py   ✅
│   ├── genetic_algorithm_search.py     ✅
│   ├── ml_feature_selector.py          ✅
│   └── signal_ranking_engine.py        ✅
│
├── research_algorithms/
│   ├── factor_models/
│   │   ├── momentum_factor.py          ✅
│   │   ├── value_factor.py             ✅
│   │   ├── quality_factor.py           ✅
│   │   ├── low_volatility_factor.py    ✅
│   │   └── size_factor.py              ✅
│   ├── mean_reversion/
│   │   ├── zscore_reversion_strategy.py ✅
│   │   └── short_term_reversal_strategy.py ✅
│   ├── statistical_arbitrage/
│   │   ├── pairs_trading_strategy.py   ✅
│   │   ├── cointegration_engine.py     ✅
│   │   └── kalman_spread_model.py      ✅
│   ├── machine_learning/
│   │   ├── gradient_boosted_tree_model.py ✅
│   │   ├── random_forest_model.py      ✅
│   │   └── neural_network_predictor.py ✅
│   └── regime_models/
│       ├── hidden_markov_regime_model.py ✅
│       ├── volatility_regime_detector.py ✅
│       └── market_state_classifier.py  ✅
│
├── alpha_monitoring/
│   ├── alpha_performance_tracker.py    ✅
│   ├── information_coefficient_monitor.py ✅
│   ├── signal_decay_detector.py        ✅
│   └── strategy_retirement_manager.py  ✅
│
├── portfolio/
│   ├── portfolio_construction/
│   │   ├── portfolio_optimizer.py      ✅
│   │   └── constraint_engine.py        ✅
│   ├── factor_risk_model/
│   │   ├── factor_exposure_estimator.py ✅
│   │   ├── factor_covariance_estimator.py ✅
│   │   └── risk_decomposition.py       ✅
│   ├── capital_allocation/
│   │   ├── strategy_performance_tracker.py ✅
│   │   ├── strategy_correlation_matrix.py ✅
│   │   └── dynamic_strategy_allocator.py ✅
│   └── capacity_model/
│       ├── liquidity_estimator.py      ✅
│       ├── market_impact_model.py      ✅
│       └── capacity_simulator.py       ✅
│
├── risk_engine/
│   ├── drawdown_monitor.py             ✅
│   ├── exposure_monitor.py             ✅
│   ├── leverage_controller.py          ✅
│   ├── portfolio_kill_switch.py        ✅
│   └── stress_test_engine.py           ✅
│
├── execution/
│   ├── execution_algorithms/
│   │   ├── vwap_execution.py           ✅
│   │   ├── twap_execution.py           ✅
│   │   └── liquidity_seeking_execution.py ✅
│   ├── order_management/
│   │   ├── order_generator.py          ✅
│   │   └── order_router.py             ✅
│   └── microstructure_models/
│       ├── bid_ask_spread_model.py     ✅
│       ├── order_book_liquidity_model.py ✅
│       ├── adverse_selection_model.py  ✅
│       ├── fill_probability_model.py   ✅
│       └── queue_position_estimator.py ✅
│
├── broker_interface/
│   ├── broker_abstraction_layer.py     ✅
│   ├── interactive_brokers_adapter.py  ✅
│   ├── alpaca_adapter.py              ✅
│   └── simulation_broker.py           ✅
│
├── research_cluster/
│   ├── distributed_backtest_runner.py  ✅
│   ├── experiment_scheduler.py         ✅
│   └── parallel_signal_evaluator.py    ✅
│
├── monitoring/
│   ├── pnl_dashboard.py                ✅
│   ├── risk_dashboard.py               ✅
│   ├── execution_quality_monitor.py    ✅
│   ├── alerting_system.py              ✅
│   └── system_health_monitor.py        ✅
│
├── governance/
│   ├── strategy_review_pipeline.py     ✅
│   ├── approval_workflow.py            ✅
│   └── deployment_controller.py        ✅
│
├── infrastructure/
│   ├── docker_environment_setup.py     ⬜
│   ├── ci_cd_pipeline_manager.py       ⬜
│   ├── dataset_version_control.py      ⬜
│   └── disaster_recovery_manager.py    ⬜
│
└── main/
    ├── research_runner.py              ✅
    ├── paper_trading_runner.py         ✅
    └── live_trading_runner.py          ✅
```

-----

## Recommended implementation order

Work in this sequence. Each group depends on the previous.

```
Group A — Foundation (DONE ✅)
  config/                           All five YAML files with defaults
  data_layer/data_validator.py      Data contract and validation rules
  data_layer/data_storage_manager.py  Parquet/HDF5 schema and access layer
  data_layer/historical_data_loader.py
  data_layer/corporate_action_adjuster.py  Dividend/split adjustment (critical)

Group B — Feature pipeline (DONE ✅)
  feature_factory/base_feature_generator.py   Abstract base class
  feature_factory/data_alignment_engine.py    Point-in-time alignment (look-ahead guard)
  feature_factory/technical_indicator_engine.py
  feature_factory/feature_normalizer.py       Cross-sectional z-score, winsorize

Group C — Research algorithms (DONE ✅)
  research_algorithms/factor_models/          All five factors
  research_algorithms/mean_reversion/
  research_algorithms/statistical_arbitrage/  cointegration_engine first, then kalman
  research_algorithms/machine_learning/       GBT first (most reliable in practice)
  research_algorithms/regime_models/          HMM first

Group D — Alternative data (DONE ✅)
  alternative_data/analyst_estimates/         Highest alpha per implementation cost
  alternative_data/options_data/
  alternative_data/news_sentiment/            NLP pipeline; implement last in this group
  alternative_data/etf_flows/
  alternative_data/macro_data/

Group E — Alpha discovery and monitoring (DONE ✅)
  alpha_discovery/                            Full pipeline: combinator → ranking
  alpha_monitoring/                           IC monitor, decay detector
  representation_learning/                    Autoencoder first; transformers last

Group F — Portfolio construction (DONE ✅)
  portfolio/factor_risk_model/               Factor exposures and covariance first
  portfolio/portfolio_construction/          Optimizer and constraints
  portfolio/capital_allocation/              Strategy allocator
  portfolio/capacity_model/                  Market impact model

Group G — Risk engine (DONE ✅)
  risk_engine/                               Kill switch first, then monitors

Group H — Execution
  broker_interface/simulation_broker.py      Implement before live adapters
  execution/microstructure_models/           Spread and fill probability models first
  execution/execution_algorithms/            VWAP/TWAP first
  execution/order_management/
  broker_interface/interactive_brokers_adapter.py
  broker_interface/alpaca_adapter.py

Group I — Research infrastructure
  research_cluster/distributed_backtest_runner.py
  research_cluster/parallel_signal_evaluator.py
  research_cluster/experiment_scheduler.py

Group J — Monitoring and governance
  monitoring/                                PnL dashboard first
  governance/                                Review pipeline then approval workflow

Group K — Entry points
  main/research_runner.py                    First
  main/paper_trading_runner.py               Second
  main/live_trading_runner.py                Last — only after paper validation
```

-----

## Strict rules — never violate these

### Look-ahead bias (most critical rule in quant systems)

- **All feature computation must be strictly point-in-time.** Features at time T
  may only use data available before T. This applies to normalisation, covariance
  estimation, and universe selection as well as raw signals.
- `data_alignment_engine.py` is the enforcer. Every feature pipeline routes through it.
- Backtest performance that cannot be replicated in paper trading is the symptom.
  The cause is almost always look-ahead bias. Treat any backtest Sharpe > 3.0 with suspicion.

### Data layer rules

- Raw data is never modified. All adjustments (splits, dividends) produce new
  adjusted series stored separately. `corporate_action_adjuster.py` owns this.
- Every data fetch returns a point-in-time snapshot with an `as_of` timestamp.
- `data_validator.py` must check: no future timestamps, no NaN in price columns,
  no negative prices, survivor-bias-free universe (delisted stocks included).

### Signal and alpha rules

- All signals must be cross-sectionally normalised (z-score, rank, or percentile)
  before entering the portfolio optimizer. Raw scores must never directly become weights.
- IC (information coefficient) is the primary evaluation metric for a signal.
  A signal must demonstrate statistically significant IC > 0.03 over at least
  252 trading days before being considered for deployment.
- Signals are evaluated in out-of-sample periods only. No in-sample evaluation
  ever counts as validation.

### Portfolio construction rules

- The portfolio optimizer must always enforce:
  - `max_position_size = 0.02` (2% per stock)
  - `sector_exposure_limit = 0.20` (20% per GICS sector)
  - `max_leverage = 2.0`
  - Long/short dollar-neutral or gross exposure within configured bounds
- `constraint_engine.py` is the single place where all constraints live.
  No constraint logic anywhere else.

### Risk engine rules

- `portfolio_kill_switch.py` is a hard stop. It has no dependencies on any
  other module except position data and drawdown calculation. It must be
  the simplest, most reliable module in the repo.
- Drawdown is measured peak-to-trough on NAV. A breach of `max_drawdown = 0.20`
  triggers an immediate halt to new position-taking (not immediate liquidation).
- The kill switch must be checked on every portfolio update cycle, before
  any new orders are generated.

### Execution rules

- `simulation_broker.py` must be functionally complete and realistic before
  any live adapter is written. All execution logic is tested against it first.
- Orders are never sent directly to a broker. They flow:
  `order_generator → order_router → broker_abstraction_layer → adapter`
- Slippage and market impact must be modelled in simulation.
  Use `market_impact_model.py` estimates to adjust expected fill prices.

### Governance rules

- No strategy reaches live trading without passing `governance/approval_workflow.py`.
- The approval workflow requires: out-of-sample Sharpe ≥ 1.0, max drawdown ≤ 25%,
  minimum 6-month paper trading period, risk team sign-off flag set to True.
- `deployment_controller.py` is the only module that can flip a strategy
  from paper to live. It checks the approval workflow output first.

### Config rules

- All parameters live in the five YAML config files. No magic numbers in code.
- Each module receives only the config sub-section it needs.
  Nothing reads the full config object except the entry points.

-----

## Key architectural decisions

|Decision              |Resolution                                                                     |
|----------------------|-------------------------------------------------------------------------------|
|Universe              |200–400 US equities, liquidity-filtered, survivor-bias-free                    |
|Data alignment        |Strict point-in-time via data_alignment_engine. No exceptions.                 |
|Feature normalisation |Cross-sectional z-score + winsorisation at ±3σ before optimizer                |
|Signal validation     |Out-of-sample IC > 0.03 over 252 days minimum                                  |
|Portfolio construction|Mean-variance / Black-Litterman with factor risk model covariance              |
|Risk model            |Multi-factor: momentum, value, quality, low-vol, size, sector, market          |
|Capital allocation    |Dynamic: Sharpe-weighted across active strategies, correlation-penalised       |
|Execution             |VWAP default; liquidity-seeking for large orders; TWAP for scheduled rebalances|
|Broker abstraction    |All brokers implement a common interface. Simulation broker first.             |
|Kill switch           |Hard stop on 20% drawdown. Simple, no dependencies, always checked first.      |
|Deployment gate       |paper trading ≥ 6 months + approval_workflow sign-off required                 |
|Market impact         |Almgren-Chriss model or calibrated empirical model in market_impact_model.py   |

-----

## Data flow through the system

```
RESEARCH / DAILY CYCLE:

  historical_data_loader  ──► corporate_action_adjuster
                                        │
                                        ▼
  alternative_data/*  ───────► data_alignment_engine  ◄─── data_validator
                                        │
                                        ▼
                             feature_factory (normalised features)
                                        │
                          ┌─────────────┴──────────────┐
                          ▼                            ▼
               research_algorithms/         alpha_discovery/
               (factor, MR, stat-arb,       (symbolic regression,
                ML, regime models)           genetic search, ML selector)
                          │                            │
                          └──────────┬─────────────────┘
                                     ▼
                          signal_ranking_engine
                          alpha_monitoring (IC, decay)
                                     │
                                     ▼
                          portfolio/factor_risk_model
                          portfolio/portfolio_optimizer
                          portfolio/constraint_engine
                          portfolio/dynamic_strategy_allocator
                                     │
                                     ▼
                          risk_engine (kill switch checked here)
                                     │
                                     ▼
                          order_generator → order_router
                          → broker_abstraction_layer
                          → simulation_broker / IB / Alpaca

LIVE CYCLE (same pipeline, live_data_stream_adapter replaces historical_data_loader):

  live_data_stream_adapter → [same pipeline as above] → live broker adapter
```

-----

## Module contracts

### `base_feature_generator.py` — abstract base all features inherit

```python
class BaseFeatureGenerator(ABC):
    feature_name: str
    lookback_days: int        # enforced by data_alignment_engine
    recompute_frequency: str  # "daily" | "hourly" | "on_data"

    @abstractmethod
    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """
        Return cross-sectional feature values for all assets as of `as_of`.
        data must contain only rows with timestamp < as_of (enforced by caller).
        Returns pd.Series indexed by ticker.
        """

    @abstractmethod
    def validate(self, feature_output: pd.Series) -> bool:
        """Return True if output passes sanity checks (no NaN > threshold, etc.)"""
```

### `portfolio_optimizer.py` — interface

```python
class PortfolioOptimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        alpha_scores: pd.Series,        # cross-sectionally normalised, indexed by ticker
        factor_covariance: pd.DataFrame,
        factor_exposures: pd.DataFrame,
        constraints: ConstraintSet,
        current_positions: pd.Series,
    ) -> pd.Series:                     # target weights indexed by ticker
        ...
```

### `broker_abstraction_layer.py` — interface all adapters implement

```python
class BrokerInterface(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> OrderAcknowledgement: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def get_positions(self) -> pd.Series: ...          # indexed by ticker

    @abstractmethod
    def get_account_value(self) -> float: ...

    @abstractmethod
    def get_fills(self, since: pd.Timestamp) -> List[Fill]: ...
```

### `portfolio_kill_switch.py` — must remain simple

```python
class KillSwitch:
    def __init__(self, max_drawdown: float, peak_nav: float): ...

    def check(self, current_nav: float) -> bool:
        """Return True if drawdown limit breached. Called before every order batch."""
        drawdown = (self.peak_nav - current_nav) / self.peak_nav
        return drawdown >= self.max_drawdown

    def update_peak(self, current_nav: float) -> None:
        """Call after every NAV update when not in halt state."""
        self.peak_nav = max(self.peak_nav, current_nav)
```

-----

## Testing strategy

**After each group, run this validation sequence:**

|Group              |Test                                                                                                                                                                          |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|A (data)           |Load 5 years of OHLCV for 10 tickers. Assert no NaN prices, no negative prices, correct adjustment for a known split.                                                         |
|B (features)       |Compute each feature with `as_of = T`. Assert no data after T was used (check via mock with future data injected).                                                            |
|C (research algos) |Each strategy produces a signal series. Assert IC is computed without look-ahead. Run walk-forward backtest on 2 years, check Sharpe and turnover are within plausible bounds.|
|D (alt data)       |Each ingestion module returns data with `as_of` timestamp. Assert analyst estimate revisions are not available before their announcement date.                                |
|E (alpha discovery)|Symbolic regression and GA search complete within time budget. Signal ranking produces ranked list with IC-weighted scores.                                                   |
|F (portfolio)      |Optimizer produces weights that satisfy all constraints. Assert max single position ≤ 0.02. Assert sector exposures ≤ 0.20. Assert leverage ≤ 2.0.                            |
|G (risk)           |Kill switch triggers at exactly 20% drawdown. Exposure monitor catches a simulated breach.                                                                                    |
|H (execution)      |Simulation broker fills orders with realistic slippage model. VWAP algo spreads a large order correctly across the day.                                                       |
|Full pipeline      |Run `paper_trading_runner.py` for 30 simulated trading days. Assert PnL attribution sums to total return. Assert no look-ahead flags raised.                                  |

-----

## Common pitfalls to avoid

|Pitfall                               |Prevention                                                                 |
|--------------------------------------|---------------------------------------------------------------------------|
|Look-ahead bias in normalisation      |Always compute z-scores using only historical mean/std up to `as_of`       |
|Survivor bias                         |Include delisted tickers in universe; use point-in-time index membership   |
|Overfitting in alpha discovery        |Out-of-sample validation only; penalise complexity in signal ranking       |
|Transaction cost neglect              |Every backtest must include bid-ask spread + market impact estimates       |
|Factor crowding                       |Monitor factor exposures in risk decomposition; penalise crowded factors   |
|Overstated Sharpe from short lookback |Minimum 252-day evaluation window for any signal                           |
|Signal decay ignored                  |`signal_decay_detector.py` runs daily; IC half-life < 20 days → review flag|
|Correlation between strategies ignored|`strategy_correlation_matrix.py` must gate capital allocation              |

-----

## Session workflow for Claude Code

At the start of each session:

1. Read this file.
1. Check which files already exist.
1. Implement the next group per the order above.
1. For any signal or feature: include a `validate_no_lookahead()` test.
1. For any portfolio output: assert all constraints are satisfied.
1. Update the ✅/⬜ list in this file before ending the session.

When implementing any module, verify:

- All parameters come from config, not hardcoded?
- Feature computation uses only data strictly before `as_of`?
- New signals have an IC computation alongside them?
- Broker calls go through `broker_abstraction_layer`, not directly to adapters?

-----

-----

# PART 2: FULL SYSTEM SPECIFICATION

-----

## S1. System Overview

QuantFund V8 is a medium-frequency quantitative equity trading platform.
It combines systematic factor investing, statistical arbitrage, machine learning
alpha discovery, and alternative data signals into a unified research-to-execution
pipeline with institutional-grade risk controls and governance.

**Deployment environments:**

- `local_research`: development and backtesting on historical data
- `paper_trading_environment`: live data feed, simulated execution, no real capital
- `live_trading_environment`: real capital, real broker, full governance required

**Containerisation:** Docker
**Orchestration:** scheduled jobs and task scheduler

-----

## S2. Configuration Files

### `config/system_config.yaml`

```yaml
system_name: QuantFund V8
environment: local_research        # local_research | paper_trading | live_trading
log_level: INFO
base_currency: USD
timezone: America/New_York
data_root: ./data
output_root: ./output
```

### `config/trading_config.yaml`

```yaml
universe:
  min_tickers: 200
  max_tickers: 400
  liquidity_filter:
    min_adv_usd: 5_000_000         # minimum average daily volume in USD
    min_price: 5.0                  # exclude penny stocks
  rebalance_frequency: daily        # daily | hourly
  execution_window: market_hours

position_limits:
  max_position_size: 0.02           # 2% of portfolio per stock
  max_sector_exposure: 0.20         # 20% per GICS sector
  max_leverage: 2.0
  dollar_neutral: true              # gross long ≈ gross short (optional flag)

performance_targets:
  target_sharpe: 1.5
  max_drawdown: 0.20
  turnover_profile: moderate
```

### `config/data_config.yaml`

```yaml
data_sources:
  price_data: polygon_io             # or alpaca, yfinance, proprietary
  fundamentals: compustat
  analyst_estimates: ibes
  options: cboe
  etf_flows: factset
  news: refinitiv
  macro: fred

schedules:
  price_update: "09:30-16:00/5min"
  daily_close: "16:30"
  alternative_data_refresh: "06:00"

storage:
  format: parquet
  partitioning: by_date
  lookback_years: 10
```

### `config/execution_config.yaml`

```yaml
default_algorithm: vwap
vwap_participation_rate: 0.05       # 5% of volume
twap_interval_minutes: 30
large_order_threshold_adv: 0.005    # >0.5% ADV → liquidity seeking
max_order_size_usd: 500_000
slippage_model: bid_ask_half_spread
market_impact_model: almgren_chriss
```

### `config/risk_config.yaml`

```yaml
drawdown_limit: 0.20
daily_var_limit: 0.02               # 2% daily VaR at 95%
max_single_name_exposure: 0.02
max_sector_exposure: 0.20
max_factor_exposure:
  market_beta: 0.20
  momentum: 1.5
  value: 1.5
kill_switch:
  enabled: true
  halt_on_drawdown: true
  halt_on_var_breach: true
  auto_resume: false                 # manual resume only
stress_tests:
  - 2008_financial_crisis
  - 2020_covid_crash
  - 2022_rate_shock
```

-----

## S3. Data Layer

### `historical_data_loader.py`

Loads OHLCV price data, fundamental data, and reference data for the configured
universe and lookback period. Returns a point-in-time snapshot indexed by
(date, ticker). Must include delisted tickers to avoid survivor bias.

Key method:

```python
def load(
    self,
    tickers: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    fields: List[str],               # ["open","high","low","close","volume",...]
    adjusted: bool = True,           # use corporate-action-adjusted prices
) -> pd.DataFrame:                   # MultiIndex (date, ticker)
    ...
```

### `corporate_action_adjuster.py`

Adjusts historical prices for dividends, splits, and spin-offs.
Raw unadjusted data is always preserved. Adjusted series are stored separately.
Adjustment factors are applied backward from the most recent date.

```python
AdjustmentType = Literal["split", "dividend", "spinoff"]

@dataclass
class CorporateAction:
    ticker: str
    ex_date: pd.Timestamp
    action_type: AdjustmentType
    adjustment_factor: float        # multiplicative for price; divisive for shares
```

### `data_validator.py`

Validates every data batch before it enters the feature pipeline.

```python
class DataValidator:
    def validate(self, df: pd.DataFrame, as_of: pd.Timestamp) -> ValidationResult:
        # Checks:
        # - No timestamps after as_of (look-ahead guard)
        # - No NaN in close prices
        # - No negative prices
        # - Volume > 0 on trading days
        # - No duplicate (date, ticker) rows
        # Returns ValidationResult with pass/fail flags and offending rows
```

### `data_alignment_engine.py`

**This is the look-ahead bias enforcer.** All feature generators call this.

```python
def get_aligned_data(
    self,
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame:
    """
    Return only rows where timestamp < as_of and within lookback window.
    Raises LookAheadError if any row has timestamp >= as_of.
    """
```

### `live_data_stream_adapter.py`

Wraps a real-time data feed (WebSocket or REST polling). Produces the same
DataFrame schema as `historical_data_loader` so the rest of the pipeline
is feed-agnostic. Emits an event on each new bar.

-----

## S4. Feature Factory

### `base_feature_generator.py`

Abstract base class. All features (technical, alternative, factor) inherit from this.

```python
class BaseFeatureGenerator(ABC):
    feature_name: str
    lookback_days: int
    recompute_frequency: str        # "daily" | "hourly" | "on_data"

    @abstractmethod
    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Cross-sectional feature values. data must be pre-aligned to as_of."""

    @abstractmethod
    def validate(self, output: pd.Series) -> bool:
        """Sanity check on output (nan rate, range, etc.)"""
```

### `technical_indicator_engine.py`

Computes price/volume based technical features:

- Returns: 1d, 5d, 20d, 60d, 252d
- Volatility: rolling 20d, 60d realised vol
- Volume: relative volume vs 20d average
- Price momentum: 12-1 month (exclude last month)
- Short-term reversal: 1-week return
- RSI, MACD signal, Bollinger Band position

All lookbacks configurable. All computed in-sample up to `as_of` only.

### `feature_normalizer.py`

Cross-sectional normalisation applied after feature computation.

```python
class FeatureNormalizer:
    def normalize(
        self,
        raw_scores: pd.Series,         # indexed by ticker
        method: str,                   # "zscore" | "rank" | "percentile"
        winsorize_std: float = 3.0,    # clip outliers before normalizing
    ) -> pd.Series:
        ...
```

Winsorisation happens before z-scoring. Rank normalisation produces uniform
distribution. Both are valid; z-score is default.

-----

## S5. Alternative Data

### Analyst Estimates (`analyst_estimates/`)

`estimate_revision_features.py` produces:

- SUE (Standardised Unexpected Earnings): (actual - consensus) / std_dev
- Revision momentum: % of estimates revised up over last 4 weeks
- Estimate dispersion: std_dev of analyst estimates / |consensus|

**Critical:** Use only estimates available before the `as_of` date.
Earnings announcement dates must be lagged by at least 1 trading day.

### Options Data (`options_data/`)

`implied_volatility_surface_builder.py` constructs a smooth IV surface
from option chain data using SVI or SABR parameterisation.

`options_feature_generation.py` produces:

- IV rank (IVR): current IV percentile vs trailing 252-day range
- IV skew: 25-delta put IV minus 25-delta call IV
- Put/call ratio: open interest and volume based
- Options-implied move: expected 1-week range from ATM straddle

### News Sentiment (`news_sentiment/`)

`sentiment_model.py` runs a fine-tuned FinBERT or similar model on headlines
and article summaries. Produces per-article sentiment scores in [-1, +1].

`sentiment_feature_generation.py` aggregates to per-ticker daily signal:

- Volume-weighted average sentiment over trailing 1-day and 5-day windows
- Sentiment momentum: change in sentiment vs prior 5-day average
- Sentiment divergence: difference from sector average

### ETF Flows (`etf_flows/`)

`flow_feature_generation.py` produces:

- Net flow as % of AUM over trailing 5 and 20 days
- Flow surprise: vs trailing 60-day average
- Sector rotation signal: relative flow between sector ETFs

### Macro Data (`macro_data/`)

`macro_regime_classifier.py` classifies the current macro regime:

- Growth regime: GDP growth trajectory (expanding / contracting)
- Inflation regime: CPI trend (rising / falling)
- Combined: 2x2 = four regimes (risk-on growth, stagflation, goldilocks, deflation)

These regime labels gate strategy weights in `dynamic_strategy_allocator.py`.

-----

## S6. Representation Learning

Extracts latent representations from high-dimensional feature spaces.
These embeddings become additional features fed into `alpha_discovery`.

### `autoencoder_model.py`

Standard bottleneck autoencoder trained on the full feature matrix.
Latent dimension: configurable (default 16). Trained offline; embeddings
stored in `embedding_feature_store.py` per date.

### `temporal_model_lstm.py` / `temporal_model_transformer.py`

Sequence models over rolling 60-day feature windows. Output: per-ticker
latent vector representing recent behaviour trajectory.
Transformer model preferred for longer contexts; LSTM for lower latency.

### `cross_asset_embedding_model.py`

Learns joint representations across the 200-400 stock universe.
Captures cross-sectional relationships (sector, style, co-movement).
Output: pairwise similarity matrix used by stat-arb pair selection.

### `embedding_feature_store.py`

Persistent store of computed embeddings, keyed by (model_id, date, ticker).
Embeddings are treated as features — all point-in-time rules apply.

-----

## S7. Alpha Discovery

Automated pipeline for discovering candidate signals from the feature space.

### `feature_combinator.py`

Generates candidate signal expressions by combining existing features:

- Linear combinations with random coefficients
- Ratio features (momentum / volatility, etc.)
- Interaction terms (feature_A x regime_label)

### `symbolic_regression_engine.py`

Uses a symbolic regression library (e.g. PySR or gplearn) to search
for mathematical expressions over the feature set that predict
forward returns. Fitness function: out-of-sample IC.

### `genetic_algorithm_search.py`

GA-based search over signal parameter space (lookback windows,
combination weights, normalisation methods). Population evolves
toward higher IC; penalises complexity to avoid overfitting.

### `ml_feature_selector.py`

Applies SHAP values, permutation importance, and mutual information
to rank features by predictive power. Removes redundant features
(|correlation| > 0.85 with a higher-ranked feature).

### `signal_ranking_engine.py`

Scores all candidate signals on:

- Out-of-sample IC (primary)
- IC t-statistic (statistical significance)
- IC stability: IC standard deviation over rolling windows
- Turnover implied by the signal (penalise very high turnover)
- Correlation with existing live signals (penalise crowding)

Signals below IC threshold or below t-stat threshold are rejected.

-----

## S8. Research Algorithms

### Factor Models

All factors follow `BaseFeatureGenerator` interface.
Factors are combined into a composite alpha via `signal_ranking_engine`.

|Factor        |Signal definition                                    |
|--------------|-----------------------------------------------------|
|Momentum      |12-1 month return (skip last month to avoid reversal)|
|Value         |Book-to-market, earnings yield, or composite         |
|Quality       |ROE, earnings stability, low accruals composite      |
|Low Volatility|Trailing 252-day realised vol, inverted              |
|Size          |Log market cap, inverted (small-cap tilt)            |

### Mean Reversion

`zscore_reversion_strategy.py`: cross-sectional z-score of 5-day return,
fade extremes. Long bottom decile, short top decile. Hold 1-5 days.

`short_term_reversal_strategy.py`: 1-week reversal with volume filter.
Only trade reversals accompanied by above-average volume (reduces noise).

### Statistical Arbitrage

`cointegration_engine.py`: Engle-Granger or Johansen test on pairs
within the same sector. Identifies pairs with stable spread (p < 0.05).
Refreshed monthly.

`kalman_spread_model.py`: Dynamic hedge ratio estimated via Kalman filter
on the spread between cointegrated pairs. State: [spread, hedge_ratio].
Entry: |z-score| > 2.0. Exit: |z-score| < 0.5. Stop: |z-score| > 3.5.

### Machine Learning

All ML models use a rolling expanding-window train/test split.
No future data leaks. Features are pre-normalised by `feature_normalizer`.

`gradient_boosted_tree_model.py` (LightGBM): predicts 5-day forward return quintile.
Primary production ML model. Features: full feature set from feature_factory.

`random_forest_model.py`: ensemble classifier for directional prediction.
Used as validation signal alongside GBT.

`neural_network_predictor.py`: MLP or temporal model. Uses embeddings from
representation_learning as input. Higher capacity; more regularisation required.

### Regime Models

`hidden_markov_regime_model.py`: Gaussian HMM on market returns and volatility.
Identifies 2-4 hidden regimes. State labels used to gate strategy weights.

`volatility_regime_detector.py`: VIX-based or realised-vol-based classification.
Low vol regime: factor strategies favoured. High vol regime: defensive tilt.

`market_state_classifier.py`: Combines HMM output, macro regime, and
volatility regime into a single market state enum. This state is read by
`dynamic_strategy_allocator.py` to adjust strategy weights.

-----

## S9. Alpha Monitoring

### `alpha_performance_tracker.py`

Tracks per-signal realised performance metrics on a rolling basis:

- Realised IC: correlation of signal with forward return
- Annualised IC x sqrt(252): information ratio per signal
- Cumulative PnL attribution from each signal

### `information_coefficient_monitor.py`

Tracks IC on a daily basis. Raises a `SignalDegradationAlert` when:

- Rolling 20-day IC falls below 0.0 (signal has gone flat or reversed)
- IC t-statistic falls below 1.5 over trailing 60 days

### `signal_decay_detector.py`

Computes IC half-life: the time for IC to decay to half its initial value.
If IC half-life < 20 trading days -> raises a review flag.
If IC half-life < 10 trading days -> raises a retirement recommendation.

### `strategy_retirement_manager.py`

Manages the lifecycle of live signals:

- Review: IC degraded -> reduce allocation, notify
- Suspend: IC persistently negative -> reduce to zero allocation
- Retire: manual sign-off required before removing from codebase

-----

## S10. Portfolio Construction

### Factor Risk Model

`factor_exposure_estimator.py`: For each stock, estimates exposures (betas) to:
market, momentum, value, quality, low-vol, size, and sector factors.
Estimated via cross-sectional regression of stock returns on factor returns.

`factor_covariance_estimator.py`: Estimates the factor covariance matrix
using a shrinkage estimator (Ledoit-Wolf). Stock-specific (idiosyncratic)
variance estimated from residuals.

Full covariance: Sigma = B F B^T + D
where B = factor exposure matrix (NxK),
F = factor covariance matrix (KxK),
D = diagonal idiosyncratic variance matrix (NxN)

`risk_decomposition.py`: Decomposes portfolio variance into factor and
idiosyncratic components. Used for attribution and exposure monitoring.

### Portfolio Optimizer

`portfolio_optimizer.py` minimises:

```
min_w  lambda * w^T Sigma w - alpha^T w
subject to:
  sum w_i = 0            (dollar neutral, if configured)
  |w_i| <= 0.02         (max position size)
  |sector_exposure| <= 0.20
  ||w||_1 <= 2.0        (leverage)
  + any additional constraints from constraint_engine
```

`alpha` is the combined alpha score from `signal_ranking_engine`.
`lambda` is the risk aversion parameter (tunable).
Solver: CVXPY with OSQP backend.

`constraint_engine.py`: the single place where all portfolio constraints
are defined. Accepts a `ConstraintSet` config object and returns CVXPY
constraint objects for the optimizer.

### Capital Allocation

`strategy_correlation_matrix.py`: Computes rolling pairwise correlation
of strategy daily returns. High-correlation strategies compete for capital.

`dynamic_strategy_allocator.py`: Allocates capital across active strategies.
Allocation algorithm:

1. Compute trailing Sharpe for each strategy (minimum 60-day window)
1. Penalise strategies with high pairwise correlation
1. Apply regime overlay (reduce allocation to strategies that historically
   underperform in current market_state)
1. Normalise allocations to sum to 1.0
1. Apply minimum allocation floor and maximum allocation cap

### Capacity Model

`market_impact_model.py`: Implements Almgren-Chriss model or calibrated
empirical model. Estimates expected market impact in bps as a function of
order size / ADV.

```
impact_bps = sigma * (order_size / ADV)^0.6
```

where sigma is daily volatility and the exponent is empirically calibrated.

`capacity_simulator.py`: For a given set of signals and target turnover,
estimates the maximum AUM the strategy can trade without significant
self-impact. Used to gate maximum capital allocation per strategy.

-----

## S11. Risk Engine

### `drawdown_monitor.py`

Tracks peak-to-trough NAV drawdown in real time. Emits `DrawdownAlert`
at 10% (warning), 15% (alert), 20% (trigger kill switch).

### `exposure_monitor.py`

Checks on every portfolio update:

- Net market exposure (beta x portfolio value)
- Gross exposure (sum of |weights|)
- Per-sector gross exposure
- Per-factor exposure vs configured limits

Emits `ExposureBreachAlert` on violation.

### `leverage_controller.py`

Enforces leverage <= 2.0 by scaling down all positions proportionally
if gross exposure would exceed the limit. Applied as a post-optimisation
adjustment before order generation.

### `portfolio_kill_switch.py`

**Simplest module in the system. Must remain simple.**

```python
class KillSwitch:
    max_drawdown: float = 0.20
    peak_nav: float

    def check(self, current_nav: float) -> bool:
        return (self.peak_nav - current_nav) / self.peak_nav >= self.max_drawdown

    def update_peak(self, nav: float) -> None:
        self.peak_nav = max(self.peak_nav, nav)
```

Called before every order batch. If True, `order_generator` produces
no new orders. Existing positions are not automatically liquidated
(manual decision required).

### `stress_test_engine.py`

Runs portfolio against historical stress scenarios:

- 2008 financial crisis (factor return shocks)
- 2020 COVID crash (liquidity shock + factor rotation)
- 2022 rate shock (duration and growth factor drawdown)

Reports estimated portfolio drawdown and factor P&L attribution under each.
Run offline (not in live trading loop).

-----

## S12. Execution

### Execution Algorithms

`vwap_execution.py`: Slices a parent order into child orders targeting
a volume-weighted average price. Participation rate: 5% of volume (configurable).
Adjusts slice sizes based on intraday volume profile (U-shaped).

`twap_execution.py`: Equally-sized slices over the execution horizon.
Used for scheduled rebalances where price impact is secondary.

`liquidity_seeking_execution.py`: Opportunistic execution that waits for
liquidity spikes (volume above 2x average) to fill larger slices.
Used for orders > 0.5% ADV.

### Order Management

`order_generator.py`: Converts target weights from the portfolio optimizer
into Order objects. Computes required trades given current positions.
Applies a minimum trade size filter (no trades below min_order_usd from config).

```python
@dataclass
class Order:
    ticker: str
    side: Literal["buy", "sell"]
    quantity: int
    order_type: Literal["market", "limit", "vwap", "twap"]
    algo_params: dict
    strategy_id: str
    timestamp: pd.Timestamp
```

`order_router.py`: Routes orders to the appropriate execution algorithm
based on order size, urgency, and time-of-day. Large orders -> liquidity seeking.
Scheduled rebalances -> TWAP. Default -> VWAP.

### Microstructure Models

`bid_ask_spread_model.py`: Estimates effective bid-ask spread from
quote data or as a function of volatility and market cap.
Used in transaction cost estimation for backtests and live monitoring.

`adverse_selection_model.py`: Estimates the adverse selection component
of the spread using VPIN (Volume-Synchronized Probability of Informed Trading)
or a simpler order imbalance metric. Flags high adverse-selection environments.

`fill_probability_model.py`: Given a limit order price relative to mid,
estimates the probability of fill within the order's time limit.

`queue_position_estimator.py`: Estimates position in the order queue
for limit orders based on order arrival time and visible queue depth.

-----

## S13. Broker Interface

### `broker_abstraction_layer.py`

All broker-specific code is hidden behind this interface. The rest of
the system only calls `broker_abstraction_layer`. Never calls adapters directly.

```python
class BrokerInterface(ABC):
    def submit_order(self, order: Order) -> OrderAcknowledgement: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_positions(self) -> pd.Series: ...
    def get_account_value(self) -> float: ...
    def get_fills(self, since: pd.Timestamp) -> List[Fill]: ...
    def get_market_data(self, tickers: List[str]) -> pd.DataFrame: ...
```

### `simulation_broker.py`

Full-featured simulation that must be implemented before live adapters.
Features:

- Realistic fill simulation: fills at mid +/- half_spread + market impact
- Partial fills for large orders
- Latency simulation (configurable delay in ms)
- Order book depth simulation
- Configurable slippage model

### `interactive_brokers_adapter.py`

Wraps the IB TWS API. Implements `BrokerInterface`.
Uses `ib_insync` or native IB API. Handles reconnection, pacing limits.

### `alpaca_adapter.py`

Wraps the Alpaca REST/WebSocket API. Implements `BrokerInterface`.
Used for paper trading and smaller live accounts.

-----

## S14. Research Cluster

### `distributed_backtest_runner.py`

Distributes backtests across multiple cores or machines.
Uses joblib (local) or Ray/Dask (cluster). Each backtest job is isolated:
no shared state between workers.

### `experiment_scheduler.py`

Manages a queue of research experiments (parameter sweeps, signal searches).
Assigns jobs to available workers, tracks status, stores results in a
database keyed by (experiment_id, parameters_hash).

### `parallel_signal_evaluator.py`

Evaluates a list of candidate signals in parallel.
Each signal is evaluated on the same out-of-sample test period.
Returns a ranked DataFrame of signals sorted by IC t-statistic.

-----

## S15. Governance

### `strategy_review_pipeline.py`

Automated pre-deployment checklist:

- Out-of-sample Sharpe >= 1.0
- Out-of-sample max drawdown <= 25%
- Minimum evaluation period: 252 trading days
- Transaction costs included in performance metrics
- No look-ahead bias flags raised by data_validator
- Capacity check: strategy not self-impacting at proposed AUM
- Factor exposure check: no extreme factor bets

### `approval_workflow.py`

Enforces the deployment gate. State machine:

```
SUBMITTED -> UNDER_REVIEW -> APPROVED -> DEPLOYED
                          -> REJECTED
DEPLOYED  -> SUSPENDED (on kill switch or IC degradation)
          -> RETIRED (manual)
```

Requires `risk_team_approved: bool = True` before transition to DEPLOYED.
`deployment_controller.py` checks this flag before enabling live orders.

### `deployment_controller.py`

The only module that can flip a strategy from paper to live.
Reads the approval workflow state. Checks kill switch status.
Maintains a registry of deployed strategies and their allocation weights.

-----

## S16. Monitoring

### `pnl_dashboard.py`

Real-time P&L tracking:

- Total NAV and daily return
- P&L attribution by strategy, by factor, by sector
- Realised vs unrealised P&L
- Drawdown from peak NAV

### `risk_dashboard.py`

- Current leverage and gross exposure
- Per-sector exposure vs limits
- Per-factor exposure vs limits
- VaR (historical simulation, 95% confidence)
- Top 10 risk contributors

### `execution_quality_monitor.py`

- Implementation shortfall vs VWAP benchmark
- Fill rate (% of target trades executed)
- Slippage vs model estimates
- Adverse selection flags

### `alerting_system.py`

Centralised alert dispatcher. Alert levels: INFO, WARNING, CRITICAL.
CRITICAL alerts: kill switch trigger, exposure limit breach, data feed outage.
Delivery: email, Slack webhook, PagerDuty (configurable per level).

-----

## S17. Pipeline Pseudocode

### Research cycle (daily, after market close)

```python
# 1. Update data
raw_data = historical_data_loader.load(universe, as_of=today)
adj_data = corporate_action_adjuster.adjust(raw_data)
validated = data_validator.validate(adj_data, as_of=today)

# 2. Compute features (all via data_alignment_engine)
features = {}
for generator in feature_factory.generators:
    aligned = data_alignment_engine.get_aligned_data(validated, as_of=today,
                                                     lookback_days=generator.lookback_days)
    features[generator.feature_name] = generator.compute(aligned, as_of=today)
feature_matrix = pd.DataFrame(features)             # shape: (n_tickers, n_features)
normalised_features = feature_normalizer.normalize(feature_matrix)

# 3. Alt data features (same alignment rules)
alt_features = alternative_data_pipeline.compute_all(as_of=today)

# 4. Combined alpha scores
alpha_scores = signal_ranking_engine.combine(
    research_algorithms.compute_all(normalised_features, as_of=today),
    alt_features,
)

# 5. Risk model
factor_exposures = factor_exposure_estimator.estimate(as_of=today)
factor_cov = factor_covariance_estimator.estimate(as_of=today)

# 6. Portfolio construction
regime = market_state_classifier.classify(as_of=today)
allocations = dynamic_strategy_allocator.allocate(regime=regime)
target_weights = portfolio_optimizer.optimize(
    alpha_scores=alpha_scores,
    factor_covariance=factor_cov,
    factor_exposures=factor_exposures,
    constraints=constraint_engine.build_constraints(),
    current_positions=broker.get_positions(),
)

# 7. Risk checks
if kill_switch.check(broker.get_account_value()):
    alerting_system.send("CRITICAL: Kill switch triggered")
    return
exposure_monitor.check(target_weights)      # raises on breach
leverage_controller.enforce(target_weights)

# 8. Execution
orders = order_generator.generate(target_weights, broker.get_positions())
for order in orders:
    routed_order = order_router.route(order)
    broker_abstraction_layer.submit_order(routed_order)

# 9. Monitoring
pnl_dashboard.update(broker.get_account_value(), orders)
alpha_monitoring.update(alpha_scores, forward_returns_if_available)
```

-----

## S18. Decision Ledger

|Decision                   |Resolution                                                                   |
|---------------------------|-----------------------------------------------------------------------------|
|Look-ahead bias enforcement|data_alignment_engine enforces all feature computations strictly before as_of|
|Survivor bias              |Delisted tickers included; point-in-time index membership required           |
|Feature normalisation      |Cross-sectional z-score + +/-3 sigma winsorisation before optimizer           |
|Signal validation threshold|Out-of-sample IC > 0.03 over 252 days minimum                                |
|Portfolio construction     |Mean-variance with factor risk model covariance (CVXPY + OSQP)               |
|Risk model                 |Multi-factor: momentum, value, quality, low-vol, size, sector, market        |
|Covariance estimation      |Ledoit-Wolf shrinkage on factor cov; diagonal idiosyncratic                  |
|Capital allocation         |Sharpe-weighted + correlation-penalised + regime-gated                       |
|Execution default          |VWAP at 5% participation; liquidity-seeking for large orders                 |
|Market impact model        |Almgren-Chriss or empirical calibration                                      |
|Kill switch                |Hard stop at 20% drawdown; simplest module; no downstream dependencies       |
|Broker abstraction         |All broker calls via BrokerInterface; never call adapters directly           |
|Deployment gate            |252-day OOS + paper trading >= 6 months + manual approval                    |
|Regime overlay             |HMM + macro + vol regime -> market_state -> strategy allocator weights       |

-----

## S19. Open Items — Deferred

|Item                               |Notes                                                    |
|-----------------------------------|---------------------------------------------------------|
|Options strategy (delta-neutral)   |Framework exists; no options strategies in v1            |
|Crypto / FX universe               |Config-ready; adapters not implemented                   |
|Intraday execution (sub-minute)    |Execution window is market hours; intraday model deferred|
|RL-based execution                 |Placeholder; VWAP/TWAP cover v1                          |
|Fully automated IC-gated deployment|Approval workflow requires manual sign-off for now       |
