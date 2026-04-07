# QuantFund V8

Institutional multi-strategy quantitative trading platform targeting 200–400 US
equities at medium frequency. Research-driven automated alpha discovery and
execution with institutional-grade risk controls and governance.

**Performance targets:** Sharpe ≥ 1.5 · Max drawdown ≤ 20% · Moderate turnover

**Trading parameters:** Hourly/daily rebalance · Max position 2% · Sector limit
20% · Max leverage 2.0× · Dollar-neutral (optional)

**Environments:** `local_research` · `paper_trading` · `live_trading`

---

## Architecture Overview

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
                          risk_cascade_coordinator (unified gate)
                          capacity_model (market impact filter)
                                     │
                                     ▼
                          order_generator → order_router
                          → broker_abstraction_layer
                          → simulation_broker / IB / Alpaca
                                     │
                                     ▼
                          execution_quality_monitor (IS, fill rate, slippage)
                          pnl_dashboard (NAV, returns, drawdown)

LIVE CYCLE (TradingEngine — always-on event loop):

  live_data_stream_adapter → TradingEngine._main_loop():
    market_hours_enforcer gate → _run_research_and_optimize() →
    _convergence_tick() → risk_cascade → capacity filter →
    order_generator → order_router → broker → execution_quality_monitor

PAPER TRADING (main_run.py):

  YFinanceFeedProvider → LiveDataStreamAdapter → SimulationBroker.set_market_data()
  → TradingEngine.run() → console dashboard (NAV, positions, execution quality)
```

### Lifecycle States

```
INITIALIZING → DATA_READY → TRADING_ENABLED → RISK_HALT → SHUTDOWN
```

- **INITIALIZING**: Load state snapshots, connect data feeds
- **DATA_READY**: Data feed validated, run readiness checks
- **TRADING_ENABLED**: Process market data → signals → risk → orders
- **RISK_HALT**: Suspend trading, wait for manual reset
- **SHUTDOWN**: Persist state, disconnect, exit

### Key Architectural Decisions

| Decision | Resolution |
|---|---|
| Data alignment | Strict point-in-time via `data_alignment_engine`. No exceptions. |
| Feature normalisation | Cross-sectional z-score + winsorisation at ±3σ before optimizer |
| Signal validation | Out-of-sample IC > 0.03 over 252 days minimum |
| Portfolio construction | Mean-variance / Black-Litterman with factor risk model covariance (CVXPY + OSQP) |
| Risk model | Multi-factor: momentum, value, quality, low-vol, size, sector, market |
| Capital allocation | Sharpe-weighted across active strategies, correlation-penalised, regime-gated |
| Execution | VWAP default; liquidity-seeking for large orders; TWAP for scheduled rebalances |
| Kill switch | Hard stop on 20% drawdown. Simple, no dependencies, always checked first. |
| Deployment gate | Paper trading ≥ 6 months + approval_workflow sign-off required |
| Market impact | Almgren-Chriss model in `market_impact_model.py` |
| State persistence | All runtime state persisted via `StateStore`; snapshots every 5 min |
| Reconciliation | Broker is source of truth. Positions reconciled every 5 min, auto-corrected |
| Broker resilience | `BrokerReconnectionManager` with exponential backoff + failover |
| Risk cascade | `RiskCascadeCoordinator`: kill_switch → drawdown → leverage → exposure in order |

---

## Code Structure

```
quant_fund/                         # 155 Python modules across 17 subsystems
├── config/                         # 5 YAML config files + validator
├── data_layer/                     # Historical loader, live adapter, validator, connectors
│   └── connectors/                 #   YFinance, CSV, abstract base
├── alternative_data/               # 5 alt data subsystems
│   ├── analyst_estimates/          #   SUE, revision momentum, estimate dispersion
│   ├── etf_flows/                  #   Net flow, flow surprise, sector rotation
│   ├── macro_data/                 #   FRED data, regime classifier (growth × inflation)
│   ├── news_sentiment/             #   FinBERT sentiment, volume-weighted aggregation
│   └── options_data/               #   IV surface, skew, put/call ratio
├── feature_factory/                # Base generator, technical indicators, normalizer, alignment
├── representation_learning/        # Autoencoder, LSTM, transformer, cross-asset embeddings
├── alpha_discovery/                # Symbolic regression, genetic search, feature selection, signal ranking
├── alpha_monitoring/               # IC monitor, signal decay detector, strategy retirement
├── research_algorithms/            # Strategy implementations
│   ├── factor_models/              #   Momentum, value, quality, low-vol, size
│   ├── mean_reversion/             #   Z-score reversion, short-term reversal
│   ├── statistical_arbitrage/      #   Pairs trading, cointegration, Kalman spread
│   ├── machine_learning/           #   GBT, random forest, neural net, ensemble, validation
│   └── regime_models/              #   HMM, volatility regime, market state classifier
├── portfolio/                      # Portfolio construction and risk modeling
│   ├── portfolio_construction/     #   Optimizer (CVXPY), constraint engine
│   ├── factor_risk_model/          #   Factor exposures, covariance (Ledoit-Wolf), decomposition
│   ├── capital_allocation/         #   Dynamic strategy allocator, correlation matrix
│   └── capacity_model/             #   Market impact (Almgren-Chriss), capacity simulator
├── risk_engine/                    # Kill switch, drawdown monitor, exposure, leverage, stress tests
├── execution/                      # Order management and execution algorithms
│   ├── execution_algorithms/       #   VWAP, TWAP, liquidity-seeking
│   ├── order_management/           #   Generator, router, market hours enforcer, safety validator
│   ├── microstructure_models/      #   Spread, adverse selection, fill probability, queue position
│   └── reconciliation_engine.py    #   Continuous position/NAV/order reconciliation
├── broker_interface/               # Abstraction layer + IB, Alpaca, simulation adapters
├── research_cluster/               # Distributed backtesting, experiment scheduling
├── monitoring/                     # PnL dashboard, risk dashboard, execution quality, alerting, health
├── governance/                     # Strategy review, approval workflow, deployment controller
├── infrastructure/                 # Event bus, state machine, persistence, model store, trade recorder
│   ├── event_bus.py                #   In-process pub/sub with idempotency + replay
│   ├── system_state_machine.py     #   Lifecycle states + readiness checks
│   ├── state_persistence_manager.py #  Wires all components to StateStore
│   ├── model_store.py              #   Model persistence + champion/challenger
│   ├── data_builder.py             #   Training dataset construction
│   ├── promotion_gate.py           #   Model promotion gate
│   └── ...                         #   Docker, CI/CD, disaster recovery, ingestion scheduler
└── main/                           # Entry points
    ├── trading_engine.py           #   Always-on event-driven main loop
    ├── research_runner.py          #   Research cycle orchestrator
    ├── paper_trading_runner.py     #   Legacy paper trading (superseded by TradingEngine)
    └── live_trading_runner.py      #   Legacy live trading (superseded by TradingEngine)

main_run.py                         # Click-to-run paper trading with live prices + dashboard
train_pipeline.py                   # Offline ML training pipeline (DataBuilder → Train → Promote)
```

### Config Files

| File | Purpose |
|---|---|
| `config/system_config.yaml` | Environment, log level, timezone, paths |
| `config/trading_config.yaml` | Universe size, position limits, leverage, rebalance frequency |
| `config/data_config.yaml` | Data sources, schedules, storage format |
| `config/execution_config.yaml` | VWAP/TWAP params, order size limits, slippage model |
| `config/risk_config.yaml` | Drawdown limit, VaR, factor exposure limits, kill switch, stress tests |

---

## Setup and Installation

**Requirements:** Python 3.11+

```bash
pip install -r requirements.txt
```

### Dependencies

Core: `pandas`, `numpy`, `pyarrow`, `pyyaml`, `scipy`, `scikit-learn`, `cvxpy`, `joblib`

Testing: `pytest`, `pytest-cov`, `pytest-timeout`, `pytest-xdist`

Optional: `yfinance` (for live paper trading data)

### Running Paper Trading

```bash
python main_run.py                              # defaults: 10 tickers, $1M
python main_run.py --tickers AAPL MSFT GOOG     # custom tickers
python main_run.py --cash 500000                 # custom starting capital
python main_run.py --research-interval 1800      # research every 30 min
python main_run.py --skip-market-hours           # trade anytime (testing)
```

### Running Offline Training

```bash
python train_pipeline.py
```

---

## Trading Operations

### Market Hours (TradingEngine)

During `TRADING_ENABLED` state:
- **Research cycles** run at configured intervals (default: hourly) — feature computation, alpha scoring, portfolio optimization
- **Convergence ticks** run at price poll intervals — execute orders toward target weights with risk gates
- **Reconciliation** runs every 10 minutes — verify positions against broker
- **Health checks** run every 60 seconds — data feed lag, system metrics
- **State snapshots** persist every 5 minutes — full system state to SQLite

### Risk Cascade (every convergence tick)

```
1. KillSwitch.check()           → halt all trading if drawdown ≥ 20%
2. DrawdownMonitor.check()      → warning at 5%, alert at 10%, halt at 15%
3. LeverageController.enforce() → scale down if gross exposure > limit
4. ExposureMonitor.check()      → verify sector/factor limits
5. MarketImpactModel.filter()   → cap order sizes by ADV participation
```

### Emergency Procedures

- **Kill switch**: Triggered automatically at 20% drawdown. No new orders generated. Existing positions remain (manual liquidation decision).
- **Risk halt**: Engine transitions to `RISK_HALT` state. Requires manual restart after investigation.
- **Graceful shutdown**: `Ctrl+C` → state snapshot → disconnect feeds → exit.

---

## ML Model Lifecycle

### Training Pipeline (Offline)

```
DataBuilder → Feature Matrix → Train Models → Validate OOS → PromotionGate → ModelStore
```

- **DataBuilder**: Constructs point-in-time training datasets with no look-ahead
- **Training**: Expanding-window walk-forward. GBT is the primary production model.
- **Validation**: OOS IC, Sharpe, drawdown, capacity checks
- **PromotionGate**: Champion/challenger comparison on OOS metrics
- **ModelStore**: Versioned model artifacts with metadata

### Signal Monitoring (Live)

- **IC Monitor**: Rolling 20-day IC tracked. Alert if IC < 0 or t-stat < 1.5 over 60 days.
- **Signal Decay Detector**: IC half-life estimation. Review flag if < 20 days, retirement recommendation if < 10 days.
- **Strategy Retirement Manager**: Review → Suspend → Retire lifecycle with manual sign-off.

### Model Health Monitor (Live)

Observe-only monitor that tracks prediction drift, staleness, and distribution
changes. Logs health events for the offline pipeline to prioritize retraining.
Never triggers retraining in the live loop.

---

## Risk Management

### Portfolio Constraints (enforced by `constraint_engine.py`)

| Constraint | Limit |
|---|---|
| Max single position | 2% of portfolio |
| Max sector exposure | 20% per GICS sector |
| Max leverage | 2.0× gross |
| Dollar neutral | Optional (configurable) |

### Factor Risk Model

Full covariance: `Σ = B F Bᵀ + D`
- **B**: Factor exposure matrix (N×K) — market, momentum, value, quality, low-vol, size, sector
- **F**: Factor covariance (K×K) — Ledoit-Wolf shrinkage estimator
- **D**: Diagonal idiosyncratic variance (N×N) — from regression residuals

### Stress Tests

Run offline against historical scenarios:
- 2008 Financial Crisis (factor return shocks)
- 2020 COVID Crash (liquidity shock + factor rotation)
- 2022 Rate Shock (duration and growth factor drawdown)

---

## Testing Framework

### Tier System

| Tier | Scope | Runtime | When |
|---|---|---|---|
| tier1 | Pre-commit smoke tests | < 30s | Every commit |
| tier2 | PR gate (module integration) | < 5 min | Every PR |
| tier3 | Post-merge integration | < 15 min | After merge |
| tier4 | Nightly full suite (validation) | < 60 min | Nightly CI |
| tier5 | Weekly stability (long-duration) | > 60 min | Weekly CI |

### Marker Taxonomy

```
@pytest.mark.unit          # Fast, isolated, no I/O
@pytest.mark.integration   # Multi-module interaction
@pytest.mark.e2e           # Full pipeline
@pytest.mark.stress        # Fault injection and extreme load
@pytest.mark.benchmark     # Performance measurement
@pytest.mark.stability     # Long-duration stability
@pytest.mark.regression    # Deterministic reproducibility
@pytest.mark.validation    # Validation, robustness, risk framework
```

### Running Tests

```bash
pytest tests/ -m tier1                          # pre-commit (~30s)
pytest tests/ -m "tier1 or tier2"               # PR gate (~5 min)
pytest tests/ -m "tier1 or tier2 or tier3"      # post-merge (~15 min)
pytest tests/ -m "not tier5"                    # nightly (~60 min)
pytest tests/                                    # everything
```

### Validation Framework (9 Modules)

Located in `tests/validation/`:

| Module | Coverage |
|---|---|
| Module 1: Backtest Integrity | Data leakage, fill realism, look-ahead, survivorship bias |
| Module 2: Strategy Robustness | Cross-validation stability, parameter sensitivity, signal decay |
| Module 3: Regime Performance | Regime detection accuracy, transitions, strategy-regime interaction |
| Module 4: Execution Slippage | Market impact, order generation fidelity, spread/capacity |
| Module 5: Risk & Stress | Drawdown monitoring, exposure limits, risk decomposition, stress tests |
| Module 6: Monte Carlo & Statistical | Bootstrap confidence, deflated Sharpe, multiple hypothesis |
| Module 7: Live Monitoring & Drift | Execution drift, PnL tracking, reconciliation, signal decay lifecycle |
| Module 8: ML Code Audit | Feature engineering, model architecture, training validation |
| Module 9: Cross-Cutting Invariants | Module invariants across the entire system |

### CI/CD Pipelines

| Workflow | Trigger | Tests Run |
|---|---|---|
| `pr-gate.yml` | Pull request | tier1 + tier2 with coverage ≥ 85% |
| `nightly.yml` | Scheduled (daily) | All except tier5 |

### Test Counts

~1,600 tests across 170 test files. Every test file has a `pytestmark` tier
annotation. Default per-test timeout: 120 seconds.

---

## Development Guide

### Adding a New Alpha Signal

1. Create a class inheriting from `BaseFeatureGenerator` in the appropriate subdirectory
2. Implement `compute(data, as_of)` — must only use data strictly before `as_of`
3. Implement `validate(output)` — sanity checks on NaN rate, range, etc.
4. Add to `TechnicalIndicatorEngine.generators` or inject via `ResearchRunner`
5. Write tests including `validate_no_lookahead()` test
6. Evaluate OOS IC over ≥ 252 days before considering for deployment

### Adding a New Data Source

1. Create a connector inheriting from `BaseConnector` in `data_layer/connectors/`
2. Ensure all data has `as_of` timestamps — no future data accessible
3. Wire into `data_config.yaml` schedule
4. Add validation rules to `DataValidator`

### Adding a New Broker Adapter

1. Implement `BrokerInterface` from `broker_abstraction_layer.py`
2. All calls go through the abstraction layer — never call adapters directly
3. Test against `SimulationBroker` first
4. Handle reconnection via `BrokerReconnectionManager`

### Module Contracts

**Feature generators:**
```python
class BaseFeatureGenerator(ABC):
    feature_name: str
    lookback_days: int
    recompute_frequency: str  # "daily" | "hourly" | "on_data"

    def compute(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Cross-sectional feature values. data contains only rows before as_of."""

    def validate(self, output: pd.Series) -> bool:
        """Sanity check on output (NaN rate, range, etc.)"""
```

**Broker interface:**
```python
class BrokerInterface(ABC):
    def submit_order(self, order: Order) -> OrderAcknowledgement: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_positions(self) -> pd.Series: ...
    def get_account_value(self) -> float: ...
    def get_fills(self, since: pd.Timestamp) -> List[Fill]: ...
    def get_market_data(self, tickers: List[str]) -> pd.DataFrame: ...
```

---

## Critical Rules

### Look-Ahead Bias (most critical rule)

- All feature computation must be strictly point-in-time. Features at time T may
  only use data available before T.
- `data_alignment_engine.py` is the enforcer. Every feature pipeline routes through it.
- Applies to normalisation, covariance estimation, and universe selection.
- Treat any backtest Sharpe > 3.0 with suspicion.

### Data Immutability

- Raw data is never modified. All adjustments (splits, dividends) produce new
  adjusted series stored separately.
- Every data fetch returns a point-in-time snapshot with an `as_of` timestamp.

### Signal Rules

- All signals must be cross-sectionally normalised before entering the optimizer.
- IC > 0.03 over ≥ 252 trading days required before deployment consideration.
- Signals are evaluated in out-of-sample periods only.

### Execution Rules

- Orders flow: `order_generator → order_router → broker_abstraction_layer → adapter`
- Never call broker adapters directly.
- Slippage and market impact modelled in simulation via `market_impact_model.py`.

### Governance Rules

- No strategy reaches live trading without passing `approval_workflow.py`.
- Requirements: OOS Sharpe ≥ 1.0, max drawdown ≤ 25%, ≥ 6 months paper trading, risk team sign-off.
- `deployment_controller.py` is the only module that can flip paper → live.

### Config Rules

- All parameters live in the five YAML config files. No magic numbers in code.
- Each module receives only its config sub-section.

---

## Common Pitfalls

| Pitfall | Prevention |
|---|---|
| Look-ahead bias in normalisation | Compute z-scores using only historical mean/std up to `as_of` |
| Survivor bias | Include delisted tickers; use point-in-time index membership |
| Overfitting in alpha discovery | Out-of-sample validation only; penalise complexity in signal ranking |
| Transaction cost neglect | Every backtest must include bid-ask spread + market impact estimates |
| Factor crowding | Monitor factor exposures in risk decomposition; penalise crowded factors |
| Overstated Sharpe from short lookback | Minimum 252-day evaluation window for any signal |
| Signal decay ignored | `signal_decay_detector.py` runs daily; IC half-life < 20 days → review |
| Strategy correlation ignored | `strategy_correlation_matrix.py` must gate capital allocation |

---

## Open Items (Deferred)

| Item | Notes |
|---|---|
| Options strategy (delta-neutral) | Framework exists; no options strategies in v1 |
| Crypto / FX universe | Config-ready; adapters not implemented |
| Intraday execution (sub-minute) | Execution window is market hours; intraday model deferred |
| RL-based execution | Placeholder; VWAP/TWAP cover v1 |
| Fully automated IC-gated deployment | Approval workflow requires manual sign-off for now |
