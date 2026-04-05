# QuantFund V8 — Integration & Model Lifecycle Implementation Plan

## Overview

This plan closes the gaps identified in the architectural review, organized into
7 self-contained steps. Each step produces a working, testable increment. Steps
are ordered by dependency — later steps build on earlier ones, but each can be
merged independently.

---

## Step 1: Wire TradeRecorder into TradingEngine (Compliance Audit Trail)

**Why first:** Every subsequent change generates orders/fills. Having the audit
trail active from the start captures everything.

**Files to modify:**
- `quant_fund/main/trading_engine.py` — add `_trade_recorder` field, record
  orders in `_convergence_tick()` and fills in the fill-event loop
- `main_run.py` — instantiate `TradeRecorder` and inject via
  `engine.inject_components(trade_recorder=...)`

**Concrete changes:**

1. In `TradingEngine.__init__`, add:
   ```python
   self._trade_recorder = None
   ```

2. In `TradingEngine._convergence_tick()`, after `orders = self._order_generator.generate_orders(...)`:
   ```python
   if self._trade_recorder is not None:
       for order in orders:
           self._trade_recorder.record_order({
               "strategy_id": order.strategy_id,
               "ticker": order.ticker,
               "side": order.side.value,
               "quantity": order.quantity,
               "order_type": order.order_type.value,
           })
   ```

3. In the fill-event loop inside `_convergence_tick()`, after recording
   execution quality:
   ```python
   if self._trade_recorder is not None:
       self._trade_recorder.record_fill({
           "order_id": ack.order_id,
           "ticker": order.ticker,
           "side": order.side.value,
           "filled_qty": order.quantity,
           "fill_price": fill_price,
       })
   ```

4. In `main_run.py` `build_engine()`:
   ```python
   from quant_fund.infrastructure.trade_recorder import TradeRecorder
   trade_recorder = TradeRecorder(db_path=db_path.replace(".db", "_trades.db"))
   engine.inject_components(..., trade_recorder=trade_recorder)
   ```

**Validation:** Run `main_run.py` briefly, then query the trades SQLite DB
to confirm orders and fills are recorded.

---

## Step 2: Wire DataValidator into ResearchRunner

**Why:** Data quality is a prerequisite for all downstream model training and
signal generation. Without validation, garbage data silently corrupts signals.

**Files to modify:**
- `main_run.py` — instantiate `DataValidator` and inject into `ResearchRunner`

**Concrete changes:**

1. In `main_run.py` `build_engine()`, after creating `research_runner`:
   ```python
   from quant_fund.data_layer.data_validator import DataValidator
   research_runner.inject_components(
       ...,
       data_validator=DataValidator(),
   )
   ```

That's it — `ResearchRunner.run_cycle()` already has the code path at line 154
that calls `self._data_validator.validate(data, as_of=as_of)` if the validator
is injected. The `ValidationResult` has a `.flags` attribute that gets attached
to `ResearchResult.validation_flags`.

**Validation:** Run the research cycle and confirm `validation_flags` appear
in the `ResearchResult`.

---

## Step 3: Wire Alpha Monitoring (IC Monitor, Decay Detector, Retirement Manager)

**Why:** The system is blind to signal degradation. These modules are fully
implemented and `ResearchRunner` already accepts them — they just need wiring.

**Files to modify:**
- `quant_fund/main/research_runner.py` — add `signal_decay_detector` and
  `strategy_retirement_manager` to `inject_components()`, call them in
  `run_cycle()` after alpha scores are produced
- `main_run.py` — instantiate and inject all four alpha monitoring components

**Concrete changes:**

1. In `ResearchRunner.__init__`, add:
   ```python
   self._signal_decay_detector = None
   self._strategy_retirement_manager = None
   ```

2. In `ResearchRunner.inject_components()`, add parameters:
   ```python
   signal_decay_detector=None,
   strategy_retirement_manager=None,
   ```
   And the corresponding setter blocks.

3. In `ResearchRunner.run_cycle()`, after the existing alpha_monitor update
   block (line 216-217), add:
   ```python
   # Update IC monitor
   if self._ic_monitor is not None and result.alpha_scores is not None:
       if data is not None and "close" in data.columns:
           fwd_ret = self._get_selection_target(data, feature_matrix)
           if fwd_ret is not None:
               alerts = self._ic_monitor.update(
                   signal_name="composite_alpha",
                   signal_values=result.alpha_scores,
                   forward_returns=fwd_ret,
                   date=as_of,
               )
               if alerts:
                   for a in alerts:
                       logger.warning("IC alert: %s", a.message)

   # Update signal decay detector
   if self._signal_decay_detector is not None and result.alpha_scores is not None:
       if hasattr(self._ic_monitor, '_ic_history') and 'composite_alpha' in self._ic_monitor._ic_history:
           latest_ic_list = self._ic_monitor._ic_history['composite_alpha']
           if latest_ic_list:
               self._signal_decay_detector.update(
                   "composite_alpha", latest_ic_list[-1], as_of
               )
               decay = self._signal_decay_detector.detect("composite_alpha")
               if decay is not None:
                   result.signal_metrics = result.signal_metrics or {}
                   result.signal_metrics["decay_half_life"] = decay.ic_half_life_days
                   result.signal_metrics["decay_action"] = decay.action.value
   ```

4. In `main_run.py` `build_engine()`:
   ```python
   from quant_fund.alpha_monitoring.alpha_performance_tracker import AlphaPerformanceTracker
   from quant_fund.alpha_monitoring.information_coefficient_monitor import InformationCoefficientMonitor
   from quant_fund.alpha_monitoring.signal_decay_detector import SignalDecayDetector
   from quant_fund.alpha_monitoring.strategy_retirement_manager import StrategyRetirementManager

   research_runner.inject_components(
       ...,
       alpha_monitor=AlphaPerformanceTracker(),
       ic_monitor=InformationCoefficientMonitor(),
       signal_decay_detector=SignalDecayDetector(),
       strategy_retirement_manager=StrategyRetirementManager(),
   )
   ```

**Validation:** Run a research cycle with historical data; confirm
`ResearchResult.signal_metrics` contains decay half-life; confirm IC monitor
logs warnings when IC drops.

---

## Step 4: Wire ModelStore into ResearchRunner (Model Persistence)

**Why:** Without this, all trained models are lost on restart. This is the
critical link between the well-designed `ModelStore` and the runtime pipeline.

**Files to modify:**
- `quant_fund/main/research_runner.py` — add `_model_store` field, use it in
  `_retrain_models()` to save after training and in a new `_load_champion_models()`
  method to restore on startup
- `main_run.py` — instantiate `ModelStore` and inject

**Concrete changes:**

1. In `ResearchRunner.__init__`, add:
   ```python
   self._model_store = None
   ```

2. In `ResearchRunner.inject_components()`, add `model_store=None` parameter
   and setter.

3. In `ResearchRunner._retrain_models()`, after `model.train_model()` succeeds,
   add model persistence:
   ```python
   if self._model_store is not None and hasattr(model, '_model') and model._model is not None:
       feature_names = list(feature_matrix.columns)
       version_id = self._model_store.save_sklearn_model(
           model=model._model,
           model_name=model_name,
           train_start_date=str(data.index.min()) if hasattr(data.index, 'min') else "",
           train_end_date=str(data.index.max()) if hasattr(data.index, 'max') else "",
           feature_names=feature_names,
           metrics=metrics,
       )
       # Champion/challenger: promote if it beats the current champion
       beats, details = self._model_store.check_challenger_beats_champion(
           model_name, version_id, metric_name="oos_ic",
       )
       if beats:
           self._model_store.promote_to_champion(model_name, version_id)
           logger.info("Model '%s' v%s promoted to champion (OOS IC improvement: %.4f)",
                       model_name, version_id, details.get("improvement", 0))
   ```

4. Add `load_champion_models()` method to `ResearchRunner`:
   ```python
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
               logger.info("No champion model found for '%s' — will train from scratch", model_name)
           except Exception as e:
               logger.warning("Failed to load model '%s': %s", model_name, e)
       return loaded
   ```

5. In `main_run.py` `build_engine()`:
   ```python
   from quant_fund.infrastructure.model_store import ModelStore
   model_store = ModelStore(config={"model_dir": "./models"})
   research_runner.inject_components(..., model_store=model_store)
   ```

6. In `main_run.py` `run_paper_trading()`, after `engine.inject_components(historical_data=historical)`:
   ```python
   research_runner.load_champion_models()
   ```

**Validation:** Train a model, restart the system, confirm the champion is
loaded automatically. Verify `./models/<model_name>/versions/` contains
versioned artifacts.

---

## Step 5: Wire ML Models into the Live Pipeline

**Why:** The system currently runs on simple technical indicators only. The
ML models (GBT, RF, etc.) are fully implemented but never instantiated in
the live path. This step makes them active.

**Files to modify:**
- `main_run.py` — instantiate `GradientBoostedTreeModel` (and optionally
  `RandomForestModel`), inject as both feature generators and ML models

**Concrete changes:**

1. In `main_run.py` `build_engine()`, after creating `tech_engine`:
   ```python
   from quant_fund.research_algorithms.machine_learning.gradient_boosted_tree_model import (
       GradientBoostedTreeModel,
   )

   gbt_model = GradientBoostedTreeModel(config={
       "gbt_n_estimators": 200,
       "gbt_max_depth": 5,
       "gbt_learning_rate": 0.05,
       "gbt_min_train_days": 126,  # 6 months minimum
       "random_seed": 42,
   })
   ```

2. Inject as ML model for periodic retraining:
   ```python
   research_runner.inject_ml_models([gbt_model])
   ```

3. Also add as a feature generator so its predictions flow into
   `SignalRankingEngine`:
   ```python
   research_runner.inject_components(
       feature_generators=tech_engine.generators + [gbt_model],
       ...
   )
   ```

4. Configure retrain frequency in `ResearchRunner` config:
   ```python
   research_runner = ResearchRunner(config={
       "universe": tickers,
       "lookback_days": 252,
       "retrain_frequency": 24,  # retrain every 24 research cycles (~daily if hourly research)
   })
   ```

**Validation:** Run the system with historical data loaded. After
`retrain_frequency` cycles, confirm GBT model trains, produces non-NaN
alpha scores, and is persisted via `ModelStore` (from Step 4).

---

## Step 6: Wire ExposureMonitor and Factor Risk Model

**Why:** The engine already has code paths for factor exposure estimation and
the exposure monitor, but `main_run.py` never injects the components. Without
this, the system has no sector or factor exposure limits beyond the basic
`ConstraintEngine`.

**Files to modify:**
- `main_run.py` — instantiate `ExposureMonitor`, `FactorExposureEstimator`,
  `FactorCovarianceEstimator`, compute factor returns from historical data,
  and inject into the engine

**Concrete changes:**

1. In `main_run.py` `build_engine()`:
   ```python
   from quant_fund.risk_engine.exposure_monitor import ExposureMonitor
   from quant_fund.portfolio.factor_risk_model.factor_exposure_estimator import FactorExposureEstimator
   from quant_fund.portfolio.factor_risk_model.factor_covariance_estimator import FactorCovarianceEstimator

   exposure_monitor = ExposureMonitor(config={
       "max_leverage": 1.0,
       "max_sector_exposure": 0.40,
       "max_single_name_exposure": 0.10,
   })
   factor_exposure_estimator = FactorExposureEstimator()
   factor_covariance_estimator = FactorCovarianceEstimator()
   ```

2. Inject into engine:
   ```python
   engine.inject_components(
       ...,
       exposure_monitor=exposure_monitor,
       factor_exposure_estimator=factor_exposure_estimator,
       factor_covariance_estimator=factor_covariance_estimator,
   )
   ```

3. In `main_run.py` `run_paper_trading()`, after historical data is loaded,
   compute and inject stock returns and factor returns:
   ```python
   if not historical.empty:
       # Compute stock returns for factor model
       if isinstance(historical.index, pd.MultiIndex):
           close_pivot = historical["close"].unstack(level="ticker")
       else:
           close_pivot = historical[["close"]]
       stock_returns = close_pivot.pct_change().dropna()

       # Simple factor proxies (market = equal-weight return)
       market_return = stock_returns.mean(axis=1)
       factor_returns = pd.DataFrame({"market": market_return})

       engine.inject_components(
           stock_returns=stock_returns,
           factor_returns=factor_returns,
       )
   ```

**Validation:** Run the system. Confirm that `_run_research_and_optimize()`
now calls `factor_exposure_estimator.estimate()` and
`factor_covariance_estimator.estimate()` (the code path already exists at
trading_engine.py:756-773). Confirm `exposure_monitor.check()` runs
in the fallback risk-check path.

---

## Step 7: Persist Retrain Counter and Add Performance-Based Retrain Triggers

**Why:** The retrain counter (`_cycles_since_retrain`) resets on restart,
causing unpredictable retrain timing. Additionally, purely cycle-based
retraining doesn't respond to signal degradation.

**Files to modify:**
- `quant_fund/infrastructure/state_persistence_manager.py` — add
  `save_research_state()` / `restore_research_state()` for retrain counter
- `quant_fund/main/research_runner.py` — add performance-based retrain trigger
  using `SignalDecayDetector` output
- `quant_fund/main/trading_engine.py` — call persistence for research state
  in `_take_snapshot()` and `_restore_state()`

**Concrete changes:**

1. In `StatePersistenceManager`, add:
   ```python
   NS_RESEARCH = "research"

   def save_research_state(self, research_runner) -> None:
       self._store.save(NS_RESEARCH, "cycles_since_retrain",
                        research_runner._cycles_since_retrain)

   def restore_research_state(self, research_runner) -> bool:
       val = self._store.load(NS_RESEARCH, "cycles_since_retrain")
       if val is None:
           return False
       research_runner._cycles_since_retrain = val
       return True
   ```

2. In `TradingEngine._take_snapshot()`, add:
   ```python
   if self._research_runner is not None:
       self._persistence.save_research_state(self._research_runner)
   ```

3. In `TradingEngine._restore_state()`, add:
   ```python
   if self._research_runner is not None:
       self._persistence.restore_research_state(self._research_runner)
   ```

4. In `ResearchRunner._retrain_models()`, add a performance-based early-retrain
   trigger at the top (before the cycle-count check in `run_cycle`):

   Modify the retrain check in `run_cycle()` (lines 209-213) from:
   ```python
   if self._ml_models and features:
       self._cycles_since_retrain += 1
       if self._cycles_since_retrain >= self._retrain_frequency:
           self._retrain_models(feature_matrix, data)
           self._cycles_since_retrain = 0
   ```
   to:
   ```python
   if self._ml_models and features:
       self._cycles_since_retrain += 1
       force_retrain = False
       # Performance-based trigger: retrain if decay detector flags review
       if (self._signal_decay_detector is not None
               and result.signal_metrics
               and result.signal_metrics.get("decay_action") == "review"):
           force_retrain = True
           logger.info("Performance-based retrain triggered by signal decay")
       if self._cycles_since_retrain >= self._retrain_frequency or force_retrain:
           self._retrain_models(feature_matrix, data)
           self._cycles_since_retrain = 0
   ```

**Validation:** Stop and restart the system. Confirm `_cycles_since_retrain`
restores correctly. Simulate a low-IC scenario and confirm the performance-based
retrain trigger fires before the cycle-count threshold.

---

## Execution Order & Dependencies

```
Step 1: TradeRecorder           (independent — can start immediately)
Step 2: DataValidator           (independent — can start immediately)
Step 3: Alpha Monitoring        (independent — can start immediately)
Step 4: ModelStore              (depends on Step 3 for decay_action in metrics)
Step 5: ML Models               (depends on Step 4 for persistence)
Step 6: Factor Risk Model       (independent — can start immediately)
Step 7: Retrain Persistence     (depends on Steps 3, 4, 5)
```

Steps 1, 2, 3, and 6 can be done in parallel. Steps 4 and 5 are sequential.
Step 7 ties everything together.

## Estimated Scope Per Step

| Step | Files Modified | Lines Changed (approx) |
|------|---------------|----------------------|
| 1    | 2             | ~30                  |
| 2    | 1             | ~5                   |
| 3    | 2             | ~50                  |
| 4    | 2             | ~60                  |
| 5    | 1             | ~25                  |
| 6    | 1             | ~30                  |
| 7    | 3             | ~40                  |

Total: ~240 lines across 6 unique files. No new files needed — every component
already exists and is tested. This is purely integration wiring.
