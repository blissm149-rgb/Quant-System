"""CI/CD pipeline configuration and validation layer.

Manages pipeline stage generation, result validation, and deployment
readiness checks for quantitative trading strategies. This is a
configuration/validation layer only — it does not make actual CI/CD
API calls.

Enforced invariants:
- All tests (unit, integration, look-ahead bias) must pass before deployment.
- Paper trading validation is required before live deployment.
- Risk team approval is mandatory for live environments.
- A rollback plan must exist for every deployment.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Valid environment names in promotion order.
ENVIRONMENTS = ("dev", "staging", "paper", "live")


class CICDPipelineManager:
    """Generates and validates CI/CD pipeline configurations for strategy deployment.

    This class produces declarative stage definitions that an external
    CI/CD system can consume.  It also validates pipeline results and
    checks deployment readiness against the fund's governance rules.

    Usage:
        manager = CICDPipelineManager({"default_timeout_minutes": 30})
        config = manager.generate_pipeline_config("staging")
        ready, issues = manager.check_deployment_readiness(
            strategy_id="mean_rev_v2",
            approval_state={"risk_team": True, "quant_lead": True},
            test_results={"unit": "passed", "integration": "passed",
                          "lookahead_bias": "passed"},
        )
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize the pipeline manager.

        Args:
            config: Optional configuration overrides.  Recognised keys:
                - default_timeout_minutes (int): per-stage timeout, default 30.
                - parallel_test_execution (bool): run test stages in parallel,
                  default True.
                - require_paper_trading_days (int): minimum paper trading days
                  before live promotion, default 5.
                - rollback_keep_versions (int): number of previous versions
                  retained for rollback, default 3.
        """
        self._config: Dict[str, Any] = config or {}
        self._default_timeout: int = self._config.get("default_timeout_minutes", 30)
        self._parallel_tests: bool = self._config.get("parallel_test_execution", True)
        self._paper_days: int = self._config.get("require_paper_trading_days", 5)
        self._rollback_versions: int = self._config.get("rollback_keep_versions", 3)
        logger.info(
            "CICDPipelineManager initialised (timeout=%d min, parallel_tests=%s)",
            self._default_timeout,
            self._parallel_tests,
        )

    # ------------------------------------------------------------------
    # Pipeline generation
    # ------------------------------------------------------------------

    def generate_pipeline_config(self, environment: str) -> dict:
        """Generate a full CI/CD pipeline configuration for *environment*.

        The returned dict describes ordered stages that an external runner
        should execute.

        Args:
            environment: Target environment — one of ``dev``, ``staging``,
                ``paper``, or ``live``.

        Returns:
            A dict with keys ``environment``, ``stages``, and ``metadata``.

        Raises:
            ValueError: If *environment* is not recognised.
        """
        if environment not in ENVIRONMENTS:
            raise ValueError(
                f"Unknown environment '{environment}'. "
                f"Must be one of {ENVIRONMENTS}."
            )

        stages: List[dict] = []
        stages.extend(self.get_test_stages())
        stages.extend(self.get_validation_stages())
        stages.extend(self.get_deployment_stages(environment))

        pipeline: Dict[str, Any] = {
            "environment": environment,
            "stages": stages,
            "metadata": {
                "default_timeout_minutes": self._default_timeout,
                "parallel_test_execution": self._parallel_tests,
                "rollback_plan": self.get_rollback_plan(environment),
            },
        }
        logger.info(
            "Generated pipeline config for '%s' with %d stages",
            environment,
            len(stages),
        )
        return pipeline

    # ------------------------------------------------------------------
    # Stage definitions
    # ------------------------------------------------------------------

    def get_test_stages(self) -> List[dict]:
        """Return test-stage configurations.

        Stages:
        1. **unit** — fast, isolated unit tests.
        2. **integration** — cross-module integration tests.
        3. **lookahead_bias** — mandatory look-ahead bias detection tests.

        Returns:
            A list of stage dicts, each with ``name``, ``type``,
            ``required``, ``timeout_minutes``, and ``parallel`` keys.
        """
        stages = [
            {
                "name": "unit_tests",
                "type": "test",
                "required": True,
                "timeout_minutes": self._default_timeout,
                "parallel": self._parallel_tests,
                "description": "Run unit tests for strategy logic and utilities.",
            },
            {
                "name": "integration_tests",
                "type": "test",
                "required": True,
                "timeout_minutes": self._default_timeout,
                "parallel": self._parallel_tests,
                "description": "Run integration tests across data, alpha, and execution layers.",
            },
            {
                "name": "lookahead_bias_tests",
                "type": "test",
                "required": True,
                "timeout_minutes": self._default_timeout,
                "parallel": False,
                "description": (
                    "Detect look-ahead bias in features, signals, and portfolio "
                    "construction. This stage is mandatory and cannot be skipped."
                ),
            },
        ]
        logger.debug("Built %d test stages", len(stages))
        return stages

    def get_validation_stages(self) -> List[dict]:
        """Return validation-stage configurations.

        Stages:
        1. **data_validation** — schema and quality checks on input data.
        2. **risk_checks** — position-limit, drawdown, and exposure checks.

        Returns:
            A list of stage dicts.
        """
        stages = [
            {
                "name": "data_validation",
                "type": "validation",
                "required": True,
                "timeout_minutes": self._default_timeout,
                "checks": [
                    "schema_conformance",
                    "null_ratio_threshold",
                    "timestamp_continuity",
                    "value_range_bounds",
                ],
                "description": "Validate data pipeline outputs before strategy consumption.",
            },
            {
                "name": "risk_checks",
                "type": "validation",
                "required": True,
                "timeout_minutes": self._default_timeout,
                "checks": [
                    "position_limits",
                    "sector_exposure",
                    "max_drawdown_threshold",
                    "leverage_limits",
                    "concentration_limits",
                ],
                "description": "Run pre-deployment risk checks against current portfolio state.",
            },
        ]
        logger.debug("Built %d validation stages", len(stages))
        return stages

    def get_deployment_stages(self, environment: str) -> List[dict]:
        """Return deployment-stage configurations for *environment*.

        For ``paper`` and ``live`` environments additional gates are
        inserted (paper-trading validation and risk-team approval,
        respectively).

        Args:
            environment: Target environment name.

        Returns:
            A list of stage dicts.
        """
        stages: List[dict] = []

        # Paper-trading gate for live deployments.
        if environment == "live":
            stages.append(
                {
                    "name": "paper_trading_validation",
                    "type": "gate",
                    "required": True,
                    "timeout_minutes": None,  # async — awaits external signal
                    "criteria": {
                        "min_paper_trading_days": self._paper_days,
                        "max_deviation_from_backtest_pct": 10.0,
                        "positive_sharpe_required": True,
                    },
                    "description": (
                        "Verify that paper-trading performance matches backtest "
                        "expectations before live promotion."
                    ),
                }
            )
            stages.append(
                {
                    "name": "risk_team_approval",
                    "type": "gate",
                    "required": True,
                    "timeout_minutes": None,
                    "approvers": ["risk_team"],
                    "description": "Risk team must approve before live deployment.",
                }
            )

        stages.append(
            {
                "name": f"deploy_{environment}",
                "type": "deployment",
                "required": True,
                "timeout_minutes": self._default_timeout,
                "environment": environment,
                "rollback_enabled": True,
                "description": f"Deploy strategy artefacts to {environment} environment.",
            }
        )

        # Post-deployment smoke tests.
        stages.append(
            {
                "name": f"smoke_test_{environment}",
                "type": "test",
                "required": True,
                "timeout_minutes": max(self._default_timeout // 3, 5),
                "description": f"Run smoke tests against {environment} after deployment.",
            }
        )

        logger.debug(
            "Built %d deployment stages for '%s'", len(stages), environment
        )
        return stages

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_pipeline_result(
        self, stage_results: dict
    ) -> Tuple[bool, List[str]]:
        """Validate the outcome of a full pipeline run.

        Args:
            stage_results: Mapping of ``stage_name`` to result string
                (e.g. ``"passed"``, ``"failed"``, ``"skipped"``).

        Returns:
            A ``(success, issues)`` tuple where *success* is ``True``
            when every required stage passed and *issues* lists human-
            readable descriptions of any problems found.
        """
        issues: List[str] = []

        # 1. All test stages must pass.
        for test_stage in ("unit_tests", "integration_tests", "lookahead_bias_tests"):
            result = stage_results.get(test_stage)
            if result is None:
                issues.append(f"Required test stage '{test_stage}' was not executed.")
            elif result != "passed":
                issues.append(
                    f"Test stage '{test_stage}' did not pass (result='{result}')."
                )

        # 2. Look-ahead bias test is mandatory (redundant with above but
        #    called out explicitly for auditability).
        if stage_results.get("lookahead_bias_tests") != "passed":
            if not any("lookahead_bias_tests" in i for i in issues):
                issues.append("Look-ahead bias tests are mandatory and must pass.")

        # 3. Validation stages must pass.
        for val_stage in ("data_validation", "risk_checks"):
            result = stage_results.get(val_stage)
            if result is None:
                issues.append(
                    f"Required validation stage '{val_stage}' was not executed."
                )
            elif result != "passed":
                issues.append(
                    f"Validation stage '{val_stage}' did not pass (result='{result}')."
                )

        # 4. If deployment was attempted, smoke test must also pass.
        deploy_stages = [k for k in stage_results if k.startswith("deploy_")]
        for deploy_stage in deploy_stages:
            env = deploy_stage.replace("deploy_", "")
            smoke_key = f"smoke_test_{env}"
            smoke_result = stage_results.get(smoke_key)
            if smoke_result is None:
                issues.append(
                    f"Smoke test '{smoke_key}' was not executed after deployment."
                )
            elif smoke_result != "passed":
                issues.append(
                    f"Smoke test '{smoke_key}' did not pass (result='{smoke_result}')."
                )

        success = len(issues) == 0
        if success:
            logger.info("Pipeline validation passed — all stages OK.")
        else:
            logger.warning(
                "Pipeline validation failed with %d issue(s): %s",
                len(issues),
                "; ".join(issues),
            )
        return success, issues

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def get_rollback_plan(self, environment: str) -> dict:
        """Return a rollback configuration for *environment*.

        Args:
            environment: Target environment.

        Returns:
            A dict describing the rollback strategy.
        """
        plan: Dict[str, Any] = {
            "environment": environment,
            "strategy": "blue_green" if environment == "live" else "in_place",
            "keep_versions": self._rollback_versions,
            "automatic_rollback_on_failure": environment in ("live", "paper"),
            "health_check_interval_seconds": 30 if environment == "live" else 60,
            "health_check_retries": 5,
            "notifications": {
                "on_rollback": ["risk_team", "quant_lead"],
                "on_failure": ["risk_team", "quant_lead", "ops_team"],
            },
        }
        logger.debug("Generated rollback plan for '%s'", environment)
        return plan

    # ------------------------------------------------------------------
    # Deployment readiness
    # ------------------------------------------------------------------

    def check_deployment_readiness(
        self,
        strategy_id: str,
        approval_state: dict,
        test_results: dict,
    ) -> Tuple[bool, List[str]]:
        """Check whether a strategy is ready for deployment.

        This is a high-level gate that aggregates test outcomes,
        approval status, and governance rules.

        Args:
            strategy_id: Identifier for the strategy being deployed.
            approval_state: Mapping of approver role to ``bool``
                (e.g. ``{"risk_team": True, "quant_lead": False}``).
            test_results: Mapping of test category to result string
                (e.g. ``{"unit": "passed", "integration": "passed",
                "lookahead_bias": "passed", "paper_trading": "passed"}``).

        Returns:
            A ``(ready, blockers)`` tuple.  *ready* is ``True`` only
            when all governance requirements are satisfied.
        """
        blockers: List[str] = []

        # 1. All core tests must pass.
        required_tests = ("unit", "integration", "lookahead_bias")
        for test_name in required_tests:
            result = test_results.get(test_name)
            if result is None:
                blockers.append(
                    f"Required test '{test_name}' has not been executed "
                    f"for strategy '{strategy_id}'."
                )
            elif result != "passed":
                blockers.append(
                    f"Test '{test_name}' did not pass for strategy "
                    f"'{strategy_id}' (result='{result}')."
                )

        # 2. Look-ahead bias test is mandatory (explicit check).
        if test_results.get("lookahead_bias") != "passed":
            if not any("lookahead_bias" in b for b in blockers):
                blockers.append(
                    f"Look-ahead bias test is mandatory for strategy "
                    f"'{strategy_id}' and must pass."
                )

        # 3. Paper trading validation required before live.
        paper_result = test_results.get("paper_trading")
        if paper_result is None:
            blockers.append(
                f"Paper trading validation has not been completed "
                f"for strategy '{strategy_id}'."
            )
        elif paper_result != "passed":
            blockers.append(
                f"Paper trading validation did not pass for strategy "
                f"'{strategy_id}' (result='{paper_result}')."
            )

        # 4. Risk team approval required for live deployment.
        if not approval_state.get("risk_team"):
            blockers.append(
                f"Risk team approval is required for live deployment "
                f"of strategy '{strategy_id}'."
            )

        # 5. Rollback plan must exist.
        rollback = self.get_rollback_plan("live")
        if not rollback:
            blockers.append(
                f"No rollback plan defined for strategy '{strategy_id}'."
            )

        ready = len(blockers) == 0
        if ready:
            logger.info(
                "Strategy '%s' is ready for deployment.", strategy_id
            )
        else:
            logger.warning(
                "Strategy '%s' is NOT ready for deployment — %d blocker(s): %s",
                strategy_id,
                len(blockers),
                "; ".join(blockers),
            )
        return ready, blockers
