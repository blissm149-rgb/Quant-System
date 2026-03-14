"""Strategy review pipeline — automated pre-deployment checklist.

Validates that a strategy meets all requirements before it
can proceed to the approval workflow and deployment.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReviewCheck:
    """Result of a single review check."""

    check_name: str
    passed: bool
    value: float = 0.0
    threshold: float = 0.0
    message: str = ""


@dataclass
class ReviewResult:
    """Full review result for a strategy."""

    strategy_id: str
    passed: bool
    checks: List[ReviewCheck] = field(default_factory=list)
    reviewed_at: Optional[pd.Timestamp] = None

    @property
    def failed_checks(self) -> List[ReviewCheck]:
        return [c for c in self.checks if not c.passed]


class StrategyReviewPipeline:
    """Automated pre-deployment review checklist.

    Requirements:
    - Out-of-sample Sharpe >= 1.0
    - Out-of-sample max drawdown <= 25%
    - Minimum evaluation period: 252 trading days
    - Transaction costs included in performance metrics
    - No look-ahead bias flags
    - Capacity check: strategy not self-impacting at proposed AUM
    - Factor exposure check: no extreme factor bets
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._min_sharpe = cfg.get("min_sharpe", 1.0)
        self._max_drawdown = cfg.get("max_drawdown", 0.25)
        self._min_eval_days = cfg.get("min_eval_days", 252)
        self._max_factor_exposure = cfg.get("max_factor_exposure", 2.0)
        self._max_capacity_impact_pct = cfg.get("max_capacity_impact_pct", 0.20)

    def review(
        self,
        strategy_id: str,
        sharpe_ratio: float,
        max_drawdown: float,
        num_eval_days: int,
        transaction_costs_included: bool = False,
        look_ahead_bias_flags: int = 0,
        capacity_impact_pct: float = 0.0,
        max_factor_exposure: float = 0.0,
    ) -> ReviewResult:
        """Run the full review pipeline.

        Parameters
        ----------
        strategy_id : str
            Strategy identifier.
        sharpe_ratio : float
            Out-of-sample Sharpe ratio (after costs).
        max_drawdown : float
            Out-of-sample maximum drawdown (as positive fraction).
        num_eval_days : int
            Number of trading days in evaluation period.
        transaction_costs_included : bool
            Whether performance metrics include transaction costs.
        look_ahead_bias_flags : int
            Number of look-ahead bias flags from data_validator.
        capacity_impact_pct : float
            Market impact as fraction of alpha at proposed AUM.
        max_factor_exposure : float
            Maximum single-factor exposure (absolute).

        Returns
        -------
        ReviewResult
        """
        checks = []

        checks.append(ReviewCheck(
            check_name="sharpe_ratio",
            passed=sharpe_ratio >= self._min_sharpe,
            value=sharpe_ratio,
            threshold=self._min_sharpe,
            message=f"Sharpe {sharpe_ratio:.2f} vs min {self._min_sharpe}",
        ))

        checks.append(ReviewCheck(
            check_name="max_drawdown",
            passed=max_drawdown <= self._max_drawdown,
            value=max_drawdown,
            threshold=self._max_drawdown,
            message=f"Max DD {max_drawdown:.2%} vs limit {self._max_drawdown:.2%}",
        ))

        checks.append(ReviewCheck(
            check_name="min_eval_period",
            passed=num_eval_days >= self._min_eval_days,
            value=float(num_eval_days),
            threshold=float(self._min_eval_days),
            message=f"{num_eval_days} days vs min {self._min_eval_days}",
        ))

        checks.append(ReviewCheck(
            check_name="transaction_costs",
            passed=transaction_costs_included,
            value=1.0 if transaction_costs_included else 0.0,
            threshold=1.0,
            message="Costs included" if transaction_costs_included else "Costs NOT included",
        ))

        checks.append(ReviewCheck(
            check_name="look_ahead_bias",
            passed=look_ahead_bias_flags == 0,
            value=float(look_ahead_bias_flags),
            threshold=0.0,
            message=f"{look_ahead_bias_flags} look-ahead flags",
        ))

        checks.append(ReviewCheck(
            check_name="capacity_check",
            passed=capacity_impact_pct <= self._max_capacity_impact_pct,
            value=capacity_impact_pct,
            threshold=self._max_capacity_impact_pct,
            message=f"Impact {capacity_impact_pct:.1%} vs limit {self._max_capacity_impact_pct:.1%}",
        ))

        checks.append(ReviewCheck(
            check_name="factor_exposure",
            passed=max_factor_exposure <= self._max_factor_exposure,
            value=max_factor_exposure,
            threshold=self._max_factor_exposure,
            message=f"Max factor exp {max_factor_exposure:.2f} vs limit {self._max_factor_exposure}",
        ))

        all_passed = all(c.passed for c in checks)

        result = ReviewResult(
            strategy_id=strategy_id,
            passed=all_passed,
            checks=checks,
            reviewed_at=pd.Timestamp.now(),
        )

        logger.info(
            "Strategy %s review: %s (%d/%d checks passed)",
            strategy_id,
            "PASSED" if all_passed else "FAILED",
            sum(1 for c in checks if c.passed),
            len(checks),
        )
        return result
