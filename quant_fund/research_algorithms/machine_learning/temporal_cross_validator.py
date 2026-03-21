"""Temporal cross-validation with purging and embargo.

Implements purged k-fold and combinatorial purged cross-validation (CPCV)
for unbiased evaluation of time-series models. Ensures no information
leakage between train and test sets through temporal purging and embargo
periods.
"""

import logging
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def purged_kfold(
    dates: pd.DatetimeIndex,
    n_folds: int = 5,
    embargo_days: int = 5,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Purged k-fold cross-validation for time-series data.

    Splits dates into n_folds contiguous blocks. For each fold used as test,
    the training set excludes an embargo period after each test block to
    prevent information leakage from overlapping labels.

    Args:
        dates: Sorted DatetimeIndex.
        n_folds: Number of folds.
        embargo_days: Number of dates to exclude after each test block.

    Returns:
        List of (train_dates, test_dates) tuples.
    """
    dates = dates.sort_values()
    n = len(dates)
    fold_size = n // n_folds
    splits = []

    for i in range(n_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else n

        test_dates = dates[test_start:test_end]

        # Build training set: all dates not in test and not in embargo
        embargo_end = min(test_end + embargo_days, n)
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_start:embargo_end] = False
        train_dates = dates[train_mask]

        if len(train_dates) > 0 and len(test_dates) > 0:
            splits.append((train_dates, test_dates))

    return splits


def combinatorial_purged_cv(
    dates: pd.DatetimeIndex,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_days: int = 5,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Combinatorial Purged Cross-Validation (CPCV).

    Divides dates into n_groups contiguous blocks, then generates all
    C(n_groups, n_test_groups) splits where the selected groups form
    the test set and remaining groups form the training set (with embargo).

    This produces more paths than standard k-fold, enabling better
    estimation of strategy performance distribution.

    Args:
        dates: Sorted DatetimeIndex.
        n_groups: Total number of groups to divide dates into.
        n_test_groups: Number of groups to use as test in each split.
        embargo_days: Embargo period after each test group.

    Returns:
        List of (train_dates, test_dates) tuples.
    """
    dates = dates.sort_values()
    n = len(dates)
    group_size = n // n_groups

    # Define group boundaries
    group_bounds = []
    for i in range(n_groups):
        start = i * group_size
        end = (i + 1) * group_size if i < n_groups - 1 else n
        group_bounds.append((start, end))

    splits = []
    for test_groups in combinations(range(n_groups), n_test_groups):
        # Test dates: union of selected groups
        test_indices = []
        for g in test_groups:
            start, end = group_bounds[g]
            test_indices.extend(range(start, end))
        test_dates = dates[test_indices]

        # Training dates: everything else minus embargo zones
        train_mask = np.ones(n, dtype=bool)
        for g in test_groups:
            start, end = group_bounds[g]
            embargo_end = min(end + embargo_days, n)
            train_mask[start:embargo_end] = False
        train_dates = dates[train_mask]

        if len(train_dates) > 0 and len(test_dates) > 0:
            splits.append((train_dates, test_dates))

    return splits
