import pytest

pytestmark = [pytest.mark.tier2]
"""Re-export invariant tests for pytest discovery."""

from tests.invariants import (
    TestDataFormatInvariants,
    TestSignalFlowInvariants,
    TestArchitecturalInvariants,
)
