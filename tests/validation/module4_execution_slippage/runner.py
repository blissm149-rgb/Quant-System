"""Module 4 runner: Execution & Slippage validation suite.

Collects all Module 4 tests and produces a JSON/CSV report.
Can be run standalone: python -m tests.validation.module4_execution_slippage.runner
"""

import sys
import subprocess
from pathlib import Path

from tests.validation.shared.report_writer import ValidationReportWriter


MODULE_DIR = Path(__file__).parent
MODULE_NAME = "module4_execution_slippage"

TEST_FILES = [
    "test_market_impact_scaling.py",
    "test_vwap_execution_quality.py",
    "test_order_generation_fidelity.py",
    "test_spread_and_capacity.py",
]


def run_module(verbose: bool = True, report: bool = True) -> int:
    """Run all Module 4 tests via pytest."""
    test_paths = [str(MODULE_DIR / f) for f in TEST_FILES]
    cmd = [sys.executable, "-m", "pytest", *test_paths, "-m", "validation"]
    if verbose:
        cmd.append("-v")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if verbose:
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

    if report:
        writer = ValidationReportWriter(MODULE_NAME)
        writer.add_metadata("exit_code", result.returncode)
        writer.add_metadata("n_test_files", len(TEST_FILES))

        for line in result.stdout.splitlines():
            if "PASSED" in line or "FAILED" in line:
                passed = "PASSED" in line
                test_name = line.split("::")[1].split(" ")[0] if "::" in line else line
                writer.add_test_result(
                    test_name=test_name,
                    passed=passed,
                    metrics={},
                    details=line.strip(),
                )

        writer.write_json()
        writer.write_csv()

    return result.returncode


if __name__ == "__main__":
    sys.exit(run_module())
