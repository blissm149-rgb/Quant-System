"""Validation report writer for JSON and CSV output."""

import json
import csv
import os
from datetime import datetime
from typing import Any, Optional


class ValidationReportWriter:
    """Writes JSON and CSV reports for each validation module run."""

    def __init__(self, module_name: str, output_dir: str = "validation_reports"):
        self._module_name = module_name
        self._output_dir = output_dir
        self._results: list[dict] = []
        self._metadata: dict[str, Any] = {
            "module": module_name,
            "timestamp": datetime.now().isoformat(),
        }

    def add_test_result(
        self,
        test_name: str,
        passed: bool,
        metrics: dict,
        details: str = "",
    ) -> None:
        """Record a single test result."""
        self._results.append(
            {
                "test_name": test_name,
                "passed": passed,
                "metrics": metrics,
                "details": details,
            }
        )

    def add_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def summary(self) -> dict:
        """Return summary statistics."""
        total = len(self._results)
        passed = sum(1 for r in self._results if r["passed"])
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
        }

    def write_json(self) -> str:
        """Write full report as JSON. Returns file path."""
        os.makedirs(self._output_dir, exist_ok=True)
        path = os.path.join(self._output_dir, f"{self._module_name}.json")
        report = {
            "metadata": self._metadata,
            "summary": self.summary(),
            "results": self._results,
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return path

    def write_csv(self) -> str:
        """Write tabular summary as CSV. Returns file path."""
        os.makedirs(self._output_dir, exist_ok=True)
        path = os.path.join(self._output_dir, f"{self._module_name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["test_name", "passed", "details"]
            )
            writer.writeheader()
            for r in self._results:
                writer.writerow(
                    {
                        "test_name": r["test_name"],
                        "passed": r["passed"],
                        "details": r["details"],
                    }
                )
        return path
