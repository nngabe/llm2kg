"""
CI Reporter.

Output formats suitable for CI/CD pipelines (GitHub Actions, etc.).
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from ..config import EvaluationResult, EvaluationLayer

logger = logging.getLogger(__name__)


class CIReporter:
    """Generates CI/CD-friendly output for evaluation results."""

    def __init__(self):
        self.is_github_actions = os.getenv("GITHUB_ACTIONS") == "true"

    def _github_output(self, name: str, value: str) -> None:
        """Write to GitHub Actions output."""
        output_file = os.getenv("GITHUB_OUTPUT")
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"{name}={value}\n")

    def _github_summary(self, content: str) -> None:
        """Write to GitHub Actions step summary."""
        summary_file = os.getenv("GITHUB_STEP_SUMMARY")
        if summary_file:
            with open(summary_file, "a") as f:
                f.write(content + "\n")

    def report(
        self,
        result: EvaluationResult,
        fail_on_threshold: bool = True,
    ) -> int:
        """Generate CI report and return exit code.

        Args:
            result: Evaluation result
            fail_on_threshold: Whether to return non-zero exit code on failure

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        # Print summary line for logs
        status = "PASSED" if result.overall_passed else "FAILED"
        print(f"\n[CI] Evaluation {status}: {result.overall_score:.2%}")

        # GitHub Actions specific output
        if self.is_github_actions:
            self._github_actions_output(result)

        # Print detailed results
        self._print_ci_output(result)

        # Determine exit code
        if fail_on_threshold and not result.overall_passed:
            return 1
        return 0

    def _github_actions_output(self, result: EvaluationResult) -> None:
        """Generate GitHub Actions specific outputs."""
        # Set outputs
        self._github_output("status", "passed" if result.overall_passed else "failed")
        self._github_output("score", f"{result.overall_score:.4f}")
        self._github_output("test_count", str(result.test_case_count))

        # Layer scores
        for layer in EvaluationLayer:
            if layer in result.layers:
                lr = result.layers[layer]
                self._github_output(f"{layer.value}_score", f"{lr.overall_score:.4f}")
                self._github_output(f"{layer.value}_passed", str(lr.passed).lower())

        # Generate step summary
        summary = self._generate_github_summary(result)
        self._github_summary(summary)

        # Annotations for failed metrics
        for metric in result.get_failed_metrics():
            print(f"::warning title={metric.metric_name}::"
                  f"Score {metric.score:.2%} below threshold {metric.threshold:.0%}")

    def _generate_github_summary(self, result: EvaluationResult) -> str:
        """Generate GitHub step summary markdown."""
        lines = [
            "## Evaluation Results",
            "",
            f"**Overall Status:** {'PASSED' if result.overall_passed else 'FAILED'}",
            f"**Overall Score:** {result.overall_score:.2%}",
            "",
            "### Layer Results",
            "",
            "| Layer | Score | Status |",
            "|-------|-------|--------|",
        ]

        for layer in EvaluationLayer:
            if layer in result.layers:
                lr = result.layers[layer]
                status_emoji = "\u2705" if lr.passed else "\u274C"
                lines.append(
                    f"| {layer.value.title()} | {lr.overall_score:.2%} | {status_emoji} |"
                )

        lines.append("")

        # Failed metrics
        failed = result.get_failed_metrics()
        if failed:
            lines.extend([
                "### Failed Metrics",
                "",
                "| Metric | Layer | Score | Threshold |",
                "|--------|-------|-------|-----------|",
            ])

            for metric in failed:
                lines.append(
                    f"| {metric.metric_name} | {metric.layer.value} | "
                    f"{metric.score:.2%} | {metric.threshold:.0%} |"
                )

        return "\n".join(lines)

    def _print_ci_output(self, result: EvaluationResult) -> None:
        """Print CI-friendly output to stdout."""
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)

        # Overall
        print(f"\nOverall: {'PASSED' if result.overall_passed else 'FAILED'} ({result.overall_score:.2%})")

        # Per layer
        print("\nLayer Results:")
        for layer in EvaluationLayer:
            if layer in result.layers:
                lr = result.layers[layer]
                status = "PASS" if lr.passed else "FAIL"
                print(f"  {layer.value}: {status} ({lr.overall_score:.2%})")

        # Failed metrics
        failed = result.get_failed_metrics()
        if failed:
            print("\nFailed Metrics:")
            for metric in failed:
                print(f"  - {metric.metric_name}: {metric.score:.2%} < {metric.threshold:.0%}")

        print("\n" + "=" * 50)

    def write_junit_xml(
        self,
        result: EvaluationResult,
        output_path: str = "test-results.xml",
    ) -> Path:
        """Generate JUnit XML format for CI systems.

        Args:
            result: Evaluation result
            output_path: Path for XML output

        Returns:
            Path to generated file
        """
        output_path = Path(output_path)

        # Build XML
        test_cases = []
        total_tests = 0
        failures = 0

        for layer in EvaluationLayer:
            if layer not in result.layers:
                continue

            layer_result = result.layers[layer]

            for metric in layer_result.metrics:
                total_tests += 1
                test_case = f'    <testcase name="{metric.metric_name}" classname="{layer.value}"'

                if metric.passed:
                    test_case += " />"
                else:
                    failures += 1
                    test_case += ">"
                    test_case += f'\n      <failure message="Score {metric.score:.2%} below threshold {metric.threshold:.0%}">'
                    if metric.error:
                        test_case += f"\n        {metric.error}"
                    test_case += "\n      </failure>"
                    test_case += "\n    </testcase>"

                test_cases.append(test_case)

        xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="Enterprise Evaluation" tests="{total_tests}" failures="{failures}" time="{result.duration_ms / 1000:.2f}">
  <testsuite name="GraphRAG Evaluation" tests="{total_tests}" failures="{failures}" timestamp="{result.timestamp}">
{chr(10).join(test_cases)}
  </testsuite>
</testsuites>
'''

        with open(output_path, "w") as f:
            f.write(xml)

        logger.info(f"JUnit XML saved to {output_path}")
        return output_path

    def write_badge_json(
        self,
        result: EvaluationResult,
        output_path: str = "evaluation-badge.json",
    ) -> Path:
        """Generate shields.io compatible badge JSON.

        Args:
            result: Evaluation result
            output_path: Path for badge JSON

        Returns:
            Path to generated file
        """
        output_path = Path(output_path)

        # Determine color based on score
        score = result.overall_score
        if score >= 0.9:
            color = "brightgreen"
        elif score >= 0.8:
            color = "green"
        elif score >= 0.7:
            color = "yellow"
        elif score >= 0.5:
            color = "orange"
        else:
            color = "red"

        badge = {
            "schemaVersion": 1,
            "label": "evaluation",
            "message": f"{score:.0%}",
            "color": color,
        }

        with open(output_path, "w") as f:
            json.dump(badge, f, indent=2)

        logger.info(f"Badge JSON saved to {output_path}")
        return output_path
