"""
Human Reporter.

Human-readable console and markdown output for evaluation results.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import EvaluationResult, EvaluationLayer

logger = logging.getLogger(__name__)


class HumanReporter:
    """Generates human-readable reports for evaluation results."""

    # ANSI color codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors enabled."""
        if self.use_colors:
            return f"{color}{text}{self.RESET}"
        return text

    def _status_icon(self, passed: bool) -> str:
        """Get status icon for pass/fail."""
        if passed:
            return self._color("[PASS]", self.GREEN)
        return self._color("[FAIL]", self.RED)

    def _score_color(self, score: float, threshold: float) -> str:
        """Color score based on threshold."""
        if score >= threshold:
            return self._color(f"{score:.2%}", self.GREEN)
        elif score >= threshold * 0.8:
            return self._color(f"{score:.2%}", self.YELLOW)
        return self._color(f"{score:.2%}", self.RED)

    def print_report(self, result: EvaluationResult) -> None:
        """Print evaluation report to console.

        Args:
            result: Evaluation result to report
        """
        print("\n" + "=" * 70)
        print(self._color("ENTERPRISE EVALUATION REPORT", self.BOLD))
        print("=" * 70)

        # Metadata
        print(f"\nTimestamp: {result.timestamp}")
        print(f"Duration: {result.duration_ms}ms")
        print(f"Test Cases: {result.test_case_count}")
        print(f"Judge Model: {result.config.judge_model}")

        # Overall Summary
        print("\n" + "-" * 70)
        print(self._color("OVERALL SUMMARY", self.BOLD))
        print("-" * 70)

        overall_status = self._status_icon(result.overall_passed)
        overall_score = self._score_color(result.overall_score, 0.7)
        print(f"Status: {overall_status}")
        print(f"Score: {overall_score}")

        # Layer Results
        for layer in EvaluationLayer:
            if layer not in result.layers:
                continue

            layer_result = result.layers[layer]

            print("\n" + "-" * 70)
            layer_name = layer.value.upper()
            print(f"{self._color(f'LAYER: {layer_name}', self.BOLD)}")
            print("-" * 70)

            layer_status = self._status_icon(layer_result.passed)
            layer_score = self._score_color(layer_result.overall_score, 0.7)
            print(f"Status: {layer_status}  Score: {layer_score}")

            # Individual metrics
            print("\nMetrics:")
            for metric in layer_result.metrics:
                status = self._status_icon(metric.passed)
                score = self._score_color(metric.score, metric.threshold)
                threshold = f"(threshold: {metric.threshold:.0%})"

                print(f"  {status} {metric.metric_name}: {score} {threshold}")

                if metric.error:
                    print(f"       {self._color(f'Error: {metric.error}', self.RED)}")

        # Failed Metrics Summary
        failed = result.get_failed_metrics()
        if failed:
            print("\n" + "-" * 70)
            print(self._color("FAILED METRICS", self.RED + self.BOLD))
            print("-" * 70)

            for metric in failed:
                print(f"  - {metric.metric_name} ({metric.layer.value}): "
                      f"{metric.score:.2%} < {metric.threshold:.0%}")

        print("\n" + "=" * 70 + "\n")

    def generate_markdown(
        self,
        result: EvaluationResult,
        output_dir: str = "benchmarks/enterprise_eval/reports",
        filename: Optional[str] = None,
    ) -> Path:
        """Generate markdown report.

        Args:
            result: Evaluation result
            output_dir: Output directory
            filename: Optional custom filename

        Returns:
            Path to markdown file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_report_{timestamp}.md"

        file_path = output_path / filename

        lines = [
            "# Enterprise Evaluation Report",
            "",
            f"**Timestamp:** {result.timestamp}",
            f"**Duration:** {result.duration_ms}ms",
            f"**Test Cases:** {result.test_case_count}",
            f"**Judge Model:** {result.config.judge_model}",
            "",
            "---",
            "",
            "## Overall Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Status** | {'PASS' if result.overall_passed else 'FAIL'} |",
            f"| **Score** | {result.overall_score:.2%} |",
            "",
        ]

        # Layer results
        for layer in EvaluationLayer:
            if layer not in result.layers:
                continue

            layer_result = result.layers[layer]

            lines.extend([
                f"## Layer: {layer.value.title()}",
                "",
                f"**Status:** {'PASS' if layer_result.passed else 'FAIL'}  ",
                f"**Score:** {layer_result.overall_score:.2%}",
                "",
                "| Metric | Score | Threshold | Status |",
                "|--------|-------|-----------|--------|",
            ])

            for metric in layer_result.metrics:
                status = "PASS" if metric.passed else "FAIL"
                lines.append(
                    f"| {metric.metric_name} | {metric.score:.2%} | "
                    f"{metric.threshold:.0%} | {status} |"
                )

            lines.append("")

        # Failed metrics
        failed = result.get_failed_metrics()
        if failed:
            lines.extend([
                "## Failed Metrics",
                "",
                "| Metric | Layer | Score | Threshold |",
                "|--------|-------|-------|-----------|",
            ])

            for metric in failed:
                lines.append(
                    f"| {metric.metric_name} | {metric.layer.value} | "
                    f"{metric.score:.2%} | {metric.threshold:.0%} |"
                )

            lines.append("")

        # Write file
        with open(file_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown report saved to {file_path}")
        return file_path

    def print_summary(self, result: EvaluationResult) -> None:
        """Print condensed one-line summary.

        Args:
            result: Evaluation result
        """
        status = self._status_icon(result.overall_passed)
        score = self._score_color(result.overall_score, 0.7)

        layer_summaries = []
        for layer in EvaluationLayer:
            if layer in result.layers:
                lr = result.layers[layer]
                icon = "+" if lr.passed else "-"
                layer_summaries.append(f"{layer.value[0].upper()}:{icon}")

        layers_str = " ".join(layer_summaries)

        print(f"{status} Score: {score} | Layers: {layers_str} | "
              f"Tests: {result.test_case_count} | Time: {result.duration_ms}ms")
