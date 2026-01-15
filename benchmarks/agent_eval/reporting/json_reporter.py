"""
JSON Reporter.

Machine-readable output for evaluation results.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import EvaluationResult

logger = logging.getLogger(__name__)


class JSONReporter:
    """Generates JSON reports for evaluation results."""

    def __init__(self, output_dir: str = "benchmarks/agent_eval/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        result: EvaluationResult,
        filename: Optional[str] = None,
    ) -> Path:
        """Generate JSON report.

        Args:
            result: Evaluation result to report
            filename: Optional custom filename

        Returns:
            Path to generated report file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_report_{timestamp}.json"

        output_path = self.output_dir / filename

        report = result.to_dict()

        # Add additional metadata
        report["report_metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "report_type": "json",
            "version": "1.0",
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"JSON report saved to {output_path}")
        return output_path

    def generate_summary(
        self,
        result: EvaluationResult,
        filename: Optional[str] = None,
    ) -> Path:
        """Generate condensed JSON summary.

        Args:
            result: Evaluation result
            filename: Optional custom filename

        Returns:
            Path to summary file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_summary_{timestamp}.json"

        output_path = self.output_dir / filename

        # Build summary
        summary = {
            "timestamp": result.timestamp,
            "duration_ms": result.duration_ms,
            "overall_score": result.overall_score,
            "overall_passed": result.overall_passed,
            "test_case_count": result.test_case_count,
            "layers": {},
        }

        for layer, layer_result in result.layers.items():
            summary["layers"][layer.value] = {
                "score": layer_result.overall_score,
                "passed": layer_result.passed,
                "metrics": {
                    m.metric_name: {
                        "score": m.score,
                        "passed": m.passed,
                        "threshold": m.threshold,
                    }
                    for m in layer_result.metrics
                },
            }

        # Failed metrics
        failed = result.get_failed_metrics()
        summary["failed_metrics"] = [
            {
                "name": m.metric_name,
                "layer": m.layer.value,
                "score": m.score,
                "threshold": m.threshold,
            }
            for m in failed
        ]

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"JSON summary saved to {output_path}")
        return output_path

    def generate_metrics_timeseries(
        self,
        results: list,
        filename: str = "metrics_timeseries.json",
    ) -> Path:
        """Generate timeseries data for tracking metrics over time.

        Args:
            results: List of evaluation results
            filename: Output filename

        Returns:
            Path to timeseries file
        """
        output_path = self.output_dir / filename

        timeseries = {
            "timestamps": [],
            "overall_scores": [],
            "layer_scores": {layer: [] for layer in ["retrieval", "agentic", "integrity", "generation"]},
            "metric_scores": {},
        }

        for result in results:
            timeseries["timestamps"].append(result.timestamp)
            timeseries["overall_scores"].append(result.overall_score)

            for layer, layer_result in result.layers.items():
                timeseries["layer_scores"][layer.value].append(layer_result.overall_score)

                for metric in layer_result.metrics:
                    if metric.metric_name not in timeseries["metric_scores"]:
                        timeseries["metric_scores"][metric.metric_name] = []
                    timeseries["metric_scores"][metric.metric_name].append(metric.score)

        with open(output_path, "w") as f:
            json.dump(timeseries, f, indent=2)

        logger.info(f"Timeseries data saved to {output_path}")
        return output_path
