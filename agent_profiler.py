"""
Agent Profiler for ReActQAAgent Performance Tracking.

This module provides performance profiling capabilities inspired by
NVIDIA NeMo-Agent-Toolkit's profiler, tracking:
- Per-tool execution latency
- Token usage per reasoning step
- Bottleneck identification
- Overall workflow metrics
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ToolMetrics:
    """Metrics for a single tool call."""
    tool_name: str
    latency_ms: float
    success: bool
    input_size: int = 0  # Approximate input size
    output_size: int = 0  # Approximate output size
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class StepMetrics:
    """Metrics for a single ReAct reasoning step."""
    step_number: int
    thought_latency_ms: float
    action_latency_ms: float
    total_latency_ms: float
    tool_name: Optional[str] = None
    thought_length: int = 0
    observation_length: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "thought_latency_ms": self.thought_latency_ms,
            "action_latency_ms": self.action_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "tool_name": self.tool_name,
            "thought_length": self.thought_length,
            "observation_length": self.observation_length,
            "timestamp": self.timestamp,
        }


@dataclass
class LLMCallMetrics:
    """Metrics for an LLM API call."""
    call_type: str  # "think", "synthesize", "compress", etc.
    latency_ms: float
    prompt_length: int
    response_length: int
    model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_type": self.call_type,
            "latency_ms": self.latency_ms,
            "prompt_length": self.prompt_length,
            "response_length": self.response_length,
            "model": self.model,
            "timestamp": self.timestamp,
        }


@dataclass
class ProfileReport:
    """Complete profiling report for an agent run."""
    question: str
    answer: str = ""
    total_latency_ms: float = 0.0
    tool_metrics: List[ToolMetrics] = field(default_factory=list)
    step_metrics: List[StepMetrics] = field(default_factory=list)
    llm_metrics: List[LLMCallMetrics] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = ""

    def identify_bottlenecks(self, threshold_percentile: float = 0.9) -> List[str]:
        """
        Identify performance bottlenecks.

        Args:
            threshold_percentile: Percentile above which to flag as bottleneck

        Returns:
            List of bottleneck descriptions
        """
        bottlenecks = []

        # Tool bottlenecks
        if self.tool_metrics:
            tool_latencies = [t.latency_ms for t in self.tool_metrics]
            avg_tool = sum(tool_latencies) / len(tool_latencies)
            max_tool = max(tool_latencies)
            threshold = avg_tool * 2  # 2x average as bottleneck threshold

            for tool in self.tool_metrics:
                if tool.latency_ms > threshold:
                    bottlenecks.append(
                        f"Tool '{tool.tool_name}' took {tool.latency_ms:.0f}ms "
                        f"(avg: {avg_tool:.0f}ms)"
                    )

        # Step bottlenecks
        if self.step_metrics:
            step_latencies = [s.total_latency_ms for s in self.step_metrics]
            avg_step = sum(step_latencies) / len(step_latencies)
            threshold = avg_step * 2

            for step in self.step_metrics:
                if step.total_latency_ms > threshold:
                    bottlenecks.append(
                        f"Step {step.step_number} took {step.total_latency_ms:.0f}ms "
                        f"(avg: {avg_step:.0f}ms)"
                    )

        # LLM call bottlenecks
        if self.llm_metrics:
            llm_latencies = [l.latency_ms for l in self.llm_metrics]
            avg_llm = sum(llm_latencies) / len(llm_latencies)
            threshold = avg_llm * 2

            for llm in self.llm_metrics:
                if llm.latency_ms > threshold:
                    bottlenecks.append(
                        f"LLM call '{llm.call_type}' took {llm.latency_ms:.0f}ms "
                        f"(avg: {avg_llm:.0f}ms)"
                    )

        self.bottlenecks = bottlenecks
        return bottlenecks

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the profiling results."""
        tool_latency = sum(t.latency_ms for t in self.tool_metrics)
        llm_latency = sum(l.latency_ms for l in self.llm_metrics)
        step_latency = sum(s.total_latency_ms for s in self.step_metrics)

        return {
            "total_latency_ms": self.total_latency_ms,
            "n_steps": len(self.step_metrics),
            "n_tool_calls": len(self.tool_metrics),
            "n_llm_calls": len(self.llm_metrics),
            "tool_latency_ms": tool_latency,
            "llm_latency_ms": llm_latency,
            "tool_latency_pct": (tool_latency / self.total_latency_ms * 100)
                if self.total_latency_ms > 0 else 0,
            "llm_latency_pct": (llm_latency / self.total_latency_ms * 100)
                if self.total_latency_ms > 0 else 0,
            "avg_step_latency_ms": step_latency / len(self.step_metrics)
                if self.step_metrics else 0,
            "bottlenecks": self.bottlenecks,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "total_latency_ms": self.total_latency_ms,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "summary": self.get_summary(),
            "tool_metrics": [t.to_dict() for t in self.tool_metrics],
            "step_metrics": [s.to_dict() for s in self.step_metrics],
            "llm_metrics": [l.to_dict() for l in self.llm_metrics],
            "bottlenecks": self.bottlenecks,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class AgentProfiler:
    """
    Track agent performance metrics during execution.

    Usage:
        profiler = AgentProfiler()
        profiler.start("What is inflation?")

        with profiler.track_tool("graph_lookup"):
            # Tool execution

        with profiler.track_step(1):
            # Step execution

        report = profiler.finish()
    """

    def __init__(self):
        """Initialize the profiler."""
        self._report: Optional[ProfileReport] = None
        self._active = False
        self._current_step: Optional[int] = None
        self._step_start_time: Optional[float] = None
        self._thought_latency: float = 0.0

    def start(self, question: str) -> None:
        """Start profiling a new agent run."""
        self._report = ProfileReport(question=question)
        self._active = True
        self._current_step = None
        self._step_start_time = time.perf_counter()
        logger.debug(f"Profiler started for: {question[:50]}...")

    def finish(self, answer: str = "") -> ProfileReport:
        """Finish profiling and return the report."""
        if not self._report:
            raise ValueError("Profiler not started. Call start() first.")

        self._report.answer = answer
        self._report.end_time = datetime.now().isoformat()
        self._report.total_latency_ms = (
            time.perf_counter() - self._step_start_time
        ) * 1000
        self._report.identify_bottlenecks()
        self._active = False

        report = self._report
        self._report = None
        return report

    @contextmanager
    def track_tool(self, tool_name: str, input_data: Any = None):
        """
        Context manager to track tool execution.

        Args:
            tool_name: Name of the tool being called
            input_data: Input data for size estimation
        """
        if not self._active:
            yield
            return

        start = time.perf_counter()
        input_size = len(str(input_data)) if input_data else 0
        error = None
        success = True

        try:
            yield
        except Exception as e:
            error = str(e)
            success = False
            raise
        finally:
            latency = (time.perf_counter() - start) * 1000
            metric = ToolMetrics(
                tool_name=tool_name,
                latency_ms=latency,
                success=success,
                input_size=input_size,
                error=error,
            )
            self._report.tool_metrics.append(metric)
            logger.debug(f"Tool {tool_name}: {latency:.0f}ms")

    @contextmanager
    def track_llm_call(
        self,
        call_type: str,
        prompt: str = "",
        model: str = "",
    ):
        """
        Context manager to track LLM API calls.

        Args:
            call_type: Type of call (think, synthesize, compress, etc.)
            prompt: The prompt being sent
            model: Model name
        """
        if not self._active:
            yield
            return

        start = time.perf_counter()
        prompt_length = len(prompt)
        response_length = 0

        try:
            yield
        finally:
            latency = (time.perf_counter() - start) * 1000
            metric = LLMCallMetrics(
                call_type=call_type,
                latency_ms=latency,
                prompt_length=prompt_length,
                response_length=response_length,
                model=model,
            )
            self._report.llm_metrics.append(metric)

            # Track thought latency for step metrics
            if call_type == "think":
                self._thought_latency = latency

            logger.debug(f"LLM {call_type}: {latency:.0f}ms")

    def track_step_start(self, step_number: int) -> None:
        """Mark the start of a reasoning step."""
        if not self._active:
            return
        self._current_step = step_number
        self._step_start_time = time.perf_counter()
        self._thought_latency = 0.0

    def track_step_end(
        self,
        tool_name: Optional[str] = None,
        thought_length: int = 0,
        observation_length: int = 0,
    ) -> None:
        """Mark the end of a reasoning step."""
        if not self._active or self._current_step is None:
            return

        total_latency = (time.perf_counter() - self._step_start_time) * 1000
        action_latency = total_latency - self._thought_latency

        metric = StepMetrics(
            step_number=self._current_step,
            thought_latency_ms=self._thought_latency,
            action_latency_ms=max(0, action_latency),
            total_latency_ms=total_latency,
            tool_name=tool_name,
            thought_length=thought_length,
            observation_length=observation_length,
        )
        self._report.step_metrics.append(metric)
        logger.debug(f"Step {self._current_step}: {total_latency:.0f}ms")

        self._current_step = None

    @contextmanager
    def track_step(self, step_number: int):
        """
        Context manager to track a complete reasoning step.

        Args:
            step_number: The step number (1-indexed)
        """
        self.track_step_start(step_number)
        try:
            yield
        finally:
            self.track_step_end()

    def is_active(self) -> bool:
        """Check if profiler is currently active."""
        return self._active

    def get_current_report(self) -> Optional[ProfileReport]:
        """Get the current in-progress report (for debugging)."""
        return self._report


def print_profile_report(report: ProfileReport) -> None:
    """Pretty print a profile report to console."""
    print("\n" + "=" * 60)
    print("AGENT PROFILE REPORT")
    print("=" * 60)

    summary = report.get_summary()

    print(f"\nQuestion: {report.question[:80]}...")
    print(f"Answer: {report.answer[:80]}..." if report.answer else "No answer")

    print(f"\n--- Timing Summary ---")
    print(f"Total time: {summary['total_latency_ms']:.0f}ms")
    print(f"Steps: {summary['n_steps']}")
    print(f"Tool calls: {summary['n_tool_calls']}")
    print(f"LLM calls: {summary['n_llm_calls']}")

    print(f"\n--- Latency Breakdown ---")
    print(f"Tool latency: {summary['tool_latency_ms']:.0f}ms ({summary['tool_latency_pct']:.1f}%)")
    print(f"LLM latency: {summary['llm_latency_ms']:.0f}ms ({summary['llm_latency_pct']:.1f}%)")
    print(f"Avg step latency: {summary['avg_step_latency_ms']:.0f}ms")

    if report.tool_metrics:
        print(f"\n--- Tool Calls ---")
        for tool in report.tool_metrics:
            status = "OK" if tool.success else f"FAIL: {tool.error}"
            print(f"  {tool.tool_name}: {tool.latency_ms:.0f}ms [{status}]")

    if report.step_metrics:
        print(f"\n--- Steps ---")
        for step in report.step_metrics:
            tool_info = f" ({step.tool_name})" if step.tool_name else ""
            print(f"  Step {step.step_number}{tool_info}: {step.total_latency_ms:.0f}ms "
                  f"(think: {step.thought_latency_ms:.0f}ms, action: {step.action_latency_ms:.0f}ms)")

    if report.bottlenecks:
        print(f"\n--- Bottlenecks Identified ---")
        for bottleneck in report.bottlenecks:
            print(f"  ! {bottleneck}")

    print("\n" + "=" * 60)


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent Profiler CLI")
    parser.add_argument("--demo", action="store_true", help="Run demo profiling")
    args = parser.parse_args()

    if args.demo:
        # Demo profiling
        profiler = AgentProfiler()
        profiler.start("What is the capital of France?")

        # Simulate some operations
        with profiler.track_step(1):
            with profiler.track_llm_call("think", "Sample prompt"):
                time.sleep(0.1)  # Simulate LLM call
            with profiler.track_tool("graph_lookup", {"entity": "France"}):
                time.sleep(0.05)  # Simulate tool call

        with profiler.track_step(2):
            with profiler.track_llm_call("synthesize", "Another prompt"):
                time.sleep(0.15)

        report = profiler.finish("Paris is the capital of France.")
        print_profile_report(report)

        # Save to file
        with open("demo_profile.json", "w") as f:
            f.write(report.to_json())
        print("\nReport saved to demo_profile.json")
    else:
        print("Use --demo to run a demo profiling session")
