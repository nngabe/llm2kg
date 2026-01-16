"""
Optuna-based Hyperparameter Optimizer for ReActQAAgent.

This module provides automatic hyperparameter tuning using Optuna's
multi-objective optimization framework, inspired by NVIDIA NeMo-Agent-Toolkit.

Optimizable Parameters:
- temperature: LLM sampling temperature (0.0-1.0)
- top_p: Nucleus sampling parameter (0.5-1.0)
- max_iterations: Maximum ReAct loop iterations (3-10)
- context_window_size: Context size limit (4000-16000)
- parse_response_max_retries: JSON parse retries (1-5)
- tool_call_max_retries: Tool call retries (1-3)
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Optional, Tuple
from datetime import datetime

import optuna
from optuna.visualization import plot_pareto_front, plot_optimization_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizableParams:
    """Parameters that can be tuned with their search spaces."""
    # LLM parameters
    temperature: float = 0.0          # Range: 0.0-1.0
    top_p: float = 1.0                # Range: 0.5-1.0

    # Agent parameters
    max_iterations: int = 5           # Range: 3-10
    context_window_size: int = 8000   # Range: 4000-16000

    # Retry parameters
    parse_response_max_retries: int = 2  # Range: 1-5
    tool_call_max_retries: int = 1       # Range: 1-3


@dataclass
class SearchSpace:
    """Define the search space for a parameter."""
    param_type: str  # "float", "int", "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    values: Optional[List[Any]] = None
    log: bool = False


# Default search spaces for agent parameters
DEFAULT_SEARCH_SPACES = {
    "temperature": SearchSpace(param_type="float", low=0.0, high=1.0),
    "top_p": SearchSpace(param_type="float", low=0.5, high=1.0),
    "max_iterations": SearchSpace(param_type="int", low=3, high=10),
    "context_window_size": SearchSpace(param_type="int", low=4000, high=16000),
    "parse_response_max_retries": SearchSpace(param_type="int", low=1, high=5),
    "tool_call_max_retries": SearchSpace(param_type="int", low=1, high=3),
}


@dataclass
class OptimizationMetric:
    """Define an optimization objective."""
    name: str
    direction: str = "maximize"  # "maximize" or "minimize"
    weight: float = 1.0


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    n_trials: int = 50
    timeout_seconds: Optional[int] = None
    n_jobs: int = 1  # Parallel trials
    study_name: str = "agent_optimization"
    storage: Optional[str] = None  # SQLite path for persistence
    sampler: str = "tpe"  # "tpe", "random", "cmaes"
    pruner: str = "median"  # "median", "hyperband", "none"

    # Search spaces to use
    search_spaces: Dict[str, SearchSpace] = field(
        default_factory=lambda: DEFAULT_SEARCH_SPACES.copy()
    )

    # Metrics to optimize
    metrics: List[OptimizationMetric] = field(
        default_factory=lambda: [
            OptimizationMetric(name="accuracy", direction="maximize", weight=0.7),
            OptimizationMetric(name="latency", direction="minimize", weight=0.3),
        ]
    )

    # Output
    output_dir: str = "optimization_results"
    save_trials: bool = True


class AgentOptimizer:
    """
    Optimize agent hyperparameters using Optuna.

    Supports single and multi-objective optimization with various
    sampling strategies and pruning algorithms.
    """

    def __init__(
        self,
        eval_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        config: Optional[OptimizationConfig] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            eval_fn: Function that takes params dict and returns metrics dict.
                     Example: {"accuracy": 0.85, "latency": 120.5}
            config: Optimization configuration.
        """
        self.eval_fn = eval_fn
        self.config = config or OptimizationConfig()
        self.study: Optional[optuna.Study] = None
        self._trial_history: List[Dict[str, Any]] = []

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create the specified sampler."""
        if self.config.sampler == "tpe":
            return optuna.samplers.TPESampler()
        elif self.config.sampler == "random":
            return optuna.samplers.RandomSampler()
        elif self.config.sampler == "cmaes":
            return optuna.samplers.CmaEsSampler()
        else:
            logger.warning(f"Unknown sampler {self.config.sampler}, using TPE")
            return optuna.samplers.TPESampler()

    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Create the specified pruner."""
        if self.config.pruner == "median":
            return optuna.pruners.MedianPruner()
        elif self.config.pruner == "hyperband":
            return optuna.pruners.HyperbandPruner()
        elif self.config.pruner == "none":
            return None
        else:
            return optuna.pruners.MedianPruner()

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters for a trial based on search spaces."""
        params = {}

        for name, space in self.config.search_spaces.items():
            if space.param_type == "float":
                params[name] = trial.suggest_float(
                    name, space.low, space.high, log=space.log
                )
            elif space.param_type == "int":
                params[name] = trial.suggest_int(
                    name, int(space.low), int(space.high), log=space.log
                )
            elif space.param_type == "categorical":
                params[name] = trial.suggest_categorical(name, space.values)

        return params

    def objective(self, trial: optuna.Trial) -> Tuple[float, ...]:
        """
        Objective function for Optuna optimization.

        For multi-objective, returns tuple of metric values.
        """
        # Suggest parameters
        params = self._suggest_params(trial)

        # Evaluate
        try:
            metrics = self.eval_fn(params)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Return worst possible values
            return tuple(
                float("inf") if m.direction == "minimize" else float("-inf")
                for m in self.config.metrics
            )

        # Record trial
        trial_record = {
            "trial_number": trial.number,
            "params": params,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self._trial_history.append(trial_record)

        # Return metric values in order
        values = []
        for metric in self.config.metrics:
            value = metrics.get(metric.name, 0.0)
            values.append(value)

        return tuple(values)

    def optimize(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            verbose: Print progress information

        Returns:
            Dict with best parameters and optimization stats
        """
        # Determine directions
        directions = [
            "maximize" if m.direction == "maximize" else "minimize"
            for m in self.config.metrics
        ]

        # Create study
        sampler = self._create_sampler()
        pruner = self._create_pruner()

        if len(self.config.metrics) == 1:
            # Single objective
            self.study = optuna.create_study(
                study_name=self.config.study_name,
                storage=self.config.storage,
                sampler=sampler,
                pruner=pruner,
                direction=directions[0],
                load_if_exists=True,
            )
        else:
            # Multi-objective
            self.study = optuna.create_study(
                study_name=self.config.study_name,
                storage=self.config.storage,
                sampler=sampler,
                directions=directions,
                load_if_exists=True,
            )

        # Set verbosity
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            n_jobs=self.config.n_jobs,
            show_progress_bar=verbose,
        )

        # Get results
        if len(self.config.metrics) == 1:
            # Single objective - best trial
            best_params = self.study.best_params
            best_value = self.study.best_value
            results = {
                "best_params": best_params,
                "best_value": best_value,
                "best_trial": self.study.best_trial.number,
            }
        else:
            # Multi-objective - Pareto front
            pareto_trials = self.study.best_trials
            results = {
                "pareto_front": [
                    {
                        "params": t.params,
                        "values": t.values,
                        "trial": t.number,
                    }
                    for t in pareto_trials
                ],
                "n_pareto_solutions": len(pareto_trials),
            }

        # Add stats
        results["n_trials"] = len(self.study.trials)
        results["trial_history"] = self._trial_history

        # Save results if configured
        if self.config.save_trials:
            self._save_results(results)

        return results

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save optimization results to output directory."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        results_path = os.path.join(
            self.config.output_dir,
            f"optimization_{timestamp}.json"
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_path}")

        # Save trials dataframe
        if self.study:
            df = self.study.trials_dataframe()
            df_path = os.path.join(
                self.config.output_dir,
                f"trials_{timestamp}.csv"
            )
            df.to_csv(df_path, index=False)
            logger.info(f"Trials dataframe saved to {df_path}")

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from completed study."""
        if not self.study:
            raise ValueError("No study available. Run optimize() first.")

        if len(self.config.metrics) == 1:
            return self.study.best_params
        else:
            # For multi-objective, return first Pareto solution
            return self.study.best_trials[0].params if self.study.best_trials else {}

    def plot_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        if not self.study:
            raise ValueError("No study available. Run optimize() first.")

        fig = plot_optimization_history(self.study)
        if save_path:
            fig.write_html(save_path)
        return fig

    def plot_pareto(self, save_path: Optional[str] = None):
        """Plot Pareto front for multi-objective optimization."""
        if not self.study or len(self.config.metrics) < 2:
            raise ValueError("Pareto plot requires multi-objective study")

        fig = plot_pareto_front(self.study)
        if save_path:
            fig.write_html(save_path)
        return fig


def create_agent_eval_function(
    agent_class,
    test_dataset: List[Dict[str, Any]],
    base_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable[[Dict[str, Any]], Dict[str, float]]:
    """
    Create an evaluation function for agent optimization.

    Args:
        agent_class: Agent class to instantiate (e.g., ReActQAAgent)
        test_dataset: List of {"question": str, "expected_answer": str}
        base_kwargs: Base kwargs for agent (will be overridden by optimized params)

    Returns:
        Function that evaluates agent with given params and returns metrics
    """
    import time

    base_kwargs = base_kwargs or {}

    def eval_fn(params: Dict[str, Any]) -> Dict[str, float]:
        # Merge params with base kwargs
        kwargs = {**base_kwargs, **params}

        # Handle LLM params specially
        llm_params = {}
        for key in ["temperature", "top_p"]:
            if key in kwargs:
                llm_params[key] = kwargs.pop(key)

        # Create agent
        if llm_params:
            from langchain_ollama import ChatOllama
            kwargs["llm"] = ChatOllama(
                model=os.getenv("OLLAMA_MAIN_MODEL", "nemotron-3-nano:30b"),
                base_url=os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"),
                **llm_params,
            )

        agent = agent_class(**kwargs)

        # Evaluate
        correct = 0
        total_time = 0

        try:
            for item in test_dataset:
                start = time.perf_counter()
                response = agent.answer_question(item["question"])
                elapsed = time.perf_counter() - start
                total_time += elapsed

                # Simple accuracy check (word overlap)
                expected_words = set(item["expected_answer"].lower().split())
                actual_words = set(response.answer.lower().split())
                if expected_words:
                    overlap = len(expected_words & actual_words) / len(expected_words)
                    if overlap > 0.5:
                        correct += 1
        finally:
            agent.close()

        accuracy = correct / len(test_dataset) if test_dataset else 0.0
        avg_latency = (total_time / len(test_dataset) * 1000) if test_dataset else 0.0

        return {
            "accuracy": accuracy,
            "latency": avg_latency,  # ms
        }

    return eval_fn


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter Optimizer CLI")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    parser.add_argument("--output-dir", type=str, default="optimization_results")
    parser.add_argument("--study-name", type=str, default="agent_optimization")
    parser.add_argument("--storage", type=str, help="SQLite path for persistence")
    args = parser.parse_args()

    # Example: Optimize a simple function
    def dummy_eval(params: Dict[str, Any]) -> Dict[str, float]:
        """Dummy evaluation function for testing."""
        temp = params.get("temperature", 0.5)
        # Simulate: lower temp = higher accuracy but higher latency
        accuracy = 1.0 - temp * 0.5 + (0.1 * (params.get("max_iterations", 5) / 10))
        latency = 100 + temp * 200 + params.get("context_window_size", 8000) / 100
        return {"accuracy": accuracy, "latency": latency}

    config = OptimizationConfig(
        n_trials=args.n_trials,
        timeout_seconds=args.timeout,
        output_dir=args.output_dir,
        study_name=args.study_name,
        storage=args.storage,
    )

    optimizer = AgentOptimizer(eval_fn=dummy_eval, config=config)
    results = optimizer.optimize(verbose=True)

    print("\n=== Optimization Results ===")
    if "best_params" in results:
        print(f"Best params: {results['best_params']}")
        print(f"Best value: {results['best_value']:.4f}")
    else:
        print(f"Pareto front solutions: {results['n_pareto_solutions']}")
        for sol in results["pareto_front"][:3]:
            print(f"  Trial {sol['trial']}: {sol['params']}")
