"""
Shared training callbacks for metric logging and progress reporting.
"""

import sys
from tqdm import tqdm
from transformers import TrainerCallback


def format_metric(key: str, value: float) -> str:
    """Format a metric value with appropriate precision."""
    if not isinstance(value, (int, float)):
        return f"{key}={value}"
    if key == "epoch":
        return f"{key}={value:.2f}"
    if key == "learning_rate" or abs(value) < 0.001 or abs(value) > 10000:
        return f"{key}={value:.2e}"
    if isinstance(value, float):
        return f"{key}={value:.4f}"
    return f"{key}={value}"


class MetricsCallback(TrainerCallback):
    """Callback to format and log training metrics to console."""

    def __init__(self, print_steps: int = 5):
        """
        Args:
            print_steps: Print to console every N steps (wandb logs every step)
        """
        self.print_steps = print_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Skip if no meaningful metrics
        if "loss" not in logs and "eval_loss" not in logs:
            return

        step = state.global_step

        # Only print to console every print_steps
        if step % self.print_steps != 0 and step != 1:
            return

        max_steps = state.max_steps
        epoch = logs.get("epoch", 0)

        # Priority metrics to show first
        priority_keys = ["loss", "eval_loss", "learning_rate", "grad_norm"]
        other_keys = ["mean_token_accuracy", "entropy"]

        parts = [f"[{step}/{max_steps}] epoch {epoch:.2f}"]

        for key in priority_keys:
            if key in logs:
                parts.append(format_metric(key, logs[key]))

        for key in other_keys:
            if key in logs:
                parts.append(format_metric(key, logs[key]))

        tqdm.write(" | ".join(parts))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation results."""
        if metrics is None:
            return

        step = state.global_step
        max_steps = state.max_steps
        parts = [f"[{step}/{max_steps}] EVAL"]

        eval_keys = [
            "eval_loss", "eval_grad_norm", "eval_entropy",
            "eval_mean_token_accuracy", "eval_runtime", "eval_samples_per_second"
        ]
        for key in eval_keys:
            if key in metrics:
                parts.append(format_metric(key, metrics[key]))

        tqdm.write(" | ".join(parts))
