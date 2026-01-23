#!/usr/bin/env python3
"""
Centralized metric calculation system.
Imports metrics from task modules.
"""

# Import metrics from task modules
from src.tasks.distance import METRIC as distance_metric
from src.tasks.randomwalk import METRIC as randomwalk_metric
from src.tasks.trianglearea import METRIC as trianglearea_metric
from src.tasks.angle import METRIC as angle_metric
from src.tasks.compass import METRIC as compass_metric
from src.tasks.crossing import METRIC as crossing_metric
from src.tasks.inside import METRIC as inside_metric
from src.tasks.perimeter import METRIC as perimeter_metric
from src.tasks.nearest_neighbor import METRIC as nearest_neighbor_metric
from src.tasks.center import METRIC as center_metric
from src.tasks.circlecount import METRIC as circlecount_metric
from src.tasks.randring import METRIC as randring_metric


# Registry of all task metrics
TASK_METRICS = {
    'distance': distance_metric,
    'randomwalk': randomwalk_metric,
    'trianglearea': trianglearea_metric,
    'angle': angle_metric,
    'compass': compass_metric,
    'crossing': crossing_metric,
    'inside': inside_metric,
    'perimeter': perimeter_metric,
    'nearest_neighbor': nearest_neighbor_metric,
    'center': center_metric,
    'circlecount': circlecount_metric,
    'randring': randring_metric,
}


def get_metric(task_type: str):
    """Get the metric calculator for a task type."""
    if task_type not in TASK_METRICS:
        raise ValueError(f"FATAL: No metric implementation for task type '{task_type}'")
    return TASK_METRICS[task_type]


def calculate_metric(
    task_type: str,
    prompt: str,
    true_completion: str,
    generated: str,
    **kwargs
) -> float:
    """
    Calculate metric for a task.

    Args:
        task_type: Type of task (distance, randomwalk, etc.)
        prompt: The input prompt
        true_completion: The expected completion
        generated: The model's generated text
        **kwargs: Additional task-specific arguments (e.g., cities_df)

    Returns:
        The metric value
    """
    metric = get_metric(task_type)
    return metric.calculate(prompt, true_completion, generated, **kwargs)


def get_failure_value(task_type: str) -> float:
    """Get the failure value for a task type."""
    metric = get_metric(task_type)
    return metric.failure_value


def format_metric_for_display(task_type: str, value: float) -> str:
    """Format a metric value for display."""
    metric = get_metric(task_type)
    return metric.format_for_print(value)


def get_metric_display_name(task_type: str) -> str:
    """Get the display name for a metric."""
    metric = get_metric(task_type)
    return metric.display_name
