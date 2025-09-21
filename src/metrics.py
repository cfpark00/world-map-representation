#!/usr/bin/env python3
"""
Centralized metric calculation system.
Each task type's logic is in EXACTLY ONE place.
"""

import re
import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List, Set


class TaskMetric(ABC):
    """Base class for all task-specific metrics."""

    @abstractmethod
    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        """
        Calculate the metric for this task.

        Args:
            prompt: The input prompt
            true_completion: The expected completion
            generated: The model's generated text
            **kwargs: Additional task-specific arguments (e.g., cities_df)

        Returns:
            The metric value
        """
        pass

    @abstractmethod
    def get_failure_value(self) -> float:
        """Get the value to use when parsing/calculation fails."""
        pass

    @abstractmethod
    def get_display_name(self) -> str:
        """Get human-readable name for this metric."""
        pass

    def format_for_print(self, value: float) -> str:
        """Format the metric value for display."""
        return f"{value:.2f}"


class DistanceMetric(TaskMetric):
    """Metric for distance tasks: absolute error in distance calculation."""

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        true_dist = self._parse_distance(true_completion)
        gen_dist = self._parse_distance(generated)

        if true_dist is not None and gen_dist is not None:
            return abs(true_dist - gen_dist)
        else:
            return self.get_failure_value()

    def _parse_distance(self, text: str) -> Optional[int]:
        """Parse distance from text (exact copy of parse_distance from utils.py)."""
        text = text.replace(' ', '')

        match = re.search(r'=(\d+)', text)
        if match:
            return int(match.group(1))
        match = re.search(r'^(\d+)', text)
        if match:
            return int(match.group(1))
        return None

    def get_failure_value(self) -> float:
        return math.sqrt(3600**2 + 1800**2)

    def get_display_name(self) -> str:
        return "Absolute Distance Error (km)"


class RandomWalkMetric(TaskMetric):
    """Metric for random walk tasks: validity ratio × length penalty."""

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        cities_df = kwargs.get('cities_df')
        if cities_df is None:
            return self.get_failure_value()

        # Parse expected parameters from prompt
        rw_match = re.search(r'rw\((\d+),(\d+)\)', prompt.replace(' ', ''))
        if not rw_match:
            return self.get_failure_value()

        expected_max_dist = int(rw_match.group(1))
        expected_chain_len = int(rw_match.group(2))

        # Parse generated walk and count all attempted transitions
        transitions, total_attempted = self._parse_walk_transitions(generated)

        if total_attempted == 0:
            return self.get_failure_value()

        # Validate transitions
        valid_trans = self._validate_transitions(
            transitions, cities_df, expected_max_dist
        )

        # Calculate validity ratio (valid transitions / total attempted transitions)
        validity_ratio = valid_trans / total_attempted

        # Calculate length penalty based on actual parsed cities (not attempted)
        actual_chain_len = len(transitions) + 1 if transitions else 0
        chain_len_diff = abs(actual_chain_len - expected_chain_len)
        length_penalty = np.exp(-chain_len_diff / expected_chain_len) if expected_chain_len > 0 else 0.0

        # Combined score
        return validity_ratio * length_penalty

    def _parse_walk_transitions(self, text: str) -> Tuple[List[Tuple[int, int]], int]:
        """
        Parse transitions from walk text.
        Returns: (list of valid transitions, total attempted transitions)
        """
        text = text.replace(' ', '')

        match = re.search(r'=(.+)', text)
        if not match:
            return [], 0

        sequence = match.group(1)

        # Find ALL city-like tokens (valid or invalid)
        all_city_tokens = re.findall(r'c_\w+', sequence)

        if len(all_city_tokens) < 2:
            return [], len(all_city_tokens)

        # Total attempted transitions is number of consecutive city pairs
        total_attempted = len(all_city_tokens) - 1

        # Now extract only the valid transitions (with numeric IDs)
        city_matches = list(re.finditer(r'c_(\d+)', sequence))

        transitions = []
        for i in range(len(city_matches) - 1):
            city1_id = int(city_matches[i].group(1))
            city2_id = int(city_matches[i + 1].group(1))
            transitions.append((city1_id, city2_id))

        return transitions, total_attempted

    def _validate_transitions(self, transitions, cities_df, distance_threshold_km):
        """
        Validate transitions.
        Returns: number of valid transitions (both cities exist AND distance is valid)
        """
        if not transitions:
            return 0

        valid_transitions = 0

        for city1_id, city2_id in transitions:
            # Check if both cities exist
            city1_rows = cities_df[cities_df['city_id'] == city1_id]
            city2_rows = cities_df[cities_df['city_id'] == city2_id]

            # If either city doesn't exist, this transition fails
            if len(city1_rows) == 0 or len(city2_rows) == 0:
                continue

            city1 = city1_rows.iloc[0]
            city2 = city2_rows.iloc[0]

            distance = np.sqrt((city2['x'] - city1['x'])**2 + (city2['y'] - city1['y'])**2)

            if distance <= distance_threshold_km:
                valid_transitions += 1

        return valid_transitions

    def get_failure_value(self) -> float:
        return 0.0

    def get_display_name(self) -> str:
        return "Walk Validity Score"

    def format_for_print(self, value: float) -> str:
        return f"{value:.3f}"


class TriangleAreaMetric(TaskMetric):
    """Metric for triangle area tasks: absolute error in area."""

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        # Look for answer after = in format: triarea(...)=AREA
        true_match = re.search(r'=(\d+)', (prompt + true_completion).replace(' ', ''))
        gen_match = re.search(r'=(\d+)', generated.replace(' ', ''))

        if true_match and gen_match:
            true_area = int(true_match.group(1))
            gen_area = int(gen_match.group(1))
            return abs(gen_area - true_area)
        else:
            return self.get_failure_value()

    def get_failure_value(self) -> float:
        return (3600 * 1800) / 2

    def get_display_name(self) -> str:
        return "Area Absolute Error"


class AngleMetric(TaskMetric):
    """Metric for angle tasks: absolute error in degrees."""

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        # Use pattern that matches after closing paren to avoid parameter = signs
        true_match = re.search(r'\)=(\d+)', (prompt + true_completion).replace(' ', ''))
        gen_match = re.search(r'\)=(\d+)', generated.replace(' ', ''))

        if true_match and gen_match:
            true_angle = int(true_match.group(1))
            gen_angle = int(gen_match.group(1))
            return abs(gen_angle - true_angle)
        else:
            return self.get_failure_value()

    def get_failure_value(self) -> float:
        return 180.0

    def get_display_name(self) -> str:
        return "Angle Error (degrees)"


class CompassMetric(TaskMetric):
    """Metric for compass direction tasks: binary accuracy."""

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        true_direction = true_completion.replace(' ', '').strip().upper()
        if '<EOS>' in true_direction:
            true_direction = true_direction.replace('<EOS>', '').strip()

        # Extract answer after the last '=' in generated text
        gen_no_space = generated.replace(' ', '')
        eq_pos = gen_no_space.rfind('=')
        if eq_pos != -1:
            gen_direction = gen_no_space[eq_pos+1:].strip().upper()
        else:
            gen_direction = gen_no_space.strip().upper()

        return 1.0 if true_direction == gen_direction else 0.0

    def get_failure_value(self) -> float:
        return 0.0

    def get_display_name(self) -> str:
        return "Direction Accuracy"


class BooleanMetric(TaskMetric):
    """Metric for TRUE/FALSE tasks (crossing, inside): binary accuracy."""

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        true_value = true_completion.replace(' ', '').strip().upper()
        if '<EOS>' in true_value:
            true_value = true_value.replace('<EOS>', '').strip()

        # Extract answer after the last '=' in generated text
        gen_no_space = generated.replace(' ', '')
        eq_pos = gen_no_space.rfind('=')
        if eq_pos != -1:
            gen_value = gen_no_space[eq_pos+1:].strip().upper()
        else:
            gen_value = gen_no_space.strip().upper()

        return 1.0 if true_value == gen_value else 0.0

    def get_failure_value(self) -> float:
        return 0.0

    def get_display_name(self) -> str:
        return "Binary Accuracy"


class PerimeterMetric(TaskMetric):
    """Metric for perimeter tasks: absolute error."""

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        # Use pattern that matches after closing paren
        true_match = re.search(r'\)=(\d+)', (prompt + true_completion).replace(' ', ''))
        gen_match = re.search(r'\)=(\d+)', generated.replace(' ', ''))

        if true_match and gen_match:
            true_perim = int(true_match.group(1))
            gen_perim = int(gen_match.group(1))
            return abs(gen_perim - true_perim)
        else:
            return self.get_failure_value()

    def get_failure_value(self) -> float:
        return 20000.0

    def get_display_name(self) -> str:
        return "Perimeter Error"


class NearestNeighborMetric(TaskMetric):
    """Metric for nearest neighbor tasks: Jaccard similarity."""

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        # Extract city IDs from completions
        true_cities = set(re.findall(r'c_\d+', true_completion.replace(' ', '')))

        # Extract answer after )= pattern ONLY
        gen_no_space = generated.replace(' ', '')
        pattern_match = re.search(r'\)=(.+)$', gen_no_space)
        if pattern_match:
            gen_completion = pattern_match.group(1)
            gen_cities = set(re.findall(r'c_\d+', gen_completion))
        else:
            gen_cities = set()

        expected_k = len(true_cities)

        if len(gen_cities) == expected_k and expected_k > 0:
            intersection = true_cities.intersection(gen_cities)
            union = true_cities.union(gen_cities)
            return len(intersection) / len(union) if union else 0.0
        else:
            return 0.0

    def get_failure_value(self) -> float:
        return 0.0

    def get_display_name(self) -> str:
        return "Jaccard Similarity"

    def format_for_print(self, value: float) -> str:
        return f"{value:.3f}"


class CenterMetric(TaskMetric):
    """Metric for center tasks: distance error to true center."""

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        cities_df = kwargs.get('cities_df')

        # Extract city IDs
        true_match = re.search(r'c_(\d+)', true_completion.replace(' ', ''))

        # Extract answer after the last '=' in generated text
        gen_no_space = generated.replace(' ', '')
        eq_pos = gen_no_space.rfind('=')
        if eq_pos != -1:
            gen_text = gen_no_space[eq_pos+1:]
        else:
            gen_text = gen_no_space
        gen_match = re.search(r'c_(\d+)', gen_text)

        if true_match and gen_match and cities_df is not None:
            true_city_id = int(true_match.group(1))
            gen_city_id = int(gen_match.group(1))

            try:
                true_city = cities_df[cities_df['city_id'] == true_city_id].iloc[0]
                gen_city = cities_df[cities_df['city_id'] == gen_city_id].iloc[0]

                # Calculate Euclidean distance
                distance = np.sqrt(
                    (gen_city['x'] - true_city['x'])**2 +
                    (gen_city['y'] - true_city['y'])**2
                )
                return float(distance)
            except (IndexError, KeyError):
                return self.get_failure_value()
        else:
            return self.get_failure_value()

    def get_failure_value(self) -> float:
        return math.sqrt(3600**2 + 1800**2)

    def get_display_name(self) -> str:
        return "Distance to True Center"


class CircleCountMetric(TaskMetric):
    """Metric for circle count tasks: absolute error in count."""

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        # Match )=COUNT pattern to get the actual count, not r=radius
        true_text = (prompt + true_completion).replace(' ', '')
        gen_text = generated.replace(' ', '')

        true_match = re.search(r'\)=(\d+)', true_text)
        gen_match = re.search(r'\)=(\d+)', gen_text)

        if true_match and gen_match:
            true_count = int(true_match.group(1))
            gen_count = int(gen_match.group(1))
            return abs(gen_count - true_count)
        else:
            return self.get_failure_value()

    def get_failure_value(self) -> float:
        return 1000.0

    def get_display_name(self) -> str:
        return "Count Error"


class RandRingMetric(TaskMetric):
    """Metric for random ring tasks: validity ratio × length penalty."""

    def calculate(self, prompt: str, true_completion: str, generated: str, **kwargs) -> float:
        cities_df = kwargs.get('cities_df')
        if cities_df is None:
            return self.get_failure_value()

        # Parse parameters from prompt
        param_match = re.search(
            r'randring\(c_(\d+),r=(\d+),R=(\d+),n=(\d+)\)',
            prompt.replace(' ', '')
        )

        if not param_match:
            return self.get_failure_value()

        center_id = int(param_match.group(1))
        r_min = int(param_match.group(2))
        r_max = int(param_match.group(3))
        expected_n = int(param_match.group(4))

        # Extract generated cities
        gen_cities = re.findall(r'c_(\d+)', generated.replace(' ', ''))

        if not gen_cities:
            return self.get_failure_value()

        # Get center city coordinates
        center = cities_df[cities_df['city_id'] == center_id].iloc[0]

        # Check each generated city
        valid_count = 0
        for city_id_str in gen_cities:
            city_id = int(city_id_str)
            city = cities_df[cities_df['city_id'] == city_id].iloc[0]
            dist = np.sqrt(
                (city['x'] - center['x'])**2 +
                (city['y'] - center['y'])**2
            )
            if r_min <= dist <= r_max:
                valid_count += 1

        # Calculate validity ratio
        validity_ratio = valid_count / len(gen_cities)

        # Calculate length penalty
        actual_n = len(gen_cities)
        n_diff = abs(actual_n - expected_n)
        length_penalty = np.exp(-n_diff / expected_n) if expected_n > 0 else 0.0

        # Combined score
        return validity_ratio * length_penalty

    def get_failure_value(self) -> float:
        return 0.0

    def get_display_name(self) -> str:
        return "Ring Validity Score"

    def format_for_print(self, value: float) -> str:
        return f"{value:.3f}"


# Registry of all task metrics
TASK_METRICS = {
    'distance': DistanceMetric(),
    'randomwalk': RandomWalkMetric(),
    'trianglearea': TriangleAreaMetric(),
    'angle': AngleMetric(),
    'compass': CompassMetric(),
    'crossing': BooleanMetric(),
    'inside': BooleanMetric(),
    'perimeter': PerimeterMetric(),
    'nearest_neighbor': NearestNeighborMetric(),
    'center': CenterMetric(),
    'circlecount': CircleCountMetric(),
    'randring': RandRingMetric(),
}


def get_metric(task_type: str) -> TaskMetric:
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
    return metric.get_failure_value()


def format_metric_for_display(task_type: str, value: float) -> str:
    """Format a metric value for display."""
    metric = get_metric(task_type)
    return metric.format_for_print(value)


def get_metric_display_name(task_type: str) -> str:
    """Get the display name for a metric."""
    metric = get_metric(task_type)
    return metric.get_display_name()