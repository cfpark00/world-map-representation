"""
Experiment registry - single source of truth for experiment metadata.
"""
from pathlib import Path
from typing import Dict, List


# Task mapping
TASK_NAMES = {
    1: 'distance',
    2: 'trianglearea',
    3: 'angle',
    4: 'compass',
    5: 'inside',
    6: 'perimeter',
    7: 'crossing',
}


def get_pt1_experiments(base_dir: Path = None) -> Dict[str, dict]:
    """Get PT1 (single-task) experiment registry."""
    if base_dir is None:
        base_dir = Path('/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1')

    experiments = {}

    for i in range(1, 8):
        exp_name = f'pt1-{i}'
        exp_dir = base_dir / 'data' / 'experiments' / exp_name

        if exp_dir.exists():
            experiments[exp_name] = {
                'name': exp_name,
                'base_dir': str(exp_dir),
                'type': 'single_task',
                'tasks': [TASK_NAMES[i]],
                'task_id': i,
                'seed': None,
            }

    return experiments


def get_pt2_experiments(base_dir: Path = None) -> Dict[str, dict]:
    """Get PT2 (two-task) experiment registry."""
    if base_dir is None:
        base_dir = Path('/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1')

    # PT2 task combinations (from research context)
    pt2_tasks = {
        1: ['distance', 'trianglearea'],
        2: ['angle', 'compass'],
        3: ['inside', 'perimeter'],
        4: ['crossing', 'distance'],
        5: ['trianglearea', 'angle'],
        6: ['compass', 'inside'],
        7: ['perimeter', 'crossing'],
        8: ['distance', 'angle'],  # Example, adjust based on actual experiments
    }

    experiments = {}

    for i in range(1, 9):
        exp_name = f'pt2-{i}'
        exp_dir = base_dir / 'data' / 'experiments' / exp_name

        if exp_dir.exists():
            experiments[exp_name] = {
                'name': exp_name,
                'base_dir': str(exp_dir),
                'type': 'two_task',
                'tasks': pt2_tasks.get(i, []),
                'task_id': i,
                'seed': None,
            }

    return experiments


def get_pt3_experiments(base_dir: Path = None) -> Dict[str, dict]:
    """Get PT3 (multi-task) experiment registry."""
    if base_dir is None:
        base_dir = Path('/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1')

    experiments = {}

    for i in range(1, 9):
        exp_name = f'pt3-{i}'
        exp_dir = base_dir / 'data' / 'experiments' / exp_name

        if exp_dir.exists():
            experiments[exp_name] = {
                'name': exp_name,
                'base_dir': str(exp_dir),
                'type': 'multi_task',
                'tasks': list(TASK_NAMES.values()),
                'task_id': i,
                'seed': None,
            }

    return experiments


def get_all_experiments(base_dir: Path = None) -> Dict[str, dict]:
    """Get all experiment registries."""
    experiments = {}
    experiments.update(get_pt1_experiments(base_dir))
    experiments.update(get_pt2_experiments(base_dir))
    experiments.update(get_pt3_experiments(base_dir))
    return experiments


def get_repr_path(exp_name: str, task: str, layer: int, prompt_type: str = 'firstcity_last_and_trans',
                  base_dir: Path = None) -> Path:
    """
    Get representation path for experiment, task, and layer.

    Args:
        exp_name: Experiment name (e.g., 'pt1-1', 'pt2-3')
        task: Task name (e.g., 'distance', 'trianglearea')
        layer: Layer number (3, 4, 5, or 6)
        prompt_type: Prompt type for representation extraction
        base_dir: Base directory (defaults to WM_1)

    Returns:
        Path to representations directory
    """
    if base_dir is None:
        base_dir = Path('/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1')

    exp_dir = base_dir / 'data' / 'experiments' / exp_name
    repr_dir = exp_dir / 'analysis_higher' / f'{task}_{prompt_type}_l{layer}' / 'representations'

    return repr_dir


def generate_experiment_pairs(experiment_type: str, base_dir: Path = None) -> List[tuple]:
    """
    Generate all unique pairs for a given experiment type.

    Args:
        experiment_type: 'pt1', 'pt2', or 'pt3'
        base_dir: Base directory

    Returns:
        List of (exp1, exp2) tuples
    """
    if experiment_type == 'pt1':
        experiments = get_pt1_experiments(base_dir)
    elif experiment_type == 'pt2':
        experiments = get_pt2_experiments(base_dir)
    elif experiment_type == 'pt3':
        experiments = get_pt3_experiments(base_dir)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    exp_names = sorted(experiments.keys())

    # Generate all unique pairs (including diagonal for self-comparison)
    pairs = []
    for i, exp1 in enumerate(exp_names):
        for exp2 in exp_names[i:]:
            pairs.append((exp1, exp2))

    return pairs
