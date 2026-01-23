#!/usr/bin/env python3
"""
Generate evaluation configs for revision/exp6 (scattered Atlantis experiment).
Creates configs for:
- Base model: pt1 (1 model)
- FTWB1 models: pt1_ftwb1-{1-7} (7 models)
- FTWB2 models: pt1_ftwb2-{1-21} (21 models)

Total: 29 models × 15 evaluations each = 435 configs

Key difference from other experiments: Uses exp6 datasets with scattered Atlantis.
"""

from pathlib import Path
import yaml

# Base paths
CONFIG_BASE = Path("/n/home12/cfpark00/datadir/WM_1/configs/revision/exp6/eval")
EXPERIMENT_BASE = Path("data/experiments/revision/exp6")
DATASET_BASE = Path("data/experiments/revision/exp6/datasets")

# All tasks to evaluate
TASKS = ["distance", "trianglearea", "angle", "compass", "inside", "perimeter", "crossing"]


def create_eval_config(experiment_name: str, task: str, is_atlantis: bool = True) -> dict:
    """Create evaluation config for a specific experiment and task."""

    if is_atlantis:
        dataset_path = DATASET_BASE / f"{task}_100k_atlantis_required"
        output_dir = EXPERIMENT_BASE / experiment_name / "evals" / f"atlantis_{task}"
    else:
        dataset_path = DATASET_BASE / f"{task}_1M_no_atlantis"
        output_dir = EXPERIMENT_BASE / experiment_name / "evals" / task

    config = {
        "checkpoints": "last",
        "dataset_path": str(dataset_path),
        "device": "cuda",
        "do_sample": False,
        "eval_batch_size": 512,
        "experiment_dir": str(EXPERIMENT_BASE / experiment_name),
        "max_generation_length": 256,
        "output_dir": str(output_dir),
        "plot_log_scale": False,
        "save_full_results": False,
        "seed": 42,
        "temperature": 0.0,
        "top_k": 1,
    }

    return config


def create_multitask_eval_config(experiment_name: str, is_base_model: bool = False) -> dict:
    """Create multi-task evaluation config.

    Args:
        experiment_name: Name of the experiment directory
        is_base_model: If True, use no-atlantis dataset (PT1 base was trained without Atlantis)
    """

    output_dir = EXPERIMENT_BASE / experiment_name / "evals" / "multi_task"

    if is_base_model:
        # PT1 base model: use original no-atlantis dataset
        cities_csv = "data/datasets/cities/cities.csv"
        dataset_path = "data/datasets/multitask_pt1"
    else:
        # FTWB models: use exp6 scattered atlantis dataset
        cities_csv = str(DATASET_BASE / "cities_scattered_atlantis" / "cities.csv")
        dataset_path = str(DATASET_BASE / "multitask_pt1_with_atlantis")

    config = {
        "checkpoints": "last",
        "cities_csv": cities_csv,
        "dataset_path": dataset_path,
        "device": "cuda",
        "do_sample": False,
        "eval_batch_size": 512,
        "experiment_dir": str(EXPERIMENT_BASE / experiment_name),
        "max_generation_length": 256,
        "output_dir": str(output_dir),
        "plot_log_scale": False,
        "save_full_results": False,
        "seed": 42,
        "temperature": 0.0,
        "top_k": 1,
    }

    return config


def create_configs_for_model(model_name: str, model_dir: Path, is_base_model: bool = False) -> int:
    """Create all eval configs for a single model. Returns number of configs created."""
    model_dir.mkdir(parents=True, exist_ok=True)
    configs_created = 0

    # Atlantis tasks
    for task in TASKS:
        config = create_eval_config(model_name, task, is_atlantis=True)
        config_path = model_dir / f"atlantis_{task}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        configs_created += 1

    # Normal tasks
    for task in TASKS:
        config = create_eval_config(model_name, task, is_atlantis=False)
        config_path = model_dir / f"{task}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        configs_created += 1

    # Multi-task
    config = create_multitask_eval_config(model_name, is_base_model=is_base_model)
    config_path = model_dir / "multi_task.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    configs_created += 1

    return configs_created


def main():
    print("Generating evaluation configs for revision/exp6 (scattered Atlantis)...")

    # Create base config directory
    CONFIG_BASE.mkdir(parents=True, exist_ok=True)

    configs_created = 0

    # Base model (pt1)
    base_dir = CONFIG_BASE / "pt1"
    n = create_configs_for_model("pt1", base_dir)
    configs_created += n
    print(f"  Created {n} configs for pt1 (base)")

    # FTWB1 models (1-7)
    for exp_num in range(1, 8):
        model_name = f"pt1_ftwb1-{exp_num}"
        model_dir = CONFIG_BASE / f"ftwb1-{exp_num}"
        n = create_configs_for_model(model_name, model_dir)
        configs_created += n
    print(f"  Created {7 * 15} configs for ftwb1-1 through ftwb1-7")

    # FTWB2 models (1-21)
    for exp_num in range(1, 22):
        model_name = f"pt1_ftwb2-{exp_num}"
        model_dir = CONFIG_BASE / f"ftwb2-{exp_num}"
        n = create_configs_for_model(model_name, model_dir)
        configs_created += n
    print(f"  Created {21 * 15} configs for ftwb2-1 through ftwb2-21")

    print("=" * 60)
    print(f"TOTAL: Created {configs_created} evaluation configs")
    print(f"Location: {CONFIG_BASE}")
    print("=" * 60)
    print(f"\nBreakdown:")
    print(f"  pt1 (base): 15 configs")
    print(f"  FTWB1 (7 models): {7 * 15} configs")
    print(f"  FTWB2 (21 models): {21 * 15} configs")
    print(f"  Total: {(1 + 7 + 21) * 15} configs")


if __name__ == "__main__":
    main()
