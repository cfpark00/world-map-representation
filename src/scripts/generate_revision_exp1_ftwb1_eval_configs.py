#!/usr/bin/env python3
"""
Generate evaluation configs for revision/exp1 FTWB1 models.
Creates 15 eval configs per model (7 atlantis + 7 normal + 1 multi) × 21 models = 315 configs.
"""

from pathlib import Path
import yaml

# Base paths
CONFIG_BASE = Path("/configs/revision/exp1/eval")
EXPERIMENT_BASE = Path("data/experiments/revision/exp1")
DATASET_BASE = Path("data/datasets")

# All tasks to evaluate
TASKS = ["distance", "trianglearea", "angle", "compass", "inside", "perimeter", "crossing"]

# Seeds
SEEDS = [1, 2, 3]

def create_eval_config(experiment_name: str, task: str, seed: int, is_atlantis: bool = True) -> dict:
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

def create_multitask_eval_config(experiment_name: str, seed: int) -> dict:
    """Create multi-task evaluation config."""

    output_dir = EXPERIMENT_BASE / experiment_name / "evals" / "multi_task"

    config = {
        "checkpoints": "last",
        "cities_csv": "/data/datasets/cities/cities.csv",
        "dataset_path": "/data/datasets/multitask_pt1",
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

def main():
    print("Generating FTWB1 evaluation configs for revision/exp1...")

    configs_created = 0

    # Generate configs for each seed
    for seed in SEEDS:
        seed_dir = CONFIG_BASE / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # FTWB1 model configs (1-7)
        for exp_num in range(1, 8):
            ftwb1_exp = f"pt1_seed{seed}_ftwb1-{exp_num}"
            ftwb1_dir = seed_dir / f"ftwb1-{exp_num}"
            ftwb1_dir.mkdir(parents=True, exist_ok=True)

            # Atlantis tasks
            for task in TASKS:
                config = create_eval_config(ftwb1_exp, task, seed, is_atlantis=True)
                config_path = ftwb1_dir / f"atlantis_{task}.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                configs_created += 1

            # Normal tasks
            for task in TASKS:
                config = create_eval_config(ftwb1_exp, task, seed, is_atlantis=False)
                config_path = ftwb1_dir / f"{task}.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                configs_created += 1

            # Multi-task
            config = create_multitask_eval_config(ftwb1_exp, seed)
            config_path = ftwb1_dir / "multi_task.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            configs_created += 1

        print(f"  Created {7 * 15} configs for seed{seed} (ftwb1-1 through ftwb1-7)")

    print("\n" + "="*60)
    print(f"TOTAL: Created {configs_created} evaluation configs")
    print(f"Location: {CONFIG_BASE}")
    print("="*60)
    print(f"\nBreakdown: 21 models × 15 evaluations = 315 configs")

if __name__ == "__main__":
    main()
